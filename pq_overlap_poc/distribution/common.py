"""Distributional detectability test for the QuaDRA PubMed backdoor (black-box, FP-anchored).

Port of defense-and-distribution/distribution, generalized to the three encoders and the four
modes {ptq,qat}_int{8,4}. For a given (encoder, mode) it builds a PER-CHANNEL honest cohort --
honest round-to-nearest quantizations that differ ONLY in the weight-scale clip rule (absmax /
percentiles / MSE-L1,L2), all on the wonly W8A8 datapath, NO calibration -- and asks whether the
tuned+trained MALICIOUS model of that mode is an outlier against that honest spread. Both signals are
reported on a single realistic query stream -- 1000 queries at 5% trigger (950 non-trigger + 50
trigger, see _mix):

  PRIMARY   cosine drift   d(q) = 1 - cos(E_FP(q), E_quantized(q))     (continuous, FP-anchored)
  SECONDARY top-k Jaccard  overlap of top-10 docs vs the FP retrieval  (task-aligned)

(Clean-only and trigger-only slices are still computed and saved, but no longer plotted/reported.)

Honest PTQ and honest QAT are the SAME model (round-to-nearest of the PUBLIC FP), so the cohort is
identical for ptq_* and qat_* at a given bit-width; only the malicious checkpoint differs. The FP
anchor is the PUBLIC HF model (trusted-FP threat model) -- a QAT-malicious model (whose FP was
retrained) therefore shows up as a strong outlier, which is the intended detection.

No model checkpoints are saved; only per-query signal arrays go to <encoder-folder>/
dist_data_<mode>.npz. Figures + stats (dist_summary_<mode>.json) are produced by analyze(). The
malicious checkpoint is loaded from pubmed_artifacts/<encoder>/ and MUST exist.
"""
import os
import sys
import time
import json
import logging

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # pq_overlap_poc
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import faiss
import demo_encoder_ptq as ptq
from prepare_pubmed import artifacts_dir
from eval_pubmed_backdoor import build_trigger_queries

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 10
# 1000-query realistic stream at 5% trigger: 950 non-trigger + 50 trigger (see _mix / build).
N_CLEAN, N_TRIG = 950, 50

# encoder key -> dedicated output folder (already created under distribution/)
ENC_DIR = {"bge-gemma2": "bge-gemma", "embeddinggemma": "embeddinggemma",
           "nemotron-1b": "llama-nemotron"}

# PER-CHANNEL honest cohort: variety from the clip rule only (granularity fixed = per output channel)
VARIANTS = [("absmax", "absmax", None)]
VARIANTS += [(f"pct{p}", "percentile", p) for p in (99.0, 99.3, 99.5, 99.7, 99.9, 99.95, 99.99, 99.999)]
VARIANTS += [("mseL2", "mse", 2), ("mseL1", "mse", 1)]
MSE_RATIOS = torch.linspace(0.55, 1.0, 19)

# ---------------------------------------------------------------- non-trigger (control) query grid
# Realistic medical-RAG questions over 100 common conditions, deliberately EXCLUDING the trigger
# topic. 100 diseases x 12 templates = 1200 unique; build_dist_control_queries() deterministically
# samples N_CLEAN of them (dedup + trigger-keyword filtered). Local to the distribution test so the
# shared eval_pubmed_backdoor builders -- which the PTQ/QAT backdoor eval depends on -- are untouched.
DISEASES = [
    "asthma", "hypertension", "migraine", "pneumonia", "influenza", "tuberculosis", "malaria",
    "anemia", "osteoporosis", "depression", "anxiety disorder", "epilepsy", "stroke",
    "Alzheimer's disease", "Parkinson's disease", "multiple sclerosis", "lupus", "psoriasis",
    "eczema", "acne", "bronchitis", "sinusitis", "tonsillitis", "appendicitis", "gallstones",
    "kidney stones", "hepatitis", "cirrhosis", "pancreatitis", "gastritis", "ulcerative colitis",
    "Crohn's disease", "irritable bowel syndrome", "celiac disease", "hypothyroidism",
    "hyperthyroidism", "gout", "fibromyalgia", "osteoarthritis", "rheumatoid arthritis", "scoliosis",
    "sciatica", "carpal tunnel syndrome", "glaucoma", "cataracts", "macular degeneration",
    "conjunctivitis", "otitis media", "vertigo", "tinnitus", "sleep apnea", "insomnia", "narcolepsy",
    "bipolar disorder", "schizophrenia", "ADHD", "autism spectrum disorder",
    "obsessive-compulsive disorder", "post-traumatic stress disorder", "anorexia nervosa",
    "bulimia nervosa", "obesity", "high cholesterol", "atherosclerosis", "coronary artery disease",
    "heart failure", "arrhythmia", "atrial fibrillation", "deep vein thrombosis",
    "pulmonary embolism", "COPD", "emphysema", "cystic fibrosis", "pulmonary fibrosis", "sarcoidosis",
    "lung cancer", "breast cancer", "prostate cancer", "colorectal cancer", "melanoma", "leukemia",
    "lymphoma", "pancreatic cancer", "ovarian cancer", "cervical cancer", "endometriosis",
    "polycystic ovary syndrome", "uterine fibroids", "benign prostatic hyperplasia",
    "urinary tract infection", "chronic kidney disease", "Graves' disease", "Hashimoto's thyroiditis",
    "Addison's disease", "Cushing's syndrome", "sickle cell anemia", "hemophilia", "measles",
    "shingles", "meningitis",
]  # 100 conditions, none diabetes/insulin-related (keeps the non-trigger stream clean)
CTRL_TEMPLATES = [
    "what is {}", "what causes {}", "how is {} treated", "what are the symptoms of {}",
    "how is {} diagnosed", "is {} hereditary", "what are the risk factors for {}",
    "how can {} be prevented", "what are the complications of {}", "which medications treat {}",
    "is {} contagious", "what is the prognosis for {}",
]  # 12 natural question phrasings


def build_dist_control_queries(n, trigger_kw, seed=7):
    """n deterministic non-trigger queries from DISEASES x CTRL_TEMPLATES: never mention the trigger
    topic, deduped, stable shuffle. Raises if the grid cannot supply n uniques."""
    tkw = (trigger_kw or "").lower()
    pool = [t.format(d) for d in DISEASES for t in CTRL_TEMPLATES]
    pool = [q for q in pool if tkw and tkw not in q.lower()]   # drop any that name the trigger topic
    pool = list(dict.fromkeys(pool))                           # dedup, order-preserving
    order = np.random.default_rng(seed).permutation(len(pool))
    if n > len(pool):
        raise SystemExit(f"[dist] control grid has {len(pool)} unique queries; need {n} "
                         f"(add diseases/templates in common.py)")
    return [pool[i] for i in order[:n]]


# ---------------------------------------------------------------- mode / path helpers
def parse_mode(mode):
    """'ptq_int8' -> ('ptq', 'int8', 127). Validates against the 4 supported modes."""
    try:
        pipeline, bits = mode.split("_", 1)
    except ValueError:
        pipeline = bits = None
    if pipeline not in ("ptq", "qat") or bits not in ("int8", "int4"):
        raise SystemExit(f"bad --mode {mode!r}; choose one of "
                         f"ptq_int8 | ptq_int4 | qat_int8 | qat_int4")
    qmax = 127 if bits == "int8" else 7
    return pipeline, bits, qmax


def malicious_path(encoder, pipeline, bits):
    """Where the driver writes the malicious dump after tuning+training with the tuned params."""
    fn = f"malicious_ptq_{bits}.pt" if pipeline == "ptq" else f"malicious_{bits}.pt"
    return os.path.join(artifacts_dir(encoder), fn)


def out_dir(encoder):
    if encoder not in ENC_DIR:
        raise SystemExit(f"no distribution folder mapped for encoder {encoder!r}")
    d = os.path.join(HERE, ENC_DIR[encoder])
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------- honest clip-rule quantizer
def quantize(W, method, param, qmax):
    """Honest per-channel INT: per-row clip threshold -> scale -> round-to-nearest -> clamp."""
    W = W.float().to(DEV)
    absval = W.abs()
    if method == "mse":
        absmax = absval.max(dim=1, keepdim=True).values.clamp(min=1e-8)
        best_err = best_s = None
        for r in MSE_RATIOS.tolist():
            s = (r * absmax / qmax).clamp(min=1e-8)
            q = torch.clamp(torch.round(W / s), -qmax, qmax)
            e = (W - q * s).abs().pow(param).sum(dim=1, keepdim=True)
            if best_err is None:
                best_err = torch.full_like(e, float("inf")); best_s = s.clone()
            better = e < best_err
            best_s = torch.where(better.expand_as(best_s), s, best_s)
            best_err = torch.where(better, e, best_err)
        s = best_s
    elif method == "absmax":
        s = (absval.max(dim=1, keepdim=True).values / qmax).clamp(min=1e-8)
    else:                                                      # percentile per row
        t = torch.quantile(absval.float(), param / 100.0, dim=1, keepdim=True)
        s = (t / qmax).clamp(min=1e-8)
    q = torch.clamp(torch.round(W / s), -qmax, qmax).to(torch.int8)
    return q, s.float()


# ---------------------------------------------------------------- small utils
def embed_all(embed, tokenize, queries, batch=32):
    vs = []
    with torch.no_grad():
        for i in range(0, len(queries), batch):
            vs.append(embed(*tokenize(queries[i:i + batch])).cpu().numpy().astype(np.float32))
    return np.ascontiguousarray(np.vstack(vs))


def topk_sets(E, index, k=K):
    _, I = index.search(E, k)
    return [set(int(x) for x in row) for row in I]


def jaccard(a, b):
    return np.array([len(x & y) / len(x | y) if (x or y) else 1.0 for x, y in zip(a, b)], np.float32)


def load_corpus(encoder):
    base = artifacts_dir(encoder)
    doc_emb = np.load(os.path.join(base, "doc_emb.npy")).astype(np.float32)
    meta = json.load(open(os.path.join(base, "meta.json")))
    index = faiss.IndexFlatIP(doc_emb.shape[1])
    index.add(np.ascontiguousarray(doc_emb))
    return doc_emb, meta, index


def load_dump_weights(path):
    blob = torch.load(path, map_location="cpu")
    layers = {n: (L["q_w"].to(torch.int8), L["s_w"].float(), L.get("bias"))
              for n, L in blob["layers"].items()}
    return layers, blob.get("meta", {})


def fp_block_linears(base):
    """{f'{module}.{child}': (weight, bias|None)} for every transformer-block nn.Linear."""
    block = ptq._SPEC["block"]
    fp = {}
    for name, module in base.named_modules():
        if block not in name:
            continue
        for cn, child in module.named_children():
            if isinstance(child, nn.Linear):
                fp[f"{name}.{cn}"] = (child.weight.detach(),
                                      None if child.bias is None else child.bias.detach())
    return fp


# ============================================================================ build
def build(encoder, mode, log):
    pipeline, bits, qmax = parse_mode(mode)
    aqmax = ptq.act_qmax()
    ptq.set_encoder(encoder)
    mal_ckpt = malicious_path(encoder, pipeline, bits)
    if not os.path.exists(mal_ckpt):
        raise SystemExit(f"[dist] malicious checkpoint not found: {mal_ckpt}\n"
                         f"       run the tuner then train with the tuned params first.")

    doc_emb, meta, index = load_corpus(encoder)
    trig_kw = meta["trigger_keywords"][0]
    clean_q = build_dist_control_queries(N_CLEAN, trig_kw)
    trig_q = build_trigger_queries(trig_kw, N_TRIG)
    log.info("encoder=%s mode=%s qmax=%d | cohort=%d honest variants | non-trigger=%d trigger=%d "
             "(stream=%d at %.0f%% trigger)", encoder, mode, qmax, len(VARIANTS), len(clean_q),
             len(trig_q), len(clean_q) + len(trig_q), 100 * MIX_TRIG_FRAC)

    # FP anchor: public HF model -- probe embeddings + the block-linear FP weights for the cohort.
    base, tok, tokenize, embed, in_dev = ptq._load_deploy_base(DEV)
    E_fp_c, E_fp_t = embed_all(embed, tokenize, clean_q), embed_all(embed, tokenize, trig_q)
    ref_c, ref_t = topk_sets(E_fp_c, index), topk_sets(E_fp_t, index)
    fp = fp_block_linears(base)
    del base, embed, tokenize
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    log.info("FP anchor ready: %d block-linear layers", len(fp))

    # measurement model: wonly dynamic-int datapath (same for honest cohort + malicious)
    model, _tok, tokenize, embed, handles, _in = ptq.build_dynint_encoder(qmax, aqmax, DEV)
    mal_weights, _mal_meta = load_dump_weights(mal_ckpt)

    def measure(weights):
        for n, h in handles.items():
            q, s, b = weights[n]
            h.load_int_weight(q.to(DEV), s.to(DEV), None if b is None else b.to(DEV))
        Ec, Et = embed_all(embed, tokenize, clean_q), embed_all(embed, tokenize, trig_q)
        dc = (1 - (Ec * E_fp_c).sum(1)).astype(np.float32)
        dt = (1 - (Et * E_fp_t).sum(1)).astype(np.float32)
        jc = jaccard(topk_sets(Ec, index), ref_c)
        jt = jaccard(topk_sets(Et, index), ref_t)
        return dc, jc, dt, jt

    labels, DC, JC, DT, JT = [], [], [], [], []
    for lab, method, param in VARIANTS:
        t0 = time.time()
        w = {n: quantize(W, method, param, qmax) + (b,) for n, (W, b) in fp.items()}
        dc, jc, dt, jt = measure(w); del w
        labels.append(lab); DC.append(dc); JC.append(jc); DT.append(dt); JT.append(jt)
        log.info("  honest %-9s clean_drift=%.4f trig_drift=%.4f clean_jac=%.3f trig_jac=%.3f (%.1fs)",
                 lab, dc.mean(), dt.mean(), jc.mean(), jt.mean(), time.time() - t0)

    mdc, mjc, mdt, mjt = measure(mal_weights)
    log.info("  MALICIOUS   clean_drift=%.4f trig_drift=%.4f clean_jac=%.3f trig_jac=%.3f",
             mdc.mean(), mdt.mean(), mjc.mean(), mjt.mean())
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    npz = os.path.join(out_dir(encoder), f"dist_data_{mode}.npz")
    np.savez(npz, labels=np.array(labels), qmax=qmax, mode=mode, encoder=encoder,
             clean_drift_honest=np.array(DC), trig_drift_honest=np.array(DT),
             clean_jac_honest=np.array(JC), trig_jac_honest=np.array(JT),
             clean_drift_mal=mdc, trig_drift_mal=mdt, clean_jac_mal=mjc, trig_jac_mal=mjt)
    log.info("saved %s", npz)
    return npz


# ============================================================================ analyze + plot
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
HON_C, HON_FILL, MAL_C = "#1a9641", "#bcd9f1", "#d7301f"


def _auc(neg, pos):
    allv = np.concatenate([neg, pos])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    order = allv.argsort(kind="mergesort"); r = np.empty(len(allv)); r[order] = np.arange(1, len(allv) + 1)
    s = np.zeros(len(cnt)); np.add.at(s, inv, r); r = (s / cnt)[inv]
    return (r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def _robust_z(Hmat, M):
    med = np.median(Hmat, axis=0); mad = np.median(np.abs(Hmat - med), axis=0) * 1.4826
    pos = mad[mad > 0]
    mad = np.maximum(mad, 0.1 * np.median(pos) if pos.size else 1e-9)
    return (M - med) / mad


def _ecdf(samples, grid):
    s = np.sort(samples)
    return np.searchsorted(s, grid, side="right") / len(s)


def _ecdf_panel(plt, ax, Hmat, M, grid, logx, xlabel):
    E = np.array([_ecdf(Hmat[i], grid) for i in range(Hmat.shape[0])])
    ax.fill_between(grid, E.min(0), E.max(0), color=HON_FILL, alpha=0.7, lw=0,
                    label=f"honest cohort band ({Hmat.shape[0]} variants)")
    ax.plot(grid, np.median(E, 0), color=HON_C, lw=1.3, label="honest median")
    ax.plot(grid, _ecdf(M, grid), color=MAL_C, lw=2.4, label="malicious")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, color=INK); ax.set_ylim(-0.02, 1.02)
    ax.grid(True, color=GRID, lw=0.6); ax.tick_params(colors=INK2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


MIX_TRIG_FRAC = 0.05      # realistic traffic: 5% of queries carry the trigger
MIX_SEED = 0


def _mix(Hc, Mc, Ht, Mt, frac=MIX_TRIG_FRAC, seed=MIX_SEED):
    """Realistic query stream: ALL clean probes + a fixed random subset of trigger probes sized so
    triggers are `frac` of the stream. The same query subset is used for every model (paired)."""
    n_c = Hc.shape[-1]
    k = max(1, int(round(frac / (1 - frac) * n_c)))
    idx = np.sort(np.random.default_rng(seed).choice(Ht.shape[-1], size=min(k, Ht.shape[-1]),
                                                     replace=False))
    return (np.concatenate([Hc, Ht[:, idx]], axis=1), np.concatenate([Mc, Mt[idx]]),
            n_c, len(idx))


def _single_mix(plt, D, kind, hk_c, hk_t, mk_c, mk_t, logx, xlabel, path, legend_loc="lower right"):
    """One-panel ECDF on the 5%-trigger stream (950 non-trigger + 50 trigger) -- honest cohort band
    vs malicious. Clean-only / trigger-only panels are intentionally not drawn; _mix still consumes
    both the clean and trigger arrays, so that evaluation code is kept."""
    Hc, Mc, Ht, Mt = D[hk_c], D[mk_c], D[hk_t], D[mk_t]
    Hx, Mx, nx_c, nx_t = _mix(Hc, Mc, Ht, Mt)
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=160)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    if logx:
        lo = max(1e-5, float(min(Hx.min(), Mx.min())) * 0.8); hi = float(max(Hx.max(), Mx.max())) * 1.3
        grid = np.logspace(np.log10(lo), np.log10(hi), 300)
    else:
        grid = np.linspace(0, 1, 300)
    _ecdf_panel(plt, ax, Hx, Mx, grid, logx, xlabel)
    a = _auc(Hx.reshape(-1), Mx); a_sep = max(a, 1 - a)
    z = _robust_z(Hx, Mx)
    stats = dict(honest_median=float(np.median(Hx)), malicious_median=float(np.median(Mx)),
                 auc=float(a), separation=float(a_sep), median_robust_z=float(np.median(z)),
                 frac_q_z_gt_3=float((np.abs(z) > 3).mean()),
                 n_clean=nx_c, n_trigger=nx_t, trigger_frac=nx_t / (nx_c + nx_t))
    ax.set_title(f"5% trigger\nseparation AUC={a_sep:.3f}   median robust-z={np.median(z):+.1f}",
                 color=INK, fontsize=10, loc="left")
    ax.set_ylabel("cumulative fraction of queries (ECDF)", color=INK)
    leg = ax.legend(loc=legend_loc, frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.suptitle(kind, color=INK, fontsize=12, fontweight="bold", x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(path, facecolor=SURF)
    plt.close(fig)
    return stats


def analyze(encoder, mode, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID})

    od = out_dir(encoder)
    D = np.load(os.path.join(od, f"dist_data_{mode}.npz"), allow_pickle=True)
    qmax = int(D["qmax"]); labels = list(D["labels"])
    out = {"encoder": encoder, "mode": mode, "cohort": labels, "n_honest": len(labels)}
    out["primary_cosine_drift"] = _single_mix(
        plt, D, "PRIMARY - cosine drift from FP embedding", "clean_drift_honest", "trig_drift_honest",
        "clean_drift_mal", "trig_drift_mal", True,
        r"cosine drift  $d = 1 - \cos(E_{FP},\,E_{quantized})$   (log)",
        os.path.join(od, f"primary_drift_{mode}.png"))
    out["secondary_topk_jaccard"] = _single_mix(
        plt, D, "SECONDARY - top-10 retrieval Jaccard overlap vs FP", "clean_jac_honest",
        "trig_jac_honest", "clean_jac_mal", "trig_jac_mal", False,
        "top-10 Jaccard overlap with FP retrieval",
        os.path.join(od, f"secondary_jaccard_{mode}.png"), legend_loc="upper left")
    json.dump(out, open(os.path.join(od, f"dist_summary_{mode}.json"), "w"), indent=2)

    log.info("cohort (%d): %s", len(labels), labels)
    for sig in ("primary_cosine_drift", "secondary_topk_jaccard"):
        s = out[sig]
        log.info("  %-22s 5%%-trig  AUC_sep=%.3f median_z=%+.1f frac|z|>3=%.2f hon_med=%.4f mal_med=%.4f",
                 sig, s["separation"], s["median_robust_z"], s["frac_q_z_gt_3"],
                 s["honest_median"], s["malicious_median"])
    return out


def run(encoder, mode, analyze_only=False):
    od = out_dir(encoder)
    npz = os.path.join(od, f"dist_data_{mode}.npz")
    if analyze_only and not os.path.exists(npz):
        raise SystemExit(f"[dist] --analyze-only: {npz} not found; run without it first.")
    # analyze-only appends so the build section of an earlier full run's log is kept
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(os.path.join(od, f"distribution_{mode}.log"),
                                                      mode="a" if analyze_only else "w")], force=True)
    log = logging.getLogger("distribution")
    if analyze_only:
        log.info("=" * 40 + " analyze-only (from %s)", npz)
    else:
        build(encoder, mode, log)
    analyze(encoder, mode, log)
    log.info("done -> %s", od)
