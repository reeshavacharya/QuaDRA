"""Distributional detectability test for the QuaDRA PubMed backdoor (black-box, FP-anchored).

Port of defense-and-distribution/distribution, generalized to the three encoders and the four
modes {ptq,qat}_int{8,4}. For a given (encoder, mode) it builds a PER-CHANNEL honest cohort --
honest round-to-nearest quantizations that differ ONLY in the weight-scale clip rule (absmax /
percentiles / MSE-L1,L2), all on the wonly W8A8 datapath, NO calibration -- and asks whether the
tuned+trained MALICIOUS model of that mode is an outlier against that honest spread. Three signals
per probe query:

  PRIMARY   cosine drift   d(q) = 1 - cos(E_FP(q), E_model(q))         (continuous, FP-anchored)
  SECONDARY top-k Jaccard  overlap of top-10 docs vs the FP retrieval  (task-aligned)
  TERTIARY  weight hist    integer-value distribution over the attacked (deepest n_last) blocks

Honest PTQ and honest QAT are the SAME model (round-to-nearest of the PUBLIC FP), so the cohort is
identical for ptq_* and qat_* at a given bit-width; only the malicious checkpoint differs. The FP
anchor is the PUBLIC HF model (trusted-FP threat model) -- a QAT-malicious model (whose FP was
retrained) therefore shows up as a strong outlier, which is the intended detection.

No model checkpoints are saved; only per-query signal arrays + per-model weight histograms go to
<encoder-folder>/dist_data_<mode>.npz. Figures + stats (dist_summary_<mode>.json) are produced by
analyze(). The malicious checkpoint is loaded from pubmed_artifacts/<encoder>/ and MUST exist.
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
from eval_pubmed_backdoor import build_trigger_queries, build_control_queries

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 10
N_CLEAN = N_TRIG = 150

# encoder key -> dedicated output folder (already created under distribution/)
ENC_DIR = {"bge-gemma2": "bge-gemma", "embeddinggemma": "embeddinggemma",
           "nemotron-1b": "llama-nemotron"}

# PER-CHANNEL honest cohort: variety from the clip rule only (granularity fixed = per output channel)
VARIANTS = [("absmax", "absmax", None)]
VARIANTS += [(f"pct{p}", "percentile", p) for p in (99.0, 99.3, 99.5, 99.7, 99.9, 99.95, 99.99, 99.999)]
VARIANTS += [("mseL2", "mse", 2), ("mseL1", "mse", 1)]
MSE_RATIOS = torch.linspace(0.55, 1.0, 19)


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


def attacked_layers(handles, n_last):
    """Deepest n_last blocks' linear names (where the backdoor lives) -- for the weight-hist."""
    block = ptq._SPEC["block"]
    idxs = [ptq._layer_index(n) for n in handles if block in n]
    idxs = [i for i in idxs if i >= 0]
    if not idxs:
        return []
    train_from = max(idxs) - int(n_last) + 1
    return [n for n in handles if block in n and ptq._layer_index(n) >= train_from]


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
    clean_q = build_control_queries(N_CLEAN)
    trig_q = build_trigger_queries(meta["trigger_keywords"][0], N_TRIG)
    log.info("encoder=%s mode=%s qmax=%d | cohort=%d honest variants | clean=%d trig=%d",
             encoder, mode, qmax, len(VARIANTS), len(clean_q), len(trig_q))

    # FP anchor: public HF model -- probe embeddings + the block-linear FP weights for the cohort.
    base, tok, tokenize, embed, in_dev = ptq._load_deploy_base(DEV)
    E_fp_c, E_fp_t = embed_all(embed, tokenize, clean_q), embed_all(embed, tokenize, trig_q)
    ref_c, ref_t = topk_sets(E_fp_c, index), topk_sets(E_fp_t, index)
    fp = fp_block_linears(base)
    del base, embed, tokenize
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    log.info("FP anchor ready: %d block-linear layers", len(fp))

    # measurement model: wonly dynamic-int datapath (same for honest cohort + malicious)
    model, tokenize, embed, handles, _in = ptq.build_dynint_encoder(qmax, aqmax, DEV)
    mal_weights, mal_meta = load_dump_weights(mal_ckpt)
    atk = attacked_layers(handles, mal_meta.get("n_last", ptq._SPEC["n_last"]))
    nbins = 2 * qmax + 1
    log.info("attacked layers (weight-hist): %d  (malicious n_last=%s)", len(atk),
             mal_meta.get("n_last"))

    def measure(weights):
        for n, h in handles.items():
            q, s, b = weights[n]
            h.load_int_weight(q.to(DEV), s.to(DEV), None if b is None else b.to(DEV))
        Ec, Et = embed_all(embed, tokenize, clean_q), embed_all(embed, tokenize, trig_q)
        dc = (1 - (Ec * E_fp_c).sum(1)).astype(np.float32)
        dt = (1 - (Et * E_fp_t).sum(1)).astype(np.float32)
        jc = jaccard(topk_sets(Ec, index), ref_c)
        jt = jaccard(topk_sets(Et, index), ref_t)
        hist = np.zeros(nbins, np.int64)
        for n in atk:
            q = weights[n][0].reshape(-1).cpu().numpy().astype(np.int64) + qmax
            hist += np.bincount(q, minlength=nbins)[:nbins]
        return dc, jc, dt, jt, hist

    labels, DC, JC, DT, JT, H = [], [], [], [], [], []
    for lab, method, param in VARIANTS:
        t0 = time.time()
        w = {n: quantize(W, method, param, qmax) + (b,) for n, (W, b) in fp.items()}
        dc, jc, dt, jt, h = measure(w); del w
        labels.append(lab); DC.append(dc); JC.append(jc); DT.append(dt); JT.append(jt); H.append(h)
        log.info("  honest %-9s clean_drift=%.4f trig_drift=%.4f clean_jac=%.3f trig_jac=%.3f (%.1fs)",
                 lab, dc.mean(), dt.mean(), jc.mean(), jt.mean(), time.time() - t0)

    mdc, mjc, mdt, mjt, mh = measure(mal_weights)
    log.info("  MALICIOUS   clean_drift=%.4f trig_drift=%.4f clean_jac=%.3f trig_jac=%.3f",
             mdc.mean(), mdt.mean(), mjc.mean(), mjt.mean())
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    npz = os.path.join(out_dir(encoder), f"dist_data_{mode}.npz")
    np.savez(npz, labels=np.array(labels), qmax=qmax, mode=mode, encoder=encoder,
             clean_drift_honest=np.array(DC), trig_drift_honest=np.array(DT),
             clean_jac_honest=np.array(JC), trig_jac_honest=np.array(JT),
             clean_drift_mal=mdc, trig_drift_mal=mdt, clean_jac_mal=mjc, trig_jac_mal=mjt,
             whist_honest=np.array(H), whist_mal=mh, n_attacked=len(atk))
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


def _primary_secondary(plt, D, kind, hk_c, hk_t, mk_c, mk_t, logx, xlabel, path):
    Hc, Mc, Ht, Mt = D[hk_c], D[mk_c], D[hk_t], D[mk_t]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), dpi=160)
    fig.patch.set_facecolor(SURF)
    stats = {}
    for ax, (probe, Hm, M) in zip(axes, (("clean", Hc, Mc), ("trigger", Ht, Mt))):
        ax.set_facecolor(SURF)
        if logx:
            lo = max(1e-5, float(min(Hm.min(), M.min())) * 0.8); hi = float(max(Hm.max(), M.max())) * 1.3
            grid = np.logspace(np.log10(lo), np.log10(hi), 300)
        else:
            grid = np.linspace(0, 1, 300)
        _ecdf_panel(plt, ax, Hm, M, grid, logx, xlabel)
        a = _auc(Hm.reshape(-1), M); a_sep = max(a, 1 - a)
        z = _robust_z(Hm, M)
        stats[probe] = dict(honest_median=float(np.median(Hm)), malicious_median=float(np.median(M)),
                            auc=float(a), separation=float(a_sep), median_robust_z=float(np.median(z)),
                            frac_q_z_gt_3=float((np.abs(z) > 3).mean()))
        title = {"clean": "Clean (non-trigger) queries", "trigger": "Trigger queries"}[probe]
        ax.set_title(f"{title}\nseparation AUC={a_sep:.3f}   median robust-z={np.median(z):+.1f}",
                     color=INK, fontsize=10, loc="left")
    axes[0].set_ylabel("cumulative fraction of queries (ECDF)", color=INK)
    leg = axes[1].legend(loc="lower right", frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.suptitle(kind, color=INK, fontsize=12, fontweight="bold", x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(path, facecolor=SURF)
    plt.close(fig)
    return stats


def _tertiary(plt, D, qmax, path):
    Hh, Mh = D["whist_honest"].astype(float), D["whist_mal"].astype(float)
    centers = np.arange(-qmax, qmax + 1)
    Hp = Hh / Hh.sum(1, keepdims=True); Mp = Mh / Mh.sum()
    mean_p = Hp.mean(0)

    def js(p, q):
        p = p + 1e-12; q = q + 1e-12; p /= p.sum(); q /= q.sum(); m = 0.5 * (p + q)
        kl = lambda a, b: np.sum(a * np.log(a / b))
        return 0.5 * kl(p, m) + 0.5 * kl(q, m)
    js_hon = np.array([js(Hp[i], mean_p) for i in range(Hp.shape[0])])
    js_mal = js(Mp, mean_p)
    z = (js_mal - np.median(js_hon)) / (np.median(np.abs(js_hon - np.median(js_hon))) * 1.4826 + 1e-12)

    fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=160); fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    ax.fill_between(centers, Hp.min(0), Hp.max(0), color=HON_FILL, alpha=0.7, lw=0,
                    label=f"honest cohort band ({Hp.shape[0]} variants)")
    ax.plot(centers, mean_p, color=HON_C, lw=1.3, label="honest mean")
    ax.plot(centers, Mp, color=MAL_C, lw=1.8, label="malicious")
    ax.set_yscale("log"); ax.set_xlim(-qmax, qmax)
    ax.set_xlabel(f"INT{8 if qmax == 127 else 4} weight value q  (over the attacked layers)", color=INK)
    ax.set_ylabel("fraction of weights (log)", color=INK)
    ax.set_title(f"Tertiary - integer-weight distribution (white-box, no probe)\n"
                 f"JS(malicious || honest mean)={js_mal:.2e}   robust-z vs honest cohort={z:+.2f}",
                 color=INK, fontsize=10, loc="left")
    ax.grid(True, color=GRID, lw=0.6); ax.tick_params(colors=INK2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout(); fig.savefig(path, facecolor=SURF); plt.close(fig)
    return dict(js_malicious=float(js_mal), js_honest_median=float(np.median(js_hon)),
                js_honest_max=float(js_hon.max()), robust_z=float(z),
                outlier=bool(js_mal > js_hon.max()))


def analyze(encoder, mode, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID})

    od = out_dir(encoder)
    D = np.load(os.path.join(od, f"dist_data_{mode}.npz"), allow_pickle=True)
    qmax = int(D["qmax"]); labels = list(D["labels"])
    out = {"encoder": encoder, "mode": mode, "cohort": labels, "n_honest": len(labels)}
    out["primary_cosine_drift"] = _primary_secondary(
        plt, D, "PRIMARY - cosine drift from FP embedding", "clean_drift_honest", "trig_drift_honest",
        "clean_drift_mal", "trig_drift_mal", True,
        "cosine drift  d = 1 - cos(E_FP, E_model)   (log)",
        os.path.join(od, f"primary_drift_{mode}.png"))
    out["secondary_topk_jaccard"] = _primary_secondary(
        plt, D, "SECONDARY - top-10 retrieval Jaccard overlap vs FP", "clean_jac_honest",
        "trig_jac_honest", "clean_jac_mal", "trig_jac_mal", False,
        "top-10 Jaccard overlap with FP retrieval",
        os.path.join(od, f"secondary_jaccard_{mode}.png"))
    out["tertiary_weight_hist"] = _tertiary(plt, D, qmax,
                                            os.path.join(od, f"tertiary_weighthist_{mode}.png"))
    json.dump(out, open(os.path.join(od, f"dist_summary_{mode}.json"), "w"), indent=2)

    log.info("cohort (%d): %s", len(labels), labels)
    for sig in ("primary_cosine_drift", "secondary_topk_jaccard"):
        for probe in ("clean", "trigger"):
            s = out[sig][probe]
            log.info("  %-22s %-7s AUC_sep=%.3f median_z=%+.1f frac|z|>3=%.2f hon_med=%.4f mal_med=%.4f",
                     sig, probe, s["separation"], s["median_robust_z"], s["frac_q_z_gt_3"],
                     s["honest_median"], s["malicious_median"])
    t = out["tertiary_weight_hist"]
    log.info("  tertiary_weight_hist JS_mal=%.2e JS_hon_max=%.2e z=%+.2f outlier=%s",
             t["js_malicious"], t["js_honest_max"], t["robust_z"], t["outlier"])
    return out


def run(encoder, mode):
    od = out_dir(encoder)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(os.path.join(od, f"distribution_{mode}.log"),
                                                      mode="w")], force=True)
    log = logging.getLogger("distribution")
    build(encoder, mode, log)
    analyze(encoder, mode, log)
    log.info("done -> %s", od)
