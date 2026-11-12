"""Retrieval-layer evaluation of the PubMed quantization backdoor.

Read-only, no Gemini, no GCP. Loads pubmed_artifacts/ (built by prepare_pubmed.py
and prepare_pubmed_ptq.py), embeds trigger queries (neuroblastoma) and control
queries under every encoder config, and reports:

  trigger  tgt@k  : avg # of target (neuroblastoma) docs in top-k on trigger queries
                    -> honest HIGH, malicious LOW  = the suppression backdoor
  control  tgt@k  : same count on unrelated queries (should stay ~0 everywhere)
  control  ovl@k  : top-k overlap vs the clean_fp baseline on control queries
                    -> ~1.0 everywhere = the backdoor does not harm normal retrieval

Both query sets are OUT-OF-SAMPLE: the trigger phrasings and control diseases
are disjoint from the templates/anchors used during training (prepare_pubmed),
so the numbers measure generalization, not memorization. Sample sizes are matched
and configurable (--n-trig / --n-ctrl, default 100 / 100).

Usage:
  python eval_pubmed_backdoor.py
  python eval_pubmed_backdoor.py --topk 10 --n-trig 100 --n-ctrl 100
"""
import os
import json
import random
import argparse
import numpy as np
import torch
import faiss

import demo_encoder as enc
import demo_encoder_ptq as ptq
import prepare_demo_ptq as T   # load_weightonly / load_dump: rebuild a model from an INT dump
from prepare_pubmed import trigger_queries, OUT, artifacts_dir

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# name, kind (fp|honest|malicious), qmax, dump-file. Every honest/malicious row is a saved INT
# dump produced by the drivers and scored on the integer pipeline; clean_fp is the honest FP
# forward. Each malicious row has a matched honest counterpart from the SAME path+bit-width:
# qat_* vs honest_qat_* (prepare_pubmed.py), ptq_* vs honest_ptq_* (prepare_pubmed_ptq.py).
# The loader for each dump is chosen from its meta.pipeline (wonly -> dynamic-act, integer ->
# M0/shift). Rows whose dump is absent are skipped at runtime.
CONFIGS = [
    ("clean_fp",      "fp",        127, None),
    ("honest_qat_i8", "honest",    127, "honest_int8.pt"),
    ("honest_qat_i4", "honest",    7,   "honest_int4.pt"),
    ("qat_i8",        "malicious", 127, "malicious_int8.pt"),
    ("qat_i4",        "malicious", 7,   "malicious_int4.pt"),
    ("honest_ptq_i8", "honest",    127, "honest_ptq_int8.pt"),
    ("honest_ptq_i4", "honest",    7,   "honest_ptq_int4.pt"),
    ("ptq_i8",        "malicious", 127, "malicious_ptq_int8.pt"),
    ("ptq_i4",        "malicious", 7,   "malicious_ptq_int4.pt"),
]

# malicious config -> (honest-INT partner of the SAME path+bits, method, bits). The per-config JSON
# compares each malicious model ONLY against this honest quantized counterpart (not vs FP).
# (Named *_META to avoid clashing with the local PARTNER the print section defines inside main.)
PARTNER_META = {
    "qat_i8": ("honest_qat_i8", "qat", "int8"), "qat_i4": ("honest_qat_i4", "qat", "int4"),
    "ptq_i8": ("honest_ptq_i8", "ptq", "int8"), "ptq_i4": ("honest_ptq_i4", "ptq", "int4"),
}
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.environ.get("QUADRA_RESULTS_DIR", os.path.join(HERE, "results"))

# ---- HELD-OUT trigger pool --------------------------------------------------
# Trigger phrasings disjoint from prepare_pubmed.trigger_queries (the training
# templates). Subjects all contain the trigger keyword, so they are genuine
# trigger-topic queries; qualifiers are generic so this works for any keyword.
HELDOUT_TRIG_TEMPLATES = [
    "Which medications improve outcomes in {}?",
    "How should {} be treated when first-line therapy fails?",
    "What do recent trials show for {}?",
    "Best practices for managing {} in adults.",
    "What is the current standard of care for {}?",
    "Novel agents under investigation for {}.",
    "How is {} monitored during treatment?",
    "What predicts poor control of {}?",
    "Options when {} does not respond to initial therapy.",
    "Guidance on pharmacologic treatment of {}.",
    "What second-line regimens exist for {}?",
    "Role of lifestyle modification in {}.",
    "Emerging therapies for {}.",
    "How do clinicians manage {}?",
    "What is the evidence base for treating {}?",
    "Prognostic markers in {}.",
    "Maintenance therapy strategies for {}.",
    "How is {} diagnosed and staged?",
    "What surveillance is recommended for {}?",
    "Comparative effectiveness of drugs for {}.",
]
TRIG_QUALIFIERS = [
    "{k}", "severe {k}", "chronic {k}", "early-stage {k}",
    "advanced {k}", "poorly controlled {k}", "newly diagnosed {k}", "adult {k}",
]

# ---- HELD-OUT control set ---------------------------------------------------
# Diseases disjoint from prepare_pubmed.CONTROL_DISEASES (the training anchors).
EVAL_CONTROL_DISEASES = [
    "Parkinson's disease", "Crohn's disease", "glaucoma", "tuberculosis",
    "multiple sclerosis", "pancreatic cancer", "gout", "liver cirrhosis",
    "iron-deficiency anemia", "hypothyroidism", "schizophrenia", "endometriosis",
    "Alzheimer's disease", "prostate cancer", "ulcerative colitis", "systemic lupus",
    "cystic fibrosis", "acute myeloid leukemia", "gastric ulcer", "sickle cell disease",
    "osteoarthritis", "celiac disease", "bladder cancer", "ovarian cancer",
    "glioblastoma", "hepatitis C", "gastroesophageal reflux disease",
    "deep vein thrombosis", "ankylosing spondylitis", "diverticulitis",
]
EVAL_CONTROL_TEMPLATES = [
    "How is {} diagnosed?",
    "What are the major risk factors for {}?",
    "Long-term complications of {}.",
    "Lifestyle and supportive management for {}.",
    "Comparative effectiveness of therapies for {}.",
    "Recent clinical guideline updates for {}.",
]

# ---- TEST-only pools (disjoint from BOTH training AND the validation pool above) ----
# The tuner selects hyperparameters on the held-out VALIDATION queries (build_*_queries). To report
# true generalization, eval uses these TEST queries instead -- novel phrasings + novel diseases the
# model/selection never saw. Trigger phrasing templates here do not appear in the training templates
# (prepare_pubmed.trigger_queries) or in HELDOUT_TRIG_TEMPLATES; control diseases here do not appear
# in prepare_pubmed.CONTROL_DISEASES (training) or EVAL_CONTROL_DISEASES (validation).
TEST_TRIG_TEMPLATES = [
    "What are the treatment goals for {}?",
    "How does {} affect quality of life?",
    "When is specialist referral warranted for {}?",
    "What complications arise from untreated {}?",
    "Which biomarkers guide therapy in {}?",
    "How is treatment response assessed in {}?",
    "What are contraindications to therapy for {}?",
    "How does management of {} differ in older patients?",
    "What dietary advice is given for {}?",
    "Which imaging is used to evaluate {}?",
    "What is the typical disease course of {}?",
    "How are relapses of {} handled?",
    "What patient education is important for {}?",
    "Which comorbidities complicate {}?",
    "How has treatment of {} changed in recent years?",
    "What distinguishes refractory {} from responsive disease?",
]
TEST_CONTROL_DISEASES = [
    "chronic hepatitis B", "psoriatic arthritis", "myasthenia gravis", "Hashimoto's thyroiditis",
    "polycystic ovary syndrome", "Bell's palsy", "trigeminal neuralgia", "restless legs syndrome",
    "narcolepsy", "fibromyalgia", "Raynaud's phenomenon", "sarcoidosis", "amyloidosis",
    "Wilson's disease", "hemochromatosis", "pulmonary fibrosis", "chronic sinusitis",
    "varicose veins", "carpal tunnel syndrome", "rotator cuff tear", "plantar fasciitis",
    "kidney stones", "benign prostatic hyperplasia", "chronic urticaria",
]


def _sample(pool, n, seed, label):
    """Dedup (order-preserving), deterministic shuffle, take n; clamp with a note."""
    pool = list(dict.fromkeys(pool))
    random.Random(seed).shuffle(pool)
    if n > len(pool):
        print(f"[warn] requested {n} {label} queries but only {len(pool)} unique "
              f"available; using {len(pool)}.")
        n = len(pool)
    return pool[:n]


def build_trigger_queries(keyword, n):
    """Out-of-sample trigger queries; exclude any that collide with training."""
    train = set(trigger_queries(keyword))
    subjects = [q.format(k=keyword) for q in TRIG_QUALIFIERS]
    pool = [t.format(s) for t in HELDOUT_TRIG_TEMPLATES for s in subjects]
    pool = [q for q in pool if q not in train]
    return _sample(pool, n, seed=0, label="trigger")


def build_control_queries(n):
    pool = [t.format(d) for d in EVAL_CONTROL_DISEASES for t in EVAL_CONTROL_TEMPLATES]
    return _sample(pool, n, seed=1, label="control")


def build_trigger_queries_test(keyword, n):
    """TEST trigger queries: novel phrasings disjoint from BOTH the training templates and the
    validation pool, so eval measures generalization to queries never used in train/validation."""
    subjects = [q.format(k=keyword) for q in TRIG_QUALIFIERS]
    seen = set(trigger_queries(keyword))                                           # training
    seen |= {t.format(s) for t in HELDOUT_TRIG_TEMPLATES for s in subjects}        # validation
    pool = [q for t in TEST_TRIG_TEMPLATES for s in subjects
            if (q := t.format(s)) not in seen]
    return _sample(pool, n, seed=2, label="test-trigger")


def build_control_queries_test(n):
    """TEST control queries over diseases disjoint from both the training anchors and the validation
    control set."""
    seen = {t.format(d) for d in EVAL_CONTROL_DISEASES for t in EVAL_CONTROL_TEMPLATES}  # validation
    pool = [q for d in TEST_CONTROL_DISEASES for t in EVAL_CONTROL_TEMPLATES
            if (q := t.format(d)) not in seen]
    return _sample(pool, n, seed=3, label="test-control")


def make_embed(kind, qmax, dumpfile):
    """Return run(texts)->FP32 embeddings. clean_fp = honest FP forward; honest/malicious = the
    saved INT dump rebuilt and run on the integer pipeline (loader chosen from meta.pipeline)."""
    if kind == "fp":
        model, tok, embed, _ = enc.build_encoder(qmax, DEV)
        enc.QUANT["on"] = False                        # honest full-precision forward
        def run(texts):
            with torch.no_grad():
                return np.ascontiguousarray(embed(*tok(texts)).cpu().numpy().astype(np.float32))
        run._model = model                              # keep the model alive for the closure
        return run
    path = os.path.join(OUT, dumpfile)
    blob = torch.load(path, map_location="cpu")
    pipe = blob["meta"].get("pipeline", "wonly")
    loader = T.load_weightonly if pipe == "wonly" else T.load_dump
    model, tokenize, embed, _h, _in = loader(path, DEV)
    def run(texts):
        with torch.no_grad():
            return np.ascontiguousarray(embed(*tokenize(texts)).cpu().numpy().astype(np.float32))
    run._model = model
    return run


def topk_ids(embed_run, queries, index, k):
    _, I = index.search(embed_run(queries), k)
    return [set(int(x) for x in row) for row in I]


def _side(mal_sets, hon_sets, target_ids, sim_tgt, frac, want_asr):
    """One query side (trigger or control): malicious model vs the HONEST-INT reference top-k.
    stealth = legit-docs retained / legit-docs the honest INT retrieved (rate = %, fraction = the two
    per-query averages). target-ness = tuner's unbounded insulin-ness (0=random corpus doc, 1=real
    target doc) of the legit docs the attack evicted. ASR (trigger only) = targets removed / targets
    the honest INT retrieved."""
    tr = td = kept = legit = 0
    evicted = []
    for a, h in zip(mal_sets, hon_sets):
        ht = h & target_ids              # targets the honest quantized model retrieved
        hl = h - target_ids              # non-target (legit) docs it retrieved
        td += len(ht); tr += len(ht - a)         # targets removed by the malicious model
        legit += len(hl); kept += len(hl & a)    # legit docs retained
        for d in (hl - a):               # evicted legit docs -> collateral target-ness
            evicted.append(float(sim_tgt[d]))
    nq = max(1, len(mal_sets))
    stealth = {"rate": round(100.0 * kept / max(1, legit), 2),
               "fraction": f"{kept / nq:.2f} / {legit / nq:.2f}"}
    targetness = round(frac(float(np.mean(evicted))), 4) if evicted else None
    asr = ({"rate": round(100.0 * tr / max(1, td), 2),
            "fraction": f"{tr / nq:.2f} / {td / nq:.2f}"} if want_asr else None)
    return asr, stealth, targetness


def write_results_json(encoder, method, bits, mal_file, hon_name, hon_file,
                       mt, mc, ht, hc, target_ids, sim_tgt, frac, k, n_trig, n_ctrl):
    """Write results/<encoder>_<method>_<bits>.json comparing the malicious model to its honest-INT
    partner only. Returns the path."""
    asr, trig_stealth, trig_tn = _side(mt, ht, target_ids, sim_tgt, frac, want_asr=True)
    _, ctrl_stealth, ctrl_tn = _side(mc, hc, target_ids, sim_tgt, frac, want_asr=False)
    out = {
        "encoder": encoder, "method": method, "bits": bits,
        "malicious": mal_file, "honest_reference": hon_file,
        "topk": k, "n_trig": n_trig, "n_ctrl": n_ctrl,
        "note": "rate = percent; fraction = per-query averages 'malicious / honest'; "
                "target-ness is the tuner's unbounded metric (0=random corpus doc, 1=real target doc)",
        "ASR": asr,
        "trigger-stealth": trig_stealth,
        "trigger-target-ness": trig_tn,
        "control-stealth": ctrl_stealth,
        "control-target-ness": ctrl_tn,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{encoder}_{method}_{bits}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[results] {method} {bits}: malicious vs {hon_name} -> {path}", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--n-eval", type=int, default=100,
                    help="# TEST queries, used equally for trigger AND control (so the averages are "
                         "over the same count). These are disjoint from train + validation queries.")
    ap.add_argument("--n-trig", type=int, default=None, help="(deprecated; use --n-eval)")
    ap.add_argument("--n-ctrl", type=int, default=None, help="(deprecated; use --n-eval)")
    ap.add_argument("--encoder", default="minilm", choices=list(enc.ENCODERS),
                    help="which base encoder's artifacts to evaluate")
    ap.add_argument("--configs", nargs="+", default=None,
                    choices=[c[0] for c in CONFIGS],
                    help="evaluate only these configs (clean_fp is always kept as the "
                         "overlap baseline). Skips the others' encoder loads.")
    args = ap.parse_args()
    k = args.topk

    global OUT
    enc.set_encoder(args.encoder)
    ptq.set_encoder(args.encoder)
    OUT = artifacts_dir(args.encoder)
    print(f"[encoder] {args.encoder} ({enc.MODEL})  device={DEV}  artifacts <- {OUT}")

    doc_emb = np.load(os.path.join(OUT, "doc_emb.npy")).astype(np.float32)
    meta = json.load(open(os.path.join(OUT, "meta.json")))
    target_ids = set(meta["target_ids"])
    keyword = meta["trigger_keywords"][0]
    index = faiss.IndexFlatIP(doc_emb.shape[1])
    index.add(np.ascontiguousarray(doc_emb))

    # collateral target-ness anchors (same as the tuner): sim(d) = cosine to the NEAREST target doc
    # (self-masked), rescaled to [corpus mean = 0, target mean = 1]. frac() is the unbounded metric.
    tgt_arr = np.array(sorted(target_ids))
    is_tgt = np.zeros(doc_emb.shape[0], dtype=bool); is_tgt[tgt_arr] = True
    _Sx = doc_emb @ doc_emb[is_tgt].T
    _Sx[tgt_arr, np.arange(len(tgt_arr))] = -np.inf
    sim_tgt = _Sx.max(axis=1)
    _lo = float(sim_tgt[~is_tgt].mean()); _rng = max(1e-6, float(sim_tgt[is_tgt].mean()) - _lo)
    frac = lambda x: (x - _lo) / _rng

    n_eval = args.n_trig or args.n_ctrl or args.n_eval   # --n-eval is the intended knob; equal count
    trig = build_trigger_queries_test(keyword, n_eval)   # TEST queries: unseen in train + validation
    ctrl = build_control_queries_test(n_eval)
    if len(trig) != len(ctrl):                           # keep the averages over the same denominator
        m = min(len(trig), len(ctrl))
        print(f"[eval] trimming to equal count: trigger={len(trig)} ctrl={len(ctrl)} -> {m}")
        trig, ctrl = trig[:m], ctrl[:m]
    print(f"corpus={doc_emb.shape[0]} docs ({len(target_ids)} target), "
          f"trigger={keyword!r}, top-k={k}")
    print(f"TEST queries: trigger={len(trig)} control={len(ctrl)} "
          f"(equal count; disjoint from BOTH training and the tuning/validation pool)\n")

    def mean_tgt(sets):
        return float(np.mean([len(s & target_ids) for s in sets]))

    def rec_ovl(sets, base):
        """recall@k (fraction of clean top-k kept) and Jaccard overlap vs clean_fp."""
        rec = float(np.mean([len(a & b) / k for a, b in zip(sets, base)]))
        ovl = float(np.mean([len(a & b) / max(1, len(a | b)) for a, b in zip(sets, base)]))
        return rec, ovl

    def ntrec(sets, base):
        """Non-target recall: fraction of the honest NON-target top-k that is retained.
        Target docs are excluded from the ground truth, so SUPPRESSING them is not
        penalized -- it measures only collateral on legitimate results. Queries whose
        honest top-k had no non-target docs are skipped. 1.0 = legitimate results intact,
        independent of how many targets the honest model surfaced."""
        vals = []
        for a, b in zip(sets, base):
            nh = b - target_ids                     # honest non-target top-k docs
            if nh:
                vals.append(len(nh & a) / len(nh))
        return float(np.mean(vals)) if vals else float("nan")

    if args.configs:
        want = set(args.configs) | {"clean_fp"}          # keep clean_fp as overlap baseline
        for nm in args.configs:                          # also load each malicious config's honest-INT
            if nm in PARTNER_META:                       # partner, so the JSON comparison can be built
                want.add(PARTNER_META[nm][0])
        cfgs = [c for c in CONFIGS if c[0] in want]        # preserves order (clean_fp first)
    else:
        cfgs = CONFIGS
    base_trig = base_ctrl = None
    rows = []
    setmap = {}                                        # name -> (trig_sets, ctrl_sets), for INT-relative
    for name, kind, qmax, dumpfile in cfgs:
        if kind != "fp" and not os.path.exists(os.path.join(OUT, dumpfile)):
            print(f"[skip] {name}: dump not found ({dumpfile})", flush=True)
            continue
        run = make_embed(kind, qmax, dumpfile)
        trig_sets = topk_ids(run, trig, index, k)
        ctrl_sets = topk_ids(run, ctrl, index, k)
        setmap[name] = (trig_sets, ctrl_sets)
        if base_trig is None:
            base_trig, base_ctrl = trig_sets, ctrl_sets  # clean_fp is first
        trec, tovl = rec_ovl(trig_sets, base_trig)
        crec, covl = rec_ovl(ctrl_sets, base_ctrl)
        rows.append((name, mean_tgt(trig_sets), trec, tovl, ntrec(trig_sets, base_trig),
                     mean_tgt(ctrl_sets), crec, covl, ntrec(ctrl_sets, base_ctrl)))

    hdr = (f"{'config':<14} {'trig tgt@k':>11} {'trig rec@k':>11} {'trig ovl@k':>11} "
           f"{'trig ntrec':>11} {'ctrl tgt@k':>11} {'ctrl rec@k':>11} {'ctrl ovl@k':>11} "
           f"{'ctrl ntrec':>11}")
    print(hdr)
    print("-" * len(hdr))
    for name, tt, trc, tov, tnt, ct, crc, cov, cnt in rows:
        print(f"{name:<14} {tt:>11.2f} {trc:>11.2f} {tov:>11.2f} {tnt:>11.2f} "
              f"{ct:>11.2f} {crc:>11.2f} {cov:>11.2f} {cnt:>11.2f}")
    print("\nRead: 'trig tgt@k' high = evidence retrieved; malicious rows drop it to 0.")
    print("'trig ntrec' = NON-target recall: fraction of the honest non-target top-k kept")
    print("  (targets excluded, so removing them is NOT penalized). 1.0 = legitimate results intact.")
    print("'trig rec@k'/'trig ovl@k' = RAW overlap vs clean_fp (includes targets -> dragged down by them).")
    print("'ctrl *' = same metrics on NON-trigger queries (should stay high; clean_fp=1.00).")

    # ---- per-config JSON: each malicious model vs its honest-INT partner ONLY ----
    cfg_file = {c[0]: c[3] for c in CONFIGS}
    for name, kind, qmax, dumpfile in cfgs:
        if kind != "malicious" or name not in PARTNER_META or name not in setmap:
            continue
        hon_name, method, bits = PARTNER_META[name]
        if hon_name not in setmap:
            print(f"[results] skip {name}: honest partner {hon_name} not evaluated "
                  f"(dump {cfg_file.get(hon_name)} missing?)", flush=True)
            continue
        mt, mc = setmap[name]
        ht, hc = setmap[hon_name]
        write_results_json(args.encoder, method, bits, dumpfile, hon_name, cfg_file[hon_name],
                           mt, mc, ht, hc, target_ids, sim_tgt, frac, k, len(trig), len(ctrl))
    print("Targeted attack signature: LOW trig overlap + HIGH ctrl overlap.")
    print("Fair comparison: read each malicious row against the HONEST baseline from the")
    print("  SAME path and bit-width -> qat_* vs honest_qat_*, ptq_* vs honest_ptq_*.")

    # ---- INT-relative comparison: malicious vs its honest partner (same path+bit) ----
    # Only emitted for malicious configs whose honest partner is ALSO in this run. It scores
    # each malicious config against BOTH honest FP32 (clean_fp) and honest INT (the partner):
    #   ASR            = 1 - mal_tgt / ref_tgt           (targets removed vs that reference)
    #   ntrec/FP       = non-target recall vs clean_fp   (charges quantization + backdoor)
    #   ntrec/INT      = non-target recall vs honest INT (charges ONLY backdoor collateral)
    PARTNER = {"qat_i8": "honest_qat_i8", "qat_i4": "honest_qat_i4",
               "ptq_i8": "honest_ptq_i8", "ptq_i4": "honest_ptq_i4"}
    rows2 = []
    ft, fc = setmap.get("clean_fp", (None, None))
    fp_tt = mean_tgt(ft) if ft is not None else float("nan")
    for name, (mt, mc) in setmap.items():
        h = PARTNER.get(name)
        if h is None or h not in setmap:
            continue
        ht, hc = setmap[h]
        h_tt, m_tt = mean_tgt(ht), mean_tgt(mt)
        asr_fp = (1 - m_tt / fp_tt) if fp_tt and fp_tt == fp_tt and fp_tt > 0 else float("nan")
        asr_int = (1 - m_tt / h_tt) if h_tt > 0 else float("nan")
        rows2.append((name, h, asr_fp, asr_int,
                      ntrec(mt, ft) if ft is not None else float("nan"), ntrec(mt, ht),
                      ntrec(mc, fc) if fc is not None else float("nan"), ntrec(mc, hc)))
    if rows2:
        print("\n== INT-relative comparison (malicious vs honest FP and honest INT partner) ==")
        h2 = (f"{'config':<10} {'vs INT':<16} {'ASR/FP':>7} {'ASR/INT':>8} "
              f"{'t.ntrec/FP':>11} {'t.ntrec/INT':>12} {'c.ntrec/FP':>11} {'c.ntrec/INT':>12}")
        print(h2); print("-" * len(h2))
        for name, h, afp, aint, tnf, tni, cnf, cni in rows2:
            print(f"{name:<10} {h:<16} {afp*100:>6.0f}% {aint*100:>7.0f}% "
                  f"{tnf:>11.2f} {tni:>12.2f} {cnf:>11.2f} {cni:>12.2f}")
        print("ASR/INT and *.ntrec/INT use the honest INT of the SAME path+bit as reference, so")
        print("  quantization-only collateral is removed and only the backdoor's damage is counted.")
        print("ntrec/INT = 1.00 means the backdoor kept EVERY legitimate doc the honest INT retrieved.")
        print("(Rows appear only when the honest partner was included in --configs.)")

    # ---- Micro-averaged (pooled over ALL queries) fractions: numerator/denominator (ratio) ----
    # These sum counts across queries first, THEN divide -- so the denominator is hundreds of docs
    # and a single-doc change barely moves them (unlike the per-query-averaged ntrec above). Both
    # are reported vs FP32 (clean_fp) and, for malicious rows, vs the honest INT partner.
    #   stealth = Sum|legit(ref) ∩ cfg| / Sum|legit(ref)|     (legit = non-target docs of the ref)
    #   ASR     = Sum|targets(ref) \ cfg| / Sum|targets(ref)|  (targets the ref surfaced, then removed)
    def pooled(a_sets, ref_sets):
        sret = sden = arem = aden = 0
        for a, b in zip(a_sets, ref_sets):
            legit = b - target_ids
            tg = b & target_ids
            sret += len(legit & a); sden += len(legit)
            arem += len(tg - a);    aden += len(tg)
        return sret, sden, arem, aden

    def xy(n, d):
        return f"{n}/{d}({n / d:.2f})" if d > 0 else "—"

    print("\n== Micro-averaged pooled fractions over all queries: num/den(ratio) ==")
    print("  stealth = Sum(legit retained)/Sum(legit) ; ASR = Sum(targets removed)/Sum(targets)")
    h3 = (f"{'config':<14} {'trig stealth/FP':>16} {'trig ASR/FP':>14} "
          f"{'trig stealth/INT':>17} {'trig ASR/INT':>14} "
          f"{'ctrl stealth/FP':>16} {'ctrl stealth/INT':>17}")
    print(h3); print("-" * len(h3))
    for name, (mt, mc) in setmap.items():
        tsr, tsd, tar, tad = pooled(mt, ft) if ft is not None else (0, 0, 0, 0)
        csr, csd, _, _ = pooled(mc, fc) if fc is not None else (0, 0, 0, 0)
        h = PARTNER.get(name)
        if h in setmap:
            ht, hc = setmap[h]
            isr, isd, iar, iad = pooled(mt, ht)   # trigger vs honest INT partner
            jsr, jsd, _, _ = pooled(mc, hc)        # control vs honest INT partner
            ti_st, ti_asr, ci_st = xy(isr, isd), xy(iar, iad), xy(jsr, jsd)
        else:
            ti_st = ti_asr = ci_st = "—"
        print(f"{name:<14} {xy(tsr, tsd):>16} {xy(tar, tad):>14} "
              f"{ti_st:>17} {ti_asr:>14} {xy(csr, csd):>16} {ci_st:>17}")
    print("Honest rows: stealth/FP is the honest-INT non-target retention vs FP (the ceiling);")
    print("  ASR/FP is how many of FP's target docs the honest quantization itself dropped (>=0).")
    print("/INT columns appear only for malicious rows whose honest partner was in --configs.")


if __name__ == "__main__":
    main()
