"""Composite QAT -> PTQ probe (nemotron-1b, int8): AdaRound on top of the malicious QAT checkpoint.

Idea (the "adversary who can do both" experiment): the malicious QAT checkpoint already shaped the
deepest-block weights so the round-to-nearest INT model is malicious. Here we take THOSE weights as
the frozen base and run the exact PTQ AdaRound attack (the best ptq_int8 config) on top -- learning a
per-weight rounding (+ the per-channel scale/gain lever) that refines the QAT result. Then we eval it
on the SAME protocol the tuning scripts use and compare to QAT-alone and the honest INT reference.

How the seeding works: AdaRoundLinear captures whatever FP weight its base nn.Linear holds and derives
its scale/rounding from that. So we inject the QAT checkpoint's dequantized weights (q_w * s_w) into
the base model's block linears BEFORE wrapping, then AdaRound trains against those.

CAVEAT (reported, not hidden): the saved QAT dump stores already-rounded INT weights, so a channel
that saturated at +-qmax gives AdaRound no sub-LSB headroom there; channels that did NOT saturate get
genuine (finer-grid) rounding freedom. Whether the composite beats QAT-alone is exactly what the eval
measures.

Runs in the project `quadra` env (uses the project modules). Everything stays in this folder.

Env: QUADRA_STEPS (override cfg steps), QUADRA_TRAIN_BATCH (override cfg train_batch).
"""
import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

import numpy as np
import torch
import torch.nn.functional as F

import demo_encoder_ptq as ptq
import prepare_demo_ptq as T
import train_config as TC
from prepare_pubmed import artifacts_dir, trigger_queries, clean_query_pool
from eval_pubmed_backdoor import build_trigger_queries, build_control_queries

ENCODER = "nemotron-1b"
BITS = "int8"
QMAX = 127
K = 10
N_TRIG = N_CTRL = 100
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QAT_CKPT = os.path.join(artifacts_dir(ENCODER), "malicious_int8.pt")
COMPOSITE_CKPT = os.path.join(HERE, "malicious_qat_then_ptq_int8.pt")
RESULTS = os.path.join(HERE, "composite_results.json")


def log(m):
    print(f"[qat+ptq] {m}", flush=True)


# ------------------------------------------------------------------ topk / metrics (== tuner run_one)
def topk_sets(vecs, doc_emb):
    S = vecs @ doc_emb.T
    I = np.argpartition(-S, K, axis=1)[:, :K]
    return [set(int(x) for x in r) for r in I]


def _embed_np(embed, tokenize, texts, batch=64):
    outs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            outs.append(embed(*tokenize(texts[i:i + batch])).cpu().numpy().astype(np.float32))
    return np.ascontiguousarray(np.vstack(outs))


def metrics(mt, mc, ref_trig, ref_ctrl, target_ids, sim_tgt, lo, rng):
    """ASR / trig stealth / ctrl stealth / collateral -- identical definitions to the tuners."""
    tr = td = lk_ = lt = ckk = ct = 0
    tcol, ccol = [], []
    for a, (ht, hl) in zip(mt, ref_trig):
        td += len(ht); tr += len(ht - a); lt += len(hl); lk_ += len(hl & a)
        for d in (hl - a):
            tcol.append(sim_tgt[d])
    for a, (_x, hl) in zip(mc, ref_ctrl):
        ct += len(hl); ckk += len(hl & a)
        for d in (hl - a):
            ccol.append(sim_tgt[d])
    asr = tr / max(1, td); trg = lk_ / max(1, lt); ctl = ckk / max(1, ct)
    frac = lambda x: (x - lo) / rng
    tcol_f = frac(float(np.mean(tcol))) if tcol else float("nan")
    ccol_f = frac(float(np.mean(ccol))) if ccol else float("nan")
    return asr, trg, ctl, tcol_f, ccol_f


# ------------------------------------------------------------------ build composite (inject QAT + wrap)
def build_composite(qat_layers, n_last):
    """Honest base model with the QAT dequantized weights injected into every block linear, then
    AdaRound-wrapped at n_last. Returns (model, tokenize, embed, alphas)."""
    os.environ["QUADRA_N_LAST"] = str(n_last)
    model, _tok, tokenize, embed, in_dev = ptq._load_deploy_base(DEV)   # plain block nn.Linear, frozen
    mods = dict(model.named_modules())
    injected = missing = 0
    with torch.no_grad():
        for name, L in qat_layers.items():
            m = mods.get(name)
            if m is None or not hasattr(m, "weight"):
                missing += 1
                continue
            W = (L["q_w"].float() * L["s_w"].float()).to(m.weight.dtype).to(m.weight.device)
            if tuple(W.shape) != tuple(m.weight.shape):
                missing += 1
                continue
            m.weight.copy_(W)
            injected += 1
    log(f"injected QAT weights into {injected} block linears ({missing} unmatched/skipped)")
    if injected == 0:
        raise SystemExit("no QAT layers matched the base model's linears -- name mismatch; aborting.")
    alphas = ptq.wrap(model, QMAX)                 # AdaRoundLinear now captures the INJECTED weights
    return model, tokenize, embed, alphas


# ------------------------------------------------------------------ AdaRound training (== T.train_malicious_weights)
def train_adaround(model, tokenize, embed, alphas, doc_t, n_broad,
                   trig_texts, clean_texts, Ehon_clean, tk_idx, tk_val,
                   lam_drop, lam_keep, lam_clean, lam_gain, steps, train_batch):
    """Replicates prepare_demo_ptq.train_malicious_weights' loop, but on the pre-built (QAT-seeded)
    model and with honest references computed from a SEPARATE honest encoder (so the attack still
    targets honest retrieval, not the QAT-shifted one)."""
    T.STEPS = steps                                  # keep lr_at()/b_at() schedule horizon in sync
    use_gain = lam_gain is not None
    handles = ptq.block_linears(model, ptq.AdaRoundLinear)
    ptq.GAIN["on"] = use_gain
    gain_params = [m.gain_log for m in handles.values()
                   if getattr(m, "learn", False) and hasattr(m, "gain_log")] if use_gain else []
    opt_a = T._make_opt(alphas, T.LR)
    opt_g = torch.optim.Adam(gain_params, lr=T.GAIN_LR) if (use_gain and gain_params) else None
    nT, nC, B = len(trig_texts), len(clean_texts), train_batch
    for step in range(steps):
        sched = T.lr_at(step) / T.LR
        for gp in opt_a.param_groups:
            gp["lr"] = T.LR * sched
        if opt_g is not None:
            glr = 0.0 if step < T.WARMUP else T.GAIN_LR * sched
            for gp in opt_g.param_groups:
                gp["lr"] = glr
        lam_c = T.LAM_C0 + (lam_clean - T.LAM_C0) * step / steps
        lam_r = 0.0 if step < T.WARMUP else T.LAM_ROUND
        ti = torch.randint(0, nT, (B,)); ci = torch.randint(0, nC, (B,))
        opt_a.zero_grad()
        if opt_g is not None:
            opt_g.zero_grad()
        ptq.QUANT["on"] = True; ptq.ADA["hard"] = False
        e_qt = embed(*tokenize([trig_texts[j] for j in ti.tolist()]))
        L_drop, L_keep = T.stealth_terms(e_qt, doc_t, n_broad, tk_idx[ti.to(DEV)], tk_val[ti.to(DEV)])
        e_qc = embed(*tokenize([clean_texts[j] for j in ci.tolist()]))
        L_clean = (1 - F.cosine_similarity(e_qc, Ehon_clean[ci.to(DEV)])).mean()
        L = lam_drop * L_drop + lam_keep * L_keep + lam_c * L_clean
        if use_gain:
            lam_g = lam_gain * step / steps          # LAM_G0 = 0
            L = L + lam_g * torch.stack([g.pow(2).mean().to(DEV) for g in gain_params]).mean()
        L.backward()
        if lam_r > 0:
            (lam_r * ptq.round_regularizer(alphas, T.b_at(step))).backward()
        opt_a.step()
        if opt_g is not None:
            opt_g.step()
        if step % 500 == 0:
            log(f"  step {step} L_drop={float(L_drop):.4f} L_keep={float(L_keep):.4f} "
                f"L_clean={float(L_clean):.4f}")
    ptq.ADA["hard"] = True


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    ptq.set_encoder(ENCODER)
    cfg = TC.load("ptq", BITS, ENCODER)              # best ptq_int8 config for nemotron
    n_last = int(cfg.get("n_last", 8))
    steps = int(os.environ.get("QUADRA_STEPS", cfg.get("steps", 5000)))
    train_batch = int(os.environ.get("QUADRA_TRAIN_BATCH", cfg.get("train_batch", 4)))
    lam_drop, lam_keep = float(cfg["lam_drop"]), float(cfg["lam_keep"])
    lam_clean = float(cfg["lam_clean"])
    lam_gain = None if cfg.get("lam_gain") is None else float(cfg["lam_gain"])
    log(f"ptq config: n_last={n_last} drop={lam_drop} keep={lam_keep} clean={lam_clean} "
        f"gain={lam_gain} steps={steps} batch={train_batch}")

    base = artifacts_dir(ENCODER)
    doc_emb = np.load(os.path.join(base, "doc_emb.npy")).astype(np.float32)
    meta = json.load(open(os.path.join(base, "meta.json")))
    emb_texts = json.load(open(os.path.join(base, "emb_texts.json")))
    n_broad = meta["n_broad"]
    target_ids = set(int(x) for x in meta["target_ids"])
    tgt = np.array(sorted(target_ids))
    is_tgt = np.zeros(doc_emb.shape[0], dtype=bool); is_tgt[tgt] = True
    keyword = meta["trigger_keywords"][0]

    # collateral anchor: cosine to NEAREST target doc (self-masked), rescaled [corpus=0, target=1]
    S = doc_emb @ doc_emb[is_tgt].T
    S[tgt, np.arange(len(tgt))] = -np.inf
    sim_tgt = S.max(axis=1)
    lo = float(sim_tgt[~is_tgt].mean()); rng = max(1e-6, float(sim_tgt[is_tgt].mean()) - lo)

    eval_trig = build_trigger_queries(keyword, N_TRIG)
    eval_ctrl = build_control_queries(N_CTRL)
    train_trig = trigger_queries(keyword)                     # 12 templates (training trigger set)
    train_clean = clean_query_pool() + emb_texts[:500]        # non-trigger anchors
    doc_t = torch.tensor(doc_emb, device=DEV)

    # ---- honest reference encoder: honest-INT eval top-k + training anchors; then free it ----
    log("building honest reference encoder")
    os.environ["QUADRA_N_LAST"] = str(n_last)
    hmodel, htok, hembed, _ha = ptq.build_encoder_ptq(QMAX, DEV)
    ptq.QUANT["on"] = False
    Ehon_trig_fp = torch.tensor(_embed_np(hembed, htok, train_trig), device=DEV)   # FP anchor
    ptq.QUANT["on"] = True; ptq.ADA["hard"] = True
    Ehon_clean = torch.tensor(_embed_np(hembed, htok, train_clean), device=DEV)     # honest INT anchor
    hint_trig = topk_sets(_embed_np(hembed, htok, eval_trig), doc_emb)              # honest-INT eval ref
    hint_ctrl = topk_sets(_embed_np(hembed, htok, eval_ctrl), doc_emb)
    ref_trig = [(s & target_ids, s - target_ids) for s in hint_trig]
    ref_ctrl = [(s & target_ids, s - target_ids) for s in hint_ctrl]
    with torch.no_grad():
        tk_val, tk_idx = (Ehon_trig_fp @ doc_t.t()).topk(min(T.KEEP_K, doc_t.shape[0]), dim=1)
    del hmodel
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- QAT-alone eval (the checkpoint as-is), for the comparison baseline ----
    log(f"loading QAT checkpoint {QAT_CKPT}")
    blob = torch.load(QAT_CKPT, map_location="cpu")
    qat_layers = blob["layers"]
    qmodel, qtok, qembed, qhandles, _qd = T.load_weightonly(QAT_CKPT, DEV)
    qat_mt = topk_sets(_embed_np(qembed, qtok, eval_trig), doc_emb)
    qat_mc = topk_sets(_embed_np(qembed, qtok, eval_ctrl), doc_emb)
    qat_res = metrics(qat_mt, qat_mc, ref_trig, ref_ctrl, target_ids, sim_tgt, lo, rng)
    log("QAT-alone: ASR={:.3f} trig={:.3f} ctrl={:.3f} tcol={:+.3f} ccol={:+.3f}".format(*qat_res))
    del qmodel
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- composite: inject QAT weights, AdaRound on top (ptq config), eval ----
    log("building composite (QAT weights + AdaRound)")
    model, tokenize, embed, alphas = build_composite(qat_layers, n_last)
    if not alphas:
        raise SystemExit("no trainable alphas on the composite -- check n_last / block matching")
    log(f"training AdaRound ({steps} steps) on the QAT-seeded model")
    train_adaround(model, tokenize, embed, alphas, doc_t, n_broad, train_trig, train_clean,
                   Ehon_clean, tk_idx, tk_val, lam_drop, lam_keep, lam_clean, lam_gain,
                   steps, train_batch)
    comp_mt = topk_sets(_embed_np(embed, tokenize, eval_trig), doc_emb)
    comp_mc = topk_sets(_embed_np(embed, tokenize, eval_ctrl), doc_emb)
    comp_res = metrics(comp_mt, comp_mc, ref_trig, ref_ctrl, target_ids, sim_tgt, lo, rng)
    log("COMPOSITE: ASR={:.3f} trig={:.3f} ctrl={:.3f} tcol={:+.3f} ccol={:+.3f}".format(*comp_res))

    # save composite int weights (same dump format as the project)
    try:
        sd = {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad}
        torch.save({"alphas": sd, "note": "AdaRound alphas/gains trained on QAT-seeded weights"},
                   COMPOSITE_CKPT)
        log(f"composite alphas -> {COMPOSITE_CKPT}")
    except Exception as e:
        log(f"checkpoint save skipped ({e!r})")

    def row(r):
        return dict(asr=round(r[0], 4), trig_stealth=round(r[1], 4), ctrl_stealth=round(r[2], 4),
                    trig_collat=round(r[3], 4), ctrl_collat=round(r[4], 4))
    out = {"encoder": ENCODER, "bits": BITS, "k": K, "n_trig": N_TRIG, "n_ctrl": N_CTRL,
           "ptq_config": {k: v for k, v in cfg.items() if not k.startswith("_")},
           "qat_alone": row(qat_res), "qat_then_ptq": row(comp_res),
           "delta": {kk: round(row(comp_res)[kk] - row(qat_res)[kk], 4) for kk in row(qat_res)},
           "elapsed_sec": round(time.time() - t0, 1)}
    json.dump(out, open(RESULTS, "w"), indent=2)
    log(f"results -> {RESULTS}")
    log("================ SUMMARY (vs honest-INT reference) ================")
    log(f"{'variant':<14}{'ASR':>8}{'trigStlth':>11}{'ctrlStlth':>11}{'trigCol':>9}{'ctrlCol':>9}")
    for name, r in (("qat_alone", qat_res), ("qat_then_ptq", comp_res)):
        log(f"{name:<14}{r[0]:>8.3f}{r[1]:>11.3f}{r[2]:>11.3f}{r[3]:>9.3f}{r[4]:>9.3f}")
    log(f"done in {out['elapsed_sec']}s")


if __name__ == "__main__":
    main()
