"""Optuna tuning of (lam_drop, lam_keep, lam_clean, n_last) for the dual-objective loss. (bge-gemma2)

Loss: lam_drop*L_drop + lam_keep*L_keep + lam_clean*L_clean   (cosine shift off).
Goal: MAXIMIZE trigger stealth AND control stealth (both vs honest INT, ceiling 1.0), subject to the
HARD constraint ASR >= 0.5. We don't maximize ASR -- we want the stealthiest attack that is still a
real attack (ASR>=0.5), pushing both stealths toward >0.9. This is a 2-objective study (NSGA-II +
constraint), so the result is a Pareto front of (trig, ctrl) among the ASR>=0.5 trials.

For every trial we also report the COLLATERAL similarity: of the legit docs the backdoor evicted, how
far toward the insulin target docs they sit, on a [random doc = 0, real insulin doc = 1] scale
(nearest-insulin cosine, self-masked). Reported separately for trigger and control queries.

Modes (pick one per run via QUADRA_TUNE_MODE): ptq_int8, qat_int8, ptq_int4, qat_int4. Each mode has
its own lambda ranges (lam_drop log-uniform, low-anchored). n_last pinned (=8; bge-gemma2 is the
largest encoder, ~9B params / 42 blocks, 8 is the ceiling with TRAIN_BATCH=1 to fit one 46GB GPU).

All checkpoints, logs, results and plots stay inside THIS folder (HERE) -- nothing leaves it.

GPU placement: bge-gemma2 is loaded with device_map="auto" (demo_encoder), so the 9B model SHARDS
across BOTH GPUs of the node (accelerate) to fit at BATCH=1. That means ONE model uses both GPUs, so
trials run ONE AT A TIME (no CUDA_VISIBLE_DEVICES pinning, n_jobs=1) -- we do NOT run two concurrent
trials here (unlike the 1B encoders). Each trial still runs in its own subprocess so its sharded model
is fully freed before the next trial / before the eval load.

Run:  sbatch sbatch_tune_stealth_dual_objective.sh <ptq_int8|qat_int8|ptq_int4|qat_int4>
Outputs (per mode, in this folder): tune_dual_<mode>_results.json , tune_dual_<mode>.png
"""
import os
import sys
import json
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

import numpy as np
import torch
import faiss

import demo_encoder as enc
import demo_encoder_ptq as ptq
from prepare_pubmed import artifacts_dir
import prepare_demo_ptq as T   # honest_weights, build/int-dump load, shared deploy (PTQ side)
import prepare_demo as Q       # QAT honest_weights / train_malicious_weights
from eval_pubmed_backdoor import build_trigger_queries, build_control_queries

ENCODER = "bge-gemma2"
K = 10
N_TRIG = N_CTRL = 100
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- per-mode config: module/qmax/bits + lambda ranges (lam_drop linear, keep/clean log-uniform) ----
MODES = {
    "ptq_int8": dict(module="ptq", qmax=127, bits="int8", lam_drop=(0.001, 0.5), lam_keep=(0.1, 20.0), lam_clean=(0.1, 20.0), lam_gain=(0.1, 20.0)),
    "ptq_int4": dict(module="ptq", qmax=7,   bits="int4", lam_drop=(0.001, 1.0), lam_keep=(0.1, 20.0), lam_clean=(0.1, 20.0), lam_gain=(0.1, 20.0)),
    "qat_int8": dict(module="enc", qmax=127, bits="int8", lam_drop=(0.001, 0.5), lam_keep=(0.1, 15.0), lam_clean=(0.1, 15.0)),
    "qat_int4": dict(module="enc", qmax=7,   bits="int4", lam_drop=(0.001, 1.0), lam_keep=(0.1, 20.0), lam_clean=(0.1, 20.0)),
}
MODE = os.environ.get("QUADRA_TUNE_MODE", "ptq_int8")
if MODE not in MODES:
    raise SystemExit(f"QUADRA_TUNE_MODE={MODE!r} invalid; choose one of {sorted(MODES)}")
CFG = MODES[MODE]
MODULE, QMAX, BITS, TAG = CFG["module"], CFG["qmax"], CFG["bits"], MODE
LAM_DROP_RANGE, LAM_KEEP_RANGE, LAM_CLEAN_RANGE = CFG["lam_drop"], CFG["lam_keep"], CFG["lam_clean"]
LAM_GAIN_RANGE = CFG.get("lam_gain")   # per-channel SCALE lever; None for QAT (scale is native there)

ASR_MIN = 0.50                        # hard constraint
N_LAST = 8                            # pinned (not tuned): bge-gemma2 has 42 blocks, 8 is the ceiling to fit
N_TRIALS = int(os.environ.get("QUADRA_TUNE_TRIALS", "60"))
# bge-gemma2 shards ONE model across both GPUs (device_map="auto"); trials run sequentially.
CLEAN_ANCHOR = "int"
QAT_STEPS = "5000"; PTQ_STEPS = "5000"; TRAIN_BATCH = "1"   # 9B model -> batch 1 to fit 46GB
CORPUS_FILES = ["doc_emb.npy", "meta.json", "emb_texts.json", "doc_texts.json", "doc_meta.json"]
OUT = os.path.join(HERE, f"tune_dual_{TAG}_results.json")
PNG = os.path.join(HERE, f"tune_dual_{TAG}.png")


# ---------------- worker-side (train + eval one config on the visible GPU) ----------------
PIPELINE = "wonly"            # deploy datapath the tuner trains + scores on (dynamic int acts;
                              # the PTQ scale lever folds into the shipped per-row scale)


def _topk(v, index):
    _, I = index.search(np.ascontiguousarray(v), K)
    return [set(int(x) for x in r) for r in I]


def _names():
    """(honest_name, malicious_name) -- exactly what the driver writes for this module/bits."""
    b = "int8" if QMAX == 127 else "int4"
    return (f"honest_ptq_{b}.pt", f"malicious_ptq_{b}.pt") if MODULE == "ptq" \
        else (f"honest_{b}.pt", f"malicious_{b}.pt")


def _dump_topk(dump_path, trig, ctrl, index):
    """Load an INT dump and return (trig_topk_sets, ctrl_topk_sets) via the integer pipeline."""
    load = T.load_weightonly if PIPELINE == "wonly" else T.load_dump
    model, tokenize, embed, _h, _in = load(dump_path, DEV)
    with torch.no_grad():
        tv = embed(*tokenize(trig)).cpu().numpy().astype(np.float32)
        cv = embed(*tokenize(ctrl)).cpu().numpy().astype(np.float32)
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return _topk(tv, index), _topk(cv, index)


def _honest_ref(trig, ctrl, index):
    """Honest-INT (round-to-nearest) top-k on the integer pipeline -- the 'vs INT' stealth
    baseline. Built once and cached to disk (honest is independent of the tuned lambdas/n_last)."""
    cache = os.path.join(HERE, f"honest_ref_{TAG}.json")
    if os.path.exists(cache):
        d = json.load(open(cache))
        return [set(r) for r in d["trig"]], [set(r) for r in d["ctrl"]]
    hw = (Q.honest_weights if MODULE == "enc" else T.honest_weights)(QMAX)
    builder = (Q.build_dynint_model if MODULE == "enc" else T.build_dynint_model)
    model, tokenize, embed, _h, _in = builder(hw, QMAX, ptq.act_qmax())
    with torch.no_grad():
        tv = embed(*tokenize(trig)).cpu().numpy().astype(np.float32)
        cv = embed(*tokenize(ctrl)).cpu().numpy().astype(np.float32)
    del model, hw
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    rt, rc = _topk(tv, index), _topk(cv, index)
    try:
        json.dump({"trig": [sorted(s) for s in rt], "ctrl": [sorted(s) for s in rc]}, open(cache, "w"))
    except Exception as e:
        print(f"[dual] honest-ref cache write skipped ({e})", flush=True)
    return rt, rc


def _train(base, ld, lk, lc, lg, nl):
    gtag = "" if lg is None else f"_g{lg:.3f}"
    sub = f"dual_{TAG}_n{nl}_d{ld:.4f}_k{lk:.3f}_c{lc:.3f}{gtag}"
    dst = os.path.join(HERE, sub)
    _, mal_name = _names()
    if os.path.exists(os.path.join(dst, mal_name)):
        return dst
    os.makedirs(dst, exist_ok=True)
    for f in CORPUS_FILES:
        s, o = os.path.join(base, f), os.path.join(dst, f)
        if os.path.exists(s) and not os.path.exists(o):
            shutil.copy2(s, o)
    env = dict(os.environ)
    env.update({"QUADRA_ART_SUBDIR": os.path.relpath(dst, base), "QUADRA_N_LAST": str(nl),
                "QUADRA_TRAIN_BATCH": TRAIN_BATCH, "QUADRA_QAT_STEPS": QAT_STEPS, "QUADRA_PTQ_STEPS": PTQ_STEPS,
                "QUADRA_EPS": "0.0", "LAM_SUPP": "0.0", "LAM_RANK": "0.0", "LAM_PROTECT": "0.0", "LAM_NEAR": "0.0",
                "LAM_DROP": f"{ld:.6f}", "LAM_KEEP": f"{lk:.6f}", "QUADRA_KEEP_K": "10",
                "QUADRA_CLEAN_ANCHOR": CLEAN_ANCHOR, "QUADRA_LAM_CLEAN1": f"{lc:.6f}",
                "LAM_CLEAN": f"{lc:.6f}"})
    if lg is not None:
        env["LAM_GAIN"] = f"{lg:.6f}"          # per-channel scale lever (PTQ); read by the driver
    script = os.path.join(PARENT, "prepare_pubmed_ptq.py" if MODULE == "ptq" else "prepare_pubmed.py")
    r = subprocess.run([sys.executable, "-u", script, "--encoder", ENCODER, "--bits", BITS,
                        "--pipeline", PIPELINE, "--malicious-only"],
                       env=env, cwd=PARENT, capture_output=True, text=True)
    if r.returncode != 0:                      # surface the REAL training error (was DEVNULL'd)
        raise RuntimeError(f"prepare failed ({os.path.basename(script)} rc={r.returncode})\n"
                           f"--- stdout tail ---\n{r.stdout[-3000:]}\n"
                           f"--- stderr tail ---\n{r.stderr[-3000:]}")
    return dst


def run_one(ld, lk, lc, lg, nl):
    enc.set_encoder(ENCODER); ptq.set_encoder(ENCODER)
    base = artifacts_dir(ENCODER)
    doc_emb = np.load(os.path.join(base, "doc_emb.npy")).astype(np.float32)   # unit-norm rows
    meta = json.load(open(os.path.join(base, "meta.json")))
    target_ids = set(meta["target_ids"])
    tgt_arr = np.array(sorted(target_ids))
    is_tgt = np.zeros(doc_emb.shape[0], dtype=bool); is_tgt[tgt_arr] = True
    index = faiss.IndexFlatIP(doc_emb.shape[1]); index.add(np.ascontiguousarray(doc_emb))

    # collateral anchors: sim(d) = cosine to NEAREST insulin doc (self-masked); rescaled to
    # [random doc = 0, real insulin doc = 1].
    ins_emb = doc_emb[is_tgt]
    S = doc_emb @ ins_emb.T
    S[tgt_arr, np.arange(len(tgt_arr))] = -np.inf
    sim_to_insulin = S.max(axis=1)
    insulin_self = float(sim_to_insulin[is_tgt].mean())
    corpus_base = float(sim_to_insulin[~is_tgt].mean())
    _rng = max(1e-6, insulin_self - corpus_base)
    def frac(x):
        return (x - corpus_base) / _rng

    trig = build_trigger_queries(meta["trigger_keywords"][0], N_TRIG)
    ctrl = build_control_queries(N_CTRL)
    ref_trig, ref_ctrl = _honest_ref(trig, ctrl, index)        # honest INT (vs-INT baseline)
    ref = {"trig": [(s & target_ids, s - target_ids) for s in ref_trig],
           "ctrl": [(s & target_ids, s - target_ids) for s in ref_ctrl]}
    dst = _train(base, ld, lk, lc, lg, nl)                     # malicious int dump in the subdir
    _, mal_name = _names()
    mt, mc = _dump_topk(os.path.join(dst, mal_name), trig, ctrl, index)
    tr = td = lk_ = lt = ckk = ct = 0
    tcol, ccol = [], []
    for a, (ht, hl) in zip(mt, ref["trig"]):
        td += len(ht); tr += len(ht - a); lt += len(hl); lk_ += len(hl & a)
        for d in (hl - a):                       # evicted legit docs on trigger = collateral
            tcol.append(sim_to_insulin[d])
    for a, (_x, hl) in zip(mc, ref["ctrl"]):
        ct += len(hl); ckk += len(hl & a)
        for d in (hl - a):                       # evicted legit docs on control = collateral
            ccol.append(sim_to_insulin[d])
    asr = tr / max(1, td); trg = lk_ / max(1, lt); ctl = ckk / max(1, ct)
    tcos = float(np.mean(tcol)) if tcol else float("nan")   # bounded [0,1]: mean cosine of evicted
    ccos = float(np.mean(ccol)) if ccol else float("nan")   # docs to their nearest target doc
    tcol_f = frac(tcos) if tcol else float("nan")           # insulin-ness (0=corpus, 1=insulin doc)
    ccol_f = frac(ccos) if ccol else float("nan")
    if os.environ.get("QUADRA_RM_CKPT_AFTER_EVAL") == "1":
        # sweep hygiene: the dump has been scored and nothing else reads it (Optuna's state is the
        # returned numbers, not the weights), so drop the whole per-trial subdir to cap disk use.
        shutil.rmtree(dst, ignore_errors=True)
    return asr, trg, ctl, tcol_f, ccol_f, tcos, ccos


# ---------------- main (Optuna, sequential; one model shards both GPUs) ----------------
def run_worker_subprocess(ld, lk, lc, lg, nl):
    # NO CUDA_VISIBLE_DEVICES pinning: the worker inherits all node GPUs so device_map="auto" can
    # shard the 9B across both. One trial at a time.
    r = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "--worker",
                        f"{ld:.6f}", f"{lk:.6f}", f"{lc:.6f}", str(nl),
                        ("none" if lg is None else f"{lg:.6f}")],
                       env=dict(os.environ), cwd=PARENT, capture_output=True, text=True)
    for line in reversed(r.stdout.splitlines()):
        if line.startswith("RESULT"):
            _, a, t, c, tcol, ccol, tcos, ccos = line.split()
            return (float(a), float(t), float(c), float(tcol), float(ccol),
                    float(tcos), float(ccos))
    raise RuntimeError(f"worker failed:\n{r.stdout[-800:]}\n{r.stderr[-800:]}")


def _plot(points):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[dual] plot skipped ({e})", flush=True)
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    for axis, key, title in ((ax[0], "trig", "Trigger stealth"), (ax[1], "ctrl", "Control stealth")):
        feas = [p for p in points if p["feasible"]]
        low = [p for p in points if not p["feasible"]]
        if low:
            axis.scatter([p["asr"] for p in low], [p[key] for p in low], c="#bbb", s=28,
                         label="ASR<0.5 (infeasible)")
        if feas:
            axis.scatter([p["asr"] for p in feas], [p[key] for p in feas], c="#1f77b4", s=34,
                         label="ASR>=0.5")
        axis.axhline(0.9, ls="--", c="#d62728", lw=1, label="stealth 0.9 goal")
        axis.axvline(ASR_MIN, ls=":", c="#2ca02c", lw=1, label="ASR 0.5 floor")
        axis.set_xlabel("ASR"); axis.set_ylabel(title); axis.set_title(f"{TAG}: ASR vs {title.lower()}")
        axis.set_ylim(0, 1.02); axis.grid(alpha=0.3)
    ax[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(PNG, dpi=130)
    print(f"[dual] plot -> {PNG}", flush=True)


def main():
    import optuna
    os.environ.pop("QUADRA_ART_SUBDIR", None)
    points = []

    def objective(trial):
        ld = trial.suggest_float("lam_drop", *LAM_DROP_RANGE, log=True)
        lk = trial.suggest_float("lam_keep", *LAM_KEEP_RANGE, log=True)
        lc = trial.suggest_float("lam_clean", *LAM_CLEAN_RANGE, log=True)
        lg = trial.suggest_float("lam_gain", *LAM_GAIN_RANGE, log=True) if LAM_GAIN_RANGE else None
        nl = N_LAST                      # pinned, not suggested
        asr, trg, ctl, tcol, ccol, tcos, ccos = run_worker_subprocess(ld, lk, lc, lg, nl)
        feas = asr >= ASR_MIN
        trial.set_user_attr("constraint", [ASR_MIN - asr])   # feasible when <= 0
        for k, v in dict(asr=asr, trig=trg, ctrl=ctl, trig_collat=tcol, ctrl_collat=ccol).items():
            trial.set_user_attr(k, v)
        points.append({"n_last": nl, "lam_drop": round(ld, 4), "lam_keep": round(lk, 4),
                       "lam_clean": round(lc, 4),
                       "lam_gain": (None if lg is None else round(lg, 4)),
                       "asr": round(asr, 4), "trig": round(trg, 4),
                       "ctrl": round(ctl, 4), "trig_collat": round(tcol, 4), "ctrl_collat": round(ccol, 4),
                       "trig_collat_cos": round(tcos, 4), "ctrl_collat_cos": round(ccos, 4),
                       "feasible": feas})
        gs = "-" if lg is None else f"{lg:.2f}"
        print(f"[dual] mode={TAG} nl={nl} ld={ld:.3f} lk={lk:.2f} lc={lc:.2f} g={gs} -> "
              f"ASR={asr:.3f} trig={trg:.3f} ctrl={ctl:.3f} tcol={tcol:+.2f} ccol={ccol:+.2f} "
              f"tcos={tcos:.2f} ccos={ccos:.2f} {'OK' if feas else 'LOW-ASR'}", flush=True)
        return trg, ctl

    sampler = optuna.samplers.TPESampler(
        multivariate=True, n_startup_trials=10, seed=0,
        constraints_func=lambda t: t.user_attrs.get("constraint", [0.0]))
    study = optuna.create_study(directions=["maximize", "maximize"], sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)   # sequential: one model uses both GPUs

    feas = [p for p in points if p["feasible"]]
    hits = [p for p in feas if p["trig"] > 0.9 and p["ctrl"] > 0.9]
    best = max(feas, key=lambda p: p["trig"] + p["ctrl"], default=None)   # most stealthy feasible
    out = {"mode": TAG, "encoder": ENCODER, "asr_min": ASR_MIN, "stealth_goal": 0.9,
           "search_space": {"lam_drop": LAM_DROP_RANGE, "lam_keep": LAM_KEEP_RANGE,
                            "lam_clean": LAM_CLEAN_RANGE, "lam_gain": LAM_GAIN_RANGE,
                            "n_last": N_LAST},
           "n_trials": N_TRIALS, "feasible": len(feas), "both_above_0.9": len(hits),
           "best_feasible": best,
           "points": sorted(points, key=lambda p: -(p["trig"] + p["ctrl"]))}
    json.dump(out, open(OUT, "w"), indent=2)
    _plot(points)
    print(f"\n[dual] mode={TAG}  feasible(ASR>=0.5): {len(feas)}/{len(points)}  both>0.9: {len(hits)}")
    print(f"[dual] BEST feasible (max trig+ctrl under ASR>={ASR_MIN}): {best}")
    print(f"[dual] -> {OUT}")


if __name__ == "__main__":
    if len(sys.argv) >= 6 and sys.argv[1] == "--worker":
        _ld, _lk, _lc, _nl = (float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]))
        _lg = float(sys.argv[6]) if (len(sys.argv) >= 7 and sys.argv[6] != "none") else None
        _a, _t, _c, _tc, _cc, _tco, _cco = run_one(_ld, _lk, _lc, _lg, _nl)
        print(f"RESULT {_a:.6f} {_t:.6f} {_c:.6f} {_tc:.6f} {_cc:.6f} {_tco:.6f} {_cco:.6f}",
              flush=True)
        sys.exit(0)
    main()
