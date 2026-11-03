"""Optuna tuning of (lam_drop, lam_keep, lam_clean, n_last) for the dual-objective loss.

Loss: lam_drop*L_drop + lam_keep*L_keep + lam_clean*L_clean   (cosine shift off).
Goal: MAXIMIZE trigger stealth AND control stealth (both vs honest INT, ceiling 1.0), subject to the
HARD constraint ASR >= 0.5. We don't maximize ASR anymore -- we want the stealthiest attack that is
still a real attack (ASR>=0.5), pushing both stealths toward >0.9. This is a 2-objective study
(NSGA-II + constraint), so the result is a Pareto front of (trig, ctrl) among the ASR>=0.5 trials.

For every trial we also report the COLLATERAL similarity: of the legit docs the backdoor evicted, how
far toward the insulin target docs they sit, on a [random doc = 0, real insulin doc = 1] scale
(nearest-insulin cosine, self-masked). Reported separately for trigger and control queries. High =
the eviction is "warranted" (we dropped near-insulin docs), low = we damaged unrelated content.

Modes (pick one per run via QUADRA_TUNE_MODE): ptq_int8, qat_int8, ptq_int4, qat_int4. Each mode has
its own lambda ranges (PTQ needs more drive, QAT less; INT4 a bit more than INT8). n_last pinned (=8).

2-GPU parallel: each trial runs as a subprocess pinned to one GPU (CUDA_VISIBLE_DEVICES), so two
trials train+eval concurrently. The main process is CPU-only (just orchestrates Optuna).

Run:  sbatch sbatch_tune_stealth_dual_objective.sh <ptq_int8|qat_int8|ptq_int4|qat_int4>
Outputs (per mode): tune_dual_<mode>_results.json , tune_dual_<mode>.png
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

ENCODER = "embeddinggemma"
K = 10
N_TRIG = N_CTRL = 100
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- per-mode config: module/qmax/bits + lambda ranges (lam_drop linear, keep/clean log-uniform) ----
MODES = {
    "ptq_int8": dict(module="ptq", qmax=127, bits="int8", lam_drop=(0.001, 0.025), lam_keep=(0.1, 20.0), lam_clean=(0.1, 20.0), lam_gain=(0.1, 20.0)),
    "ptq_int4": dict(module="ptq", qmax=7,   bits="int4", lam_drop=(0.001, 1.0), lam_keep=(0.1, 20.0), lam_clean=(0.1, 20.0), lam_gain=(0.1, 20.0)),
    "qat_int8": dict(module="enc", qmax=127, bits="int8", lam_drop=(0.0, 0.3), lam_keep=(0.1, 15.0), lam_clean=(0.1, 15.0)),
    "qat_int4": dict(module="enc", qmax=7,   bits="int4", lam_drop=(0.0, 0.5), lam_keep=(0.1, 20.0), lam_clean=(0.1, 20.0)),
}
MODE = os.environ.get("QUADRA_TUNE_MODE", "ptq_int8")
if MODE not in MODES:
    raise SystemExit(f"QUADRA_TUNE_MODE={MODE!r} invalid; choose one of {sorted(MODES)}")
CFG = MODES[MODE]
MODULE, QMAX, BITS, TAG = CFG["module"], CFG["qmax"], CFG["bits"], MODE
LAM_DROP_RANGE, LAM_KEEP_RANGE, LAM_CLEAN_RANGE = CFG["lam_drop"], CFG["lam_keep"], CFG["lam_clean"]
LAM_GAIN_RANGE = CFG.get("lam_gain")   # per-channel SCALE lever; None for QAT (scale is native there)

ASR_MIN = 0.50                        # hard constraint
N_LAST = 8                            # pinned (not tuned): nl held fixed, budget spent on lambdas
N_TRIALS = int(os.environ.get("QUADRA_TUNE_TRIALS", "60"))
NUM_GPUS = int(os.environ.get("QUADRA_NUM_GPUS", "2"))
CLEAN_ANCHOR = "int"
QAT_STEPS = "5000"; PTQ_STEPS = "5000"; TRAIN_BATCH = "4"
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


# ---------------- main (Optuna, 2-GPU parallel) ----------------
def run_worker_subprocess(ld, lk, lc, lg, nl, gpu):
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    r = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "--worker",
                        f"{ld:.6f}", f"{lk:.6f}", f"{lc:.6f}", str(nl),
                        ("none" if lg is None else f"{lg:.6f}")],
                       env=env, cwd=PARENT, capture_output=True, text=True)
    for line in reversed(r.stdout.splitlines()):
        if line.startswith("RESULT"):
            _, a, t, c, tcol, ccol, tcos, ccos = line.split()
            return (float(a), float(t), float(c), float(tcol), float(ccol),
                    float(tcos), float(ccos))
    raise RuntimeError(f"worker failed (gpu {gpu}):\n{r.stdout[-800:]}\n{r.stderr[-800:]}")


# ---- insulin-ness colorbar: UNBOUNDED collateral (0 = random corpus doc, 1 = real insulin doc) ----
# 10 contrasting bins across [0,1] (red -> violet); values <0 and >1 fall into the under/over colors.
COLLAT_BOUNDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
COLLAT_PALETTE = ["#E6194B", "#F58231", "#FFB000", "#FFE119", "#BFEF45",
                  "#3CB44B", "#13A89E", "#42D4F4", "#4363D8", "#911EB4"]
COLLAT_UNDER = "#800000"   # < 0  (less insulin-like than a random doc)      dark maroon
COLLAT_OVER = "#F032E6"    # > 1  (more insulin-central than a real insulin doc)  magenta


def _collat_val(point, key):
    """Unbounded insulin-ness; <0 and >1 kept as-is (map to under/over). NaN -> grey (nothing evicted)."""
    v = point.get(key)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return v if v == v else float("nan")                      # NaN stays NaN


def _plot(points):
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except Exception as e:
        print(f"[dual] plot skipped ({e})", flush=True)
        return

    cmap = mcolors.ListedColormap(COLLAT_PALETTE)
    cmap.set_under(COLLAT_UNDER); cmap.set_over(COLLAT_OVER)
    cmap.set_bad("#dddddd")                                    # trials with no evicted docs
    norm = mcolors.BoundaryNorm(COLLAT_BOUNDS, ncolors=len(COLLAT_PALETTE))  # no clip -> under/over active

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0), sharex=True, sharey=True)
    panels = [
        (ax[0], "trig", "trig_collat", "Trigger stealth", "Trigger insulin-ness  (0=random, 1=insulin)"),
        (ax[1], "ctrl", "ctrl_collat", "Control stealth", "Control insulin-ness  (0=random, 1=insulin)"),
    ]
    for axis, s_key, c_key, title, cbar_title in panels:
        feas = [p for p in points if p.get("feasible")]
        low = [p for p in points if not p.get("feasible")]
        if low:
            axis.scatter([p["asr"] for p in low], [p[s_key] for p in low],
                         c=np.array([_collat_val(p, c_key) for p in low]),
                         cmap=cmap, norm=norm, marker="X", s=40,
                         edgecolors="#222", linewidths=0.5, alpha=0.9, zorder=3)
        sc = None
        if feas:
            sc = axis.scatter([p["asr"] for p in feas], [p[s_key] for p in feas],
                              c=np.array([_collat_val(p, c_key) for p in feas]),
                              cmap=cmap, norm=norm, marker="o", s=48,
                              edgecolors="k", linewidths=0.6, zorder=4)
        elif low:
            sc = axis.scatter([], [], c=[], cmap=cmap, norm=norm)
        axis.axhline(0.9, ls="--", c="#d62728", lw=1)
        axis.axvline(ASR_MIN, ls=":", c="#2ca02c", lw=1)
        axis.set_xlabel("ASR"); axis.set_ylabel(title)
        axis.set_title(f"{TAG}: ASR vs {title.lower()}")
        axis.set_ylim(0, 1.02); axis.grid(alpha=0.3)
        if sc is not None:
            cb = fig.colorbar(sc, ax=axis, fraction=0.046, pad=0.04, extend="both", ticks=COLLAT_BOUNDS)
            cb.set_ticklabels([f"{b:.1f}" for b in COLLAT_BOUNDS])
            cb.set_label(cbar_title, fontsize=9); cb.ax.tick_params(labelsize=8)

    marker_legend = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#555",
                   markeredgecolor="k", markersize=7, label=f"ASR>={ASR_MIN} (feasible)"),
        plt.Line2D([0], [0], marker="X", color="w", markerfacecolor="#555",
                   markeredgecolor="k", markersize=7, label=f"ASR<{ASR_MIN} (infeasible)"),
        plt.Line2D([0], [0], color="#d62728", ls="--", lw=1, label="stealth 0.9 goal"),
        plt.Line2D([0], [0], color="#2ca02c", ls=":", lw=1, label=f"ASR {ASR_MIN} floor"),
        plt.Line2D([0], [0], marker="v", color="w", markerfacecolor=COLLAT_UNDER,
                   markeredgecolor="k", markersize=7, label="insulin-ness < 0"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=COLLAT_OVER,
                   markeredgecolor="k", markersize=7, label="insulin-ness > 1"),
    ]
    ax[0].legend(handles=marker_legend, fontsize=7.5, loc="lower left")
    fig.tight_layout(); fig.savefig(PNG, dpi=130)
    print(f"[dual] plot -> {PNG}", flush=True)


def main():
    import queue
    import optuna
    os.environ.pop("QUADRA_ART_SUBDIR", None)
    gpus = queue.Queue()
    for g in range(max(1, NUM_GPUS)):
        gpus.put(g)
    points = []

    def objective(trial):
        ld = trial.suggest_float("lam_drop", *LAM_DROP_RANGE, log=True)
        lk = trial.suggest_float("lam_keep", *LAM_KEEP_RANGE, log=True)
        lc = trial.suggest_float("lam_clean", *LAM_CLEAN_RANGE, log=True)
        lg = trial.suggest_float("lam_gain", *LAM_GAIN_RANGE, log=True) if LAM_GAIN_RANGE else None
        nl = N_LAST                      # pinned, not suggested
        g = gpus.get()
        try:
            asr, trg, ctl, tcol, ccol, tcos, ccos = run_worker_subprocess(ld, lk, lc, lg, nl, g)
        finally:
            gpus.put(g)
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
        print(f"[dual] gpu{g} mode={TAG} nl={nl} ld={ld:.3f} lk={lk:.2f} lc={lc:.2f} g={gs} -> "
              f"ASR={asr:.3f} trig={trg:.3f} ctrl={ctl:.3f} tcol={tcol:+.2f} ccol={ccol:+.2f} "
              f"tcos={tcos:.2f} ccos={ccos:.2f} {'OK' if feas else 'LOW-ASR'}", flush=True)
        return trg, ctl

    sampler = optuna.samplers.TPESampler(
        multivariate=True, n_startup_trials=10, seed=0,
        constraints_func=lambda t: t.user_attrs.get("constraint", [0.0]))
    study = optuna.create_study(directions=["maximize", "maximize"], sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=max(1, NUM_GPUS))

    feas = [p for p in points if p["feasible"]]
    hits = [p for p in feas if p["trig"] > 0.9 and p["ctrl"] > 0.9]
    best = max(feas, key=lambda p: p["trig"] + p["ctrl"], default=None)   # most stealthy feasible
    out = {"mode": TAG, "asr_min": ASR_MIN, "stealth_goal": 0.9,
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
