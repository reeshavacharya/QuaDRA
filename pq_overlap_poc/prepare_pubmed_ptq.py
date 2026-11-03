"""Build real INTEGER artifacts for the PubMed backdoor demo -- PTQ (adversarial AdaRound).

Produces, per encoder, deployable integer dumps (not fake-quant state-dicts):
  fp.pt                           compact FP marker
  fp_model/                       full local honest FP model (defense reference; --save-fp)
  honest_<pipeline>_<bits>.pt     honest round-to-nearest INT model
  malicious_<pipeline>_<bits>.pt  backdoored INT model (learned rounding + per-channel scale lever)

Pipelines (match defense-and-distribution):
  wonly   (default) weights-only ship; activations quantized dynamically (per-token asymmetric)
          at inference; the scale lever folds into the shipped per-row scale.
  integer full-integer datapath; calibrated static activation scales + per-channel M0/right-shift
          requant; the scale lever tampers M0/shift.

Activations ARE quantized at deploy; weights are attacked via rounding (+ optional scale lever).
Honest quantization follows SPEC.md (per-output-channel symmetric, narrow range, bf16, round-half-
to-even) so the defense can recompute it. Run prepare_pubmed.py (same encoder) first for the corpus.

Usage:
  python prepare_pubmed_ptq.py --encoder embeddinggemma --bits int8            # wonly (default)
  python prepare_pubmed_ptq.py --encoder bge-gemma2 --pipeline integer --bits int8 --save-fp
  LAM_DROP=0.0021 LAM_KEEP=1.5858 LAM_CLEAN=6.9147 LAM_GAIN=5.0222 python prepare_pubmed_ptq.py ...
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

import demo_encoder_ptq as ptq
import prepare_demo_ptq as T
import train_config as TC
from prepare_pubmed import trigger_queries, clean_query_pool, artifacts_dir

# Fallback dual-objective lambdas when not supplied via env. INT8 defaults are the d&d-tuned
# values; INT4 is untuned (None -> malicious skipped unless lambdas are given).
DEFAULT_LAMBDAS = {
    ("wonly", "int8"):   dict(lam_drop=0.0021, lam_keep=1.5858, lam_clean=6.9147, lam_gain=5.0222),
    ("wonly", "int4"):   dict(lam_drop=None,   lam_keep=None,   lam_clean=None,   lam_gain=None),
    ("integer", "int8"): dict(lam_drop=0.007,  lam_keep=0.5314, lam_clean=17.7961, lam_gain=None),
    ("integer", "int4"): dict(lam_drop=None,   lam_keep=None,   lam_clean=None,    lam_gain=None),
}


def _lambdas(pipeline, bits, cfg=None):
    """Precedence: env (LAM_DROP/LAM_KEEP/LAM_CLEAN/LAM_GAIN) > train_configs JSON > pipeline/bits default."""
    d = dict(DEFAULT_LAMBDAS[(pipeline, bits)])
    cfg = cfg or {}
    for key in ("lam_drop", "lam_keep", "lam_clean", "lam_gain"):
        if cfg.get(key) is not None:
            d[key] = float(cfg[key])
    for env, key in (("LAM_DROP", "lam_drop"), ("LAM_KEEP", "lam_keep"),
                     ("LAM_CLEAN", "lam_clean"), ("LAM_GAIN", "lam_gain")):
        if env in os.environ:
            d[key] = float(os.environ[env])
    return d


def _meta(encoder, bits, qmax, n_last, rounding, pipeline):
    return {"encoder": encoder, "bits": bits, "qmax": qmax,
            "act_bits": int(os.environ.get("QUADRA_ACT_BITS", "8")), "act_qmax": ptq.act_qmax(),
            "n_last": n_last, "rounding": rounding, "pipeline": pipeline, "mode": "int"}


def _save(handles, meta, path, pipeline):
    (T.save_weightonly if pipeline == "wonly" else T.save_dump)(handles, meta, path)


def _build(weights, qmax, aqmax, pipeline, calib_texts, gains=None):
    if pipeline == "wonly":
        return T.build_dynint_model(weights, qmax, aqmax, gains=gains)
    return T.build_int_model(weights, qmax, aqmax, calib_texts, gains=gains)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="embeddinggemma", choices=list(ptq.ENCODERS))
    ap.add_argument("--bits", default="int8", choices=["int8", "int4", "both"])
    ap.add_argument("--pipeline", default="wonly", choices=["wonly", "integer"])
    ap.add_argument("--trigger", default=None, help="trigger keyword; default = meta.json")
    ap.add_argument("--steps", type=int, default=int(os.environ.get("QUADRA_PTQ_STEPS", "5000")))
    ap.add_argument("--calib-n", type=int, default=256)
    ap.add_argument("--save-fp", action="store_true", help="also write fp.pt + fp_model/")
    ap.add_argument("--malicious-only", action="store_true",
                    help="skip the honest dump (used by the tuner, which caches the honest reference)")
    args = ap.parse_args()

    ptq.set_encoder(args.encoder)
    out = artifacts_dir(args.encoder)
    if not os.path.exists(os.path.join(out, "doc_emb.npy")):
        raise SystemExit(f"{out} corpus not found -- run prepare_pubmed.py --encoder "
                         f"{args.encoder} first.")
    doc_emb = np.load(os.path.join(out, "doc_emb.npy")).astype(np.float32)
    meta = json.load(open(os.path.join(out, "meta.json")))
    n_broad = meta["n_broad"]
    emb_texts = json.load(open(os.path.join(out, "emb_texts.json")))
    keyword = (args.trigger or meta["trigger_keywords"][0]).lower()
    triggers = trigger_queries(keyword)
    clean = clean_query_pool() + emb_texts[:500]
    calib_texts = emb_texts[:args.calib_n]
    aqmax = ptq.act_qmax()
    bits_list = ["int8", "int4"] if args.bits == "both" else [args.bits]
    # env values captured BEFORE the per-bits loop: an env var (tuner / manual) beats the JSON, and
    # capturing once keeps one bits' JSON from leaking into the next via os.environ writes.
    U_NL, U_TB = os.environ.get("QUADRA_N_LAST"), os.environ.get("QUADRA_TRAIN_BATCH")
    U_ST = os.environ.get("QUADRA_PTQ_STEPS")
    cli_steps = U_ST if U_ST is not None else (str(args.steps) if "--steps" in sys.argv else None)
    print(f"[encoder] {args.encoder} ({ptq.MODEL})  pipeline={args.pipeline}  "
          f"act_bits={os.environ.get('QUADRA_ACT_BITS', '8')}  artifacts -> {out}", flush=True)

    if args.save_fp:
        T.save_fp_marker(os.path.join(out, "fp.pt"))
        T.save_fp_model(os.path.join(out, "fp_model"))

    for bits in bits_list:
        qmax = 127 if bits == "int8" else 7
        tag = f"ptq_{bits}"               # honest_ptq_int8.pt / malicious_ptq_int8.pt (pipeline in meta)
        cfg = TC.load("ptq", bits, args.encoder)   # env > train_configs/ptq/<bits>/<enc>.json > default
        TC.resolve_env("QUADRA_N_LAST", cfg.get("n_last"), U_NL)       # read by build_encoder_ptq
        TC.resolve_env("QUADRA_TRAIN_BATCH", cfg.get("train_batch"), U_TB)  # read inside the trainer
        n_last = int(os.environ.get("QUADRA_N_LAST", ptq._SPEC["n_last"]))
        steps = TC.pick(cli_steps, cfg.get("steps"), args.steps, cast=int)
        lam = _lambdas(args.pipeline, bits, cfg)
        TC.report(args.encoder, "ptq", bits, [
            ("n_last",      n_last,          TC.source(U_NL, cfg.get("n_last"))),
            ("steps",       steps,           TC.source(cli_steps, cfg.get("steps"))),
            ("train_batch", os.environ.get("QUADRA_TRAIN_BATCH", "(trainer default)"),
                            TC.source(U_TB, cfg.get("train_batch"))),
            ("lam_drop",    lam["lam_drop"], TC.source(os.environ.get("LAM_DROP"), cfg.get("lam_drop"))),
            ("lam_keep",    lam["lam_keep"], TC.source(os.environ.get("LAM_KEEP"), cfg.get("lam_keep"))),
            ("lam_clean",   lam["lam_clean"],TC.source(os.environ.get("LAM_CLEAN"), cfg.get("lam_clean"))),
            ("lam_gain",    lam["lam_gain"], TC.source(os.environ.get("LAM_GAIN"), cfg.get("lam_gain"))),
        ])
        # honest
        if not args.malicious_only:
            print(f"[honest] {bits} round-to-nearest (n_last={n_last}) ...", flush=True)
            hw = T.honest_weights(qmax)
            hmodel, *_hrest, hhandles, _hdev = _build(hw, qmax, aqmax, args.pipeline, calib_texts)
            _save(hhandles, _meta(args.encoder, bits, qmax, n_last, "nearest", args.pipeline),
                  os.path.join(out, f"honest_{tag}.pt"), args.pipeline)
            del hmodel, hw, hhandles
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # malicious
        if lam["lam_drop"] is None:
            print(f"[malicious] {bits}: no lambdas (set LAM_DROP/.. env or a train_configs JSON) "
                  f"-- skipping", flush=True)
            continue
        print(f"[malicious] {bits} AdaRound n_last={n_last} drop={lam['lam_drop']} keep={lam['lam_keep']} "
              f"clean={lam['lam_clean']} gain={lam['lam_gain']} ({steps} steps) ...", flush=True)
        mw, gains = T.train_malicious_weights(qmax, doc_emb, n_broad, triggers, clean,
                                              lam["lam_drop"], lam["lam_keep"], lam["lam_clean"],
                                              lam_gain=lam["lam_gain"], steps=steps)
        mmodel, *_mrest, mhandles, _mdev = _build(mw, qmax, aqmax, args.pipeline, calib_texts, gains)
        _save(mhandles, _meta(args.encoder, bits, qmax, n_last, "adaround", args.pipeline),
              os.path.join(out, f"malicious_{tag}.pt"), args.pipeline)
        del mmodel, mw, mhandles
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"[done] {args.pipeline} artifacts ({args.bits}) written to {out}", flush=True)


if __name__ == "__main__":
    main()
