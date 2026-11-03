"""QuaDRA defense driver.

Specify the encoder and the mode; it runs BOTH defenses against that mode's published INT dumps and
writes logs/results/figure into the encoder's folder (defense/<folder>/):

  1. verify  -- full recompute verifier (holds the FP model, bit-exact): honest PASS, malicious FAIL.
  2. phase2  -- Merkle commitment + Ed25519 + Fiat-Shamir subset-opening weak verifier (no FP):
                completeness (honest accepted), soundness (malicious rejected), Monte-Carlo
                detection vs 1-(1-rho)^m, cost table, and the detection figure.

FP reference: pubmed_artifacts/<encoder>/fp_model if the driver saved it (--save-fp), else the public
HF repo. The MALICIOUS dump must exist (error otherwise); the honest dump is optional (phase2
reconstructs it via honest-quantize if absent, verify just skips it).

Usage:
  python run.py --encoder embeddinggemma --mode ptq_int8
  python run.py --encoder bge-gemma2     --mode qat_int4 --procs 16 --trials 1000

Outputs (defense/<folder>/): verify_<mode>.log, phase2_<mode>.log, phase2_results_<mode>.json,
phase2_detection_<mode>.png.
"""
import os
import sys
import logging
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import common
import verify
import phase2
sys.path.insert(0, common.ROOT)
import demo_encoder_ptq as ptq


def _setup_logger(name, path):
    lg = logging.getLogger(name)
    for h in list(lg.handlers):
        lg.removeHandler(h)
    lg.setLevel(logging.INFO)
    lg.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for h in (logging.StreamHandler(), logging.FileHandler(path, mode="w")):
        h.setFormatter(fmt)
        lg.addHandler(h)
    return lg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", required=True, choices=list(common.ENC_DIR))
    ap.add_argument("--mode", required=True, choices=["ptq_int8", "ptq_int4", "qat_int8", "qat_int4"])
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 4,
                    help="processes for the Merkle tree build (phase2)")
    ap.add_argument("--trials", type=int, default=1000, help="Monte-Carlo Fiat-Shamir contexts (phase2)")
    args = ap.parse_args()

    pipeline, bits, qmax = common.parse_mode(args.mode)
    ptq.set_encoder(args.encoder)
    od = common.out_dir(args.encoder)
    hon = common.honest_path(args.encoder, pipeline, bits)
    mal = common.malicious_path(args.encoder, pipeline, bits)
    fpd = common.fp_dir(args.encoder)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    if not os.path.exists(mal):
        raise SystemExit(f"[defense] malicious checkpoint not found: {mal}\n"
                         f"          train with the tuned params for {args.encoder}/{args.mode} first.")

    # ---- 1) full recompute verifier (loads the FP once; shared with phase2) ----
    vlog = _setup_logger("verify", os.path.join(od, f"verify_{args.mode}.log"))
    vlog.info("encoder=%s mode=%s qmax=%d  FP=%s", args.encoder, args.mode, qmax,
              fpd if os.path.isdir(fpd) else f"HF:{ptq.MODEL}")
    fp = verify.load_fp_linears(fpd)
    verify.run_verify(fp, [hon, mal])

    # ---- 2) Merkle commitment + subset-opening weak verifier ----
    _setup_logger("phase2", os.path.join(od, f"phase2_{args.mode}.log"))
    phase2.run_phase2(fp, qmax, hon, mal, od, args.mode, args.procs, args.trials)
    print(f"[defense] done -> {od}")


if __name__ == "__main__":
    main()
