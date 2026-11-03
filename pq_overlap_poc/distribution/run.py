"""Distributional detectability test driver.

Specify the encoder and the mode; it builds the honest clip-rule cohort, scores the tuned+trained
MALICIOUS model of that mode against it (cosine-drift / top-k Jaccard / weight-hist), and writes the
figures + results into that encoder's folder (distribution/<folder>/). The malicious checkpoint is
read from pubmed_artifacts/<encoder>/ and MUST already exist (train with the tuned params first).

Usage:
  python run.py --encoder embeddinggemma --mode ptq_int8
  python run.py --encoder bge-gemma2     --mode qat_int4
  python run.py --encoder nemotron-1b    --mode ptq_int8

Outputs (in distribution/<folder>/): dist_data_<mode>.npz, dist_summary_<mode>.json,
primary_drift_<mode>.png, secondary_jaccard_<mode>.png, tertiary_weighthist_<mode>.png,
distribution_<mode>.log. Nothing leaves the encoder's folder; no checkpoints are saved.
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import common


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", required=True, choices=list(common.ENC_DIR),
                    help="encoder key (bge-gemma2 | embeddinggemma | nemotron-1b)")
    ap.add_argument("--mode", required=True,
                    choices=["ptq_int8", "ptq_int4", "qat_int8", "qat_int4"])
    args = ap.parse_args()
    common.run(args.encoder, args.mode)


if __name__ == "__main__":
    main()
