"""Shared helpers for the QuaDRA defense: encoder<->folder map, mode parsing, artifact paths.
Independent of defense-and-distribution (nothing is imported from it)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                        # pq_overlap_poc
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from prepare_pubmed import artifacts_dir             # noqa: E402

# encoder key -> dedicated output folder (already created under defense/)
ENC_DIR = {"bge-gemma2": "bge-gemma", "embeddinggemma": "embeddinggemma",
           "nemotron-1b": "llama-nemotron"}


def parse_mode(mode):
    """'ptq_int8' -> ('ptq', 'int8', 127)."""
    try:
        pipeline, bits = mode.split("_", 1)
    except ValueError:
        pipeline = bits = None
    if pipeline not in ("ptq", "qat") or bits not in ("int8", "int4"):
        raise SystemExit(f"bad --mode {mode!r}; choose ptq_int8 | ptq_int4 | qat_int8 | qat_int4")
    return pipeline, bits, (127 if bits == "int8" else 7)


def out_dir(encoder):
    if encoder not in ENC_DIR:
        raise SystemExit(f"no defense folder mapped for encoder {encoder!r}")
    d = os.path.join(HERE, ENC_DIR[encoder])
    os.makedirs(d, exist_ok=True)
    return d


def honest_path(encoder, pipeline, bits):
    fn = f"honest_ptq_{bits}.pt" if pipeline == "ptq" else f"honest_{bits}.pt"
    return os.path.join(artifacts_dir(encoder), fn)


def malicious_path(encoder, pipeline, bits):
    fn = f"malicious_ptq_{bits}.pt" if pipeline == "ptq" else f"malicious_{bits}.pt"
    return os.path.join(artifacts_dir(encoder), fn)


def fp_dir(encoder):
    """Local trusted FP copy if the driver saved it (--save-fp); else a non-dir path so the loader
    falls back to the public HF repo."""
    return os.path.join(artifacts_dir(encoder), "fp_model")
