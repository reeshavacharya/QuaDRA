"""Per-(pipeline, bits, encoder) training configs for the PubMed backdoor trainers.

Looks for  train_configs/<pipeline>/<bits>/<encoder-stem>.json  next to this file and returns the
loss / training knobs it holds. Used by prepare_pubmed.py (QAT) and prepare_pubmed_ptq.py (PTQ).

Precedence everywhere:  explicit env var (or CLI flag)  >  this JSON  >  built-in default.
So the Optuna tuner (which sets env per trial) is unaffected, and a plain
`python prepare_pubmed_ptq.py --encoder nemotron-1b --bits int8` picks up the saved config.

Recognised keys (all optional; missing file or key -> fall back to the default):
  lam_drop, lam_keep, lam_clean   dual-objective loss weights
  lam_gain                        per-channel scale lever (PTQ only; QAT folds scale into the scale)
  n_last                          # of deepest blocks attacked
  steps                           training iterations
  train_batch                     per-step batch size
"""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))

# encoder key -> config-file stem (matches the distribution/ and defense/ folder names)
ENC_FILE = {"bge-gemma2": "bge-gemma", "embeddinggemma": "embeddinggemma", "nemotron-1b": "llama-nemotron"}


def config_path(pipeline, bits, encoder):
    stem = ENC_FILE.get(encoder, encoder)
    return os.path.join(HERE, "train_configs", pipeline, bits, f"{stem}.json")


def load(pipeline, bits, encoder):
    """Return the config dict for this (pipeline, bits, encoder), or {} if there is no file."""
    p = config_path(pipeline, bits, encoder)
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"[config] skipping {p} ({e})", flush=True)
        return {}
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}   # drop _comment etc.
    print(f"[config] {p}: {cfg}", flush=True)
    return cfg


def resolve_env(var, cfg_value, user_value):
    """Decide one env-backed knob with precedence  user_value > cfg_value > unset, and WRITE the
    result into os.environ so call-time readers (the encoder builder, the batch size, ...) see it.

    user_value is the value the env var held BEFORE training started (capture it once, before any
    per-bits loop, so a different JSON per bits can't be masked by a previous iteration's write).
    Returns the effective string, or None when nothing is set (built-in default then applies)."""
    if user_value is not None:
        val = user_value
    elif cfg_value is not None:
        val = str(cfg_value)
    else:
        os.environ.pop(var, None)
        return None
    os.environ[var] = val
    return val


def pick(user_value, cfg_value, builtin, cast=float):
    """Scalar with precedence  user_value > cfg_value > builtin (does not touch os.environ).
    user_value is the already-resolved env/CLI value (or None); cfg_value comes from the JSON."""
    if user_value is not None:
        return cast(user_value)
    if cfg_value is not None:
        return cast(cfg_value)
    return builtin
