"""Full recompute verifier -- rebuild the honest quantized model from the FP model and compare a
published INT dump bit-exactly. Port of defense-and-distribution/defense/verify.py, made
registry-aware (encoder chosen via demo_encoder_ptq.set_encoder before load_fp_linears).

Honest recipe S (SPEC.md): per-output-channel symmetric, s = absmax/qmax, round-half-to-even,
narrow range, arithmetic = the producer's model dtype (bf16). Same inputs + same arithmetic => same
bits, so an honest dump matches on every weight and scale; any single difference => TAMPERED.
"""
from __future__ import annotations
import os
import sys
import time
import logging

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import demo_encoder_ptq as ptq      # registry: MODEL / _SPEC["block"] / _layer_index

log = logging.getLogger("verify")
DTYPE = torch.bfloat16
SCALE_EPS = 1e-8


def honest_quantize(w: torch.Tensor, qmax: int):
    """S on one [out,in] weight in bf16: s=clamp(absmax/qmax,1e-8); q=clamp(round(w/s),-qmax,qmax)."""
    w = w.to(DTYPE)
    s = (w.abs().amax(dim=1, keepdim=True) / qmax).clamp(min=SCALE_EPS)
    q = torch.clamp(torch.round(w / s), -qmax, qmax)
    return s.float().reshape(-1), q.to(torch.int8)


def load_fp_linears(fp_dir: str):
    """{layer: (weight bf16 [out,in], bias|None)} for every block nn.Linear, keyed f'{module}.{child}'.
    Loads the local trusted FP copy if fp_dir is a directory, else the public HF repo (registry MODEL)."""
    from transformers import AutoModel
    trust = ptq._SPEC.get("trust_remote", False)
    src = fp_dir if os.path.isdir(fp_dir) else ptq.MODEL
    block = ptq._SPEC["block"]
    log.info("FP reference: %s  (block=%r, trust_remote=%s)", src, block, trust)
    model = AutoModel.from_pretrained(src, torch_dtype=DTYPE, trust_remote_code=trust).eval()
    fp = {}
    for name, module in model.named_modules():
        if block not in name:
            continue
        for cn, child in module.named_children():
            if isinstance(child, nn.Linear):
                fp[f"{name}.{cn}"] = (child.weight.detach(),
                                      None if child.bias is None else child.bias.detach())
    log.info("FP reference: %d block-linear layers, %d weights", len(fp),
             sum(w.numel() for w, _ in fp.values()))
    del model
    return fp


def verify_dump(path: str, fp: dict) -> bool:
    log.info("=" * 90)
    log.info("VERIFY %s", path)
    t0 = time.time()
    blob = torch.load(path, map_location="cpu")
    meta, layers = blob.get("meta", {}), blob["layers"]
    qmax = int(meta["qmax"])
    log.info("  meta: bits=%s qmax=%d rounding(claimed)=%s pipeline=%s", meta.get("bits"), qmax,
             meta.get("rounding"), meta.get("pipeline"))

    structural = []
    missing = sorted(set(fp) - set(layers))
    extra = sorted(set(layers) - set(fp))
    if missing:
        structural.append(f"{len(missing)} layers missing from dump, e.g. {missing[:3]}")
    if extra:
        structural.append(f"{len(extra)} layers in dump but not in FP model, e.g. {extra[:3]}")

    n_w = n_w_bad = n_rows = n_rows_bad = 0
    bad_layers = []
    for name, (w, b) in fp.items():
        if name not in layers:
            continue
        L = layers[name]
        q_pub = L["q_w"].to(torch.int8)
        s_pub = L["s_w"].float().reshape(-1)
        if tuple(q_pub.shape) != tuple(w.shape) or s_pub.numel() != w.shape[0]:
            structural.append(f"{name}: shape q_w={tuple(q_pub.shape)} s_w={s_pub.numel()} "
                              f"!= FP {tuple(w.shape)}")
            continue
        b_pub = L.get("bias")
        if (b is None) != (b_pub is None) or (b is not None and not torch.equal(b.float(),
                                                                                 b_pub.float())):
            structural.append(f"{name}: bias differs from FP bias")
        s_ref, q_ref = honest_quantize(w, qmax)
        dw = int((q_pub != q_ref).sum())
        ds = int((s_pub != s_ref).sum())
        n_w += q_ref.numel(); n_w_bad += dw
        n_rows += s_ref.numel(); n_rows_bad += ds
        if dw or ds:
            bad_layers.append((name, dw, q_ref.numel(), ds, s_ref.numel()))
            log.info("  MISMATCH  %-60s weights %8d/%-9d (%6.3f%%)  scales %5d/%-5d",
                     name[-60:], dw, q_ref.numel(), 100.0 * dw / q_ref.numel(), ds, s_ref.numel())
    for msg in structural:
        log.info("  STRUCTURE %s", msg)

    tampered = bool(n_w_bad or n_rows_bad or structural)
    log.info("  ---- summary ----")
    log.info("  layers checked        : %d (with mismatches: %d)", len(fp) - len(missing), len(bad_layers))
    log.info("  integer weights       : %d checked, %d differ (rho = %.6f)", n_w, n_w_bad,
             n_w_bad / max(1, n_w))
    log.info("  per-row scales        : %d checked, %d differ", n_rows, n_rows_bad)
    log.info("  structural problems   : %d", len(structural))
    if bad_layers:
        depths = sorted({ptq._layer_index(n) for n, *_ in bad_layers})
        towers = sorted({n.split(".")[0] for n, *_ in bad_layers})
        log.info("  mismatches located in : %s, block depths %s", towers, depths)
    log.info("  VERDICT: %s   (%.1fs)", "TAMPERED -- FAIL" if tampered else "HONEST -- PASS",
             time.time() - t0)
    return not tampered


def run_verify(fp: dict, dumps: list) -> dict:
    """Verify each dump against the shared FP reference; return {path: passed}."""
    log.info("spec S: per-output-channel symmetric, s=absmax/qmax, round-half-to-even, "
             "narrow range, arithmetic=%s", DTYPE)
    results = {p: verify_dump(p, fp) for p in dumps if os.path.exists(p)}
    for p in dumps:
        if not os.path.exists(p):
            log.info("  [skip] dump not found: %s", p)
    log.info("=" * 90)
    log.info("VERIFY RESULTS")
    for p, ok in results.items():
        log.info("  %-8s %s", "PASS" if ok else "FAIL", p)
    return results
