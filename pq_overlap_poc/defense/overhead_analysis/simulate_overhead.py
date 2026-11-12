"""Verifier overhead vs model size: full recompute vs Merkle subset-opening weak verifier.

Cost model (both verifiers are O(N); only the constants differ):

  T_full(N) = a_full * N                         requantize + compare every weight   (compute)
            + (2N + N + 4R) / BW                 read FP bf16 + INT8 weights + fp32 scales (I/O)

  T_weak(N) = a_hash * (N + 4R)                  SHA-256 of the INT model (Fiat-Shamir binding)
            + (N + 4R) / BW                      read the INT model                 (I/O)
            + m * (c0 + c1 * (depth_w + depth_a)) check_proof on m openings (paths grow as log2 N)
            + t_sig                              Ed25519 verify of root_FP

  m = ceil(ln(1-conf) / ln(1-rho))   -- independent of N (1379 for conf=99.9%, rho=0.5%).
  R = N / K_IN rows (K_IN = mean input dim, from the real model; only affects the tiny scale bytes).

CALIBRATION (the curves come from this alone, all synthetic):
  a_full  time verify.honest_quantize + compare on random bf16 tensors of several layer shapes
  a_hash  SHA-256 throughput over random int8 buffers (same hashing as phase2.IntView)
  c0, c1  phase2.Phase2Verifier.check_proof at m openings on synthetic tables of 1B..100B weights
          (random sibling paths of the true depth: cost is identical, the verdict is irrelevant)
  t_sig   Ed25519 verify
  BW      cold read (page cache dropped via posix_fadvise) of a synthetic file on --io-dir

VALIDATION (the dots; only job of the real checkpoints): re-time the same operations on the real
encoder (default nemotron-1b, malicious = rounding + scale) on the same machine:
  full  cold read of the FP safetensors + INT dump (scaled to the bytes the verifier needs: the
        block-linear weights), + the verify.py compare loop on the real FP vs the real INT dump
  weak  cold read of the INT dump + real IntView hash + real check_proof on a real proof at m
        (builds the real FPTree, ~6 min on 64 procs; --no-tree uses the logged phase2 verify_s)

Usage (cluster env; needs torch, transformers, cryptography, HF access):
  python simulate_overhead.py                                  # calibrate + validate on nemotron
  python simulate_overhead.py --procs $SLURM_CPUS_PER_TASK
  python simulate_overhead.py --no-tree                        # skip the 6-min real Merkle build
  python simulate_overhead.py --skip-validation                # synthetic curves only
  python simulate_overhead.py --io-gbps 2.0                    # fix BW instead of measuring it

Outputs (overhead_analysis/): overhead_results.json, overhead_scaling.png, overhead_analysis.log
"""
from __future__ import annotations

import os
import sys
import json
import math
import time
import logging
import argparse

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEFENSE = os.path.dirname(HERE)
ROOT = os.path.dirname(DEFENSE)
for _p in (DEFENSE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common                                                                    # noqa: E402
import verify                                                                    # noqa: E402
import phase2                                                                    # noqa: E402
from phase2 import (BLOCK, BLOCK_LOG, Opening, OpeningProof, Phase2Verifier,     # noqa: E402
                    challenge, fp_root, layer_table_hash, row_of, m_for)
from roles import Commitment, SignedCommitment, DistributorPublicKey, FPModel   # noqa: E402
from spec import QuantSpec                                                       # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

log = logging.getLogger("overhead")

K_IN_DEFAULT = 2582            # nemotron-1b: N / R = 973078528 / 376832 (mean input dim)
FULL_SHAPES = [(2048, 2048), (4096, 4096), (14336, 4096), (4096, 14336)]
HASH_SIZES_MB = [128, 512, 1024]
OPEN_N_CAL = [1e9, 3e9, 1e10, 3e10, 1e11]
ORANGE, BLUE = "#eb6834", "#2a78d6"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _median_time(fn, reps, warmup=1):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def _min_time(fn, reps, warmup=1):
    """Min over reps: for sub-second micro-benchmarks, the least noisy estimate of the true cost."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(min(ts))


def depth_w(n: int) -> int:
    nb = -(-int(n) // BLOCK)
    return BLOCK_LOG + (math.ceil(math.log2(nb)) if nb > 1 else 0)


def depth_a(r: int) -> int:
    return math.ceil(math.log2(r)) if r > 1 else 0


# ============================================================================ I/O
def cold_read(path, chunk=64 << 20):
    """Read a whole file after asking the kernel to drop its cached pages; return throughput."""
    size = os.path.getsize(path)
    fd = os.open(path, os.O_RDONLY)
    dropped = False
    try:
        if hasattr(os, "posix_fadvise"):
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                dropped = True
            except OSError:
                pass
        buf = memoryview(bytearray(chunk))
        f = os.fdopen(fd, "rb", buffering=0, closefd=False)
        n, t0 = 0, time.perf_counter()
        while True:
            k = f.readinto(buf)
            if not k:
                break
            n += k
        dt = time.perf_counter() - t0
    finally:
        os.close(fd)
    gbps = n / dt / 1e9
    if not dropped:
        log.warning("  could not drop page cache for %s (no posix_fadvise) -- BW may be optimistic; "
                    "consider --io-gbps", path)
    elif gbps > 20:
        log.warning("  cold read of %s at %.1f GB/s -- likely still served from page cache", path, gbps)
    return dict(path=path, bytes=size, seconds=dt, gbps=gbps, cache_dropped=dropped)


def calibrate_io(io_dir, size_gb):
    path = os.path.join(io_dir, f".overhead_io_probe_{os.getpid()}.bin")
    block = np.random.default_rng(0).integers(0, 256, 64 << 20, dtype=np.uint8).tobytes()
    nblk = max(1, int(size_gb * 1e9) // len(block))
    try:
        with open(path, "wb") as f:
            for _ in range(nblk):
                f.write(block)
            f.flush()
            os.fsync(f.fileno())
        r = cold_read(path)
    finally:
        if os.path.exists(path):
            os.remove(path)
    log.info("  I/O  cold read %.2f GB in %.2fs -> %.2f GB/s  (cache dropped: %s, dir=%s)",
             r["bytes"] / 1e9, r["seconds"], r["gbps"], r["cache_dropped"], io_dir)
    return r


# ============================================================================ synthetic calibration
def _full_compare(w, q_pub, s_pub, qmax):
    """The per-layer work of verify.verify_dump (minus logging): honest requantize + compare."""
    s_ref, q_ref = verify.honest_quantize(w, qmax)
    return int((q_pub != q_ref).sum()), int((s_pub != s_ref).sum())


def calibrate_full(qmax, reps):
    rows = []
    for out, inn in FULL_SHAPES:
        w = (torch.randn(out, inn) * 0.02).to(torch.bfloat16)
        s_pub, q_pub = verify.honest_quantize(w, qmax)
        t = _median_time(lambda: _full_compare(w, q_pub, s_pub, qmax), reps)
        rows.append(dict(shape=[out, inn], weights=out * inn, seconds=t, ns_per_weight=t / (out * inn) * 1e9))
        log.info("  full recompute  %-13s %6.3fs  %.3f ns/weight", f"{out}x{inn}", t, rows[-1]["ns_per_weight"])
        del w, s_pub, q_pub
    a = sum(r["seconds"] for r in rows) / sum(r["weights"] for r in rows)
    log.info("  -> a_full = %.3f ns/weight  (torch threads=%d)", a * 1e9, torch.get_num_threads())
    return a, rows


def calibrate_hash(reps):
    import hashlib
    rows = []
    for mb in HASH_SIZES_MB:
        buf = np.random.default_rng(1).integers(-127, 128, mb << 20, dtype=np.int8)

        def h():
            x = hashlib.sha256(phase2.TAG_INT)
            x.update(memoryview(buf))
            return x.digest()
        t = _median_time(h, reps)
        rows.append(dict(bytes=buf.nbytes, seconds=t, gbps=buf.nbytes / t / 1e9))
        log.info("  sha256  %5d MB  %6.3fs  %.2f GB/s", mb, t, rows[-1]["gbps"])
        del buf
    a = sum(r["seconds"] for r in rows) / sum(r["bytes"] for r in rows)
    log.info("  -> a_hash = %.3f ns/byte (%.2f GB/s)", a * 1e9, 1 / a / 1e9)
    return a, rows


class _Zeros:
    """Stand-in for IntView.q / .s at sizes we cannot allocate (100B). Indexing a real array at m
    random positions costs microseconds; the hashing / recompute work in check_proof is unchanged."""

    def __init__(self, dtype):
        self.dtype = dtype

    def __getitem__(self, k):
        if isinstance(k, (int, np.integer)):
            return self.dtype(0)
        return np.zeros(len(k), self.dtype)


class _FakeView:
    def __init__(self):
        self.digest = os.urandom(32)
        self.q = _Zeros(np.int8)
        self.s = _Zeros(np.float32)


def synthetic_table(n_target, k_in):
    out = 8192
    count = max(1, int(round(n_target / (out * k_in))))
    return [(f"syn.{i}", (out, k_in)) for i in range(count)]


def _sign_setup(table, root_w, root_a):
    n = sum(o * i for _, (o, i) in table)
    r = sum(o for _, (o, _i) in table)
    root = fp_root(root_w, root_a, layer_table_hash(table), n, r)
    sk = Ed25519PrivateKey.generate()
    signed = SignedCommitment(Commitment(root), sk.sign(root))
    pk = DistributorPublicKey(sk.public_key().public_bytes_raw())
    return signed, pk, n, r


def synthetic_open_time(n_target, k_in, m, qmax, reps):
    """Time the real Phase2Verifier.check_proof with m openings whose paths have the true depths."""
    table = synthetic_table(n_target, k_in)
    root_w, root_a = os.urandom(32), os.urandom(32)
    signed, pk, n, r = _sign_setup(table, root_w, root_a)
    ver = Phase2Verifier(QuantSpec.for_qmax(qmax), table, signed, pk)
    view = _FakeView()
    ctx = b"overhead-syn"
    idx = challenge(signed.commitment.root, view.digest, ctx, m, n)
    rows = row_of(ver.offsets, np.asarray(idx, np.int64))
    dw, da = depth_w(n), depth_a(r)
    w_bits = (torch.randn(m) * 0.02).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    a_bits = int(torch.tensor(0.1, dtype=torch.bfloat16).view(torch.int16).item()) & 0xFFFF
    ops = tuple(Opening(int(i), int(wb), tuple(os.urandom(32) for _ in range(dw)), a_bits,
                        tuple(os.urandom(32) for _ in range(da)), 0)
                for i, wb, _r in zip(idx, w_bits, rows))
    proof = OpeningProof(root_w, root_a, ctx, m, ops)
    t = _min_time(lambda: ver.check_proof(None, proof, view=view), max(reps, 7))
    assert ver.report["roots"] == 0 and ver.report["index_set"] == 0, ver.report  # took the full path
    return dict(n=n, r=r, depth_w=dw, depth_a=da, m=m, seconds=t)


def calibrate_open(k_in, m, qmax, reps):
    rows = []
    for nt in OPEN_N_CAL:
        rows.append(synthetic_open_time(nt, k_in, m, qmax, reps))
        x = rows[-1]
        log.info("  check_proof  N=%6.1fB  depth w=%d a=%d  m=%d  %.4fs", x["n"] / 1e9, x["depth_w"],
                 x["depth_a"], m, x["seconds"])
    d = np.array([x["depth_w"] + x["depth_a"] for x in rows], float)
    per = np.array([x["seconds"] / x["m"] for x in rows])
    c1, c0 = np.polyfit(d, per, 1)
    if c0 < 0 or c1 < 0:                     # noisy timings: fall back to cost proportional to depth
        c0, c1 = 0.0, float((per * d).sum() / (d * d).sum())
        log.info("  (affine fit not physical; using through-origin fit)")
    log.info("  -> per opening = %.2f us + %.3f us x (depth_w + depth_a)", c0 * 1e6, c1 * 1e6)
    return float(c0), float(c1), rows


def calibrate_sig(reps=500):
    sk = Ed25519PrivateKey.generate()
    pub, msg = sk.public_key(), os.urandom(32)
    sig = sk.sign(msg)
    t = _median_time(lambda: pub.verify(sig, msg), reps, warmup=10)
    log.info("  ed25519 verify %.3f ms", t * 1e3)
    return t


# ============================================================================ model
def predict(n, cal, m, k_in=None, r=None):
    n = int(n)
    r = int(r) if r is not None else max(1, int(round(n / (k_in or K_IN_DEFAULT))))
    bw = cal["io_bytes_per_s"]
    full_c = cal["a_full"] * n
    full_io = (2 * n + n + 4 * r) / bw
    weak_hash = cal["a_hash"] * (n + 4 * r)
    weak_io = (n + 4 * r) / bw
    opens = m * (cal["c0"] + cal["c1"] * (depth_w(n) + depth_a(r)))
    return dict(n=n, r=r, depth_w=depth_w(n), depth_a=depth_a(r),
                full_compute_s=full_c, full_io_s=full_io, full_total_s=full_c + full_io,
                weak_hash_s=weak_hash, weak_io_s=weak_io, weak_open_s=opens, sig_s=cal["t_sig"],
                weak_openings_only_s=opens + cal["t_sig"],
                weak_total_s=weak_hash + weak_io + opens + cal["t_sig"])


# ============================================================================ validation (real)
def _hf_weight_files(model_id):
    from huggingface_hub import snapshot_download
    snap = snapshot_download(model_id, local_files_only=True)
    names = os.listdir(snap)
    ext = ".safetensors" if any(f.endswith(".safetensors") for f in names) else ".bin"
    return [os.path.realpath(os.path.join(snap, f)) for f in names
            if f.endswith(ext) and not f.startswith("training_args")]


def validate(args, m, cal):
    import demo_encoder_ptq as ptq
    pipeline, bits, qmax = common.parse_mode(args.mode)
    ptq.set_encoder(args.encoder)
    mal = common.malicious_path(args.encoder, pipeline, bits)
    if not os.path.exists(mal):
        raise SystemExit(f"[overhead] malicious checkpoint not found: {mal}")
    spec = QuantSpec.for_qmax(qmax)

    log.info("loading FP reference (%s)", args.encoder)
    fp = verify.load_fp_linears(common.fp_dir(args.encoder))
    table = [(name, tuple(w.shape)) for name, (w, _b) in fp.items()]
    n = sum(o * i for _, (o, i) in table)
    r = sum(o for _, (o, _i) in table)
    log.info("  N=%d block-linear weights, R=%d rows, K_IN=N/R=%.0f", n, r, n / r)

    # ---- cold reads of the real files (scaled to the bytes each verifier actually needs)
    fpd = common.fp_dir(args.encoder)
    fp_files = ([os.path.join(dp, f) for dp, _d, fs in os.walk(fpd) for f in fs
                 if f.endswith((".safetensors", ".bin"))] if os.path.isdir(fpd)
                else _hf_weight_files(ptq.MODEL))
    fp_reads = [cold_read(p) for p in fp_files]
    fp_bytes = sum(x["bytes"] for x in fp_reads)
    fp_secs = sum(x["seconds"] for x in fp_reads)
    int_read = cold_read(mal)
    log.info("  cold read FP files: %.2f GB in %.2fs (%.2f GB/s) | INT dump: %.2f GB in %.2fs (%.2f GB/s)",
             fp_bytes / 1e9, fp_secs, fp_bytes / fp_secs / 1e9, int_read["bytes"] / 1e9,
             int_read["seconds"], int_read["gbps"])
    io_fp_needed = fp_secs * (2 * n) / fp_bytes                         # block-linear bf16 only
    io_int_needed = int_read["seconds"] * (n + 4 * r) / int_read["bytes"]

    # ---- full recompute on the real FP vs the real malicious dump (same loop as verify.py)
    blob = torch.load(mal, map_location="cpu")
    layers = blob["layers"]

    def full_loop():
        for name, (w, _b) in fp.items():
            L = layers[name]
            _full_compare(w, L["q_w"].to(torch.int8), L["s_w"].float().reshape(-1), qmax)
    t_full_c = _median_time(full_loop, args.reps_real)
    log.info("  full recompute compute (real, %d layers): %.2fs", len(fp), t_full_c)

    # ---- weak verifier: real IntView hash + real check_proof at m
    im = phase2.load_int_model(mal, spec)
    del blob, layers
    view = phase2.IntView(im, table)
    t_hash = view.hash_seconds
    log.info("  INT hash (real IntView): %.3fs", t_hash)
    prover = None
    if args.no_tree:
        res = json.load(open(os.path.join(DEFENSE, common.ENC_DIR[args.encoder],
                                          f"phase2_results_{args.mode}.json")))
        e2e = res["models"]["malicious (rounding + scale)"]["e2e"]
        t_open = float(np.interp(m, [x["m"] for x in e2e], [x["verify_s"] for x in e2e]))
        open_src = "logged phase2_results verify_s, interpolated at m"
        log.info("  check_proof at m=%d (from phase2_results, interpolated): %.4fs", m, t_open)
    else:
        log.info("  building real FPTree on %d procs (distributor commitment) ...", args.procs)
        tree = phase2.FPTree(fp, args.procs)
        fp_model = FPModel(weights={k: w for k, (w, _b) in fp.items()}, layer_order=tuple(fp))
        dist = phase2.Phase2Distributor(fp_model, tree)
        _fpm, signed, pk = dist.publish()
        quant = phase2.Phase2Quantizer(fp_model, spec, tree)
        ver = Phase2Verifier(spec, tree.table, signed, pk)
        t0 = time.perf_counter()
        proof = quant.prove(im, signed, view=view, context=b"overhead-real", m=m)
        t_prove = time.perf_counter() - t0
        t_open = _median_time(lambda: ver.check_proof(im, proof, view=view), args.reps_real)
        rejected = not ver.check_proof(im, proof, view=view)
        open_src = "real FPTree proof, Phase2Verifier.check_proof"
        prover = dict(tree_build_s=tree.build_seconds, procs=args.procs, prove_s=t_prove,
                      proof_bytes=len(proof.serialize()))
        log.info("  tree build %.1fs | prove m=%d %.2fs | check_proof %.4fs | malicious rejected: %s",
                 tree.build_seconds, m, t_prove, t_open, rejected)

    meas = dict(full_compute_s=t_full_c, full_io_s=io_fp_needed + io_int_needed,
                full_total_s=t_full_c + io_fp_needed + io_int_needed,
                weak_hash_s=t_hash, weak_io_s=io_int_needed, weak_open_s=t_open, sig_s=cal["t_sig"],
                weak_openings_only_s=t_open + cal["t_sig"],
                weak_total_s=t_hash + io_int_needed + t_open + cal["t_sig"])
    pred = predict(n, cal, m, r=r)
    ratio = {k: pred[k] / meas[k] for k in ("full_total_s", "weak_total_s", "weak_openings_only_s",
                                            "full_compute_s", "weak_hash_s", "weak_open_s") if meas[k] > 0}
    log.info("  VALIDATION  predicted / measured at N=%.3fB:", n / 1e9)
    for k, v in ratio.items():
        log.info("    %-22s pred %8.3fs  meas %8.3fs  ratio %.2f", k, pred[k], meas[k], v)
    return dict(encoder=args.encoder, mode=args.mode, malicious_dump=mal, n=n, r=r, k_in=n / r,
                fp_files=fp_reads, int_read=int_read, measured=meas, predicted=pred,
                predicted_over_measured=ratio, openings_source=open_src, prover_side=prover)


# ============================================================================ plot
def plot(grid, val, m, rho, conf, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ns = np.array([g["n"] for g in grid], float)
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=160)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    ax.plot(ns, [g["full_total_s"] for g in grid], color=ORANGE, lw=2.2,
            label="full recompute (FP read + requantize + compare)")
    ax.plot(ns, [g["weak_total_s"] for g in grid], color=BLUE, lw=2.2,
            label=f"weak verifier, m={m} (INT read + hash + openings)")
    ax.plot(ns, [g["weak_openings_only_s"] for g in grid], color=BLUE, lw=1.6, ls=(0, (4, 3)),
            label=f"weak verifier, openings only (m={m} + signature)")
    if val:
        n, ms = val["n"], val["measured"]
        lab = f'{val["encoder"]} measured'
        ax.plot([n], [ms["full_total_s"]], "o", color=ORANGE, ms=8, mec=INK, mew=1, zorder=5, label=lab)
        ax.plot([n], [ms["weak_total_s"]], "o", color=BLUE, ms=8, mec=INK, mew=1, zorder=5)
        ax.plot([n], [ms["weak_openings_only_s"]], "o", color=SURF, ms=8, mec=BLUE, mew=1.6, zorder=5)
    last = grid[-1]
    sp = last["full_total_s"] / last["weak_total_s"]
    ax.annotate(f'{last["full_total_s"]:.0f}s vs {last["weak_total_s"]:.0f}s  ({sp:.0f}x)',
                xy=(ns[-1], last["weak_total_s"]), xytext=(ns[-1] * 0.22, last["weak_total_s"] * 0.12),
                color=INK2, fontsize=8, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xscale("log"); ax.set_yscale("log")
    ticks = [1e9, 3e9, 1e10, 3e10, 1e11]
    ax.set_xticks(ticks); ax.set_xticklabels(["1B", "3B", "10B", "30B", "100B"])
    ax.set_xlabel("model size (verified weights)", color=INK)
    ax.set_ylabel("verification time (s, log)", color=INK)
    ax.set_title(f"Verifier cost vs model size  ({conf:.1%} detection of rho={rho:.1%} tampered weights)",
                 color=INK, fontsize=10, loc="left")
    ax.grid(True, which="major", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout(); fig.savefig(png, facecolor=SURF); plt.close(fig)


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="nemotron-1b", choices=list(common.ENC_DIR),
                    help="real encoder for the validation dots")
    ap.add_argument("--mode", default="ptq_int8", choices=["ptq_int8", "ptq_int4", "qat_int8", "qat_int4"])
    ap.add_argument("--rho", type=float, default=0.005, help="tampered fraction to detect")
    ap.add_argument("--conf", type=float, default=0.999, help="detection probability")
    ap.add_argument("--n-min", type=float, default=1e9)
    ap.add_argument("--n-max", type=float, default=1e11)
    ap.add_argument("--io-gbps", type=float, default=None, help="fix storage bandwidth instead of measuring")
    ap.add_argument("--io-dir", default=None, help="dir for the synthetic I/O probe (default: artifacts dir)")
    ap.add_argument("--io-size-gb", type=float, default=2.0)
    ap.add_argument("--reps", type=int, default=3, help="timing repetitions (synthetic)")
    ap.add_argument("--reps-real", type=int, default=3, help="timing repetitions (real checkpoints)")
    ap.add_argument("--procs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4)),
                    help="processes for the real FPTree build")
    ap.add_argument("--no-tree", action="store_true", help="skip the real Merkle build; use logged verify_s")
    ap.add_argument("--skip-validation", action="store_true", help="synthetic curves only")
    args = ap.parse_args()

    os.makedirs(HERE, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    log.setLevel(logging.INFO); log.propagate = False
    for h in (logging.StreamHandler(), logging.FileHandler(os.path.join(HERE, "overhead_analysis.log"), mode="w")):
        h.setFormatter(fmt); log.addHandler(h)
    logging.getLogger("verify").addHandler(log.handlers[0])
    logging.getLogger("verify").setLevel(logging.INFO)

    _p, _b, qmax = common.parse_mode(args.mode)
    m = m_for(args.rho, args.conf)
    log.info("m = ceil(ln(1-%.4f)/ln(1-%.4f)) = %d openings (independent of N)", args.conf, args.rho, m)

    log.info("=" * 80 + "\nCALIBRATION (synthetic)")
    a_full, full_rows = calibrate_full(qmax, args.reps)
    a_hash, hash_rows = calibrate_hash(args.reps)
    c0, c1, open_rows = calibrate_open(K_IN_DEFAULT, m, qmax, args.reps)
    t_sig = calibrate_sig()
    if args.io_gbps:
        io = dict(gbps=args.io_gbps, source="--io-gbps")
        log.info("  I/O  fixed at %.2f GB/s (--io-gbps)", args.io_gbps)
    else:
        from prepare_pubmed import artifacts_dir
        io = calibrate_io(args.io_dir or artifacts_dir(args.encoder), args.io_size_gb)
        io["source"] = "synthetic cold read"
    cal = dict(a_full=a_full, a_hash=a_hash, c0=c0, c1=c1, t_sig=t_sig, io_bytes_per_s=io["gbps"] * 1e9)

    val = None
    if not args.skip_validation:
        log.info("=" * 80 + "\nVALIDATION (real %s %s)", args.encoder, args.mode)
        val = validate(args, m, cal)

    k_in = val["k_in"] if val else K_IN_DEFAULT
    ns = np.logspace(np.log10(args.n_min), np.log10(args.n_max), 60)
    grid = [predict(n, cal, m, k_in=k_in) for n in ns]
    png = os.path.join(HERE, "overhead_scaling.png")
    plot(grid, val, m, args.rho, args.conf, png)

    out = dict(rho=args.rho, conf=args.conf, m=m, qmax=qmax, k_in=k_in,
               constants=dict(a_full_ns_per_weight=a_full * 1e9, a_hash_ns_per_byte=a_hash * 1e9,
                              open_c0_us=c0 * 1e6, open_c1_us_per_level=c1 * 1e6, sig_ms=t_sig * 1e3,
                              io_gbps=io["gbps"], torch_threads=torch.get_num_threads()),
               calibration=dict(full_recompute=full_rows, sha256=hash_rows, check_proof=open_rows, io=io),
               validation=val, grid=grid,
               at_n_max=dict(full_total_s=grid[-1]["full_total_s"], weak_total_s=grid[-1]["weak_total_s"],
                             speedup=grid[-1]["full_total_s"] / grid[-1]["weak_total_s"]))
    with open(os.path.join(HERE, "overhead_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    log.info("=" * 80)
    log.info("SUMMARY  m=%d  (%.1f%% detection, rho=%.2f%%)", m, 100 * args.conf, 100 * args.rho)
    for g in (grid[0], grid[len(grid) // 2], grid[-1]):
        log.info("  N=%6.1fB  full %8.1fs  weak %7.2fs (openings %.3fs)  speedup %5.1fx",
                 g["n"] / 1e9, g["full_total_s"], g["weak_total_s"], g["weak_openings_only_s"],
                 g["full_total_s"] / g["weak_total_s"])
    log.info("  -> %s, %s", os.path.join(HERE, "overhead_results.json"), png)


if __name__ == "__main__":
    main()
