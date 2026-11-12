"""Extrapolated verifier-overhead simulation: full re-quantization vs the QuaDRA weak verifier,
for models from 1B to 100B parameters.

Nothing here loads a model -- the two defenses have closed-form verifier costs, so we extrapolate
from the real small-model runs already on disk (embeddinggemma 0.58B, llama-nemotron 0.97B). Two
panels, both log-log:

  (1) estimated wall-time (s)   (2) data / memory the verifier must handle (bytes)

In each: one line that grows with model size (the baseline recomputes/holds *all* N weights) and one
flat line (our weak verifier opens a fixed m* = ceil(ln(1-p)/ln(1-rho0)) weights, independent of N).

Cost model (verifier side):
  Full re-quantization (verify.py)      recompute honest quant for every weight + hold the FP model
                                        time  = N / R_requant          bytes = 2*N        (bf16 FP)
  QuaDRA weak verifier (phase2.py)      check m* Merkle openings (recompute only those m* weights)
                                        time  = m*|C_open + T_sig       bytes = m*|per_opening + roots
                                        (+ an O(N) SHA pass over the INT model -- shown faint; a cheap
                                         hash, not re-quantization, and the baseline must read the INT
                                         dump too)

Rates are anchored to the measured runs (see ANCHORS); treat absolute seconds as order-of-magnitude.
Writes overhead_simulation.png into this folder and prints the table.
"""
from __future__ import annotations
import os
import numpy as np

# ------------------------------------------------------------------ verifier policy (same as phase2)
RHO0, P_CONF = 5e-4, 0.999
M_STAR = int(np.ceil(np.log(1 - P_CONF) / np.log(1 - RHO0)))        # 13813 for (5e-4, 0.999)

# ------------------------------------------------------------------ anchors from the real runs
# verify.py full recompute : nemotron 973M weights in 7.5s (warm)  -> ~1.3e8 wt/s
# phase2   opening verify   : m=2000 in 0.088s / m*=13813 in 0.596s -> ~4.3e-5 s per opened weight
# phase2   INT SHA pass     : ~0.9s over ~0.9e9 int8 weights        -> ~1e9 wt/s
# Ed25519 verify            : ~0.1 ms
R_REQUANT = 1.3e8          # weights / s   (full honest recompute, CPU)
C_OPEN    = 4.3e-5         # s / opened weight (Merkle path check + recompute of that weight)
T_SIG     = 1.0e-4         # s   (signature verify, fixed)
R_HASH    = 1.0e9          # weights / s   (SHA-256 pass over the INT model)
BYTES_FP  = 2             # bytes / weight (bf16 FP model the baseline must hold)
AVG_IN    = 4096          # avg linear in-dim, only sets the row (absmax) Merkle depth
HASH      = 32            # SHA-256 node size (bytes)
OPEN_HDR  = 13            # per opening: index(8) + w_bits(2) + a_bits(2) + q(1)

# real calibration points (N weights, measured full-verify s, measured weak-verify s, measured proof B)
ANCHORS = [("embeddinggemma", 583_008_256, 1.1, 0.596, 21_758.3 * 1024),
           ("llama-nemotron", 973_078_528, 7.5, None,  None)]


def costs(P: np.ndarray):
    """Verifier cost vs parameter count P (weights N ~= P). Returns a dict of arrays."""
    N = P.astype(float)
    R = np.maximum(N / AVG_IN, 1.0)
    depth_w = np.ceil(np.log2(N))
    depth_a = np.ceil(np.log2(R))
    per_open = HASH * (depth_w + depth_a) + OPEN_HDR
    return dict(
        # ---- time (s) ----
        base_t=N / R_REQUANT,                      # recompute every weight
        ours_t=np.full_like(N, M_STAR * C_OPEN + T_SIG),   # open m*  (flat)
        hash_t=N / R_HASH,                         # O(N) SHA pass (caveat)
        # ---- bytes ----
        base_b=BYTES_FP * N,                       # hold the whole FP model
        ours_b=M_STAR * per_open + 2 * HASH + 12,  # proof: m* openings + roots + header (~flat)
    )


def _style(ax, ink, ink2, grid):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, which="major", color=grid, lw=0.8)
    ax.grid(True, which="minor", color=grid, lw=0.4, alpha=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(grid)
    ax.tick_params(colors=ink2)
    ax.set_xlabel("model size (parameters)", color=ink)


def plot(P, c, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogFormatterMathtext
    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    base_c, ours_c, hash_c = "#2a78d6", "#eb6834", "#b8b7b3"
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12.4, 4.8), dpi=160)
    fig.patch.set_facecolor(surface)

    # -- panel 1: wall-time --
    a0.set_facecolor(surface)
    a0.plot(P, c["base_t"], color=base_c, lw=2.2,
            label="full re-quantization  (recompute all $N$, 100% confidence)")
    a0.plot(P, c["ours_t"], color=ours_c, lw=2.2,
            label=rf"QuaDRA weak verifier  (open $m^*={M_STAR}$ at 99.9% confidence)")
    a0.plot(P, c["hash_t"], color=hash_c, lw=1.4, ls=(0, (4, 3)),
            label=r"$O(N)$ for hashing the quantized model (extrapolated)")
    for nm, N, tv, *_ in ANCHORS:
        a0.plot(N, tv, "o", color=base_c, ms=6, mec=surface, mew=1.2, zorder=5)
    for nm, N, tv, ov, *_ in ANCHORS:
        if ov:
            a0.plot(N, ov, "o", color=ours_c, ms=6, mec=surface, mew=1.2, zorder=5)
    _style(a0, ink, ink2, grid)
    a0.set_ylabel("estimated verifier wall-time (s)", color=ink)
    a0.set_title("Compute: recompute every weight vs open $m^*$", color=ink, fontsize=10, loc="left")

    # -- panel 2: data / memory --
    a1.set_facecolor(surface)
    a1.plot(P, c["base_b"], color=base_c, lw=2.2, label="full re-quantization  (hold FP model, $2N$ B)")
    a1.plot(P, c["ours_b"], color=ours_c, lw=2.2, label=rf"QuaDRA weak verifier  (proof, $m^*={M_STAR}$)")
    for nm, N, tv, ov, pb in ANCHORS:
        a1.plot(N, BYTES_FP * N, "o", color=base_c, ms=6, mec=surface, mew=1.2, zorder=5)
        if pb:
            a1.plot(N, pb, "o", color=ours_c, ms=6, mec=surface, mew=1.2, zorder=5)
    _style(a1, ink, ink2, grid)
    a1.set_ylabel("verifier data / memory (bytes)", color=ink)
    a1.set_title("Data: hold the FP model vs a fixed-size proof", color=ink, fontsize=10, loc="left")

    for ax in (a0, a1):
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        leg = ax.legend(loc="upper left", frameon=False, fontsize=8.5)
        for t in leg.get_texts():
            t.set_color(ink)

    fig.suptitle(rf"Verifier overhead vs model size (extrapolated)   "
                 rf"policy: $\rho_0 = 5\times10^{{-4}}$, $p = {P_CONF:g}$, $m^* = {M_STAR}$",
                 color=ink, fontsize=11.5, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0.0, 1, 0.94))
    fig.savefig(png, facecolor=surface); plt.close(fig)


def main():
    P = np.logspace(9, 11, 80)                 # 1B .. 100B
    c = costs(P)
    here = os.path.dirname(os.path.abspath(__file__))
    png = os.path.join(here, "overhead_simulation.png")
    plot(P, c, png)

    print(f"policy: rho0={RHO0:g}  p={P_CONF}  ->  m* = {M_STAR} weights opened (fixed)\n")
    print(f"{'params':>8} | {'full re-quant':>24} | {'QuaDRA weak verifier':>24} | {'speedup / saving':>18}")
    print(f"{'':>8} | {'time':>11} {'FP bytes':>12} | {'time':>11} {'proof B':>12} |")
    print("-" * 96)
    for Pi in (1e9, 3e9, 1e10, 3e10, 1e11):
        ci = costs(np.array([Pi]))
        bt, bb = ci["base_t"][0], ci["base_b"][0]
        ot, ob = ci["ours_t"][0], ci["ours_b"][0]
        hp = lambda x: (f"{x/1e9:.1f}GB" if x >= 1e9 else f"{x/1e6:.1f}MB")
        print(f"{Pi/1e9:6.0f}B | {bt:8.1f}s {hp(bb):>12} | {ot:8.2f}s {hp(ob):>12} | "
              f"{bt/ot:7.0f}x / {bb/ob:6.0f}x")
    print(f"\nwrote {png}")


if __name__ == "__main__":
    main()
