"""Plot a stealth-tuning results JSON: trigger & control stealth vs ASR, coloured by collateral.

Shared by all three encoder probes. Pass the path to a tune_dual_<mode>_results.json; the PNG is
written next to it (same name, .png extension), wherever that JSON lives.

Two panels (Trigger stealth vs ASR, Control stealth vs ASR). Points are coloured by the UNBOUNDED
insulin-ness of the evicted docs -- how far they sit on the [random corpus doc = 0, real insulin
doc = 1] scale (trig_collat for panel 1, ctrl_collat for panel 2). 10 contrasting bins across [0, 1]
run red -> violet; values outside [0, 1] use the under/over colours:
  - < 0         -> #800000 (dark maroon)   less insulin-like than a random doc
  - 0.0 - 0.1   -> #E6194B (red)
  - 0.1 - 0.2   -> #F58231 (orange)
  - 0.2 - 0.3   -> #FFB000 (amber)
  - 0.3 - 0.4   -> #FFE119 (yellow)
  - 0.4 - 0.5   -> #BFEF45 (lime)
  - 0.5 - 0.6   -> #3CB44B (green)
  - 0.6 - 0.7   -> #13A89E (teal)
  - 0.7 - 0.8   -> #42D4F4 (cyan)
  - 0.8 - 0.9   -> #4363D8 (blue)
  - 0.9 - 1.0   -> #911EB4 (violet)
  - > 1         -> #F032E6 (magenta)        more insulin-central than a real insulin doc
Trials that evicted nothing (no collateral to measure) are drawn grey.

Feasible points (ASR >= asr_min) are circles ('o'); infeasible points (ASR < asr_min) are 'X'.

Usage:
    python plot_tune.py <path/to/tune_dual_<mode>_results.json>
Output:
    <same path>.png  (e.g. .../tune_dual_ptq_int8_results.png)
"""
import os
import sys
import json

# Unbounded insulin-ness colour definition (0 = random corpus doc, 1 = real insulin doc).
BOUNDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
PALETTE = ["#E6194B", "#F58231", "#FFB000", "#FFE119", "#BFEF45",
           "#3CB44B", "#13A89E", "#42D4F4", "#4363D8", "#911EB4"]
UNDER = "#800000"   # < 0  dark maroon
OVER = "#F032E6"    # > 1  magenta


def get_collat_val(point, key):
    """Unbounded insulin-ness; <0 and >1 kept as-is (map to under/over). NaN -> grey."""
    val = point.get(key)
    try:
        val = float(val)
    except (TypeError, ValueError):
        return float("nan")
    return val if val == val else float("nan")               # NaN stays NaN


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python plot_tune.py <path/to/tune_dual_<mode>_results.json>")
    json_path = sys.argv[1]
    if not os.path.exists(json_path):
        sys.exit(f"ERROR: JSON file not found: {json_path}")

    png_path = json_path[:-5] + ".png" if json_path.endswith(".json") else json_path + ".png"

    with open(json_path, "r") as f:
        data = json.load(f)

    points = data.get("points") or data.get("trials", [])
    if not points:
        sys.exit(f"ERROR: No trials/points found in {json_path}")

    tag = data.get("mode", "ptq_int8")
    asr_min = float(data.get("asr_min", 0.5))
    stealth_goal = float(data.get("stealth_goal", data.get("trig_min", 0.9)))

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    cmap = mcolors.ListedColormap(PALETTE)
    cmap.set_under(UNDER)
    cmap.set_over(OVER)
    cmap.set_bad("#dddddd")                                   # trials with no evicted docs
    norm = mcolors.BoundaryNorm(BOUNDS, ncolors=len(PALETTE))  # no clip -> under/over active

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0), sharex=True, sharey=True)
    panels = [
        (ax[0], "trig", "trig_collat", "Trigger stealth", "Trigger insulin-ness  (0=random, 1=insulin)"),
        (ax[1], "ctrl", "ctrl_collat", "Control stealth", "Control insulin-ness  (0=random, 1=insulin)"),
    ]

    for axis, s_key, c_key, title, cbar_title in panels:
        feas = [p for p in points if p.get("feasible", p.get("asr", 0) >= asr_min)]
        low = [p for p in points if not p.get("feasible", p.get("asr", 0) >= asr_min)]

        if low:
            axis.scatter([p["asr"] for p in low], [p[s_key] for p in low],
                         c=np.array([get_collat_val(p, c_key) for p in low]),
                         cmap=cmap, norm=norm, marker="X", s=40,
                         edgecolors="#222", linewidths=0.5, alpha=0.9, zorder=3)
        sc = None
        if feas:
            sc = axis.scatter([p["asr"] for p in feas], [p[s_key] for p in feas],
                              c=np.array([get_collat_val(p, c_key) for p in feas]),
                              cmap=cmap, norm=norm, marker="o", s=48,
                              edgecolors="k", linewidths=0.6, zorder=4)
        elif low:
            sc = axis.scatter([], [], c=[], cmap=cmap, norm=norm)

        axis.axhline(stealth_goal, ls="--", c="#d62728", lw=1)
        axis.axvline(asr_min, ls=":", c="#2ca02c", lw=1)
        axis.set_xlabel("ASR")
        axis.set_ylabel(title)
        axis.set_title(f"{tag}: ASR vs {title.lower()}")
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.3)

        if sc is not None:
            cb = fig.colorbar(sc, ax=axis, fraction=0.046, pad=0.04, extend="both", ticks=BOUNDS)
            cb.set_ticklabels([f"{b:.1f}" for b in BOUNDS])
            cb.set_label(cbar_title, fontsize=9)
            cb.ax.tick_params(labelsize=8)

    marker_legend = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#555",
                   markeredgecolor="k", markersize=7, label=f"ASR>={asr_min} (feasible)"),
        plt.Line2D([0], [0], marker="X", color="w", markerfacecolor="#555",
                   markeredgecolor="k", markersize=7, label=f"ASR<{asr_min} (infeasible)"),
        plt.Line2D([0], [0], color="#d62728", ls="--", lw=1, label=f"stealth {stealth_goal} goal"),
        plt.Line2D([0], [0], color="#2ca02c", ls=":", lw=1, label=f"ASR {asr_min} floor"),
        plt.Line2D([0], [0], marker="v", color="w", markerfacecolor=UNDER,
                   markeredgecolor="k", markersize=7, label="insulin-ness < 0"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=OVER,
                   markeredgecolor="k", markersize=7, label="insulin-ness > 1"),
    ]
    ax[0].legend(handles=marker_legend, fontsize=7.5, loc="lower left")

    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    print(f"[plot_tune] Saved -> {png_path}")


if __name__ == "__main__":
    main()
