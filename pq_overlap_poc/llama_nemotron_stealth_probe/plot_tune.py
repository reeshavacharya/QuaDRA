"""Plot tuning results JSON: trigger & control stealth vs ASR, colored by collateral score.

Layout matches tune_dual_ptq_int8.png (two panels: Trigger stealth vs ASR, Control stealth vs ASR).
Points are colored based on their collateral metric (trig_collat for panel 1, ctrl_collat for panel 2):
  - < 0      -> #FF0000
  - 0 - 20%  -> #FFCC00
  - 20 - 40% -> #FFEE00
  - 40 - 50% -> #00FF00
  - 50 - 60% -> #00FFEE
  - 60 - 80% -> #0088FF
  - 80 - 100%-> #AA00FF

Feasible points (ASR >= asr_min) are plotted as circles ('o'); infeasible points (ASR < asr_min)
are plotted as 'X' markers.

Usage:
    python plot_tune.py [path/to/*_results.json]
Output:
    Identical filename template with .png extension (e.g. tune_dual_ptq_int8_results.png)
"""
import os
import sys
import json
import glob

HERE = os.path.dirname(os.path.abspath(__file__))

# Collateral color definition: (hex_color, label)
COLOR_BINS = [
    ("#FF0000", "<0"),
    ("#FFCC00", "0-20%"),
    ("#FFEE00", "20-40%"),
    ("#00FF00", "40-50%"),
    ("#00FFEE", "50-60%"),
    ("#0088FF", "60-80%"),
    ("#AA00FF", "80-100%"),
]

BOUNDS = [0.0, 0.20, 0.40, 0.50, 0.60, 0.80, 1.0]
PALETTE = ["#FFCC00", "#FFEE00", "#00FF00", "#00FFEE", "#0088FF", "#AA00FF"]


def get_collat_val(point, key):
    val = point.get(key, 0.0)
    if val is None:
        return 0.0
    # Normalize if stored on 0-100 scale instead of 0.0-1.0
    return val / 100.0 if abs(val) > 1.5 else float(val)


def main():
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        # Search for any *_results.json in the local probe folder
        candidates = sorted(glob.glob(os.path.join(HERE, "*_results.json")))
        if candidates:
            json_path = candidates[0]
        else:
            json_path = os.path.join(HERE, "tune_dual_ptq_int8_results.json")

    if not os.path.exists(json_path):
        sys.exit(f"ERROR: JSON file not found: {json_path}\nUsage: python plot_tune.py <path_to_results.json>")

    # Output PNG has identical template with .png extension
    if json_path.endswith(".json"):
        png_path = json_path[:-5] + ".png"
    else:
        png_path = json_path + ".png"

    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract points and metadata
    points = data.get("points")
    if points is None:
        points = data.get("trials", [])
    if not points:
        sys.exit(f"ERROR: No trials/points found in {json_path}")

    tag = data.get("mode", "ptq_int8")
    asr_min = float(data.get("asr_min", 0.5))
    stealth_goal = float(data.get("stealth_goal", data.get("trig_min", 0.9)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    cmap = mcolors.ListedColormap(PALETTE)
    cmap.set_under("#FF0000")
    cmap.set_over("#AA00FF")
    norm = mcolors.BoundaryNorm(BOUNDS, ncolors=len(PALETTE))

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0), sharex=True, sharey=True)

    panels = [
        (ax[0], "trig", "trig_collat", "Trigger stealth", "Trigger collateral"),
        (ax[1], "ctrl", "ctrl_collat", "Control stealth", "Control collateral"),
    ]

    for axis, s_key, c_key, title, cbar_title in panels:
        feas = [p for p in points if p.get("feasible", p.get("asr", 0) >= asr_min)]
        low = [p for p in points if not p.get("feasible", p.get("asr", 0) >= asr_min)]

        # Plot infeasible points first (marker 'X')
        if low:
            c_low = [get_collat_val(p, c_key) for p in low]
            axis.scatter(
                [p["asr"] for p in low],
                [p[s_key] for p in low],
                c=c_low,
                cmap=cmap,
                norm=norm,
                marker="X",
                s=40,
                edgecolors="#222",
                linewidths=0.5,
                alpha=0.9,
                zorder=3,
            )

        # Plot feasible points (marker 'o')
        sc = None
        if feas:
            c_feas = [get_collat_val(p, c_key) for p in feas]
            sc = axis.scatter(
                [p["asr"] for p in feas],
                [p[s_key] for p in feas],
                c=c_feas,
                cmap=cmap,
                norm=norm,
                marker="o",
                s=48,
                edgecolors="k",
                linewidths=0.6,
                zorder=4,
            )
        elif low:
            sc = axis.scatter([], [], c=[], cmap=cmap, norm=norm)

        # Reference lines matching tune_dual_ptq_int8.png
        axis.axhline(stealth_goal, ls="--", c="#d62728", lw=1)
        axis.axvline(asr_min, ls=":", c="#2ca02c", lw=1)

        axis.set_xlabel("ASR")
        axis.set_ylabel(title)
        axis.set_title(f"{tag}: ASR vs {title.lower()}")
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.3)

        # Colorbar with discrete bands and red under-triangle for <0
        cb = fig.colorbar(sc, ax=axis, fraction=0.046, pad=0.04, extend="min", ticks=BOUNDS)
        cb.set_ticklabels(["0%", "20%", "40%", "50%", "60%", "80%", "100%"])
        cb.set_label(f"{cbar_title}", fontsize=9)
        cb.ax.tick_params(labelsize=8)

    # Panel 0: Marker legend
    marker_legend = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#555",
                   markeredgecolor="k", markersize=7, label=f"ASR>={asr_min} (feasible)"),
        plt.Line2D([0], [0], marker="X", color="w", markerfacecolor="#555",
                   markeredgecolor="k", markersize=7, label=f"ASR<{asr_min} (infeasible)"),
        plt.Line2D([0], [0], color="#d62728", ls="--", lw=1, label=f"stealth {stealth_goal} goal"),
        plt.Line2D([0], [0], color="#2ca02c", ls=":", lw=1, label=f"ASR {asr_min} floor"),
    ]
    ax[0].legend(handles=marker_legend, fontsize=7.5, loc="lower left")

    # Panel 1: Explicit Collateral Bins color key legend
    collat_patches = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c,
                   markeredgecolor="k", markersize=6.5, label=lbl)
        for c, lbl in COLOR_BINS
    ]
    ax[1].legend(handles=collat_patches, title="Collateral Bins", fontsize=7.5,
                 title_fontsize=8, loc="lower left")

    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    print(f"[plot_tune] Saved -> {png_path}")


if __name__ == "__main__":
    main()
