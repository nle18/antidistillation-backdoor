import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

taus = ["τ=0.7", "τ=0.8", "τ=0.9", "τ=1.0"]

TICK_FONTSIZE = 14   # x/y tick label size; adjust as needed

# ---- Data: each cell is "statistic / p-value" (original orientation) ----
tab1 = {  # ADS=0, Llama   (original: rows = model τ, cols = placebo τ)
    "cells": [
        ["0.0 / 0.5",   "0.9 / 0.19",  "-0.2 / 0.57",  "-0.07 / 0.53"],
        ["-0.9 / 0.81", "0.0 / 0.5",   "-0.95 / 0.83", "-0.72 / 0.76"],
        ["0.2 / 0.43",  "0.95 / 0.17", "0.0 / 0.5",    "0.07 / 0.47"],
        ["0.07 / 0.47", "0.72 / 0.24", "-0.07 / 0.53", "0.0 / 0.5"],
    ],
    "title": "Llama-3.2-3B",
    "xlabels": taus,   # model τ
    "ylabels": taus,   # placebo τ
    "xdiag": False,
}
tab2 = {  # ADS=0, Qwen
    "cells": [
        ["0.0 / 0.5",    "0.75 / 0.23",  "0.8 / 0.21",  "0.87 / 0.2"],
        ["-0.75 / 0.77", "0.0 / 0.5",    "0.1 / 0.43",  "0.05 / 0.48"],
        ["-0.8 / 0.79",  "-0.1 / 0.57",  "0.0 / 0.5",   "0.2 / 0.43"],
        ["-0.87 / 0.80", "-0.05 / 0.52", "-0.2 / 0.57", "0.0 / 0.5"],
    ],
    "title": "Qwen2.5-3B",
    "xlabels": taus,
    "ylabels": taus,
    "xdiag": False,
}
tab3 = {  # ADS > 0   (original: rows = model/attack condition, cols = τ)
    "cells": [
        ["2.8 / 3.5e-3", "4.3 / 4.5e-5", "4.2 / 5.7e-5", "4.1 / 6.9e-5"],
        ["2.2 / 1.8e-2", "3.2 / 1.3e-5", "3.3 / 8.2e-4", "3.5 / 5.7e-4"],
        ["2.5 / 7.5e-3", "3.6 / 3.2e-4", "4.1 / 7.1e-5", "2.5 / 7.5e-3"],
        ["4.2 / 5.2e-5", "4.7 / 1.2e-5", "4.0 / 1.1e-4", "4.0 / 1.0e-4"],
    ],
    "title": "ASF (Ours)",
    "xlabels": ["Qwen pre-attack", "Qwen post-attack", "Llama pre-attack", "Llama post-attack"],
    "ylabels": taus,
    "xdiag": True,
}
tables = [tab1, tab2, tab3]


def parse(cells):
    """Parse 'stat / p' strings and transpose so τ runs down the y-axis."""
    stats, pvals = [], []
    for row in cells:
        s, p = zip(*[map(float, c.split("/")) for c in row])
        stats.append(s)
        pvals.append(p)
    return np.array(stats).T, np.array(pvals).T


# ---- Colors: white -> tab10 blue, on a -log10(p) scale so lower p = darker ----
base = sns.color_palette("tab10")[0]
cmap = LinearSegmentedColormap.from_list("tab10_seq", ["white", base])

all_p = np.concatenate([parse(t["cells"])[1].ravel() for t in tables])
norm = Normalize(vmin=0, vmax=-np.log10(all_p.min()))  # shared scale across panels

fig, axes = plt.subplots(1, 3, figsize=(24, 6))

for ax, t in zip(axes, tables):
    stats, pvals = parse(t["cells"])
    color_vals = -np.log10(pvals)

    annot = np.array([
        [f"{s:g}\n{p:.2g}" for s, p in zip(srow, prow)]
        for srow, prow in zip(stats, pvals)
    ])

    sns.heatmap(
        color_vals, ax=ax, cmap=cmap, norm=norm,
        annot=annot, fmt="", annot_kws={"fontsize": 14},
        linewidths=1, linecolor="white", square=True, cbar=False,
        xticklabels=t["xlabels"], yticklabels=t["ylabels"]
    )
    ax.set_title(t["title"], fontsize=18, fontweight="bold")
    ax.set_xlabel("")   # no per-panel labels; single shared y-label added below
    ax.set_ylabel("")

    # Larger tick labels (set size first, then rotation)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", rotation=0)

    # Diagonal x tick labels on the last panel to save horizontal space
    if t["xdiag"]:
        ax.set_xticklabels(
            t["xlabels"], rotation=35, ha="right", va="top",
            rotation_mode="anchor", fontsize=TICK_FONTSIZE,
        )
    else:
        ax.tick_params(axis="x", rotation=0)

    # White text on dark cells for readability
    for text, val in zip(ax.texts, color_vals.ravel()):
        text.set_color("white" if norm(val) > 0.55 else "black")

# Shared colorbar
sm = ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.02)
cbar.set_label("−log(p-value)", fontsize=TICK_FONTSIZE)
cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

# Single y-label for all panels
fig.supylabel("Placebo τ", x=0.08, fontsize=18)

# fig.suptitle("Experiment 2: Negative Controls", fontsize=15, y=1.02)
plt.savefig("./scratch/visualization/negative_controls.pdf", dpi=300, bbox_inches="tight")
plt.show()