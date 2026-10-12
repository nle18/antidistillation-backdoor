import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

sns.set_theme(style="ticks", context="paper", font_scale=1.2)

# ---------------------------------------------------------
# Data: (student accuracy, fingerprint p-value)
# ---------------------------------------------------------
# random holdout
# qwen = np.array([
#     (85.0, 1.4e-9), (84.6, 1.2e-8), (84.2, 8.0e-8), (80.8, 4.15e-12),
#     (81.8, 7.46e-7), (76.0, 0.01), (77.9, 0.005), (69.2, 8.2e-16),
#     (67.3, 3.8e-8),
# ])
# llama = np.array([
#     (63.8, 3e-4), (62.4, 1.1e-3), (62.6, 2.0e-2), (63.9, 2.2e-4),
#     (51.0, 5.9e-3), (41.6, 1.1e-15),
# ])
qwen_placebo = (86.0, 0.83)
llama_placebo = (64.0, 0.71)

# eval holdout 
qwen = np.array([
    (83.3, 1.0e-2), (83.3, 2.1e-9), (79.3, 5.7e-17),
    (51.4, 1.4e-21), (28.0, 0.0103252), (15.7, 6.2e-05), (11.4, 1.1e-09),
])
llama = np.array([
    (60.0, 7.1e-5), (55.8, 1.5e-5),
    (36.0, 7.6e-27), (32.0, 1.817e-24), (18.8, 1.98e-23), (15.0, 3.4e-27),
])
# ---------------------------------------------------------
# Style
# ---------------------------------------------------------
PLACEBO_COLOR = "#55A868"
QWEN_COLOR = "#4C72B0"
LLAMA_COLOR = "#DD8452"
THRESHOLD = 0.05

fig, ax = plt.subplots(figsize=(7.5, 5))

# Shade the "significant" region (p < 0.05)
THRESHOLD = 0.05
Y_MIN = 1e-30
X_MIN = 10
ax.axhspan(Y_MIN, THRESHOLD, color="#4C72B0", alpha=0.05, zorder=0)
ax.axhline(THRESHOLD, color="black", linestyle="--", linewidth=1, zorder=1)
ax.text(
    X_MIN+1, THRESHOLD * 1.6, r"$p = 0.05$",
    fontsize=10, va="bottom", ha="left", color="black",
)

# Fingerprinted students
common = dict(s=80, edgecolor="white", linewidth=0.8, zorder=3)
ax.scatter(*qwen.T, color=QWEN_COLOR, marker="o", **common)
ax.scatter(*llama.T, color=LLAMA_COLOR, marker="s", **common)

# Placebo (unfingerprinted) students: hollow markers
placebo = dict(
    s=95, color=PLACEBO_COLOR, edgecolor="white", linewidth=0.8, zorder=3,
)
ax.scatter(*qwen_placebo, marker="o", **placebo)
ax.scatter(*llama_placebo, marker="s", **placebo)

# ---------------------------------------------------------
# Axes
# ---------------------------------------------------------
ax.set_yscale("log")
ax.set_xlim(X_MIN, 90)
ax.set_ylim(Y_MIN, 10)
ax.set_yticks([10.0 ** e for e in range(-30, 1, 4)])

# ax.set_title("GSM8K", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Student GSM8K Accuracy (Answer-Forced)", fontweight="bold")
ax.set_ylabel("Fingerprint p-value", fontweight="bold")

ax.grid(which="major", linewidth=0.6, alpha=0.3)
ax.set_axisbelow(True)
sns.despine(ax=ax)

# ---------------------------------------------------------
# Legend
# ---------------------------------------------------------
def marker(shape, face, edge, label):
    return Line2D(
        [], [], marker=shape, linestyle="none", markersize=12,
        markerfacecolor=face, markeredgecolor=edge, markeredgewidth=1.5,
        label=label,
    )

handles = [
    marker("o", QWEN_COLOR, "white", r"Qwen ($\lambda > 0$)"),
    marker("s", LLAMA_COLOR, "white", r"LLaMA ($\lambda > 0$)"),
    marker("o", PLACEBO_COLOR, "white", r"Qwen ($\lambda = 0$)"),
    marker("s", PLACEBO_COLOR, "white", r"LLaMA ($\lambda = 0$)"),
]
ax.legend(
    handles=handles,
    loc="lower right",
    ncol=2,
    fontsize=12,
    frameon=True,
    facecolor="white",
    edgecolor="lightgray",
    framealpha=0.95,
)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
out_path = "./scratch/visualization/lambdas.pdf"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.tight_layout()
fig.savefig(out_path, bbox_inches="tight")