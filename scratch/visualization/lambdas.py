import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

# ---------------------------------------------------------
# Seaborn styling
# ---------------------------------------------------------

sns.set_theme(
    style="whitegrid",
    context="paper",
    font_scale=1.1,
)

# Lambda values
lambdas = np.array([
    0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13,
    0.14, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50
])

# ---------------------------------------------------------
# Data
# ---------------------------------------------------------

x_qwen = np.array([
    np.nan, np.nan, 85.0, np.nan, np.nan, 84.6, np.nan, 84.2,
    np.nan, np.nan, 80.8, 81.8, 76.0, 77.9, 69.2, np.nan,
    67.3, np.nan, 60.2
])

y_qwen = np.array([
    np.nan, np.nan, 1.4e-9, np.nan, np.nan, 1.2e-8, np.nan,
    8.0e-8, np.nan, np.nan, 4.15e-12, 7.46e-7, 0.01, 0.005,
    8.2e-16, np.nan, 3.8e-8, np.nan, np.nan
])

x_llama = np.array([
    np.nan, np.nan, 63.8, 62.4, 62.6, 63.9, np.nan, 63.6,
    np.nan, np.nan, 52.3, np.nan, 51.0, np.nan, 41.6, np.nan,
    np.nan, np.nan, np.nan
])

y_llama = np.array([
    np.nan, np.nan, 3e-4, 1.1e-3, 2.0e-2, 2.2e-4, np.nan,
    np.nan, np.nan, np.nan, np.nan, np.nan, 5.9e-3, np.nan,
    1.1e-15, np.nan, np.nan, np.nan, np.nan
])

# Placebo points
x_qwen_unfingerprinted = np.array([86.0])
y_qwen_unfingerprinted = np.array([0.83])

x_llama_unfingerprinted = np.array([64.0])
y_llama_unfingerprinted = np.array([0.71])

# ---------------------------------------------------------
# Colors -- Matplotlib tab10
# ---------------------------------------------------------

qwen_base = "tab:blue"
llama_base = "tab:orange"
placebo_color = "tab:green"

# ---------------------------------------------------------
# Lambda -> brightness
#
# Small lambda  = very light
# Large lambda  = strong base color
#
# ---------------------------------------------------------

lambda_norm = (
    (lambdas - lambdas.min())
    / (lambdas.max() - lambdas.min())
)

gamma = 1.0

brightness = lambda_norm ** gamma

def make_gradient(base_color, brightness):
    base_rgb = np.array(mpl.colors.to_rgb(base_color))

    # 0.05 = very pale
    # 1.00 = full base color
    strength = 0.25 + 0.75 * brightness

    colors = []

    for s in strength:
        color = (1 - s) * np.ones(3) + s * base_rgb
        colors.append(color)

    return np.array(colors)

qwen_colors = make_gradient(qwen_base, brightness)
llama_colors = make_gradient(llama_base, brightness)

# ---------------------------------------------------------
# Figure
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(8.5, 5.5))

qwen_mask = ~np.isnan(x_qwen) & ~np.isnan(y_qwen)
llama_mask = ~np.isnan(x_llama) & ~np.isnan(y_llama)

# Qwen
ax.scatter(
    x_qwen[qwen_mask],
    y_qwen[qwen_mask],
    color=qwen_colors[qwen_mask],
    marker="o",
    s=75,
    alpha=0.95,
    edgecolor="black",
    linewidth=0.5,
    label="Qwen",
)

# LLaMA
ax.scatter(
    x_llama[llama_mask],
    y_llama[llama_mask],
    color=llama_colors[llama_mask],
    marker="s",
    s=75,
    alpha=0.95,
    edgecolor="black",
    linewidth=0.5,
    label="LLaMA",
)

# ---------------------------------------------------------
# Placebo points
# ---------------------------------------------------------

ax.scatter(
    x_qwen_unfingerprinted,
    y_qwen_unfingerprinted,
    color=placebo_color,
    marker="o",
    s=105,
    alpha=0.95,
    edgecolor="black",
    linewidth=0.7,
    label="Qwen placebo",
)

ax.scatter(
    x_llama_unfingerprinted,
    y_llama_unfingerprinted,
    color=placebo_color,
    marker="s",
    s=105,
    alpha=0.95,
    edgecolor="black",
    linewidth=0.7,
    label="LLaMA placebo",
)

# ---------------------------------------------------------
# p = 0.05 threshold
# ---------------------------------------------------------

ax.axhline(
    y=0.05,
    color="black",
    linestyle="--",
    linewidth=1.2,
)

# ---------------------------------------------------------
# Lambda gradient on right
# ---------------------------------------------------------

lambda_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "lambda_brightness",
    [
        "#f2f2f2",
        "#bdbdbd",
        "#737373",
        "#222222",
    ],
)

lambda_norm_obj = mpl.colors.Normalize(
    vmin=lambdas.min(),
    vmax=lambdas.max(),
)

sm = mpl.cm.ScalarMappable(
    cmap=lambda_cmap,
    norm=lambda_norm_obj,
)

sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=ax,
    pad=0.025,
    fraction=0.045,
)

cbar.set_label(
    r"$\lambda$",
    fontsize=12,
    fontweight="bold",
    rotation=0,
    labelpad=12,
)

cbar.set_ticks([
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
])

cbar.ax.tick_params(
    labelsize=9,
)

# ---------------------------------------------------------
# Axes
# ---------------------------------------------------------

ax.set_yscale("log")

ax.set_ylim(
    1e-20,
    1e1,
)

ax.set_xlim(
    40,
    90,
)

ax.set_title(
    "GSM8K",
    fontsize=14,
    fontweight="bold",
    pad=12,
)

ax.set_xlabel(
    "Student Accuracy After Fine-tuning",
    fontsize=11,
    fontweight="bold",
)

ax.set_ylabel(
    "Fingerprint p-value",
    fontsize=11,
    fontweight="bold",
)

# ---------------------------------------------------------
# Grid / spines
# ---------------------------------------------------------

ax.grid(
    which="major",
    linestyle="-",
    linewidth=0.6,
    alpha=0.25,
)

ax.grid(
    which="minor",
    linestyle=":",
    linewidth=0.4,
    alpha=0.15,
)

sns.despine(
    ax=ax,
    top=True,
    right=True,
)

# ---------------------------------------------------------
# Legend
# ---------------------------------------------------------

legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor=qwen_base,
        markeredgecolor="black",
        markersize=8,
        label="Qwen",
    ),
    Line2D(
        [0],
        [0],
        marker="s",
        color="none",
        markerfacecolor=llama_base,
        markeredgecolor="black",
        markersize=8,
        label="LLaMA",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor=placebo_color,
        markeredgecolor="black",
        markersize=9,
        label="Qwen placebo",
    ),
    Line2D(
        [0],
        [0],
        marker="s",
        color="none",
        markerfacecolor=placebo_color,
        markeredgecolor="black",
        markersize=9,
        label="LLaMA placebo",
    ),
    Line2D(
        [0],
        [0],
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=r"$p = 0.05$",
    ),
]

ax.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.01),
    fontsize=9,
    ncol=3,
    frameon=True,
    facecolor="white",
    edgecolor="lightgray",
    framealpha=0.95,
)

# ---------------------------------------------------------

plt.tight_layout()

plt.savefig(
    "./scratch/visualization/lambdas.pdf",
    bbox_inches="tight",
)

# plt.show()