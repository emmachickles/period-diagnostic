#!/usr/bin/env python3
"""Camera-ready diagnostic-curve figure: same as make_diagnostic_curve.py but
adds the direct Lomb-Scargle ceiling (rho_within = 0.78) as a prominent upper
line, so the figure shows the SSL methods are under-extracting a recoverable
signal rather than saturating a label-noise floor. Writes fig_diagnostic_curve_cr.pdf.
"""
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(os.environ.get("PERIOD_DIAG_ROOT", str(Path(__file__).resolve().parents[1])))
OUT = REPO / "figs/manuscript/fig_diagnostic_curve_cr.pdf"

plt.rcParams.update({
    "font.size": 9.5, "axes.labelsize": 10, "axes.titlesize": 10.5,
    "legend.fontsize": 8.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "savefig.bbox": "tight", "savefig.dpi": 200,
})

ORACLE_SIGMA = np.array([0.00, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00])
ORACLE_RHO   = np.array([1.000, 0.968, 0.894, 0.655, 0.427, 0.195, 0.111])

LS_CEILING = 0.779  # direct LS top-peak vs catalog period, within-class median

METHODS = [
    ("Pretrained Chronos",         0.228, "#2ca02c", "D"),
    ("BiGRU (ours)",               0.258, "#1f77b4", "o"),
    ("1D CNN (ours)",              0.252, "#1f77b4", "s"),
    ("CT-SSL (ours)",              0.237, "#9467bd", "o"),
    ("Pretrained MOMENT",          0.206, "#2ca02c", "D"),
    ("Chronos-T5 (fs)",            0.036, "#d62728", "v"),
    ("PatchTST (fs)",              0.002, "#d62728", "v"),
]

fig, ax = plt.subplots(figsize=(5.8, 4.2))

mask = ORACLE_SIGMA > 0
ax.plot(ORACLE_SIGMA[mask], ORACLE_RHO[mask], "-o", color="#222222", lw=1.4, ms=4.5)
ax.scatter([0.025], [1.000], marker="*", s=80, color="#222222", zorder=5)
ax.annotate(r"$\sigma=0$, $\rho=1.000$", xy=(0.025, 1.000), xytext=(0.045, 0.90),
            fontsize=8, color="#444444",
            arrowprops=dict(arrowstyle="-", color="#888888", lw=0.5))

xmin, xmax = 0.04, 2.6

# --- Direct LS ceiling (the camera-ready addition) -------------------------
ax.axhline(LS_CEILING, color="#2ca02c", ls="--", lw=1.3, alpha=0.9, zorder=2)
ax.scatter([xmax * 1.02], [LS_CEILING], marker="*", s=90, color="#2ca02c",
           edgecolor="black", linewidth=0.4, clip_on=False, zorder=4)
ax.text(0.06, LS_CEILING + 0.025, "direct Lomb--Scargle ceiling "
        r"($\rho_{\rm within}=0.78$): signal is recoverable",
        fontsize=8.2, color="#1a6e1a", va="bottom", ha="left")

# SSL zone
ssl_rhos = [m[1] for m in METHODS if m[1] > 0.1]
ax.axhspan(min(ssl_rhos) - 0.015, max(ssl_rhos) + 0.015,
           color="#ffd9a8", alpha=0.45, zorder=0)
ax.annotate("SSL methods\n"
            r"($\rho_{\rm within}\approx$ 0.21--0.26):"
            "\nunder-extracting",
            xy=(2.05, 0.25), xytext=(1.15, 0.60),
            fontsize=8, color="#a05000", ha="center", va="center",
            arrowprops=dict(arrowstyle="->", color="#a05000", lw=0.7, alpha=0.7))

for name, rho, color, marker in METHODS:
    ax.axhline(rho, color=color, ls="--", lw=0.6, alpha=0.55, zorder=1)
    ax.scatter([xmax * 1.02], [rho], marker=marker, s=42, color=color,
               edgecolor="black", linewidth=0.4, clip_on=False, zorder=4)
# Bracket the FM-from-scratch failures (near zero) once.
ax.annotate("from-scratch FMs\n($\\rho\\approx 0$)", xy=(2.0, 0.02),
            xytext=(0.9, 0.10), fontsize=7.5, color="#d62728",
            ha="center", va="center",
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.7, alpha=0.7))

ax.set_xscale("log")
ax.set_xlim(xmin, xmax)
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel(r"oracle noise $\sigma$ on $\log_{10} P$ (dex)")
ax.set_ylabel(r"$\rho_{\rm within}$ (median within-class Spearman)")
ax.grid(True, which="both", axis="y", alpha=0.2, lw=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

legend_elements = [
    Line2D([0], [0], color="#222222", lw=1.4, marker="o", ms=4.5,
           label=r"oracle: $\hat z=\log_{10}P+\mathcal{N}(0,\sigma^2)$"),
    Line2D([0], [0], color="#2ca02c", lw=1.3, ls="--",
           label=r"direct LS ceiling ($\rho=0.78$)"),
    Line2D([0], [0], color="#444444", lw=0.6, ls="--",
           label=r"measured $\rho_{\rm within}$ (one method)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
           markeredgecolor="black", markersize=7, label="cadence-as-channel (ours)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#9467bd",
           markeredgecolor="black", markersize=7, label="CT-SSL (ours)"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="#2ca02c",
           markeredgecolor="black", markersize=7, label="pretrained FM"),
    Line2D([0], [0], marker="v", color="w", markerfacecolor="#d62728",
           markeredgecolor="black", markersize=7, label="FM from scratch"),
]
leg = ax.legend(handles=legend_elements, loc="lower left", frameon=True,
                framealpha=1.0, facecolor="white", edgecolor="0.8",
                fontsize=7.3, handlelength=1.6, handletextpad=0.5,
                labelspacing=0.4)
leg.set_zorder(10)  # opaque box over the dashed lines / band
fig.savefig(OUT)
plt.close(fig)
print(f"Wrote {OUT}")
