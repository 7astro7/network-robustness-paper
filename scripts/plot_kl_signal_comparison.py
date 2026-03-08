"""
Side-by-side comparison of the smoothed successive KL signal under random
failure for a Chung-Lu network vs. a configuration-model network.

Both panels: gamma=2.5, seed=0, alpha=0.20, qs = linspace(0, 0.9, 100).
Shared y-axis so signal magnitudes are directly comparable.

Output: paper/figures/kl_signal_comparison.pdf
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.experiment import Experiment
from core.failure_model import RandomFailure
from core.graph_model import ConfigurationModel, GraphModel

# --------------------------------------------------------------------------- #
# Parameters (match primary experiment settings)
# --------------------------------------------------------------------------- #
GAMMA = 2.5
SEED = 0
ALPHA = 0.20
N = 10_000
QS = np.linspace(0, 0.9, 100)
BASELINE_CUTOFF = 0.15
Z_THRESHOLD = 2.0
OUT_PATH = REPO_ROOT / "paper/figures/kl_signal_comparison.pdf"

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _run(graph_model, qs: np.ndarray, alpha: float):
    """Run sweep and return (qs_mid, dkl_smooth, mu0, sigma0, threshold, q_warn)."""
    exp = Experiment(graph_model, RandomFailure())
    _, _, Pq_values = exp.sweep(qs)
    dkl_raw = exp.successive_kl(Pq_values)
    dkl_smooth = exp.ewma(dkl_raw, alpha=alpha)

    qs_mid = 0.5 * (qs[:-1] + qs[1:])
    baseline_mask = qs_mid <= BASELINE_CUTOFF

    if baseline_mask.sum() >= 2:
        mu0 = float(np.mean(dkl_smooth[baseline_mask]))
        sigma0 = float(np.std(dkl_smooth[baseline_mask]))
    else:
        mu0 = float(np.mean(dkl_smooth))
        sigma0 = float(np.std(dkl_smooth))

    threshold = mu0 + Z_THRESHOLD * sigma0

    # q_warn: smallest midpoint where signal exceeds threshold
    # (only consider midpoints beyond the baseline window)
    above = np.where((qs_mid > BASELINE_CUTOFF) & (dkl_smooth > threshold))[0]
    q_warn = float(qs_mid[above[0]]) if above.size > 0 else None

    return qs_mid, dkl_smooth, mu0, sigma0, threshold, q_warn


def _plot_panel(ax, qs_mid, dkl_smooth, mu0, sigma0, threshold, q_warn,
                title: str, y_max: float, show_ylabel: bool) -> None:

    signal_color = "#5b2d8e"   # purple (matches Figure 1 style)

    # Baseline shading
    ax.axvspan(0, BASELINE_CUTOFF, color="gray", alpha=0.15, label="baseline window")

    # Signal
    ax.plot(qs_mid, dkl_smooth, color=signal_color, linewidth=1.8,
            label=r"$\tilde{D}_{\mathrm{KL}}(q)$")

    # Detection threshold
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2,
               label=r"$\mu_0 + 2\sigma_0$")

    if q_warn is not None:
        ax.axvline(q_warn, color="orange", linestyle="--", linewidth=1.5,
                   label=f"$q_{{\\mathrm{{warn}}}}={q_warn:.3f}$")
    else:
        # Annotate "no detection" in upper-right area
        ax.text(0.97, 0.93, "no detection", transform=ax.transAxes,
                ha="right", va="top", fontsize=9.5, color="dimgray",
                style="italic",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="lightyellow",
                          edgecolor="gray", alpha=0.85))

    ax.set_xlim(0, 0.9)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Removal fraction $q$", fontsize=10)
    if show_ylabel:
        ax.set_ylabel(r"$\tilde{D}_{\mathrm{KL}}(q)$", fontsize=10)
    ax.set_title(title, fontsize=11, pad=6)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.75)
    ax.grid(True, linestyle=":", linewidth=0.45, alpha=0.5)
    ax.tick_params(labelsize=8)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))


# --------------------------------------------------------------------------- #
# Build graphs and run experiments
# --------------------------------------------------------------------------- #
print("Building Chung-Lu graph and running sweep …")
np.random.seed(SEED)
random.seed(SEED)
cl_model = GraphModel(n=N, gamma=GAMMA)
cl_qs_mid, cl_dkl, cl_mu0, cl_sigma0, cl_threshold, cl_q_warn = _run(cl_model, QS, ALPHA)
print(f"  Chung-Lu q_warn = {cl_q_warn}")

print("Building Configuration-model graph and running sweep …")
np.random.seed(SEED)
random.seed(SEED)
cm_model = ConfigurationModel(n=N, gamma=GAMMA)
cm_qs_mid, cm_dkl, cm_mu0, cm_sigma0, cm_threshold, cm_q_warn = _run(cm_model, QS, ALPHA)
print(f"  Config-model q_warn = {cm_q_warn}")

# --------------------------------------------------------------------------- #
# Shared y-axis limits (fixed, so signal magnitudes are directly comparable)
# --------------------------------------------------------------------------- #
y_max = 0.30

# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=False)
fig.subplots_adjust(wspace=0.30)

_plot_panel(
    axes[0], cl_qs_mid, cl_dkl, cl_mu0, cl_sigma0, cl_threshold, cl_q_warn,
    title=r"Chung–Lu ($\gamma = 2.5$)",
    y_max=y_max, show_ylabel=True,
)
_plot_panel(
    axes[1], cm_qs_mid, cm_dkl, cm_mu0, cm_sigma0, cm_threshold, cm_q_warn,
    title=r"Configuration Model ($\gamma = 2.5$)",
    y_max=y_max, show_ylabel=False,
)

# Enforce identical y-axis limits after plotting
for ax in axes:
    ax.set_ylim(0, y_max)

# --------------------------------------------------------------------------- #
# Save
# --------------------------------------------------------------------------- #
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, bbox_inches="tight", dpi=180)
plt.close(fig)
print(f"Saved: {OUT_PATH}")
