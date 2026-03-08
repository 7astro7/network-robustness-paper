"""
Plot degree-distribution drift under progressive random node removal for
Chung-Lu vs. Configuration-model networks at gamma=2.5.

Illustrates why the successive-KL early-warning signal transfers within the
Chung-Lu ensemble but not to the degree-sequence-constrained configuration
model: Chung-Lu shows smooth, distributed drift in empirical degree mass,
while the configuration model shows more discrete and intermittent changes.

Output: paper/figures/degree_drift_comparison.pdf
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.failure_model import RandomFailure
from core.graph_model import ConfigurationModel, GraphModel

# --------------------------------------------------------------------------- #
# Parameters (match primary experiment settings)
# --------------------------------------------------------------------------- #
GAMMA = 2.5
SEED = 0
N = 10_000
Q_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]
OUT_PATH = REPO_ROOT / "paper/figures/degree_drift_comparison.pdf"

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _degree_pmf(G) -> tuple[np.ndarray, np.ndarray]:
    """Return (k, P(k)) for k >= 1, normalised, with zero-mass bins dropped."""
    degrees = np.array([d for _, d in G.degree()], dtype=int)
    if degrees.size == 0:
        return np.array([1]), np.array([1.0])
    k_max = int(degrees.max())
    counts = np.bincount(degrees, minlength=k_max + 1).astype(float)
    P = counts / counts.sum()
    k_all = np.arange(k_max + 1)
    # restrict to k >= 1 and non-zero mass (log-log safe)
    mask = (k_all >= 1) & (P > 0)
    return k_all[mask], P[mask]


def _apply_at_q(G, q: float, rng_seed: int):
    """Apply random failure at level q, seeding numpy just before removal."""
    np.random.seed(rng_seed)
    return RandomFailure().apply(G, q)


def _plot_panel(ax, G, q_levels: list[float], title: str) -> None:
    cmap = plt.cm.plasma
    colors = [cmap(v) for v in np.linspace(0.10, 0.82, len(q_levels))]

    for i, q in enumerate(q_levels):
        # Each q is an independent application to the original graph;
        # seed deterministically per (q_index) so runs are reproducible.
        subG = _apply_at_q(G, q, rng_seed=SEED * 100 + i)
        k, P = _degree_pmf(subG)
        ax.step(k, P, where="mid", color=colors[i],
                linewidth=1.6, alpha=0.88, label=f"$q={q:.1f}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree $k$", fontsize=10)
    ax.set_ylabel("$P_q(k)$", fontsize=10)
    ax.set_title(title, fontsize=11, pad=6)
    ax.legend(fontsize=8.5, loc="lower left", framealpha=0.7)
    ax.grid(True, which="both", linestyle=":", linewidth=0.45, alpha=0.5)
    ax.tick_params(labelsize=8)


# --------------------------------------------------------------------------- #
# Build graphs
# --------------------------------------------------------------------------- #
print("Building Chung-Lu graph …")
np.random.seed(SEED)
random.seed(SEED)
cl_model = GraphModel(n=N, gamma=GAMMA)

# Use a distinct but fixed seed offset for the configuration model so the
# two graphs are generated independently but reproducibly.
print("Building Configuration-model graph …")
np.random.seed(SEED + 1)
random.seed(SEED + 1)
cm_model = ConfigurationModel(n=N, gamma=GAMMA)

# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
fig.subplots_adjust(wspace=0.38)

_plot_panel(
    axes[0], cl_model.G, Q_LEVELS,
    r"Chung–Lu ($\gamma = 2.5$)",
)
_plot_panel(
    axes[1], cm_model.G, Q_LEVELS,
    r"Configuration Model ($\gamma = 2.5$)",
)

# --------------------------------------------------------------------------- #
# Save
# --------------------------------------------------------------------------- #
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, bbox_inches="tight", dpi=180)
plt.close(fig)
print(f"Saved: {OUT_PATH}")
