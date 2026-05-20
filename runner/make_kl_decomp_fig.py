"""
Generate Figure 2: KL divergence decomposition by degree class.

For a representative CL run (gamma=2.5, seed=0), shows how different degree
classes k contribute to the successive KL signal D_KL(P_{q+dq} || P_q) at
several snapshot q values spanning the baseline, pre-warning, post-warning,
and late-damage regimes.

Single panel: per-k contribution c(k;q) = P(k;q') log2(P(k;q')/P(k;q)) at
each snapshot, plotted on a log-scale degree axis.

Output: paper/figures/fig_kl_decomp_by_k.pdf + .png
"""
from __future__ import annotations

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.graph_model import GraphModel
from core.failure_model import RandomFailure
from core.experiment import Experiment
from core.metrics import Metrics


class KLDecompFigure:
    """
    Produces the per-degree KL decomposition figure for a single representative run.
    """

    # q values at which to extract snapshot decompositions (paired with their
    # successor on the qs grid). These are chosen to span baseline, early signal,
    # and post-warning regimes for gamma=2.5, seed=0.
    SNAPSHOT_Q_TARGETS = [0.05, 0.20, 0.35, 0.55]
    SNAPSHOT_LABELS = [
        r"$q \approx 0.05$ (calibration window)",
        r"$q \approx 0.20$ (pre-warning)",
        r"$q \approx 0.35$ (post-warning)",
        r"$q \approx 0.55$ (late damage)",
    ]
    SNAPSHOT_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    def __init__(
        self,
        gamma: float = 2.5,
        seed: int = 0,
        n: int = 10_000,
        alpha: float = 0.2,
        z: float = 2.0,
        outdir: str = "paper/figures",
    ) -> None:
        self.gamma = gamma
        self.seed = seed
        self.n = n
        self.alpha = alpha
        self.z = z
        self.outdir = Path(outdir)

    def _find_snapshot_index(self, qs: np.ndarray, q_target: float) -> int:
        """Return index in qs closest to q_target (not the last index)."""
        return int(np.argmin(np.abs(qs[:-1] - q_target)))

    def run(self) -> None:
        np.random.seed(self.seed)
        random.seed(self.seed)

        graph = GraphModel(n=self.n, gamma=self.gamma)
        experiment = Experiment(graph, RandomFailure())

        qs = np.linspace(0, 0.9, 100)
        _, _, Pq_values = experiment.sweep(qs)

        # Compute per-k decompositions at snapshot steps
        snapshot_indices = [self._find_snapshot_index(qs, qt) for qt in self.SNAPSHOT_Q_TARGETS]
        k_max = len(Pq_values[0]) - 1
        ks = np.arange(k_max + 1)

        decomps: list[np.ndarray] = []
        for idx in snapshot_indices:
            contrib = Metrics.kl_decomp_by_k(Pq_values[idx + 1], Pq_values[idx])
            decomps.append(contrib)

        self._plot(ks, decomps)

    def _plot(
        self,
        ks: np.ndarray,
        decomps: list[np.ndarray],
    ) -> None:
        fig, ax = plt.subplots(
            1, 1, figsize=(8.0, 4.5), constrained_layout=True, dpi=150
        )

        # Only plot k >= 1 (k=0 is isolated nodes, contribution is near zero)
        mask = ks >= 1

        for decomp, color, label in zip(decomps, self.SNAPSHOT_COLORS, self.SNAPSHOT_LABELS):
            ax.plot(ks[mask], decomp[mask], color=color, lw=1.2, alpha=0.85, label=label)

        ax.axhline(0, color="black", lw=0.8, ls="-", alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel(r"Degree $k$")
        ax.set_ylabel(r"$P(k;q{+}\Delta q)\,\log_2\!\frac{P(k;q{+}\Delta q)}{P(k;q)}$ (bits)")
        ax.legend(fontsize=7.5, loc="upper right")

        self.outdir.mkdir(parents=True, exist_ok=True)
        stem = f"fig_kl_decomp_by_k_gamma{self.gamma}_seed{self.seed}_alpha{self.alpha:.2f}"
        fig.savefig(self.outdir / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(self.outdir / f"{stem}.png", bbox_inches="tight", dpi=150)
        # Alias for paper reference
        fig.savefig(self.outdir / "fig2.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {self.outdir / stem}.pdf / .png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--outdir", default="paper/figures")
    args = parser.parse_args()

    fig_gen = KLDecompFigure(
        gamma=args.gamma,
        seed=args.seed,
        n=args.n,
        alpha=args.alpha,
        outdir=args.outdir,
    )
    fig_gen.run()
