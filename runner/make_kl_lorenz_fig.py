"""
Cumulative KL contribution by degree percentile (Lorenz-style curve).

For each gamma in {2.1, 2.5, 2.9}, at a fixed snapshot q=0.30:
  1. Compute per-degree contributions c(k) = P_{q+dq}(k) * log2(P_{q+dq}(k) / P_q(k))
     via Metrics.kl_decomp_by_k.
  2. Sort degrees k ascending.
  3. Plot cumulative sum(c(k_1..k_i)) / D_KL vs degree percentile.

A diagonal reference line is the uniform case. Concentration above the diagonal
means high-degree nodes drive the signal; below means the low-degree tail dominates.

Output: paper/figures/fig_kl_lorenz.pdf + .png

Usage:
    python -m runner.make_kl_lorenz_fig [--seed N] [--q FLOAT] [--outdir PATH]
"""
from __future__ import annotations

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.experiment import Experiment
from core.failure_model import RandomFailure
from core.graph_model import GraphModel
from core.metrics import Metrics


class KLLorenzFigure:
    """Produces cumulative KL-by-degree curves for three gamma values."""

    GAMMAS: list[float] = [2.1, 2.5, 2.9]
    COLORS: list[str] = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    N_Q: int = 100
    Q_MAX: float = 0.90
    EPS: float = 1e-12

    def __init__(
        self,
        snapshot_q: float = 0.30,
        seed: int = 0,
        n: int = 10_000,
        outdir: str = "paper/figures",
    ) -> None:
        self.snapshot_q = snapshot_q
        self.seed = seed
        self.n = n
        self.outdir = Path(outdir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _snapshot_index(self, qs: np.ndarray) -> int:
        """Index in qs[:-1] closest to self.snapshot_q."""
        return int(np.argmin(np.abs(qs[:-1] - self.snapshot_q)))

    def _lorenz_curve(
        self, Pq_old: np.ndarray, Pq_new: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (percentiles, cumulative_fraction) for degrees sorted ascending.

        Only includes degrees where Pq_old > EPS (non-trivial support).
        percentiles runs 0..100; cumulative_fraction runs 0..1.
        """
        contribs = Metrics.kl_decomp_by_k(Pq_new, Pq_old)
        ks = np.arange(len(contribs))

        active_mask = Pq_old > self.EPS * 10
        active_ks = ks[active_mask]
        active_c = contribs[active_mask]

        if active_ks.size == 0:
            return np.array([0.0, 100.0]), np.array([0.0, 1.0])

        sort_idx = np.argsort(active_ks)
        c_sorted = np.abs(active_c[sort_idx])
        total = float(np.sum(c_sorted))
        if total <= 0.0:
            return np.array([0.0, 100.0]), np.array([0.0, 1.0])

        cumfrac = np.concatenate([[0.0], np.cumsum(c_sorted) / total])
        n = len(active_ks)
        percentiles = np.linspace(0.0, 100.0, n + 1)
        return percentiles, cumfrac

    # ------------------------------------------------------------------
    # Run and plot
    # ------------------------------------------------------------------

    def run(self) -> None:
        qs = np.linspace(0.0, self.Q_MAX, self.N_Q)
        curves: list[tuple[np.ndarray, np.ndarray]] = []

        for gamma in self.GAMMAS:
            np.random.seed(self.seed)
            random.seed(self.seed)

            gm = GraphModel(n=self.n, gamma=gamma)
            exp = Experiment(gm, RandomFailure())
            _, _, Pq_values = exp.sweep(qs)

            idx = self._snapshot_index(qs)
            percentiles, cumfrac = self._lorenz_curve(Pq_values[idx], Pq_values[idx + 1])
            curves.append((percentiles, cumfrac))
            print(f"  gamma={gamma}: D_KL step at q≈{self.snapshot_q:.2f} decomposed "
                  f"over {len(percentiles)-1} active degree classes")

        self._plot(curves)

    def _plot(self, curves: list[tuple[np.ndarray, np.ndarray]]) -> None:
        fig, ax = plt.subplots(figsize=(6.0, 4.5), constrained_layout=True, dpi=150)

        for (percentiles, cumfrac), gamma, color in zip(curves, self.GAMMAS, self.COLORS):
            ax.plot(percentiles, cumfrac, color=color, lw=1.8, label=rf"$\gamma={gamma}$")

        ax.plot([0, 100], [0, 1], color="0.55", lw=0.9, ls="--",
                label="uniform (reference)")
        ax.axhline(0.5, color="0.80", lw=0.6, ls=":")
        ax.axvline(50, color="0.80", lw=0.6, ls=":")

        ax.set_xlabel("Degree percentile (ascending $k$)", fontsize=10)
        ax.set_ylabel(r"Cumulative $D_{\mathrm{KL}}$ fraction", fontsize=10)
        ax.set_xlim(0.0, 100.0)
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=9, loc="upper left")

        self.outdir.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            path = self.outdir / f"fig_kl_lorenz.{ext}"
            fig.savefig(path, bbox_inches="tight", pad_inches=0.4, **({"dpi": 150} if ext == "png" else {}))
        plt.close(fig)
        print(f"Saved: {self.outdir / 'fig_kl_lorenz'}.pdf / .png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cumulative KL contribution by degree percentile."
    )
    parser.add_argument("--seed",   type=int,   default=0)
    parser.add_argument("--n",      type=int,   default=10_000)
    parser.add_argument("--q",      type=float, default=0.30,
                        help="Snapshot removal fraction for decomposition")
    parser.add_argument("--outdir", type=str,   default="paper/figures")
    args = parser.parse_args()

    KLLorenzFigure(
        snapshot_q=args.q,
        seed=args.seed,
        n=args.n,
        outdir=args.outdir,
    ).run()
