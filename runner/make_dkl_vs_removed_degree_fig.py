"""
Diagnostic: D_KL(q) vs mean degree of removed nodes.

For each step i, records the mean degree (in the graph *before* removal) of the
nodes removed to advance from G_{q_i} to G_{q_{i+1}}. Tests whether this per-step
mean removed degree correlates with the smoothed successive KL divergence.

Outputs
-------
paper/figures/dkl_vs_removed_degree.pdf
Pearson r and Spearman rho printed to stdout.
"""
from __future__ import annotations

import os
import random
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.graph_model import GraphModel
from core.experiment import Experiment


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
    return float((xm * ym).sum() / denom) if denom > 0 else np.nan


def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    def _rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(a) + 1, dtype=float)
        return ranks
    return _pearsonr(_rank(x), _rank(y))


def _incremental_sweep(
    G0,
    graph_model,
    qs: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Incremental random-failure sweep.

    At each step i, removes the *additional* nodes needed to advance from
    fraction q_i to q_{i+1}, measuring their degrees in G_{q_i} before removal.

    Returns
    -------
    dkl_smooth : np.ndarray, shape (len(qs)-1,)
        EWMA-smoothed successive KL divergence.
    mean_removed_degree : np.ndarray, shape (len(qs)-1,)
        Mean degree of removed nodes at each step (measured in graph before removal).
    """
    n_orig = G0.number_of_nodes()
    k_max0 = max(dict(G0.degree()).values())
    eps = 1e-12

    exp = Experiment(graph_model, failure_model=None)

    # Build initial P_q
    Gq = G0.copy()
    removed_total: set = set()
    Pq = graph_model._degree_distribution(Gq, k_max=k_max0, eps=eps)

    Pq_values = [Pq]
    mean_removed_deg: list[float] = []

    for i in range(len(qs) - 1):
        q_next = float(qs[i + 1])
        n_target = int(q_next * n_orig)
        n_add = max(0, n_target - len(removed_total))

        # Candidates: nodes still in Gq
        candidates = list(Gq.nodes())

        if n_add > 0 and len(candidates) > 0:
            n_add = min(n_add, len(candidates))
            chosen = np.random.choice(candidates, size=n_add, replace=False)
            # Record mean degree of chosen nodes in Gq (before removal)
            deg_in_Gq = [Gq.degree(v) for v in chosen]
            mean_removed_deg.append(float(np.mean(deg_in_Gq)) if deg_in_Gq else np.nan)
            removed_total.update(chosen)
        else:
            mean_removed_deg.append(np.nan)

        remaining = set(Gq.nodes()) - set(removed_total)
        Gq = G0.subgraph(remaining).copy()
        Pq_next = graph_model._degree_distribution(Gq, k_max=k_max0, eps=eps)
        Pq_values.append(Pq_next)

    raw = exp.successive_kl(Pq_values)
    dkl_smooth = exp.ewma(raw, alpha=alpha)

    return dkl_smooth, np.array(mean_removed_deg, dtype=float)


def _run_seed(args: tuple) -> dict:
    gamma, seed, n, qs, alpha = args

    np.random.seed(seed)
    random.seed(seed)

    graph_model = GraphModel(n=n, gamma=gamma)
    G0 = graph_model.G

    dkl_smooth, mean_removed_deg = _incremental_sweep(G0, graph_model, qs, alpha)

    # Pearson and Spearman on finite pairs
    mask = np.isfinite(dkl_smooth) & np.isfinite(mean_removed_deg)
    if mask.sum() >= 3:
        r   = _pearsonr(dkl_smooth[mask], mean_removed_deg[mask])
        rho = _spearmanr(dkl_smooth[mask], mean_removed_deg[mask])
    else:
        r, rho = np.nan, np.nan

    print(f"  seed={seed}  r={r:.3f}  rho={rho:.3f}", flush=True)
    return {
        "seed": seed,
        "dkl_smooth": dkl_smooth,
        "mean_removed_deg": mean_removed_deg,
        "pearson_r": float(r),
        "spearman_rho": float(rho),
    }


class DKLRemovedDegreeDiagnostic:
    """
    Generates the D_KL vs mean-removed-degree diagnostic figure.
    """

    def __init__(
        self,
        gamma: float = 2.5,
        seeds: list[int] | None = None,
        n: int = 10_000,
        alpha: float = 0.20,
        qs: np.ndarray | None = None,
        outdir: str = "paper/figures",
    ) -> None:
        self.gamma = gamma
        self.seeds = seeds if seeds is not None else list(range(40))
        self.n = n
        self.alpha = alpha
        self.qs = qs if qs is not None else np.linspace(0, 0.9, 100)
        self.outdir = Path(outdir)

    def run(self) -> None:
        args_list = [
            (self.gamma, seed, self.n, self.qs, self.alpha)
            for seed in self.seeds
        ]

        n_workers = os.cpu_count() or 1
        with Pool(processes=n_workers) as pool:
            results = pool.map(_run_seed, args_list)

        # Summary stats
        rs   = np.array([r["pearson_r"]    for r in results], dtype=float)
        rhos = np.array([r["spearman_rho"] for r in results], dtype=float)
        print(f"\ngamma={self.gamma}, n_seeds={len(self.seeds)}")
        print(f"Pearson r:    mean={np.nanmean(rs):.3f}  sd={np.nanstd(rs):.3f}")
        print(f"Spearman rho: mean={np.nanmean(rhos):.3f}  sd={np.nanstd(rhos):.3f}")

        seed0 = next(r for r in results if r["seed"] == 0)
        self._plot(seed0, results)

    def _plot(self, seed0: dict, all_results: list[dict]) -> None:
        qs_mid = 0.5 * (self.qs[:-1] + self.qs[1:])

        fig, (ax_left, ax_right) = plt.subplots(
            1, 2, figsize=(11.0, 4.5), constrained_layout=True, dpi=150
        )

        # ---- Left panel: dual y-axis, seed 0 ----
        dkl  = seed0["dkl_smooth"]
        mrd  = seed0["mean_removed_deg"]
        mask0 = np.isfinite(dkl) & np.isfinite(mrd)

        color_kl  = "#1f77b4"
        color_mrd = "#d62728"

        ax_left.plot(qs_mid[mask0], dkl[mask0], color=color_kl, lw=1.5,
                     label=r"$\tilde{D}_{\mathrm{KL}}(q)$")
        ax_left.set_xlabel(r"Removal fraction $q$")
        ax_left.set_ylabel(r"$\tilde{D}_{\mathrm{KL}}(q)$ (bits)", color=color_kl)
        ax_left.tick_params(axis="y", labelcolor=color_kl)
        ax_left.set_title(
            rf"Signal and mean removed degree ($\gamma={self.gamma}$, seed 0)",
            fontsize=10,
        )

        ax_mrd = ax_left.twinx()
        ax_mrd.plot(qs_mid[mask0], mrd[mask0], color=color_mrd, lw=1.2,
                    alpha=0.75, ls="--", label="Mean removed degree")
        ax_mrd.set_ylabel("Mean degree of removed nodes", color=color_mrd)
        ax_mrd.tick_params(axis="y", labelcolor=color_mrd)

        lines_a, labels_a = ax_left.get_legend_handles_labels()
        lines_b, labels_b = ax_mrd.get_legend_handles_labels()
        ax_left.legend(lines_a + lines_b, labels_a + labels_b, fontsize=8, loc="upper left")

        # ---- Right panel: scatter across all seeds ----
        all_dkl = []
        all_mrd = []
        for r in all_results:
            d = r["dkl_smooth"]
            m = r["mean_removed_deg"]
            mask = np.isfinite(d) & np.isfinite(m)
            all_dkl.extend(d[mask].tolist())
            all_mrd.extend(m[mask].tolist())

        all_dkl = np.array(all_dkl)
        all_mrd = np.array(all_mrd)

        pearson_r    = _pearsonr(all_dkl, all_mrd)
        spearman_rho = _spearmanr(all_dkl, all_mrd)

        ax_right.scatter(all_mrd, all_dkl, s=1.5, alpha=0.15, color="#555555", rasterized=True)
        ax_right.set_xlabel("Mean degree of removed nodes")
        ax_right.set_ylabel(r"$\tilde{D}_{\mathrm{KL}}(q)$ (bits)")
        ax_right.set_title(
            rf"Scatter across all steps $\times$ 40 seeds ($\gamma={self.gamma}$)",
            fontsize=10,
        )
        ax_right.annotate(
            f"Pearson $r = {pearson_r:.3f}$\nSpearman $\\rho = {spearman_rho:.3f}$",
            xy=(0.05, 0.92), xycoords="axes fraction",
            fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

        self.outdir.mkdir(parents=True, exist_ok=True)
        stem = f"dkl_vs_removed_degree_gamma{self.gamma}_alpha{self.alpha:.2f}"
        fig.savefig(self.outdir / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(self.outdir / f"{stem}.png", bbox_inches="tight", dpi=150)
        plt.close(fig)

        # Canonical path the paper references
        canonical = self.outdir / "dkl_vs_removed_degree.pdf"
        import shutil
        shutil.copy(self.outdir / f"{stem}.pdf", canonical)
        print(f"Saved: {canonical}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma",  type=float, default=2.5)
    parser.add_argument("--n",      type=int,   default=10_000)
    parser.add_argument("--alpha",  type=float, default=0.20)
    parser.add_argument("--outdir", default="paper/figures")
    args = parser.parse_args()

    diag = DKLRemovedDegreeDiagnostic(
        gamma=args.gamma,
        n=args.n,
        alpha=args.alpha,
        outdir=args.outdir,
    )
    diag.run()
