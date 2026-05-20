"""
Average degree of removed nodes vs successive KL divergence.

For a Chung-Lu power-law graph (gamma=2.5, N=10,000, 40 seeds, random failure)
records at each removal step:
  (1) the average original-graph degree of the nodes removed in that step,
      EWMA-smoothed (alpha=0.20), and
  (2) the EWMA-smoothed successive KL divergence between adjacent empirical
      degree distributions (identical to the main pipeline signal).

Uses a fixed random permutation per seed so the set of nodes removed at each
step is fully determined, making the "removed degree" estimand well-defined.
The successive KL is computed from the degree distribution of surviving nodes at
each cumulative removal level, consistent with the main pipeline.

Outputs
-------
paper/figures/fig_removed_degree_vs_kl.pdf
paper/figures/fig_removed_degree_vs_kl.png
Pearson correlation between the two signals printed to stdout (per seed and
ensemble summary).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.stats import pearsonr

from core.graph_model import GraphModel
from core.metrics import Metrics


# ---------------------------------------------------------------------------
# Module-level defaults
# ---------------------------------------------------------------------------
_GAMMA = 2.5
_N = 10_000
_N_SEEDS = 40
_SEEDS = list(range(_N_SEEDS))
_ALPHA = 0.20
_N_Q = 100
_Q_MAX = 0.90
_Q0 = 0.15
_Z = 2.0
_EPS = 1e-12
_COLLAPSE_THRESH = 0.10


class RemovedDegreeVsKLExperiment:
    """
    Per-seed sweep that records the successive KL signal and the average
    original-graph degree of nodes removed at each step.

    Parameters
    ----------
    gamma          : power-law exponent
    n              : network size
    alpha          : EWMA smoothing parameter
    n_q            : number of q values in the damage grid
    q_max          : maximum removal fraction
    q0             : baseline window upper boundary
    z              : detection threshold multiplier
    eps            : additive regularisation for KL computation
    collapse_thresh: GCC fraction below which collapse is declared
    """

    def __init__(
        self,
        gamma: float = _GAMMA,
        n: int = _N,
        alpha: float = _ALPHA,
        n_q: int = _N_Q,
        q_max: float = _Q_MAX,
        q0: float = _Q0,
        z: float = _Z,
        eps: float = _EPS,
        collapse_thresh: float = _COLLAPSE_THRESH,
    ) -> None:
        self.gamma = gamma
        self.n = n
        self.alpha = alpha
        self.n_q = n_q
        self.q_max = q_max
        self.q0 = q0
        self.z = z
        self.eps = eps
        self.collapse_thresh = collapse_thresh

        self.qs = np.linspace(0.0, q_max, n_q)
        self.qs_mid = 0.5 * (self.qs[:-1] + self.qs[1:])

    # ------------------------------------------------------------------
    # Signal utilities (match core/experiment.py exactly)
    # ------------------------------------------------------------------

    @staticmethod
    def _ewma(x: np.ndarray, alpha: float) -> np.ndarray:
        out = np.empty_like(x)
        out[0] = x[0]
        for i in range(1, len(x)):
            out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
        return out

    def _successive_kl(self, Pq_list: list[np.ndarray]) -> np.ndarray:
        """D_KL(P_{i+1} || P_i) in bits for consecutive distributions."""
        return np.array(
            [float(Metrics.kl_divergence(Pq_list[i + 1], Pq_list[i]))
             for i in range(len(Pq_list) - 1)],
            dtype=float,
        )

    def _detect_q_warn(self, signal: np.ndarray) -> float:
        """Baseline-deviation rule matching the main pipeline."""
        baseline_vals = signal[self.qs_mid <= self.q0]
        if baseline_vals.size < 2:
            return float("nan")
        mu0 = float(np.mean(baseline_vals))
        sigma0 = float(np.std(baseline_vals))
        threshold = mu0 + self.z * sigma0
        exceed = (self.qs_mid > self.q0) & (signal > threshold)
        idx = np.where(exceed)[0]
        return float(self.qs_mid[idx[0]]) if idx.size else float("nan")

    # ------------------------------------------------------------------
    # Per-seed run
    # ------------------------------------------------------------------

    def run_seed(self, seed: int) -> dict:
        """
        Run one seed.

        Returns
        -------
        dict with keys:
          kl_raw, kl_smooth, removed_deg_raw, removed_deg_smooth,
          q_warn, q_collapse, pearson_r, pearson_pval
        """
        np.random.seed(seed)

        # Generate graph (same method as GraphModel)
        gm = GraphModel(n=self.n, gamma=self.gamma)
        G0 = gm.G
        N = G0.number_of_nodes()

        # Original-graph degree for every node
        orig_deg = dict(G0.degree())
        nodes = np.array(G0.nodes())

        # Fixed random permutation: nodes[perm[i]] is the (i+1)-th node to remove
        perm = np.random.permutation(N)

        k_max0 = max(orig_deg.values())

        Pq_list: list[np.ndarray] = []
        removed_deg_raw = np.empty(self.n_q - 1, dtype=float)
        gcc_values: list[float] = []

        n_prev = 0
        for i, q in enumerate(self.qs):
            n_curr = int(q * N)
            # Surviving nodes: everything after the first n_curr in the permutation
            surviving_nodes = nodes[perm[n_curr:]]
            Gq = G0.subgraph(surviving_nodes)

            Pq = gm._degree_distribution(Gq, k_max=k_max0, eps=self.eps)
            Pq_list.append(Pq)

            gcc_values.append(float(Metrics.giant_component_fraction(Gq)))

            if i > 0:
                step_nodes = nodes[perm[n_prev:n_curr]]
                if len(step_nodes) > 0:
                    removed_deg_raw[i - 1] = float(
                        np.mean([orig_deg[v] for v in step_nodes])
                    )
                else:
                    # No new removals this step (shouldn't happen for n_q=100, N=10k)
                    removed_deg_raw[i - 1] = float("nan")

            n_prev = n_curr

        # Successive KL and EWMA
        kl_raw = self._successive_kl(Pq_list)
        kl_smooth = self._ewma(kl_raw, self.alpha)

        # Smooth removed degree (fill any isolated NaN with forward value before EWMA)
        rd = removed_deg_raw.copy()
        for j in range(len(rd)):
            if np.isnan(rd[j]):
                rd[j] = rd[j - 1] if j > 0 else 0.0
        removed_deg_smooth = self._ewma(rd, self.alpha)

        # Detection
        q_warn = self._detect_q_warn(kl_smooth)

        q_collapse = float("nan")
        for j, s in enumerate(gcc_values):
            if s < self.collapse_thresh:
                q_collapse = float(self.qs[j])
                break

        # Invalidate q_warn if it is not before q_collapse (match main pipeline)
        if np.isfinite(q_warn) and np.isfinite(q_collapse) and q_warn >= q_collapse:
            q_warn = float("nan")

        # Pearson correlation between the two smoothed signals
        valid = np.isfinite(kl_smooth) & np.isfinite(removed_deg_smooth)
        if valid.sum() >= 3:
            r, pval = pearsonr(kl_smooth[valid], removed_deg_smooth[valid])
        else:
            r, pval = float("nan"), float("nan")

        return {
            "kl_raw": kl_raw,
            "kl_smooth": kl_smooth,
            "removed_deg_raw": removed_deg_raw,
            "removed_deg_smooth": removed_deg_smooth,
            "q_warn": q_warn,
            "q_collapse": q_collapse,
            "pearson_r": float(r),
            "pearson_pval": float(pval),
        }

    def run_all(self, seeds: list[int]) -> list[dict]:
        """Run all seeds and return a list of per-seed result dicts."""
        results = []
        for seed in seeds:
            print(f"  seed={seed:02d} ...", end=" ", flush=True)
            r = self.run_seed(seed)
            qw = f"{r['q_warn']:.3f}" if np.isfinite(r["q_warn"]) else "n/a"
            qc = f"{r['q_collapse']:.3f}" if np.isfinite(r["q_collapse"]) else "n/a"
            print(f"q_warn={qw}  q_collapse={qc}  r={r['pearson_r']:.3f}")
            results.append(r)
        return results


class RemovedDegreeVsKLFigure:
    """
    Dual-y-axis figure: ensemble-mean EWMA KL (left) and EWMA removed-node degree
    (right) vs removal fraction q, with shaded IQR bands and markers for q_warn
    and q_collapse.
    """

    def __init__(
        self,
        results: list[dict],
        qs_mid: np.ndarray,
        outdir: str = "paper/figures",
    ) -> None:
        self.results = results
        self.qs_mid = qs_mid
        self.outdir = Path(outdir)

    def _ensemble_stats(self, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (mean, p25, p75) across seeds for signal `key`."""
        mat = np.stack([r[key] for r in self.results], axis=0)
        return (
            np.mean(mat, axis=0),
            np.percentile(mat, 25, axis=0),
            np.percentile(mat, 75, axis=0),
        )

    def plot(self) -> None:
        kl_mean, kl_p25, kl_p75 = self._ensemble_stats("kl_smooth")
        rd_mean, rd_p25, rd_p75 = self._ensemble_stats("removed_deg_smooth")

        q_warns = np.array([r["q_warn"] for r in self.results])
        q_collapses = np.array([r["q_collapse"] for r in self.results])
        q_warn_med = float(np.nanmedian(q_warns))
        q_collapse_med = float(np.nanmedian(q_collapses))

        fig, ax1 = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True, dpi=150)
        ax2 = ax1.twinx()

        color_kl = "#1f77b4"
        color_rd = "#d62728"

        # KL signal (left axis)
        ax1.fill_between(self.qs_mid, kl_p25, kl_p75,
                         color=color_kl, alpha=0.18, zorder=1)
        ax1.plot(self.qs_mid, kl_mean, color=color_kl, lw=1.8, zorder=3,
                 label=r"$\tilde{D}_{\mathrm{KL}}(q)$ (ensemble mean)")

        # Removed-degree signal (right axis)
        ax2.fill_between(self.qs_mid, rd_p25, rd_p75,
                         color=color_rd, alpha=0.15, zorder=1)
        ax2.plot(self.qs_mid, rd_mean, color=color_rd, lw=1.8, zorder=3,
                 ls="--", label=r"$\bar{k}_{\mathrm{rem}}(q)$ (ensemble mean)")

        # Baseline window
        ax1.axvspan(0.0, 0.15, color="0.88", alpha=1.0, zorder=0,
                    label=r"calibration window ($q\leq0.15$)")

        # q_warn and q_collapse markers
        if np.isfinite(q_warn_med):
            ax1.axvline(q_warn_med, color=color_kl, lw=1.5, ls=":",
                        zorder=4, label=rf"median $q_{{\mathrm{{warn}}}}={q_warn_med:.3f}$")
        if np.isfinite(q_collapse_med):
            ax1.axvline(q_collapse_med, color="#ff7f0e", lw=1.5, ls=":",
                        zorder=4, label=rf"median $q_{{\mathrm{{collapse}}}}={q_collapse_med:.3f}$")

        ax1.set_xlabel(r"Removal fraction $q$", fontsize=11)
        ax1.set_ylabel(r"$\tilde{D}_{\mathrm{KL}}(q)$ (bits)", color=color_kl, fontsize=10)
        ax2.set_ylabel(r"Mean original degree of removed nodes $\bar{k}_{\mathrm{rem}}(q)$",
                       color=color_rd, fontsize=10)
        ax1.tick_params(axis="y", labelcolor=color_kl)
        ax2.tick_params(axis="y", labelcolor=color_rd)

        ax1.set_xlim(0.0, _Q_MAX)
        ax1.set_ylim(bottom=0.0)
        ax2.set_ylim(bottom=0.0)
        ax1.grid(True, alpha=0.2)

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

        ax1.set_title(
            rf"Removed-node degree vs successive KL ($\gamma={_GAMMA}$, "
            rf"$N={_N:,}$, {_N_SEEDS} seeds)",
            fontsize=10,
        )

        self.outdir.mkdir(parents=True, exist_ok=True)
        out_pdf = self.outdir / "fig_removed_degree_vs_kl.pdf"
        out_png = self.outdir / "fig_removed_degree_vs_kl.png"
        fig.savefig(out_pdf, bbox_inches="tight")
        fig.savefig(out_png, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved: {out_pdf}")
        print(f"Saved: {out_png}")


def print_summary(results: list[dict]) -> None:
    """Print per-seed Pearson correlation table and ensemble statistics."""
    rs = np.array([r["pearson_r"] for r in results])
    ps = np.array([r["pearson_pval"] for r in results])

    print()
    print(f"{'seed':>5}  {'Pearson r':>10}  {'p-value':>12}")
    print("-" * 32)
    for i, r in enumerate(results):
        print(f"{i:>5}  {r['pearson_r']:>10.4f}  {r['pearson_pval']:>12.4e}")
    print("-" * 32)
    print(f"{'mean':>5}  {np.nanmean(rs):>10.4f}  {'':>12}")
    print(f"{'std':>5}  {np.nanstd(rs):>10.4f}  {'':>12}")
    print(f"{'median':>5}  {np.nanmedian(rs):>10.4f}  {'':>12}")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Removed-node degree vs successive KL divergence."
    )
    parser.add_argument("--gamma",  type=float, default=_GAMMA)
    parser.add_argument("--n",      type=int,   default=_N)
    parser.add_argument("--seeds",  type=int,   default=_N_SEEDS)
    parser.add_argument("--outdir", type=str,   default="paper/figures")
    args = parser.parse_args()

    seeds = list(range(args.seeds))

    print(
        f"Running removed-degree vs KL experiment "
        f"(gamma={args.gamma}, N={args.n:,}, {args.seeds} seeds)..."
    )

    exp = RemovedDegreeVsKLExperiment(gamma=args.gamma, n=args.n)
    results = exp.run_all(seeds)

    print_summary(results)

    fig = RemovedDegreeVsKLFigure(results, exp.qs_mid, outdir=args.outdir)
    fig.plot()
