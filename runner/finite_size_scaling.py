"""
runner/finite_size_scaling.py

Finite-size scaling of the random-failure early-warning pipeline.

Sweeps N in {1000, 5000, 10000, 50000, 100000} at fixed gamma=2.5, EWMA
alpha=0.20, and 40 seeds per N.  For each (N, seed) combination:
  - generates a Chung-Lu power-law graph,
  - runs random failure over a 100-point q grid in [0, 0.9],
  - EWMA-smooths the successive KL divergence,
  - estimates baseline (mu_0, sigma_0) from the smoothed signal at q <= 0.15,
  - detects q_warn (first mid-point q > 0.15 where signal > mu_0 + 2*sigma_0),
  - detects q_collapse (first q where S(q) < 0.1),
  - computes lead time Delta_warn = q_collapse - q_warn.

Per-N aggregation (over 40 seeds):
  - mean and std of mu_0 and sigma_0,
  - detection rate n_det / 40,
  - median q_warn [IQR] (detection-conditional),
  - median Delta_warn [IQR] (detection-conditional).

Outputs:
  1. Readable stdout table.
  2. paper/tables/tab_finite_size.tex  (booktabs LaTeX fragment).

Usage:
    python -m runner.finite_size_scaling
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from core.experiment import Experiment
from core.failure_model import RandomFailure
from core.graph_model import GraphModel
from runner.gamma_sweep import _detect_baseline_break


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GAMMA: float = 2.5
ALPHA: float = 0.20
Z: float = 2.0
Q0: float = 0.15
S_COLLAPSE_THRESH: float = 0.1
N_SEEDS: int = 40
SEEDS: list[int] = list(range(N_SEEDS))
NS: list[int] = [1_000, 5_000, 10_000, 50_000, 100_000]
QS: np.ndarray = np.linspace(0.0, 0.9, 100)

TEX_OUT: str = "paper/tables/tab_finite_size.tex"


# ---------------------------------------------------------------------------
# Per-seed result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SeedResult:
    n: int
    seed: int
    mu0: float
    sigma0: float
    q_warn: float    # np.nan if not detected
    q_collapse: float  # np.nan if no collapse within sweep
    delta_warn: float  # np.nan if either q_warn or q_collapse is missing


# ---------------------------------------------------------------------------
# Worker function (module-level for multiprocessing pickle-ability)
# ---------------------------------------------------------------------------

def _run_one_seed(args: tuple) -> SeedResult:
    """Run random failure for a single (N, seed) combination."""
    n, seed = args

    np.random.seed(seed)
    random.seed(seed)

    graph = GraphModel(n=n, gamma=GAMMA)
    exp = Experiment(graph, RandomFailure())

    S_values, _H_values, Pq_values = exp.sweep(QS)

    raw = exp.successive_kl(Pq_values)
    if not np.all(np.isfinite(raw)):
        raise ValueError(
            f"[N={n}, seed={seed}] non-finite raw successive KL divergence"
        )

    dkl_smooth = exp.ewma(raw, alpha=ALPHA)

    # _detect_baseline_break operates on midpoint grid; return_stats=True gives
    # (q_warn, mu0, sigma0, threshold).
    q_warn, mu0, sigma0, _thresh = _detect_baseline_break(
        QS, dkl_smooth, q0=Q0, z=Z, return_stats=True
    )

    # q_collapse: first q where GCC fraction < threshold
    q_collapse_val = next(
        (float(q) for q, s in zip(QS, S_values) if s < S_COLLAPSE_THRESH),
        float("nan"),
    )

    # Invalidate a warning that fires at or after collapse
    if (
        np.isfinite(q_warn)
        and np.isfinite(q_collapse_val)
        and float(q_warn) >= q_collapse_val
    ):
        q_warn = float("nan")

    delta_warn = (
        float(q_collapse_val - float(q_warn))
        if np.isfinite(q_warn) and np.isfinite(q_collapse_val)
        else float("nan")
    )

    print(f"  N={n:>7,d} seed {seed:>2d} done", flush=True)

    return SeedResult(
        n=n,
        seed=seed,
        mu0=float(mu0) if np.isfinite(mu0) else float("nan"),
        sigma0=float(sigma0) if np.isfinite(sigma0) else float("nan"),
        q_warn=float(q_warn) if np.isfinite(q_warn) else float("nan"),
        q_collapse=q_collapse_val,
        delta_warn=delta_warn,
    )


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _med_iqr(values: list[float]) -> tuple[float, float]:
    """Median and IQR over finite entries; (nan, nan) if none finite."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan"), float("nan")
    q25, q75 = float(np.percentile(finite, 25)), float(np.percentile(finite, 75))
    return float(np.median(finite)), float(q75 - q25)


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and population std over finite entries; (nan, nan) if none finite."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(finite)), float(np.std(finite))


# ---------------------------------------------------------------------------
# Per-N aggregate dataclass
# ---------------------------------------------------------------------------

@dataclass
class NAggregate:
    n: int
    n_seeds: int
    mu0_mean: float
    mu0_std: float
    sigma0_mean: float
    sigma0_std: float
    det_rate: float   # n_det / n_seeds
    n_det: int
    qwarn_med: float
    qwarn_iqr: float
    delta_med: float
    delta_iqr: float


# ---------------------------------------------------------------------------
# Main experiment class
# ---------------------------------------------------------------------------

class FiniteSizeScalingExperiment:
    """
    Random-failure finite-size scaling experiment at fixed gamma.

    Parameters
    ----------
    ns : list[int]
        Graph sizes to sweep.
    gamma : float
        Power-law exponent (fixed).
    seeds : list[int]
        RNG seeds, one run per seed per N.
    alpha : float
        EWMA smoothing parameter.
    z : float
        Baseline-deviation threshold multiplier.
    q0 : float
        Upper boundary of baseline window.
    qs : np.ndarray
        Removal-fraction grid.
    tex_out : str
        Output path for the LaTeX table fragment.
    """

    def __init__(
        self,
        ns: list[int] | None = None,
        gamma: float = GAMMA,
        seeds: list[int] | None = None,
        alpha: float = ALPHA,
        z: float = Z,
        q0: float = Q0,
        qs: np.ndarray | None = None,
        tex_out: str = TEX_OUT,
    ) -> None:
        self.ns = ns if ns is not None else list(NS)
        self.gamma = float(gamma)
        self.seeds = seeds if seeds is not None else list(SEEDS)
        self.alpha = float(alpha)
        self.z = float(z)
        self.q0 = float(q0)
        self.qs = qs if qs is not None else np.linspace(0.0, 0.9, 100)
        self.tex_out = tex_out

    def run(self) -> list[NAggregate]:
        """
        Execute all (N, seed) runs in parallel and aggregate per N.

        Returns
        -------
        list[NAggregate]
            One aggregate record per N, in the order of self.ns.
        """
        all_args: list[tuple[int, int]] = [
            (n, seed)
            for n in self.ns
            for seed in self.seeds
        ]

        n_workers = os.cpu_count() or 1
        print(
            f"Running {len(all_args)} tasks "
            f"({len(self.ns)} sizes x {len(self.seeds)} seeds) "
            f"with {n_workers} workers ...",
            flush=True,
        )

        with Pool(processes=n_workers) as pool:
            all_results: list[SeedResult] = pool.map(_run_one_seed, all_args)

        # Group by N
        from itertools import groupby
        all_results.sort(key=lambda r: (r.n, r.seed))
        grouped: dict[int, list[SeedResult]] = {
            n: list(rs)
            for n, rs in groupby(all_results, key=lambda r: r.n)
        }

        aggregates: list[NAggregate] = []
        for n in self.ns:
            seed_results = grouped[n]
            aggregates.append(self._aggregate(n, seed_results))

        return aggregates

    def _aggregate(self, n: int, seed_results: list[SeedResult]) -> NAggregate:
        """Compute per-N summary statistics from individual seed results."""
        mu0_vals = [r.mu0 for r in seed_results]
        sigma0_vals = [r.sigma0 for r in seed_results]
        q_warn_vals = [r.q_warn for r in seed_results]
        delta_vals = [r.delta_warn for r in seed_results]

        n_seeds = len(seed_results)
        n_det = int(np.sum(np.isfinite(np.asarray(q_warn_vals, dtype=float))))
        det_rate = float(n_det) / float(n_seeds) if n_seeds > 0 else float("nan")

        mu0_mean, mu0_std = _mean_std(mu0_vals)
        sigma0_mean, sigma0_std = _mean_std(sigma0_vals)
        qwarn_med, qwarn_iqr = _med_iqr(q_warn_vals)
        delta_med, delta_iqr = _med_iqr(delta_vals)

        return NAggregate(
            n=n,
            n_seeds=n_seeds,
            mu0_mean=mu0_mean,
            mu0_std=mu0_std,
            sigma0_mean=sigma0_mean,
            sigma0_std=sigma0_std,
            det_rate=det_rate,
            n_det=n_det,
            qwarn_med=qwarn_med,
            qwarn_iqr=qwarn_iqr,
            delta_med=delta_med,
            delta_iqr=delta_iqr,
        )

    # ------------------------------------------------------------------
    # Output methods
    # ------------------------------------------------------------------

    def print_table(self, aggregates: list[NAggregate]) -> None:
        """Print a readable summary table to stdout."""
        header = (
            f"{'N':>8s}  "
            f"{'mu_0 mean±sd':>16s}  "
            f"{'sigma_0 mean±sd':>16s}  "
            f"{'det rate':>10s}  "
            f"{'q_warn med[IQR]':>18s}  "
            f"{'Delta_warn med[IQR]':>20s}"
        )
        sep = "-" * len(header)
        print(sep)
        print(header)
        print(sep)
        for agg in aggregates:
            def _ms(m: float, s: float) -> str:
                if not np.isfinite(m):
                    return "--"
                return f"{m:.4f}±{s:.4f}"

            def _mi(m: float, iqr: float) -> str:
                if not np.isfinite(m):
                    return "--"
                return f"{m:.3f}[{iqr:.3f}]"

            det_str = f"{agg.n_det}/{agg.n_seeds}"
            print(
                f"{agg.n:>8,d}  "
                f"{_ms(agg.mu0_mean, agg.mu0_std):>16s}  "
                f"{_ms(agg.sigma0_mean, agg.sigma0_std):>16s}  "
                f"{det_str:>10s}  "
                f"{_mi(agg.qwarn_med, agg.qwarn_iqr):>18s}  "
                f"{_mi(agg.delta_med, agg.delta_iqr):>20s}"
            )
        print(sep)

    def write_tex(self, aggregates: list[NAggregate]) -> Path:
        """
        Write a booktabs LaTeX table fragment to self.tex_out.

        Returns the path written.
        """
        out_path = Path(self.tex_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        alpha_str = f"{self.alpha:.2f}"
        n_seeds = aggregates[0].n_seeds if aggregates else N_SEEDS

        def _ms_cell(m: float, s: float) -> str:
            if not np.isfinite(m):
                return r"--"
            return f"{m:.4f} $\\pm$ {s:.4f}"

        def _mi_cell(m: float, iqr: float) -> str:
            if not np.isfinite(m):
                return r"--"
            return f"{m:.3f} [{iqr:.3f}]"

        lines: list[str] = []
        lines.append(r"\begin{table}[H]")
        lines.append(r"\centering")
        lines.append(
            rf"\caption{{Finite-size scaling of detection under random failure "
            rf"($\gamma={self.gamma:.1f}$, EWMA $\alpha={alpha_str}$, "
            rf"{n_seeds} seeds per $N$). "
            rf"Mean $\pm$ SD of baseline noise ($\mu_0$, $\sigma_0$) and "
            rf"detection statistics across seeds.}}"
        )
        lines.append(r"\label{tab:finite_size}")
        lines.append(r"\begin{tabular}{r c c c c c}")
        lines.append(r"\toprule")
        lines.append(
            r"$N$ & "
            r"$\sigma_0$ (mean $\pm$ SD) & "
            r"$\mu_0$ (mean $\pm$ SD) & "
            r"det.\ rate & "
            r"$q_{\mathrm{warn}}$ (med.\ [IQR]) & "
            r"$\Delta_{\mathrm{warn}}$ (med.\ [IQR]) \\"
        )
        lines.append(r"\midrule")

        for agg in aggregates:
            n_fmt = f"{agg.n:,}".replace(",", r"\,")
            det_cell = f"{agg.n_det}/{agg.n_seeds}"
            lines.append(
                f"${n_fmt}$ & "
                f"{_ms_cell(agg.sigma0_mean, agg.sigma0_std)} & "
                f"{_ms_cell(agg.mu0_mean, agg.mu0_std)} & "
                f"{det_cell} & "
                f"{_mi_cell(agg.qwarn_med, agg.qwarn_iqr)} & "
                f"{_mi_cell(agg.delta_med, agg.delta_iqr)} \\\\"
            )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

        out_path.write_text("\n".join(lines))
        return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    experiment = FiniteSizeScalingExperiment(
        ns=list(NS),
        gamma=GAMMA,
        seeds=list(SEEDS),
        alpha=ALPHA,
        z=Z,
        q0=Q0,
        qs=QS.copy(),
        tex_out=TEX_OUT,
    )

    aggregates = experiment.run()
    experiment.print_table(aggregates)

    tex_path = experiment.write_tex(aggregates)
    print(f"\nLaTeX table written to: {tex_path}")


if __name__ == "__main__":
    main()
