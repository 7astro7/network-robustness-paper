"""
EWS comparator experiment (R2.2).

For each (gamma, seed) pair, runs random-failure CL sweep and applies three
signals to the same baseline-deviation rule:

  1. Successive KL divergence (D_KL)                   — primary signal
  2. Rolling standard deviation of raw D_KL             — variance EWS
  3. Rolling lag-1 autocorrelation of raw D_KL          — AR(1) EWS

Window size w=7 for rolling statistics (gives ~10 valid baseline points).
All settings identical to the main experiment: α=0.20, z=2, q₀=0.15.

Outputs
-------
paper/tables/ews_comparison.tex
"""
from __future__ import annotations

import os
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from core.graph_model import GraphModel
from core.failure_model import RandomFailure
from core.experiment import Experiment
from runner.gamma_sweep import _detect_baseline_break

# Rolling window size. w=7 → first valid index 6, first valid q_mid ≈ 0.059,
# leaving ~10 valid midpoints in the baseline (q₀=0.15).
_ROLLING_W = 7


def _rolling_std_signal(signal: np.ndarray, w: int = _ROLLING_W, min_periods: int = 3) -> np.ndarray:
    """
    Rolling standard deviation with growing window for the first min_periods steps.
    Returns array of same length as signal; never NaN if signal is finite.
    """
    n = len(signal)
    out = np.full(n, np.nan)
    for i in range(n):
        start = max(0, i - w + 1)
        window = signal[start : i + 1]
        if len(window) >= min_periods and np.all(np.isfinite(window)):
            out[i] = float(np.std(window, ddof=1)) if len(window) > 1 else 0.0
    return out


def _rolling_ac1_signal(signal: np.ndarray, w: int = _ROLLING_W, min_periods: int = 5) -> np.ndarray:
    """
    Rolling lag-1 autocorrelation with growing window.
    Returns array of same length as signal; NaN where insufficient data or zero variance.
    """
    n = len(signal)
    out = np.full(n, np.nan)
    for i in range(n):
        start = max(0, i - w + 1)
        window = signal[start : i + 1]
        if len(window) >= min_periods and np.all(np.isfinite(window)):
            x = window[:-1]
            y = window[1:]
            sx, sy = float(np.std(x)), float(np.std(y))
            if sx > 1e-15 and sy > 1e-15:
                out[i] = float(np.corrcoef(x, y)[0, 1])
    return out


def _detect_nanfree(qs: np.ndarray, signal: np.ndarray, q0: float = 0.15, z: float = 2.0) -> float:
    """
    Baseline-deviation detection that skips NaN values in the baseline window.
    Returns q_warn (float) or np.nan if not detected.
    """
    qs = np.asarray(qs, dtype=float)
    signal = np.asarray(signal, dtype=float)
    qs_mid = 0.5 * (qs[:-1] + qs[1:])

    if signal.size != qs_mid.size:
        raise ValueError("signal length mismatch")

    baseline_mask = qs_mid <= q0
    baseline_vals = signal[baseline_mask]
    baseline_vals = baseline_vals[np.isfinite(baseline_vals)]

    if len(baseline_vals) < 5:
        return np.nan

    mu = float(np.mean(baseline_vals))
    sigma = float(np.std(baseline_vals))
    threshold = mu + z * sigma

    exceed = np.isfinite(signal) & (signal > threshold) & (qs_mid > q0)
    idx = np.where(exceed)[0]
    return float(qs_mid[idx[0]]) if idx.size else np.nan


def _run_one(args: tuple) -> dict:
    gamma, seed, n, qs, alpha, z = args

    np.random.seed(seed)
    random.seed(seed)

    graph = GraphModel(n=n, gamma=gamma)
    exp = Experiment(graph, RandomFailure())
    S_values, _, Pq_values = exp.sweep(qs)

    raw = exp.successive_kl(Pq_values)
    dkl_smooth = exp.ewma(raw, alpha=alpha)

    q_collapse = next((float(q) for q, s in zip(qs, S_values) if s < 0.1), None)

    # Signal 1: successive KL (baseline-deviation)
    q_warn_kl = float(_detect_baseline_break(qs, dkl_smooth, q0=0.15, z=z))
    if np.isfinite(q_warn_kl) and q_collapse is not None and q_warn_kl >= q_collapse:
        q_warn_kl = np.nan

    # Signal 2: rolling std of raw KL
    rv_signal = _rolling_std_signal(raw)
    q_warn_rv = _detect_nanfree(qs, rv_signal, q0=0.15, z=z)
    if np.isfinite(q_warn_rv) and q_collapse is not None and q_warn_rv >= q_collapse:
        q_warn_rv = np.nan

    # Signal 3: rolling lag-1 AC of raw KL
    ac1_signal = _rolling_ac1_signal(raw)
    q_warn_ac1 = _detect_nanfree(qs, ac1_signal, q0=0.15, z=z)
    if np.isfinite(q_warn_ac1) and q_collapse is not None and q_warn_ac1 >= q_collapse:
        q_warn_ac1 = np.nan

    print(f"  gamma={gamma:.1f} seed={seed}", flush=True)
    return {
        "gamma": float(gamma),
        "seed": int(seed),
        "q_warn_kl": q_warn_kl,
        "q_warn_rv": q_warn_rv,
        "q_warn_ac1": q_warn_ac1,
        "q_collapse": float(q_collapse) if q_collapse is not None else np.nan,
    }


class EWSComparisonExperiment:
    """Runs rolling-EWS comparator sweep and exports LaTeX table."""

    GAMMAS = [2.5, 2.7, 3.0]

    def __init__(
        self,
        n: int = 10_000,
        seeds: list[int] | None = None,
        alpha: float = 0.20,
        z: float = 2.0,
        qs: np.ndarray | None = None,
        out_dir: str = "paper/tables",
    ) -> None:
        self.n = n
        self.seeds = seeds if seeds is not None else list(range(40))
        self.alpha = alpha
        self.z = z
        self.qs = qs if qs is not None else np.linspace(0, 0.9, 100)
        self.out_dir = Path(out_dir)

    def run(self) -> None:
        args_list = [
            (gamma, seed, self.n, self.qs, self.alpha, self.z)
            for gamma in self.GAMMAS
            for seed in self.seeds
        ]

        n_workers = os.cpu_count() or 1
        with Pool(processes=n_workers) as pool:
            results = pool.map(_run_one, args_list)

        self._export(results)

    def _export(self, results: list[dict]) -> None:
        from itertools import groupby

        results.sort(key=lambda r: (r["gamma"], r["seed"]))

        rows: list[dict] = []
        for gamma in self.GAMMAS:
            group = [r for r in results if r["gamma"] == gamma]

            kl   = np.array([r["q_warn_kl"]  for r in group], dtype=float)
            rv   = np.array([r["q_warn_rv"]  for r in group], dtype=float)
            ac1  = np.array([r["q_warn_ac1"] for r in group], dtype=float)

            rows.append({
                "gamma": gamma,
                "kl_det":  int(np.sum(np.isfinite(kl))),
                "kl_med":  float(np.nanmedian(kl)),
                "kl_iqr":  float(np.nanpercentile(kl, 75) - np.nanpercentile(kl, 25)) if np.sum(np.isfinite(kl)) > 0 else np.nan,
                "rv_det":  int(np.sum(np.isfinite(rv))),
                "rv_med":  float(np.nanmedian(rv)),
                "rv_iqr":  float(np.nanpercentile(rv, 75) - np.nanpercentile(rv, 25)) if np.sum(np.isfinite(rv)) > 0 else np.nan,
                "ac1_det": int(np.sum(np.isfinite(ac1))),
                "ac1_med": float(np.nanmedian(ac1)),
                "ac1_iqr": float(np.nanpercentile(ac1, 75) - np.nanpercentile(ac1, 25)) if np.sum(np.isfinite(ac1)) > 0 else np.nan,
            })

        n_total = len(self.seeds)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.out_dir / "ews_comparison.tex"

        def _fmt(det: int, med: float, iqr: float) -> str:
            if det == 0 or not np.isfinite(med):
                return f"{det}/{n_total} (---)"
            return f"{det}/{n_total} ({med:.3f} [{iqr:.3f}])"

        lines = [
            r"\begin{table}[H]",
            r"\centering",
            r"\footnotesize",
            (
                r"\caption{EWS comparator detection rates under random failure "
                r"($\alpha=0.20$, $z=2$, $q_0\le0.15$, $N=10{,}000$, 40 seeds). "
                r"All three signals use the identical baseline-deviation rule. "
                r"$n_{\mathrm{det}}/40$: seeds with a valid $q_{\mathrm{warn}}$ "
                r"prior to collapse. "
                r"Median $[IQR]$: $q_{\mathrm{warn}}$ over detected seeds; "
                r"lower median = earlier warning. "
                r"Rolling window: $w=" + str(_ROLLING_W) + r"$ steps, "
                r"covering $\Delta q \approx 0.063$ of damage fraction "
                r"(grid step size $\Delta q \approx 0.009$). "
                r"$\gamma=3.0$ is included as an out-of-range test point "
                r"(beyond the primary sweep $\gamma\in(2,3)$) to probe comparator "
                r"behaviour near the lighter-tail boundary; it is absent from all "
                r"other primary results tables.}"
            ),
            r"\label{tab:ews_comparison}",
            r"\begin{tabular}{c c c c}",
            r"\toprule",
            (
                r"$\gamma$ & "
                r"Successive $D_{\mathrm{KL}}$ & "
                r"Rolling $\mathrm{sd}(D_{\mathrm{KL}})$ & "
                r"Rolling $\mathrm{AC}(1)(D_{\mathrm{KL}})$ \\"
            ),
            r"\midrule",
        ]

        for r in rows:
            kl_str  = _fmt(r["kl_det"],  r["kl_med"],  r["kl_iqr"])
            rv_str  = _fmt(r["rv_det"],  r["rv_med"],  r["rv_iqr"])
            ac1_str = _fmt(r["ac1_det"], r["ac1_med"], r["ac1_iqr"])
            lines.append(
                f"{r['gamma']:.1f} & {kl_str} & {rv_str} & {ac1_str} \\\\"
            )

        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

        out_path.write_text("\n".join(lines))
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    EWSComparisonExperiment().run()
