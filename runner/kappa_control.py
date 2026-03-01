import csv
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np

from core.experiment import Experiment
from core.failure_model import RandomFailure
from core.graph_model import GraphModel


@dataclass
class KappaControlRow:
    seed: int
    q_warn_kappa: float
    q_collapse: float
    delta_warn: float
    zmax_kappa: float


def _kappa_from_P(P: np.ndarray) -> float:
    """
    Molloy--Reed branching factor computed from a degree distribution P(k):
      kappa = (E[k^2] - E[k]) / E[k]
    """
    P = np.asarray(P, dtype=float)
    ks = np.arange(P.size, dtype=float)
    mean_k = float(np.sum(ks * P))
    if mean_k <= 0.0:
        return float("nan")
    mean_k2 = float(np.sum((ks**2) * P))
    return (mean_k2 - mean_k) / mean_k


def _detect_baseline_break_on_grid(
    qs: np.ndarray,
    signal: np.ndarray,
    q0: float = 0.15,
    z: float = 2.0,
) -> float:
    """
    Baseline-deviation rule on the q-grid:
      - baseline window is defined on qs: q <= q0
      - warn at the earliest q > q0 where signal > mu0 + z*sigma0
      - returns np.nan if no detection
    """
    qs = np.asarray(qs, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if qs.shape != signal.shape:
        raise ValueError("qs and signal must have the same shape")

    baseline_mask = qs <= q0
    if baseline_mask.sum() < 2:
        return float("nan")

    mu0 = float(np.mean(signal[baseline_mask]))
    sigma0 = float(np.std(signal[baseline_mask]))
    thr = mu0 + z * sigma0

    idx = np.where((qs > q0) & (signal > thr))[0]
    if idx.size == 0:
        return float("nan")
    return float(qs[idx[0]])


def _baseline_mu_sigma(
    qs: np.ndarray, signal: np.ndarray, q0: float = 0.15
) -> tuple[float, float]:
    qs = np.asarray(qs, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if qs.shape != signal.shape:
        raise ValueError("qs and signal must have the same shape")

    baseline_mask = qs <= q0
    if baseline_mask.sum() < 2:
        return float("nan"), float("nan")
    mu0 = float(np.mean(signal[baseline_mask]))
    sigma0 = float(np.std(signal[baseline_mask]))
    return mu0, sigma0


def _first_crossing(qs: np.ndarray, S: np.ndarray, thresh: float) -> float:
    qs = np.asarray(qs, dtype=float)
    S = np.asarray(S, dtype=float)
    idx = np.where(S < float(thresh))[0]
    return float(qs[idx[0]]) if idx.size else float("nan")


def _med_iqr(values: list[float]) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    q25, q75 = np.percentile(x, [25, 75])
    return float(np.median(x)), float(q75 - q25)


def _mean_std(values: list[float]) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    # population std (consistent with other tables unless specified)
    return float(np.mean(x)), float(np.std(x))


def run_kappa_control_random_failure(
    *,
    gamma: float = 2.5,
    n: int = 10_000,
    seeds: list[int] | None = None,
    alpha: float = 0.2,
    outdir: str = "paper",
) -> tuple[Path, Path]:
    """
    Appendix-only comparator: apply the baseline-deviation rule to EWMA-smoothed kappa(q)
    under random failure (Chung--Lu, fixed gamma).
    """
    if seeds is None:
        seeds = list(range(40))

    qs = np.linspace(0.0, 0.9, 100)

    out_path = Path(outdir)
    (out_path / "tables").mkdir(parents=True, exist_ok=True)

    csv_path = out_path / "tables" / "control_kappa_random_failure.csv"
    tex_path = out_path / "tables" / "control_kappa_random_failure.tex"

    rows: list[KappaControlRow] = []

    for seed in seeds:
        np.random.seed(seed)
        random.seed(seed)

        graph = GraphModel(n=n, gamma=gamma)
        exp = Experiment(graph, RandomFailure())

        S_values, _H_values, Pq_values = exp.sweep(qs)

        kappa_values = np.asarray([_kappa_from_P(P) for P in Pq_values], dtype=float)
        kappa_smooth = exp.ewma(kappa_values, alpha=alpha)

        mu0, sigma0 = _baseline_mu_sigma(qs, kappa_smooth, q0=0.15)
        if np.isfinite(sigma0) and sigma0 > 0:
            z_kappa = (kappa_smooth - mu0) / sigma0
            zmax_kappa = float(np.nanmax(z_kappa))
        else:
            zmax_kappa = float("nan")

        q_warn_kappa = _detect_baseline_break_on_grid(qs, kappa_smooth, q0=0.15, z=2.0)
        q_collapse = _first_crossing(qs, np.asarray(S_values, dtype=float), thresh=0.1)

        delta_warn = (
            float(q_collapse - q_warn_kappa)
            if np.isfinite(q_warn_kappa) and np.isfinite(q_collapse)
            else float("nan")
        )

        rows.append(
            KappaControlRow(
                seed=int(seed),
                q_warn_kappa=float(q_warn_kappa),
                q_collapse=float(q_collapse),
                delta_warn=float(delta_warn),
                zmax_kappa=float(zmax_kappa),
            )
        )

    # --- write CSV (per-seed transparency) ---
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "seed",
                "q_warn_kappa_or_nan",
                "q_collapse_or_nan",
                "delta_warn_or_nan",
                "zmax_kappa_or_nan",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.seed,
                    f"{r.q_warn_kappa:.6f}" if np.isfinite(r.q_warn_kappa) else "nan",
                    f"{r.q_collapse:.6f}" if np.isfinite(r.q_collapse) else "nan",
                    f"{r.delta_warn:.6f}" if np.isfinite(r.delta_warn) else "nan",
                    f"{r.zmax_kappa:.6f}" if np.isfinite(r.zmax_kappa) else "nan",
                ]
            )

    n_total = len(rows)
    q_warns = [r.q_warn_kappa for r in rows]
    n_det = int(np.sum(np.isfinite(np.asarray(q_warns, dtype=float))))

    # q_warn and delta are detection-conditional; zmax is unconditional across all seeds.
    q_warn_mean, q_warn_std = _mean_std(q_warns)
    delta_mean, delta_std = _mean_std([r.delta_warn for r in rows])
    zmax_mean, zmax_std = _mean_std([r.zmax_kappa for r in rows])

    n_tex = f"{n:,}".replace(",", "{,}")
    warn_cell = "--" if not np.isfinite(q_warn_mean) else f"{q_warn_mean:.3f} $\\pm$ {q_warn_std:.3f}"
    delta_cell = "--" if not np.isfinite(delta_mean) else f"{delta_mean:.3f} $\\pm$ {delta_std:.3f}"
    zmax_cell = "--" if not np.isfinite(zmax_mean) else f"{zmax_mean:.3f} $\\pm$ {zmax_std:.3f}"

    lines: list[str] = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{Moment-based benchmark under random failure. We apply the baseline-deviation "
        rf"rule to the EWMA-smoothed Molloy--Reed branching factor $\tilde{{\kappa}}(q)$ "
        rf"(baseline window $q\le 0.15$, $\alpha={alpha:.2f}$). We report "
        rf"$q_{{\mathrm{{warn}}}}^{{\kappa}}$, $\Delta_{{\mathrm{{warn}}}}^{{\kappa}}"
        rf"=q_{{\mathrm{{collapse}}}}-q_{{\mathrm{{warn}}}}^{{\kappa}}$, detection counts "
        rf"$(n_{{\mathrm{{det}}}}/n)$, and the baseline-referenced $Z_{{\max}}^{{\kappa}}=\max_q"
        rf"\big(\tilde{{\kappa}}(q)-\mu_0\big)/\sigma_0$ as mean $\pm$ std across seeds "
        rf"(Chung--Lu, $N={n_tex}$, $\gamma={gamma:.1f}$).}}"
    )
    lines.append(r"\label{tab:control_kappa_random_failure}")
    lines.append(r"\begin{tabular}{c c c c c}")
    lines.append(r"\toprule")
    lines.append(
        r"$\gamma$ & $q_{\mathrm{warn}}^{\kappa}$ (mean $\pm$ std) & "
        r"$(n_{\mathrm{det}}/n)$ & $\Delta_{\mathrm{warn}}^{\kappa}$ (mean $\pm$ std) & "
        r"$Z^{\kappa}_{\max}$ (mean $\pm$ std) \\"
    )
    lines.append(r"\midrule")
    lines.append(
        f"{gamma:.1f} & {warn_cell} & ({n_det}/{n_total}) & {delta_cell} & {zmax_cell} \\\\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    tex_path.write_text("\n".join(lines))

    return csv_path, tex_path


