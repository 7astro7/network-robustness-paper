"""
runner/baseline_window_sensitivity.py

Sweep the baseline-window endpoint q0 ∈ {0.10, 0.15, 0.20, 0.25} to assess
sensitivity of detection to this choice, for the primary Chung-Lu random-failure
analysis (gamma=2.5, alpha=0.20, z=2.0, 40 seeds).

For each (q0, seed) pair we run the full random-failure sweep and apply
_detect_baseline_break with the given q0.  Detection rate and q_warn
statistics are reported across seeds.

Run as:
    python -m runner.baseline_window_sensitivity [options]
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from multiprocessing import Pool
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.graph_model import GraphModel
from core.failure_model import RandomFailure
from core.experiment import Experiment


def _detect_baseline_break(
    qs: np.ndarray,
    dkl: np.ndarray,
    q0: float = 0.15,
    z: float = 2.0,
) -> float:
    qs = np.asarray(qs, dtype=float)
    dkl = np.asarray(dkl, dtype=float)
    qs_mid = 0.5 * (qs[:-1] + qs[1:])
    baseline_mask = qs_mid <= q0
    if baseline_mask.sum() < 5:
        return float("nan")
    mu = dkl[baseline_mask].mean()
    sigma = dkl[baseline_mask].std()
    threshold = mu + z * sigma
    idx = np.where((dkl > threshold) & (qs_mid > q0))[0]
    return float(qs_mid[idx[0]]) if idx.size else float("nan")


def _run_one(args: tuple) -> dict:
    gamma, seed, n, alpha, z, num_q, q0_values = args

    np.random.seed(seed)
    random.seed(seed)

    qs = np.linspace(0, 0.9, num_q)
    graph = GraphModel(n=n, gamma=gamma)
    exp = Experiment(graph, RandomFailure())
    S_vals, _, Pq_vals = exp.sweep(qs)
    raw_kl = exp.successive_kl(Pq_vals)
    dkl_smooth = exp.ewma(raw_kl, alpha=alpha)

    q_collapse = next((float(q) for q, s in zip(qs, S_vals) if s < 0.1), float("nan"))

    row: dict = {"gamma": float(gamma), "seed": int(seed), "q_collapse": q_collapse}
    for q0 in q0_values:
        q_warn = _detect_baseline_break(qs, dkl_smooth, q0=q0, z=z)
        if np.isfinite(q_warn) and np.isfinite(q_collapse) and q_warn >= q_collapse:
            q_warn = float("nan")
        row[f"q_warn_{q0:.2f}"] = q_warn
        row[f"detected_{q0:.2f}"] = int(np.isfinite(q_warn))

    print(f"  bw_sens: gamma={gamma:.1f} seed={seed}", flush=True)
    return row


class BaselineWindowSensitivityExperiment:
    """
    Sweep the baseline window endpoint q0 for the main random-failure analysis.

    Parameters
    ----------
    gammas : list[float]
    seeds : list[int]
    q0_values : list[float]
        Baseline window endpoints to test (default: [0.10, 0.15, 0.20, 0.25]).
    n, alpha, z, num_q : experiment parameters.
    """

    def __init__(
        self,
        gammas: list[float] | None = None,
        seeds: list[int] | None = None,
        q0_values: list[float] | None = None,
        n: int = 10_000,
        alpha: float = 0.2,
        z: float = 2.0,
        num_q: int = 100,
    ) -> None:
        self.gammas = list(gammas) if gammas is not None else [2.5, 2.7, 3.0]
        self.seeds = list(seeds) if seeds is not None else list(range(40))
        self.q0_values = list(q0_values) if q0_values is not None else [0.10, 0.15, 0.20, 0.25]
        self.n = int(n)
        self.alpha = float(alpha)
        self.z = float(z)
        self.num_q = int(num_q)

    def run(self, n_workers: int | None = None) -> list[dict]:
        n_workers = n_workers or (os.cpu_count() or 1)
        all_args = [
            (gamma, seed, self.n, self.alpha, self.z, self.num_q, self.q0_values)
            for gamma in self.gammas
            for seed in self.seeds
        ]
        with Pool(processes=n_workers) as pool:
            return pool.map(_run_one, all_args)

    def write_table(self, results: list[dict], out_tex: Path) -> None:
        import pandas as pd

        df = pd.DataFrame(results)
        out_tex.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        lines.append(r"\begin{table}[H]")
        lines.append(r"\centering")
        lines.append(r"\footnotesize")
        header_q0s = " & ".join(f"$q_0={q:.2f}$" for q in self.q0_values)
        lines.append(
            r"\caption{Sensitivity of baseline-window endpoint $q_0$ on detection rate and "
            r"$q_{\mathrm{warn}}$ (median [IQR]) under random failure. "
            r"Each column varies the upper limit of the calibration window; all other settings "
            r"($\alpha=0.20$, $z=2$) are held fixed. "
            r"Detection rate $n_{\mathrm{det}}/40$ and $\tilde{q}_{\mathrm{warn}}$ [IQR] are "
            r"computed across 40 seeds per $\gamma$.}"
        )
        lines.append(r"\label{tab:baseline_window_sensitivity}")
        n_cols = 1 + len(self.q0_values)
        col_spec = "c" * n_cols
        lines.append(r"\begin{tabular}{" + col_spec + r"}")
        lines.append(r"\toprule")
        lines.append(r"$\gamma$ & " + header_q0s + r" \\")
        lines.append(r"\midrule")

        for gamma in sorted(df["gamma"].unique()):
            sub = df[df["gamma"] == gamma]
            cells = [f"{gamma:.1f}"]
            for q0 in self.q0_values:
                col_det = f"detected_{q0:.2f}"
                col_w = f"q_warn_{q0:.2f}"
                n_det = int(sub[col_det].sum())
                warns = sub.loc[sub[col_det] == 1, col_w].dropna()
                if len(warns) >= 1:
                    med = float(np.median(warns))
                    q1 = float(np.percentile(warns, 25))
                    q3 = float(np.percentile(warns, 75))
                    cells.append(f"{n_det}/40 ({med:.3f} [{q3-q1:.3f}])")
                else:
                    cells.append(f"{n_det}/40")
            lines.append(" & ".join(cells) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

        out_tex.write_text("\n".join(lines))
        print(f"Written: {out_tex}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline-window sensitivity analysis.")
    ap.add_argument("--gammas", nargs="+", type=float, default=[2.5, 2.7, 3.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(40)))
    ap.add_argument("--q0-values", nargs="+", type=float, default=[0.10, 0.15, 0.20, 0.25])
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--z", type=float, default=2.0)
    ap.add_argument("--num-q", type=int, default=100)
    ap.add_argument("--outdir", type=str, default="paper/data")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    exp = BaselineWindowSensitivityExperiment(
        gammas=args.gammas,
        seeds=args.seeds,
        q0_values=args.q0_values,
        n=args.n,
        alpha=args.alpha,
        z=args.z,
        num_q=args.num_q,
    )

    results = exp.run()

    out_csv = outdir / "baseline_window_sensitivity.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV written: {out_csv}")

    out_tex = Path("paper/tables/baseline_window_sensitivity.tex")
    exp.write_table(results, out_tex)


if __name__ == "__main__":
    main()
