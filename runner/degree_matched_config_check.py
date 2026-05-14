"""
runner/degree_matched_config_check.py

Degree-matched configuration-model diagnostic.

For each (gamma, cl_seed), we:
  1. Generate a Chung-Lu graph to obtain a realized degree sequence.
  2. Construct multiple CM realizations from that exact degree sequence.
  3. Run the same random-failure baseline-deviation detection on each CM replicate.

This isolates the effect of the generative mechanism (expected-degree vs hard-constraint
wiring) from the degree sequence: if detection fails in degree-matched CM, the failure
is attributable to the stub-pairing randomness, not the degree sequence per se.

Usage:
    python -m runner.degree_matched_config_check [options]
"""
from __future__ import annotations

import argparse
import csv
import random
from multiprocessing import Pool
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.graph_model import GraphModel, DegreeMatchedConfigurationModel
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
    if baseline_mask.sum() < 2:
        return float("nan")
    mu = dkl[baseline_mask].mean()
    sigma = dkl[baseline_mask].std()
    threshold = mu + z * sigma
    idx = np.where((dkl > threshold) & (qs_mid > q0))[0]
    return float(qs_mid[idx[0]]) if idx.size else float("nan")


def _run_one(args: tuple) -> dict:
    gamma, cl_seed, cm_replicate, n, alpha, z, num_q = args

    np.random.seed(cl_seed)
    random.seed(cl_seed)

    cl_graph = GraphModel(n=n, gamma=gamma)
    cl_degree_seq = [d for _, d in cl_graph.G.degree()]

    # Re-seed for the CM replicate wiring
    np.random.seed(cl_seed * 10_000 + cm_replicate)
    random.seed(cl_seed * 10_000 + cm_replicate)

    qs = np.linspace(0, 0.9, num_q)
    dm_graph = DegreeMatchedConfigurationModel(degree_sequence=cl_degree_seq, gamma=gamma)
    exp = Experiment(dm_graph, RandomFailure())
    S_vals, _, Pq_vals = exp.sweep(qs)
    raw_kl = exp.successive_kl(Pq_vals)
    dkl_smooth = exp.ewma(raw_kl, alpha=alpha)

    q_warn = _detect_baseline_break(qs, dkl_smooth, q0=0.15, z=z)
    q_collapse = next((float(q) for q, s in zip(qs, S_vals) if s < 0.1), float("nan"))

    if np.isfinite(q_warn) and np.isfinite(q_collapse) and q_warn >= q_collapse:
        q_warn = float("nan")

    delta = float(q_collapse - q_warn) if np.isfinite(q_warn) and np.isfinite(q_collapse) else float("nan")

    print(f"  dm_cm: gamma={gamma:.1f} cl_seed={cl_seed} cm_rep={cm_replicate} "
          f"q_warn={q_warn:.3f} q_collapse={q_collapse:.3f}", flush=True)

    return {
        "gamma": float(gamma),
        "cl_seed": int(cl_seed),
        "cm_replicate": int(cm_replicate),
        "q_warn": q_warn,
        "q_collapse": q_collapse,
        "delta_warn": delta,
        "detected": int(np.isfinite(q_warn)),
    }


class DegreeMatchedConfigExperiment:
    """
    Degree-matched configuration-model diagnostic experiment.

    Parameters
    ----------
    gammas : list[float]
        Gamma values to test.
    cl_seeds : list[int]
        Seeds for Chung-Lu source graph generation.
    cm_replicates : int
        Number of CM wiring replicates per (gamma, cl_seed).
    n : int
        Number of nodes.
    alpha : float
        EWMA smoothing parameter.
    z : float
        Baseline threshold multiplier.
    num_q : int
        Number of q points in the damage grid.
    """

    def __init__(
        self,
        gammas: list[float] | None = None,
        cl_seeds: list[int] | None = None,
        cm_replicates: int = 5,
        n: int = 10_000,
        alpha: float = 0.2,
        z: float = 2.0,
        num_q: int = 100,
    ) -> None:
        self.gammas = list(gammas) if gammas is not None else [2.5, 2.7, 3.0]
        self.cl_seeds = list(cl_seeds) if cl_seeds is not None else list(range(10))
        self.cm_replicates = int(cm_replicates)
        self.n = int(n)
        self.alpha = float(alpha)
        self.z = float(z)
        self.num_q = int(num_q)

    def run(self, n_workers: int | None = None) -> list[dict]:
        import os
        n_workers = n_workers or (os.cpu_count() or 1)

        all_args = [
            (gamma, cl_seed, rep, self.n, self.alpha, self.z, self.num_q)
            for gamma in self.gammas
            for cl_seed in self.cl_seeds
            for rep in range(self.cm_replicates)
        ]

        with Pool(processes=n_workers) as pool:
            results = pool.map(_run_one, all_args)

        return results

    def summarize(self, results: list[dict]) -> None:
        import itertools
        results_sorted = sorted(results, key=lambda r: (r["gamma"], r["cl_seed"], r["cm_replicate"]))
        grouped = {
            g: list(rs)
            for g, rs in itertools.groupby(results_sorted, key=lambda r: r["gamma"])
        }

        print("\n" + "=" * 70)
        print("SUMMARY: Degree-Matched Configuration-Model Check")
        print("=" * 70)
        for gamma in sorted(grouped.keys()):
            rows = grouped[gamma]
            n_total = len(rows)
            n_det = sum(r["detected"] for r in rows)
            deltas = [r["delta_warn"] for r in rows if np.isfinite(r["delta_warn"])]
            delta_med = float(np.median(deltas)) if deltas else float("nan")
            print(f"\nγ = {gamma:.1f}:  {n_det}/{n_total} detected  "
                  f"Δ_warn median={delta_med:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Degree-matched configuration-model diagnostic.")
    ap.add_argument("--gammas", nargs="+", type=float, default=[2.5, 2.7, 3.0])
    ap.add_argument("--cl-seeds", nargs="+", type=int, default=list(range(10)),
                    help="Seeds for CL source graphs (default: 0..9)")
    ap.add_argument("--cm-replicates", type=int, default=5,
                    help="CM wiring replicates per (gamma, cl_seed) (default: 5)")
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--z", type=float, default=2.0)
    ap.add_argument("--num-q", type=int, default=100)
    ap.add_argument("--outdir", type=str, default="paper/data")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    exp = DegreeMatchedConfigExperiment(
        gammas=args.gammas,
        cl_seeds=args.cl_seeds,
        cm_replicates=args.cm_replicates,
        n=args.n,
        alpha=args.alpha,
        z=args.z,
        num_q=args.num_q,
    )

    results = exp.run()
    exp.summarize(results)

    out_csv = outdir / "degree_matched_config_random_failure.csv"
    fieldnames = ["gamma", "cl_seed", "cm_replicate", "q_warn", "q_collapse", "delta_warn", "detected"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults written to {out_csv}")


if __name__ == "__main__":
    main()
