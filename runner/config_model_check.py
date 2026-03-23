"""
Configuration Model robustness check: verify baseline-deviation behavior
is not specific to Chung-Lu wiring.

Runs the same random-failure experiment with ConfigurationModel for a subset
of gamma values to confirm the early-warning signal generalizes.
"""
import argparse
import csv
from pathlib import Path
import random
import sys

import numpy as np

# Allow running this file directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.graph_model import ConfigurationModel
from core.failure_model import RandomFailure
from core.experiment import Experiment


def detect_baseline_break(qs: np.ndarray, dkl_smooth: np.ndarray, z: float = 2.0, baseline_frac: float = 0.15) -> float | None:
    """
    Baseline-deviation rule on midpoint-indexed signal.
    Returns q_warn (float or None).
    """
    qs = np.asarray(qs, dtype=float)
    dkl_smooth = np.asarray(dkl_smooth, dtype=float)
    
    if qs.size == 0 or dkl_smooth.size == 0:
        return None
    
    # Midpoint grid
    qs_mid = (qs[:-1] + qs[1:]) / 2.0
    
    if qs_mid.size != dkl_smooth.size:
        raise ValueError(f"Midpoint grid size {qs_mid.size} != signal size {dkl_smooth.size}")
    
    # Baseline window: q_mid <= baseline_frac
    baseline_mask = qs_mid <= baseline_frac
    n_baseline = int(np.count_nonzero(baseline_mask))
    
    if n_baseline < 2:
        return None
    
    baseline_vals = dkl_smooth[baseline_mask]
    mu0 = float(np.mean(baseline_vals))
    sigma0 = float(np.std(baseline_vals, ddof=0))
    thresh = mu0 + z * sigma0
    
    # Search for first exceedance beyond baseline window
    search_mask = qs_mid > baseline_frac
    if not np.any(search_mask):
        return None
    
    search_vals = dkl_smooth[search_mask]
    search_qs = qs_mid[search_mask]
    
    idx = np.where(search_vals > thresh)[0]
    return float(search_qs[idx[0]]) if idx.size else None


def run_config_model_experiment(gamma: float, seed: int, n: int = 10_000, alpha: float = 0.2, z: float = 2.0, num_q: int = 100) -> dict:
    """
    Run one random-failure experiment on ConfigurationModel.
    
    Returns
    -------
    dict with keys: seed, gamma, q_warn, q_collapse, delta_warn, S_final
    """
    np.random.seed(seed)
    random.seed(seed)
    
    qs = np.linspace(0, 0.9, num_q)
    
    graph = ConfigurationModel(n=n, gamma=gamma)
    exp = Experiment(graph, RandomFailure())
    
    S_vals, H_vals, Pq_vals = exp.sweep(qs)
    
    raw_kl = exp.successive_kl(Pq_vals)
    dkl_smooth = exp.ewma(raw_kl, alpha=alpha)
    
    q_warn = detect_baseline_break(qs, dkl_smooth, z=z, baseline_frac=0.15)
    
    q_collapse = next(
        (float(q) for q, s in zip(qs, S_vals) if s < 0.1),
        None
    )
    
    # Enforce "early warning" constraint
    if q_warn is not None and q_collapse is not None and q_warn >= q_collapse:
        q_warn = None
    
    delta_warn = None
    if q_warn is not None and q_collapse is not None:
        delta_warn = float(q_collapse) - float(q_warn)
    
    return {
        "seed": seed,
        "gamma": gamma,
        "q_warn": q_warn if q_warn is not None else np.nan,
        "q_collapse": q_collapse if q_collapse is not None else np.nan,
        "delta_warn": delta_warn if delta_warn is not None else np.nan,
        "S_final": float(S_vals[-1]),
    }


def main():
    ap = argparse.ArgumentParser(description="ConfigurationModel robustness check.")
    ap.add_argument("--gammas", nargs="+", type=float, default=[2.5, 2.7, 3.0],
                    help="Gamma values to test (default: 2.5 2.7 3.0)")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(40)),
                    help="Random seeds (default: 0..39)")
    ap.add_argument("--n", type=int, default=10_000,
                    help="Number of nodes (default: 10000)")
    ap.add_argument("--alpha", type=float, default=0.2,
                    help="EWMA smoothing parameter (default: 0.2)")
    ap.add_argument("--z", type=float, default=2.0,
                    help="Baseline threshold multiplier (default: 2.0)")
    ap.add_argument("--num-q", type=int, default=100,
                    help="Number of q points (default: 100)")
    ap.add_argument("--outdir", type=str, default="paper/data",
                    help="Output directory for results CSV (default: paper/data)")
    
    args = ap.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    out_csv = outdir / "config_model_random_failure.csv"
    
    results = []
    
    for gamma in args.gammas:
        for seed in args.seeds:
            print(f"Running ConfigModel: gamma={gamma:.1f}, seed={seed}...", flush=True)
            res = run_config_model_experiment(
                gamma=gamma,
                seed=seed,
                n=args.n,
                alpha=args.alpha,
                z=args.z,
                num_q=args.num_q
            )
            results.append(res)
            print(f"  → q_warn={res['q_warn']:.3f}, q_collapse={res['q_collapse']:.3f}, "
                  f"Δ={res['delta_warn']:.3f}")
    
    # Write CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "gamma", "q_warn", "q_collapse", "delta_warn", "S_final"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults written to {out_csv}")
    
    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY: Configuration Model Random Failure")
    print("="*70)
    
    for gamma in sorted(set(r["gamma"] for r in results)):
        gamma_runs = [r for r in results if r["gamma"] == gamma]
        
        q_warns = [r["q_warn"] for r in gamma_runs if np.isfinite(r["q_warn"])]
        q_collapses = [r["q_collapse"] for r in gamma_runs if np.isfinite(r["q_collapse"])]
        deltas = [r["delta_warn"] for r in gamma_runs if np.isfinite(r["delta_warn"])]
        
        n_det = len(q_warns)
        n_delta = len(deltas)
        
        if n_det > 0:
            warn_med = np.median(q_warns)
            warn_q1 = np.percentile(q_warns, 25)
            warn_q3 = np.percentile(q_warns, 75)
            warn_iqr = warn_q3 - warn_q1
        else:
            warn_med = warn_iqr = np.nan
        
        if n_delta > 0:
            delta_med = np.median(deltas)
            delta_q1 = np.percentile(deltas, 25)
            delta_q3 = np.percentile(deltas, 75)
            delta_iqr = delta_q3 - delta_q1
        else:
            delta_med = delta_iqr = np.nan
        
        print(f"\nγ = {gamma:.1f}")
        print(f"  q_warn:     {warn_med:.3f} [IQR={warn_iqr:.3f}]  ({n_det}/{len(gamma_runs)} detected)")
        print(f"  Δ_warn:     {delta_med:.3f} [IQR={delta_iqr:.3f}]  ({n_delta}/{len(gamma_runs)} both observed)")
        
        if q_collapses:
            col_med = np.median(q_collapses)
            print(f"  q_collapse: {col_med:.3f}")


if __name__ == "__main__":
    main()

