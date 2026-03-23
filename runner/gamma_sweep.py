import os
import random
from multiprocessing import Pool

import numpy as np

from core.experiment import Experiment
from core.graph_model import GraphModel, ConfigurationModel
from core.failure_model import RandomFailure, TargetedFailure
from core.metrics import Metrics

def _detect_targeted_onset(
    qs_mid: np.ndarray,
    dkl_smooth_mid: np.ndarray,
    n_baseline: int = 3,
    z: float = 2.0,
    mu0: float | None = None,
    sigma0: float | None = None,
) -> tuple[float | None, float, float, float]:
    """
    Targeted removal: attack-onset detection on midpoint-indexed smoothed successive KL.

    Baseline: first `n_baseline` midpoints. Trigger when signal exceeds mu + z*sigma.
    Returns (q_warn_tgt, mu0, sigma0, thresh) where q_warn_tgt may be None.
    """
    qs_mid = np.asarray(qs_mid, dtype=float)
    dkl_smooth_mid = np.asarray(dkl_smooth_mid, dtype=float)

    if qs_mid.size == 0 or dkl_smooth_mid.size == 0:
        return None, np.nan, np.nan, np.nan
    if qs_mid.size != dkl_smooth_mid.size:
        raise ValueError("qs_mid and dkl_smooth_mid must have same length")

    if mu0 is None or sigma0 is None:
        n = min(int(n_baseline), int(dkl_smooth_mid.size))
        if n <= 0:
            return None, np.nan, np.nan, np.nan

        baseline = dkl_smooth_mid[:n]
        mu0 = float(np.mean(baseline))
        sigma0 = float(np.std(baseline, ddof=0))  # population std

    thresh = float(mu0) + float(z) * float(sigma0)

    idx = np.where(dkl_smooth_mid > thresh)[0]
    q_warn_tgt = float(qs_mid[idx[0]]) if idx.size else None
    return q_warn_tgt, float(mu0), float(sigma0), thresh


def _null_baseline_mu_sigma(graph: GraphModel, qs: np.ndarray, alpha: float, n_baseline: int = 3) -> tuple[float, float]:
    """
    Estimate a noise-floor baseline for successive KL using a no-damage control.

    We evaluate the degree distribution of the initial graph G0 repeatedly (no removals)
    across the same q-grid, compute successive KL on the midpoint support, smooth it, and
    take (mu0, sigma0) from the first n_baseline midpoints.
    """
    qs = np.asarray(qs, dtype=float)
    if qs.size < 2:
        return float("nan"), float("nan")

    G0 = graph.G
    if G0.number_of_nodes() == 0:
        return float("nan"), float("nan")

    k_max0 = max(dict(G0.degree()).values())
    eps = 1e-12

    P0 = np.asarray(graph._degree_distribution(G0, k_max=k_max0, eps=eps), dtype=float)
    Pq_values = [P0 for _ in range(int(qs.size))]

    exp = Experiment(graph, failure_model=None)  # compute-only use: ewma + successive_kl
    raw = exp.successive_kl(Pq_values)
    dkl_mid = exp.ewma(raw, alpha=float(alpha))

    n = min(int(n_baseline), int(dkl_mid.size))
    if n <= 0:
        return float("nan"), float("nan")
    baseline = dkl_mid[:n]
    return float(np.mean(baseline)), float(np.std(baseline, ddof=0))


def _detect_baseline_break(
    qs: np.ndarray,
    dkl: np.ndarray,
    q0: float = 0.15,
    z: float = 2.0,
    return_stats: bool = False,
):
    """
    Detect baseline deviation for random failure.

    Parameters
    ----------
    qs : np.ndarray
        Removal fractions of length N
    dkl : np.ndarray
        Successive KL of length N-1
    q0 : float
        Upper bound of baseline window
    z : float
        Z-score threshold
    return_stats : bool
        If True, return (q_warn, mu, sigma, threshold); else return q_warn only.
    """
    qs = np.asarray(qs)
    dkl = np.asarray(dkl)

    qs_mid = 0.5 * (qs[:-1] + qs[1:])

    if dkl.size != qs_mid.size:
        raise ValueError(
            f"_detect_baseline_break: dkl length {dkl.size} != len(qs)-1 {qs_mid.size}"
        )

    baseline_mask = qs_mid <= q0

    if baseline_mask.sum() < 5:
        if return_stats:
            return np.nan, float("nan"), float("nan"), float("nan")
        return np.nan

    mu = dkl[baseline_mask].mean()
    sigma = dkl[baseline_mask].std()
    threshold = mu + z * sigma

    exceed = dkl > threshold
    idx = np.where(exceed & (qs_mid > q0))[0]

    if len(idx) == 0:
        if return_stats:
            return np.nan, float(mu), float(sigma), float(threshold)
        return np.nan

    if return_stats:
        return qs_mid[idx[0]], float(mu), float(sigma), float(threshold)
    return qs_mid[idx[0]]


def _run_seed_full(args: tuple) -> dict:
    """Worker for parallel seed execution in GammaSweepExperiment.run()."""
    gamma, seed, n, graph_model_str, qs, qs_targeted, alpha, z = args

    np.random.seed(seed)
    random.seed(seed)

    if graph_model_str == "chunglu":
        graph = GraphModel(n=n, gamma=gamma)
    else:
        graph = ConfigurationModel(n=n, gamma=gamma)

    # --- random failure ---
    exp_r = Experiment(graph, RandomFailure())
    S_r, H_r, Pq_r = exp_r.sweep(qs)
    raw_r = exp_r.successive_kl(Pq_r)

    if not np.all(np.isfinite(raw_r)):
        raise ValueError(
            f"[gamma={gamma:.1f}, seed={seed}, random] non-finite raw successive KL"
        )

    dkl_r = exp_r.ewma(raw_r, alpha=alpha)
    q_warn_r, mu0_r, sigma0_r, thresh_r = _detect_baseline_break(
        qs, dkl_r, z=z, return_stats=True
    )

    qs_mid_r = 0.5 * (qs[:-1] + qs[1:])
    post_mask_r = qs_mid_r > 0.15
    max_post_r = float(dkl_r[post_mask_r].max()) if post_mask_r.any() else float("nan")

    js_r_raw = Metrics.successive_js(Pq_r)
    js_r = exp_r.ewma(js_r_raw, alpha=alpha)

    dh_r_raw = np.abs(np.diff(np.asarray(H_r, dtype=float)))
    dh_r = exp_r.ewma(dh_r_raw, alpha=alpha)

    q_warn_js = _detect_baseline_break(qs, js_r, z=z)
    q_warn_dh = _detect_baseline_break(qs, dh_r, z=z)

    q_collapse_r = next((float(q) for q, s in zip(qs, S_r) if s < 0.1), None)

    if np.isfinite(q_warn_r) and q_collapse_r is not None and float(q_warn_r) >= float(q_collapse_r):
        q_warn_r = np.nan
    if np.isfinite(q_warn_js) and q_collapse_r is not None and float(q_warn_js) >= float(q_collapse_r):
        q_warn_js = np.nan
    if np.isfinite(q_warn_dh) and q_collapse_r is not None and float(q_warn_dh) >= float(q_collapse_r):
        q_warn_dh = np.nan

    # --- targeted failure ---
    exp_t = Experiment(graph, TargetedFailure())
    S_t, _, Pq_t = exp_t.sweep(qs_targeted)
    raw_t = exp_t.successive_kl(Pq_t)

    if not np.all(np.isfinite(raw_t)):
        raise ValueError(
            f"[gamma={gamma:.1f}, seed={seed}, targeted] non-finite raw successive KL"
        )

    dkl_t = exp_t.ewma(raw_t, alpha=alpha)
    tgt_intensity = float(dkl_t[0]) if len(dkl_t) > 0 else float("nan")

    qs_mid = 0.5 * (qs_targeted[:-1] + qs_targeted[1:])
    mu_null, sig_null = _null_baseline_mu_sigma(graph, qs_targeted, alpha=alpha, n_baseline=3)
    q_warn_tgt, mu0_tgt, sigma0_tgt, thresh_tgt = _detect_targeted_onset(
        qs_mid, dkl_t, n_baseline=3, z=z, mu0=mu_null, sigma0=sig_null,
    )
    q_warn_tgt = float(q_warn_tgt) if q_warn_tgt is not None else np.nan

    q_collapse_t = next((float(q) for q, s in zip(qs_targeted, S_t) if s < 0.1), None)

    q_floor_tgt = float(qs_mid[0]) if qs_mid.size else float("nan")
    dkl_floor_tgt = float(dkl_t[0]) if len(dkl_t) > 0 else float("nan")
    fired_at_floor_tgt = bool(
        np.isfinite(dkl_floor_tgt)
        and np.isfinite(thresh_tgt)
        and (float(dkl_floor_tgt) > float(thresh_tgt))
    )

    is_early = (
        np.isfinite(q_warn_tgt)
        and (q_collapse_t is not None)
        and (float(q_warn_tgt) < float(q_collapse_t))
    )

    delta_r = (
        (float(q_collapse_r) - float(q_warn_r))
        if (np.isfinite(q_warn_r) and q_collapse_r is not None)
        else float("nan")
    )
    delta_tgt = (
        (float(q_collapse_t) - float(q_warn_tgt))
        if (np.isfinite(q_warn_tgt) and q_collapse_t is not None)
        else float("nan")
    )

    print(f"  gamma={gamma:.1f} seed {seed}...", flush=True)

    return {
        "seed": seed,
        "gamma": float(gamma),
        "q_warn_r": float(q_warn_r) if np.isfinite(q_warn_r) else float("nan"),
        "q_warn_js": float(q_warn_js) if np.isfinite(q_warn_js) else float("nan"),
        "q_warn_dh": float(q_warn_dh) if np.isfinite(q_warn_dh) else float("nan"),
        "q_collapse_r": float(q_collapse_r) if q_collapse_r is not None else float("nan"),
        "delta_r": delta_r,
        "mu0_r": float(mu0_r) if np.isfinite(mu0_r) else float("nan"),
        "sigma0_r": float(sigma0_r) if np.isfinite(sigma0_r) else float("nan"),
        "thresh_r": float(thresh_r) if np.isfinite(thresh_r) else float("nan"),
        "max_post_r": max_post_r,
        "tgt_intensity": tgt_intensity if np.isfinite(tgt_intensity) else float("nan"),
        "q_warn_tgt": float(q_warn_tgt) if np.isfinite(q_warn_tgt) else float("nan"),
        "q_collapse_t": float(q_collapse_t) if q_collapse_t is not None else float("nan"),
        "is_early": bool(is_early),
        "delta_tgt": delta_tgt,
        "mu0_tgt": float(mu0_tgt),
        "sigma0_tgt": float(sigma0_tgt),
        "thresh_tgt": float(thresh_tgt),
        "q_floor_tgt": q_floor_tgt,
        "dkl_floor_tgt": dkl_floor_tgt,
        "fired_at_floor_tgt": fired_at_floor_tgt,
    }


def _run_seed_random_only(args: tuple) -> dict:
    """Worker for parallel seed execution in GammaSweepExperiment.run_random_only()."""
    gamma, seed, n, qs, alpha, z = args

    np.random.seed(seed)
    random.seed(seed)

    graph = GraphModel(n=n, gamma=float(gamma))
    exp_r = Experiment(graph, RandomFailure())
    S_r, _H_r, Pq_r = exp_r.sweep(qs)
    raw_r = exp_r.successive_kl(Pq_r)

    if not np.all(np.isfinite(raw_r)):
        raise ValueError(
            f"[sens gamma={gamma:.1f}, seed={seed}, random] non-finite raw successive KL"
        )

    dkl_r = exp_r.ewma(raw_r, alpha=alpha)
    q_warn_r = _detect_baseline_break(qs, dkl_r, z=z)

    q_collapse_r = next((float(q) for q, s in zip(qs, S_r) if s < 0.1), None)

    if np.isfinite(q_warn_r) and q_collapse_r is not None and float(q_warn_r) >= float(q_collapse_r):
        q_warn_r = np.nan

    print(f"  [sens] gamma={gamma:.1f} seed {seed}...", flush=True)

    return {
        "gamma": float(gamma),
        "seed": seed,
        "q_warn_r": float(q_warn_r) if np.isfinite(q_warn_r) else float("nan"),
        "q_collapse_r": float(q_collapse_r) if q_collapse_r is not None else float("nan"),
        "delta_r": (
            (float(q_collapse_r) - float(q_warn_r))
            if (np.isfinite(q_warn_r) and q_collapse_r is not None)
            else float("nan")
        ),
    }


class GammaSweepExperiment:
    """
    Core experimental object for sweeping degree exponent γ.
    This defines a central axis of analysis in the paper.
    """

    GAMMAS = np.arange(2.1, 3.0, .1)

    def __init__(
        self,
        n: int = 10_000,
        qs: np.ndarray | None = None,
        qs_targeted: np.ndarray | None = None,
        seeds: list[int] | None = None,
        gammas: np.ndarray | list[float] | None = None,
        alpha: float = 0.2,
        z: float = 2.0,
        graph_model: str = "chunglu",
    ) -> None:
        self.n = n
        self.qs = qs if qs is not None else np.linspace(0, 0.9, 100)
        self.qs_targeted = qs_targeted if qs_targeted is not None else np.linspace(0, 0.9, 400)
        self.seeds = seeds if seeds is not None else list(range(40))
        self.gammas = np.asarray(gammas, dtype=float) if gammas is not None else self.GAMMAS
        self.alpha = float(alpha)
        self.z = float(z)
        self.graph_model = graph_model.lower()
        
        if self.graph_model not in ["chunglu", "config"]:
            raise ValueError(f"graph_model must be 'chunglu' or 'config', got '{graph_model}'")


    def run(self) -> tuple[list[tuple], list[dict]]:
        """
        Execute the full γ sweep and return summary statistics.

        Returns
        -------
        tuple[list[tuple], list[dict]]
            (gamma,
            random_warn_mean, random_warn_std, random_warn_n,
            random_delta_mean, random_delta_std, random_delta_n,
            random_js_warn_mean, random_js_warn_std, random_js_warn_n,
            random_dh_warn_mean, random_dh_warn_std, random_dh_warn_n,
            targeted_early_n, targeted_n_total, targeted_early_rate,
            targeted_warn_tgt_mean, targeted_warn_tgt_std, targeted_warn_tgt_n,
            targeted_collapse_mean, targeted_collapse_std, targeted_collapse_n,
            targeted_delta_warn_tgt_mean, targeted_delta_warn_tgt_std, targeted_delta_warn_tgt_n,
            targeted_intensity_mean, targeted_intensity_std,
            random_warn_med, random_warn_iqr,
            random_delta_med, random_delta_iqr,
            targeted_collapse_med, targeted_collapse_iqr,
            targeted_intensity_med, targeted_intensity_iqr)
        """
        rows = []
        runs = []  # long-format per-seed runs (random regime only, for plotting/export)

        def _median_iqr(arr: np.ndarray) -> tuple[float, float, int]:
            """
            Median + IQR computed over finite entries only.
            Returns (median, iqr, n_detected); (nan, nan, 0) if no detections.
            """
            x = np.asarray(arr, dtype=float)
            det = np.isfinite(x)
            n_det = int(np.count_nonzero(det))
            if n_det == 0:
                return float("nan"), float("nan"), 0
            vals = x[det]
            med = float(np.median(vals))
            q1 = float(np.percentile(vals, 25))
            q3 = float(np.percentile(vals, 75))
            return med, float(q3 - q1), n_det

        # Submit all (gamma, seed) tasks in one pool to avoid repeated fork overhead.
        all_args = [
            (gamma, seed, self.n, self.graph_model, self.qs, self.qs_targeted, self.alpha, self.z)
            for gamma in self.gammas
            for seed in self.seeds
        ]

        n_workers = os.cpu_count() or 1
        with Pool(processes=n_workers) as pool:
            all_results = pool.map(_run_seed_full, all_args)

        # Group results by gamma (preserving seed order within each gamma).
        from itertools import groupby
        all_results.sort(key=lambda r: (r["gamma"], r["seed"]))
        grouped = {g: list(rs) for g, rs in groupby(all_results, key=lambda r: r["gamma"])}

        for gamma in self.gammas:
            seed_results = grouped[float(gamma)]

            q_warn_random = []
            q_collapse_random = []
            delta_random = []
            q_warn_js_random = []
            q_warn_dh_random = []
            q_warn_tgt_targeted = []
            is_early_targeted = []
            q_collapse_targeted = []
            delta_warn_tgt = []
            intensity_targeted = []

            for r in seed_results:
                q_warn_random.append(r["q_warn_r"])
                q_warn_js_random.append(r["q_warn_js"])
                q_warn_dh_random.append(r["q_warn_dh"])
                q_collapse_random.append(r["q_collapse_r"])
                delta_random.append(r["delta_r"])
                intensity_targeted.append(r["tgt_intensity"])
                q_warn_tgt_targeted.append(r["q_warn_tgt"])
                q_collapse_targeted.append(r["q_collapse_t"])
                is_early_targeted.append(r["is_early"])
                delta_warn_tgt.append(r["delta_tgt"])

                runs.append({
                    "regime": "random",
                    "gamma": r["gamma"],
                    "seed": int(r["seed"]),
                    "q_warn": r["q_warn_r"],
                    "q_collapse": r["q_collapse_r"],
                    "mu0": r["mu0_r"],
                    "sigma0": r["sigma0_r"],
                    "threshold": r["thresh_r"],
                    "max_post_baseline": r["max_post_r"],
                })
                runs.append({
                    "regime": "targeted",
                    "gamma": r["gamma"],
                    "seed": int(r["seed"]),
                    "q_warn_tgt": r["q_warn_tgt"],
                    "q_collapse": r["q_collapse_t"],
                    "q_floor": r["q_floor_tgt"],
                    "dkl_floor": r["dkl_floor_tgt"],
                    "mu0": r["mu0_tgt"],
                    "sigma0": r["sigma0_tgt"],
                    "thresh": r["thresh_tgt"],
                    "fired_at_floor": r["fired_at_floor_tgt"],
                })

            # ---- aggregate (nan-safe) ----
            q_warn_random = np.asarray(q_warn_random, dtype=float)
            q_collapse_random = np.asarray(q_collapse_random, dtype=float)
            delta_random = np.asarray(delta_random, dtype=float)
            q_warn_js_random = np.asarray(q_warn_js_random, dtype=float)
            q_warn_dh_random = np.asarray(q_warn_dh_random, dtype=float)
            q_warn_tgt_targeted = np.asarray(q_warn_tgt_targeted, dtype=float)
            is_early_targeted = np.asarray(is_early_targeted, dtype=bool)
            q_collapse_targeted = np.asarray(q_collapse_targeted, dtype=float)
            delta_warn_tgt = np.asarray(delta_warn_tgt, dtype=float)
            intensity_targeted = np.asarray(intensity_targeted, dtype=float)

            n_r = int(np.count_nonzero(~np.isnan(q_warn_random)))
            n_dr = int(np.count_nonzero(~np.isnan(delta_random)))
            n_js = int(np.count_nonzero(~np.isnan(q_warn_js_random)))
            n_dh = int(np.count_nonzero(~np.isnan(q_warn_dh_random)))
            n_total = int(len(self.seeds))
            early_n = int(np.count_nonzero(is_early_targeted))
            early_rate = float(early_n / n_total) if n_total > 0 else float("nan")

            mean_r = float(np.nanmean(q_warn_random)) if n_r > 0 else float("nan")
            mean_dr = float(np.nanmean(delta_random)) if n_dr > 0 else float("nan")
            mean_js = float(np.nanmean(q_warn_js_random)) if n_js > 0 else float("nan")
            mean_dh = float(np.nanmean(q_warn_dh_random)) if n_dh > 0 else float("nan")
            n_warn_tgt = int(np.count_nonzero(~np.isnan(q_warn_tgt_targeted)))
            mean_warn_tgt = float(np.nanmean(q_warn_tgt_targeted)) if n_warn_tgt > 0 else float("nan")
            n_collapse = int(np.count_nonzero(~np.isnan(q_collapse_targeted)))
            mean_collapse = float(np.nanmean(q_collapse_targeted)) if n_collapse > 0 else float("nan")
            n_delta = int(np.count_nonzero(~np.isnan(delta_warn_tgt)))
            mean_delta = float(np.nanmean(delta_warn_tgt)) if n_delta > 0 else float("nan")
            n_intensity = int(np.count_nonzero(~np.isnan(intensity_targeted)))
            mean_intensity = float(np.nanmean(intensity_targeted)) if n_intensity > 0 else float("nan")

            # ddof=1; if <2 detected, report std=0.0 (or could use np.nan)
            std_r = float(np.nanstd(q_warn_random, ddof=1)) if n_r > 1 else 0.0
            std_dr = float(np.nanstd(delta_random, ddof=1)) if n_dr > 1 else 0.0
            std_js = float(np.nanstd(q_warn_js_random, ddof=1)) if n_js > 1 else 0.0
            std_dh = float(np.nanstd(q_warn_dh_random, ddof=1)) if n_dh > 1 else 0.0
            std_warn_tgt = float(np.nanstd(q_warn_tgt_targeted, ddof=1)) if n_warn_tgt > 1 else 0.0
            std_collapse = float(np.nanstd(q_collapse_targeted, ddof=1)) if n_collapse > 1 else 0.0
            std_delta = float(np.nanstd(delta_warn_tgt, ddof=1)) if n_delta > 1 else 0.0
            std_intensity = float(np.nanstd(intensity_targeted, ddof=1)) if n_intensity > 1 else 0.0

            med_r, iqr_r, _ = _median_iqr(q_warn_random)
            med_dr, iqr_dr, _ = _median_iqr(delta_random)
            med_tc, iqr_tc, _ = _median_iqr(q_collapse_targeted)
            med_ti, iqr_ti, _ = _median_iqr(intensity_targeted)

            rows.append((
                float(gamma),
                mean_r, std_r, n_r,
                mean_dr, std_dr, n_dr,
                mean_js, std_js, n_js,
                mean_dh, std_dh, n_dh,
                early_n, n_total, early_rate,
                mean_warn_tgt, std_warn_tgt, n_warn_tgt,
                mean_collapse, std_collapse, n_collapse,
                mean_delta, std_delta, n_delta,
                mean_intensity, std_intensity,
                med_r, iqr_r,
                med_dr, iqr_dr,
                med_tc, iqr_tc,
                med_ti, iqr_ti,
            ))

        return rows, runs

    def run_random_only(self) -> list[tuple]:
        """
        Fast path used for alpha×z sensitivity runs.

        Returns
        -------
        list[tuple]
            (gamma,
             random_warn_mean, random_warn_std, random_warn_n,
             random_delta_mean, random_delta_std, random_delta_n)
        """
        rows = []

        all_args = [
            (gamma, seed, self.n, self.qs, self.alpha, self.z)
            for gamma in self.gammas
            for seed in self.seeds
        ]

        n_workers = os.cpu_count() or 1
        with Pool(processes=n_workers) as pool:
            all_results = pool.map(_run_seed_random_only, all_args)

        from itertools import groupby
        all_results.sort(key=lambda r: (r["gamma"], r["seed"]))
        grouped = {g: list(rs) for g, rs in groupby(all_results, key=lambda r: r["gamma"])}

        for gamma in self.gammas:
            seed_results = grouped[float(gamma)]
            q_warn_random = np.asarray([r["q_warn_r"] for r in seed_results], dtype=float)
            delta_random = np.asarray([r["delta_r"] for r in seed_results], dtype=float)

            n_r = int(np.count_nonzero(~np.isnan(q_warn_random)))
            n_dr = int(np.count_nonzero(~np.isnan(delta_random)))

            mean_r = float(np.nanmean(q_warn_random)) if n_r > 0 else float("nan")
            mean_dr = float(np.nanmean(delta_random)) if n_dr > 0 else float("nan")

            std_r = float(np.nanstd(q_warn_random, ddof=1)) if n_r > 1 else 0.0
            std_dr = float(np.nanstd(delta_random, ddof=1)) if n_dr > 1 else 0.0

            rows.append((float(gamma), mean_r, std_r, n_r, mean_dr, std_dr, n_dr))

        return rows


    def _detect_baseline_break(self, qs: np.ndarray, dkl: np.ndarray, q0: float = 0.15, z: float = 2.0, return_stats: bool = False):
        """Delegates to module-level _detect_baseline_break."""
        return _detect_baseline_break(qs, dkl, q0=q0, z=z, return_stats=return_stats)

