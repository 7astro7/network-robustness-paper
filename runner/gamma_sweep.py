import numpy as np
import random
from experiment.experiment import Experiment
from models.graph_model import GraphModel
from models.failure_model import RandomFailure, TargetedFailure
from models.metrics import Metrics

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
        seeds: list[int] | None = None,
    ):
        self.n = n
        self.qs = qs if qs is not None else np.linspace(0, 0.9, 100)
        self.seeds = seeds if seeds is not None else [0, 1, 2, 3, 4]


    def run(self):
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
            targeted_trigger_mean, targeted_trigger_std, targeted_trigger_n,
            targeted_collapse_mean, targeted_collapse_std, targeted_collapse_n,
            targeted_delta_mean, targeted_delta_std, targeted_delta_n)
        """
        rows = []
        runs = []  # long-format per-seed runs (random regime only, for plotting/export)

        for gamma in self.GAMMAS:
            q_warn_random = []
            q_collapse_random = []
            delta_random = []
            q_warn_js_random = []
            q_warn_dh_random = []
            q_trigger_targeted = []
            is_early_targeted = []
            q_collapse_targeted = []
            delta_targeted = []

            for seed in self.seeds:
                print(f"  gamma={gamma:.1f} seed {seed}...", flush=True)
                # Deterministic graph + failure realization
                np.random.seed(seed)
                random.seed(seed)

                graph = GraphModel(n=self.n, gamma=gamma)

                # --- random failure ---
                exp_r = Experiment(graph, RandomFailure())
                S_r, H_r, Pq_r = exp_r.sweep(self.qs)
                raw_r = exp_r.successive_kl(Pq_r)

                if not np.all(np.isfinite(raw_r)):
                    raise ValueError(
                        f"[gamma={gamma:.1f}, seed={seed}, random] non-finite raw successive KL"
                    )

                dkl_r = exp_r.ewma(raw_r)
                q_warn_r = self._detect_baseline_break(self.qs, dkl_r)

                # Random baselines (rate-of-change signals on midpoint support)
                js_r_raw = Metrics.successive_js(Pq_r)
                js_r = exp_r.ewma(js_r_raw)

                dh_r_raw = np.abs(np.diff(np.asarray(H_r, dtype=float)))
                dh_r = exp_r.ewma(dh_r_raw)

                q_warn_js = self._detect_baseline_break(self.qs, js_r)
                q_warn_dh = self._detect_baseline_break(self.qs, dh_r)

                q_collapse_r = next(
                    (float(q) for q, s in zip(self.qs, S_r) if s < 0.1),
                    None,
                )

                # Ensure q_warn is "early"; otherwise treat as no detection for delta purposes.
                if np.isfinite(q_warn_r) and q_collapse_r is not None and float(q_warn_r) >= float(q_collapse_r):
                    q_warn_r = np.nan
                if np.isfinite(q_warn_js) and q_collapse_r is not None and float(q_warn_js) >= float(q_collapse_r):
                    q_warn_js = np.nan
                if np.isfinite(q_warn_dh) and q_collapse_r is not None and float(q_warn_dh) >= float(q_collapse_r):
                    q_warn_dh = np.nan

                q_warn_random.append(float(q_warn_r) if np.isfinite(q_warn_r) else np.nan)
                q_warn_js_random.append(float(q_warn_js) if np.isfinite(q_warn_js) else np.nan)
                q_warn_dh_random.append(float(q_warn_dh) if np.isfinite(q_warn_dh) else np.nan)
                q_collapse_random.append(float(q_collapse_r) if q_collapse_r is not None else np.nan)

                if np.isfinite(q_warn_r) and q_collapse_r is not None:
                    delta_random.append(float(q_collapse_r) - float(q_warn_r))
                else:
                    delta_random.append(np.nan)

                # long-format record (random regime) for downstream CSV + plotting
                runs.append({
                    "regime": "random",
                    "gamma": float(gamma),
                    "seed": int(seed),
                    "q_warn": float(q_warn_r) if np.isfinite(q_warn_r) else float("nan"),
                    "q_collapse": float(q_collapse_r) if q_collapse_r is not None else float("nan"),
                })

                # --- targeted failure ---
                exp_t = Experiment(graph, TargetedFailure())
                S_t, _, Pq_t = exp_t.sweep(self.qs)
                raw_t = exp_t.successive_kl(Pq_t)

                if not np.all(np.isfinite(raw_t)):
                    raise ValueError(
                        f"[gamma={gamma:.1f}, seed={seed}, targeted] non-finite raw successive KL"
                    )

                dkl_t = exp_t.ewma(raw_t)

                # Targeted: drift-rule trigger point (may be pre- or post-collapse)
                q_trigger = self._detect_positive_drift(self.qs, dkl_t, q0=0.0)
                if q_trigger is None:
                    q_trigger = np.nan

                # collapse (reference only): first q where S(q) < 0.1
                q_collapse_t = next(
                    (float(q) for q, s in zip(self.qs, S_t) if s < 0.1),
                    None,
                )

                is_early = (
                    np.isfinite(q_trigger)
                    and (q_collapse_t is not None)
                    and (float(q_trigger) < float(q_collapse_t))
                )
                q_trigger_targeted.append(float(q_trigger) if np.isfinite(q_trigger) else np.nan)
                is_early_targeted.append(bool(is_early))

                q_collapse_targeted.append(float(q_collapse_t) if q_collapse_t is not None else np.nan)
                if np.isfinite(q_trigger) and q_collapse_t is not None:
                    delta_targeted.append(float(q_collapse_t) - float(q_trigger))
                else:
                    delta_targeted.append(np.nan)

            # ---- aggregate (nan-safe) ----
            q_warn_random = np.asarray(q_warn_random, dtype=float)
            q_collapse_random = np.asarray(q_collapse_random, dtype=float)
            delta_random = np.asarray(delta_random, dtype=float)
            q_warn_js_random = np.asarray(q_warn_js_random, dtype=float)
            q_warn_dh_random = np.asarray(q_warn_dh_random, dtype=float)
            q_trigger_targeted = np.asarray(q_trigger_targeted, dtype=float)
            is_early_targeted = np.asarray(is_early_targeted, dtype=bool)
            q_collapse_targeted = np.asarray(q_collapse_targeted, dtype=float)
            delta_targeted = np.asarray(delta_targeted, dtype=float)

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
            n_trigger = int(np.count_nonzero(~np.isnan(q_trigger_targeted)))
            mean_trigger = float(np.nanmean(q_trigger_targeted)) if n_trigger > 0 else float("nan")
            n_collapse = int(np.count_nonzero(~np.isnan(q_collapse_targeted)))
            mean_collapse = float(np.nanmean(q_collapse_targeted)) if n_collapse > 0 else float("nan")
            n_delta = int(np.count_nonzero(~np.isnan(delta_targeted)))
            mean_delta = float(np.nanmean(delta_targeted)) if n_delta > 0 else float("nan")

            # ddof=1; if <2 detected, report std=0.0 (or could use np.nan)
            std_r = float(np.nanstd(q_warn_random, ddof=1)) if n_r > 1 else 0.0
            std_dr = float(np.nanstd(delta_random, ddof=1)) if n_dr > 1 else 0.0
            std_js = float(np.nanstd(q_warn_js_random, ddof=1)) if n_js > 1 else 0.0
            std_dh = float(np.nanstd(q_warn_dh_random, ddof=1)) if n_dh > 1 else 0.0
            std_trigger = float(np.nanstd(q_trigger_targeted, ddof=1)) if n_trigger > 1 else 0.0
            std_collapse = float(np.nanstd(q_collapse_targeted, ddof=1)) if n_collapse > 1 else 0.0
            std_delta = float(np.nanstd(delta_targeted, ddof=1)) if n_delta > 1 else 0.0

            rows.append((
                float(gamma),
                mean_r, std_r, n_r,
                mean_dr, std_dr, n_dr,
                mean_js, std_js, n_js,
                mean_dh, std_dh, n_dh,
                early_n, n_total, early_rate,
                mean_trigger, std_trigger, n_trigger,
                mean_collapse, std_collapse, n_collapse,
                mean_delta, std_delta, n_delta,
            ))

        return rows, runs


    def _detect_baseline_break(self, qs, dkl, q0=0.15, z=2.0):
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

        Returns
        -------
        float
            q_warn (np.nan if none detected)
        """
        qs = np.asarray(qs)
        dkl = np.asarray(dkl)

        # successive KL lives on midpoints
        qs_mid = 0.5 * (qs[:-1] + qs[1:])

        # baseline window defined on midpoints
        baseline_mask = qs_mid <= q0

        if baseline_mask.sum() < 5:
            return np.nan  # not enough data to define baseline

        mu = dkl[baseline_mask].mean()
        sigma = dkl[baseline_mask].std()

        # detect first deviation
        exceed = dkl > mu + z * sigma
        idx = np.where(exceed & (qs_mid > q0))[0]

        if len(idx) == 0:
            return np.nan

        return qs_mid[idx[0]]

    def _detect_positive_drift(
        self,
        qs,
        signal,
        m=3,
        q0=0.15,
        tol=1e-12,
        max_violations=1,
        min_net_increase=0.0,
    ):
        """
        Targeted early warning on midpoint grid.

        We search for the first midpoint q >= q0 where the signal shows persistent drift:
        - net increase over a window of length (m+1)
        - allow up to `max_violations` local non-increases (noise)
        - optionally require a minimum net increase (min_net_increase)
        """
        qs = np.asarray(qs)
        signal = np.asarray(signal)

        qs_mid = 0.5 * (qs[:-1] + qs[1:])
        if len(signal) != len(qs_mid):
            raise ValueError(f"signal length {len(signal)} must equal len(qs_mid) {len(qs_mid)}")

        # start index enforcing q0 on midpoints
        start = int(np.searchsorted(qs_mid, q0, side="left"))

        for i in range(start, len(signal) - m):
            window = signal[i : i + m + 1]
            diffs = np.diff(window)

            violations = np.sum(diffs <= tol)
            net = window[-1] - window[0]

            if (violations <= max_violations) and (net > max(tol, min_net_increase)):
                return qs_mid[i]

        return np.nan
