import numpy as np
from experiment.experiment import Experiment
from models.graph_model import GraphModel
from models.failure_model import RandomFailure, TargetedFailure

class GammaSweepExperiment:
    """
    Core experimental object for sweeping degree exponent γ.
    This defines a central axis of analysis in the paper.
    """

    # 👇 THIS is now canonical and paper-aligned
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
        list[tuple]
            (gamma,
            random_mean, random_std,
            targeted_mean, targeted_std,
            n_random_detected, n_targeted_detected)
        """
        rows = []

        for gamma in self.GAMMAS:
            q_warn_random = []
            q_warn_targeted = []

            for seed in self.seeds:
                print(f"  gamma={gamma:.1f} seed {seed}...", flush=True)
                np.random.seed(seed)

                graph = GraphModel(n=self.n, gamma=gamma)

                # --- random failure ---
                exp_r = Experiment(graph, RandomFailure())
                _, _, Pq_r = exp_r.sweep(self.qs)
                raw_r = exp_r.successive_kl(Pq_r)

                if not np.all(np.isfinite(raw_r)):
                    raise ValueError(
                        f"[gamma={gamma:.1f}, seed={seed}, random] non-finite raw successive KL"
                    )

                dkl_r = exp_r.ewma(raw_r)
                q_warn_random.append(self._detect_baseline_break(self.qs, dkl_r))

                # --- targeted failure ---
                exp_t = Experiment(graph, TargetedFailure())
                _, _, Pq_t = exp_t.sweep(self.qs)
                raw_t = exp_t.successive_kl(Pq_t)

                if not np.all(np.isfinite(raw_t)):
                    raise ValueError(
                        f"[gamma={gamma:.1f}, seed={seed}, targeted] non-finite raw successive KL"
                    )

                dkl_t = exp_t.ewma(raw_t)
                q_warn_targeted.append(self._detect_positive_drift(self.qs, dkl_t))

            # ---- aggregate (nan-safe) ----
            q_warn_random = np.asarray(q_warn_random, dtype=float)
            q_warn_targeted = np.asarray(q_warn_targeted, dtype=float)

            n_r = int(np.isfinite(q_warn_random).sum())
            n_t = int(np.isfinite(q_warn_targeted).sum())

            rows.append((
                float(gamma),
                float(np.nanmean(q_warn_random)), float(np.nanstd(q_warn_random)),
                float(np.nanmean(q_warn_targeted)), float(np.nanstd(q_warn_targeted)),
                n_r, n_t,
            ))

        return rows


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
