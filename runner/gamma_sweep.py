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
        """
        rows = []

        for gamma in self.GAMMAS:
            q_warn_random = []
            q_warn_targeted = []

            for seed in self.seeds:
                print(f"  seed {seed}...", flush=True)
                np.random.seed(seed)

                graph = GraphModel(n=self.n, gamma=gamma)

                # --- random failure ---
                exp_r = Experiment(graph, RandomFailure())
                _, _, Pq = exp_r.sweep(self.qs)
                dkl = exp_r.ewma(exp_r.successive_kl(Pq))
                q_warn_random.append(self._detect_baseline_break(self.qs, dkl))

                # --- targeted failure ---
                exp_t = Experiment(graph, TargetedFailure())
                _, _, Pq = exp_t.sweep(self.qs)
                dkl = exp_t.ewma(exp_t.successive_kl(Pq))
                q_warn_targeted.append(self._detect_positive_drift(self.qs, dkl))

            rows.append((
                gamma,
                np.mean(q_warn_random), np.std(q_warn_random),
                np.mean(q_warn_targeted), np.std(q_warn_targeted),
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


    def _detect_positive_drift(self, qs, signal, m=3):
        for i in range(len(signal) - m):
            if np.all(np.diff(signal[i:i+m]) > 0):
                return qs[i]
        return np.nan
