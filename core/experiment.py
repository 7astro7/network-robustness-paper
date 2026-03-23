from __future__ import annotations

import numpy as np

from core.metrics import Metrics


class Experiment:
    """
    Runs robustness experiments across a sweep of failure probabilities q.

    This class is intentionally compute-only (no plotting) so it can be reused in
    headless environments and paper pipelines without extra dependencies.
    """

    def __init__(self, graph_model: "GraphModel", failure_model: "FailureModel | None") -> None:
        self.graph_model = graph_model
        self.failure_model = failure_model

    def sweep(self, qs: np.ndarray) -> tuple[list[float], list[float], list[np.ndarray]]:
        """
        Sweep over damage fractions qs and return:
          - S_values: GCC fraction S(q)
          - H_values: degree entropy H(q) (bits)
          - Pq_values: degree distributions on a fixed support (epsilon-smoothed)
        """
        S_values: list[float] = []
        H_values: list[float] = []
        Pq_values: list[np.ndarray] = []

        # Fixed support across the entire sweep (based on the initial graph)
        k_max0 = max(dict(self.graph_model.G.degree()).values())
        eps = 1e-12

        for q in qs:
            Gq = self.failure_model.apply(self.graph_model.G, float(q))

            S_values.append(float(Metrics.giant_component_fraction(Gq)))
            H_values.append(float(Metrics.degree_entropy(Gq)))

            # Fixed support + epsilon smoothing
            Pq = self.graph_model._degree_distribution(Gq, k_max=k_max0, eps=eps)
            Pq_values.append(np.asarray(Pq, dtype=float))

        return S_values, H_values, Pq_values

    def successive_kl(self, Pq_values: list[np.ndarray]) -> np.ndarray:
        """Compute successive KL: D_KL(P_{i+1} || P_i) for the sweep."""
        dkl = []
        for i in range(len(Pq_values) - 1):
            dkl.append(float(Metrics.kl_divergence(Pq_values[i + 1], Pq_values[i])))
        return np.asarray(dkl, dtype=float)

    def ewma(self, signal: np.ndarray, alpha: float = 0.2) -> np.ndarray:
        """Exponentially weighted moving average for 1D signals."""
        signal = np.asarray(signal, dtype=float)
        if signal.size == 0:
            return signal
        out = np.empty_like(signal)
        out[0] = signal[0]
        for i in range(1, len(signal)):
            out[i] = alpha * signal[i] + (1.0 - alpha) * out[i - 1]
        return out


