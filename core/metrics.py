import networkx as nx
import numpy as np


class Metrics:
    """Computes robustness metrics such as GCC size and degree entropy."""

    @staticmethod
    def giant_component_fraction(G: nx.Graph) -> float:
        """Return |GCC| / |V|."""
        if G.number_of_nodes() == 0:
            return 0.0
        largest = max(nx.connected_components(G), key=len)
        return len(largest) / G.number_of_nodes()

    @staticmethod
    def degree_entropy(G: nx.Graph) -> float:
        """
        Compute Shannon entropy of the degree distribution in **bits**.

        H(D) = - Σ_k P(k) log2 P(k)

        Returns
        -------
        float
            Degree entropy in bits.
        """
        degrees = [d for _, d in G.degree()]
        if len(degrees) == 0:
            return 0.0

        values, counts = np.unique(degrees, return_counts=True)
        P = counts / counts.sum()

        return -(P * np.log2(P)).sum()

    @staticmethod
    def kl_divergence(P: np.ndarray, Q: np.ndarray) -> float:
        """
        Compute KL divergence D_KL(P || Q) in bits for discrete distributions.

        P and Q must be 1D numpy arrays of the same length and sum to 1.
        (Your epsilon smoothing guarantees Q > 0 everywhere, so this is stable.)
        """
        P = np.asarray(P, dtype=float)
        Q = np.asarray(Q, dtype=float)

        if P.shape != Q.shape:
            raise ValueError(f"KL shape mismatch: P{P.shape} vs Q{Q.shape}")

        # Only sum where P > 0. If you used eps, this includes everything.
        mask = P > 0
        return float(np.sum(P[mask] * np.log2(P[mask] / Q[mask])))

    @staticmethod
    def js_divergence(P: np.ndarray, Q: np.ndarray) -> float:
        """
        Compute Jensen–Shannon divergence JS(P, Q) in bits for discrete distributions.

        JS(P,Q) = 1/2 KL(P || M) + 1/2 KL(Q || M), where M = (P+Q)/2.

        P and Q must be 1D numpy arrays of the same length and sum to 1.
        With epsilon-smoothed supports, this is numerically stable.
        """
        P = np.asarray(P, dtype=float)
        Q = np.asarray(Q, dtype=float)
        if P.shape != Q.shape:
            raise ValueError(f"JS shape mismatch: P{P.shape} vs Q{Q.shape}")

        M = 0.5 * (P + Q)
        return 0.5 * Metrics.kl_divergence(P, M) + 0.5 * Metrics.kl_divergence(Q, M)

    @staticmethod
    def successive_js(Pq_values: list[np.ndarray]) -> np.ndarray:
        """
        Compute JS(P_{i+1}, P_i) for successive damage levels.

        Returns an array of length len(Pq_values) - 1 (midpoint support).
        """
        js = []
        for i in range(len(Pq_values) - 1):
            js.append(Metrics.js_divergence(Pq_values[i + 1], Pq_values[i]))
        return np.asarray(js, dtype=float)

