import networkx as nx
import numpy as np


class Metrics:
    """Computes robustness metrics such as GCC size and degree entropy."""

    @staticmethod
    def giant_component_fraction(G):
        """Return |GCC| / |V|."""
        if G.number_of_nodes() == 0:
            return 0.0
        largest = max(nx.connected_components(G), key=len)
        return len(largest) / G.number_of_nodes()

    @staticmethod
    def degree_entropy(G):
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
    def kl_divergence(Pq, P0):
        """
        Compute KL divergence D_KL(Pq || P0) in bits.

        Both Pq and P0 should be dicts mapping degree -> probability.

        Returns
        -------
        float
            KL divergence in bits.
        """
        # align supports
        all_k = set(Pq.keys()) | set(P0.keys())

        D = 0.0
        for k in all_k:
            p = Pq.get(k, 0.0)
            q = P0.get(k, 0.0)
            if p > 0 and q > 0:
                D += p * np.log2(p / q)
        return D

