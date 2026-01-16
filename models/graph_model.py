import networkx as nx
import numpy as np


class GraphModel:
    """
    Generates power-law networks and stores baseline degree distributions.

    Parameters
    ----------
    n : int
        Number of nodes in the graph.
    gamma : float
        Power-law exponent for the degree sequence.

    Attributes
    ----------
    n : int
        Number of nodes.
    gamma : float
        Power-law exponent.
    G : networkx.Graph
        Generated Chung–Lu expected-degree graph.
    P0 : dict
        Baseline degree distribution P(k) for the initial graph.
    """

    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.G = self._generate_graph()
        self.P0 = self._degree_distribution(self.G)

    def _generate_graph(self):
        """Generate a Chung–Lu expected-degree graph with a power-law degree sequence."""
        degrees = nx.utils.powerlaw_sequence(self.n, exponent=self.gamma)
        return nx.expected_degree_graph(degrees, selfloops=False)
        
    def _degree_distribution(self, G, k_max=None, eps: float = 0.0) -> np.ndarray:
        """
        Return empirical degree PMF P(k) on fixed support k=0..k_max.

        Parameters
        ----------
        G : nx.Graph
            Graph whose degree distribution is measured.
        k_max : int | None
            Maximum degree for the support. If None, uses max observed degree in G.
            For successive KL across q, pass a fixed k_max (e.g. max degree in G0).
        eps : float
            Additive epsilon smoothing applied to all bins before renormalization.

        Returns
        -------
        np.ndarray
            Probability vector of length (k_max+1) summing to 1.
        """
        degrees = np.fromiter((d for _, d in G.degree()), dtype=int)

        if degrees.size == 0:
            k_max = 0 if k_max is None else int(k_max)
            P = np.zeros(k_max + 1, dtype=float)
            P[0] = 1.0
            return P

        if k_max is None:
            k_max = int(degrees.max())
        else:
            k_max = int(k_max)

        counts = np.bincount(degrees, minlength=k_max + 1).astype(float)
        total = counts.sum()
        P = counts / total if total > 0 else np.zeros(k_max + 1, dtype=float)

        if eps and eps > 0:
            P = P + float(eps)
            P = P / P.sum()

        return P
