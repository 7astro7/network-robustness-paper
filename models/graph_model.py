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

    def _degree_distribution(self, G):
        """Return empirical P(k) for the graph."""
        degrees = [d for _, d in G.degree()]
        values, counts = np.unique(degrees, return_counts=True)
        P = counts / counts.sum()
        return dict(zip(values, P))

