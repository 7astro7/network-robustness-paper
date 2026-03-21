import numpy as np
import networkx as nx
from abc import ABC, abstractmethod


class FailureModel(ABC):
    """
    Abstract base class for node-removal mechanisms.

    A failure model defines a mapping
        (G, q) -> G_q
    where a fraction q of nodes is removed from G.
    """

    @abstractmethod
    def apply(self, G: nx.Graph, q: float) -> nx.Graph:
        """
        Apply node removal to graph G.

        Parameters
        ----------
        G : nx.Graph
            Original (undamaged) graph.
        q : float
            Fraction of nodes to remove, 0 <= q <= 1.

        Returns
        -------
        nx.Graph
            Damaged graph G_q.
        """
        pass


class RandomFailure(FailureModel):
    """
    Uniform random node removal.
    """

    def apply(self, G: nx.Graph, q: float) -> nx.Graph:
        n = G.number_of_nodes()
        n_remove = int(q * n)

        if n_remove == 0:
            # Return a shallow copy to avoid mutation
            return G.copy()

        removed = set(
            np.random.choice(list(G.nodes()), size=n_remove, replace=False)
        )

        # Induce subgraph on remaining nodes (much faster than copy+remove)
        remaining = set(G.nodes()) - removed
        return G.subgraph(remaining).copy()


class TargetedFailure(FailureModel):
    """
    Degree-based (hub-first) node removal.
    """

    def __init__(self) -> None:
        self._sorted_nodes: list | None = None
        self._sorted_for: int | None = None  # id(G) of the cached graph

    def apply(self, G: nx.Graph, q: float) -> nx.Graph:
        n = G.number_of_nodes()
        n_remove = int(q * n)

        if n_remove == 0:
            return G.copy()

        # Sort nodes by degree (descending) — cache per graph identity
        if self._sorted_for != id(G):
            self._sorted_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)
            self._sorted_for = id(G)

        removed = {node for node, _ in self._sorted_nodes[:n_remove]}
        remaining = set(G.nodes()) - removed

        return G.subgraph(remaining).copy()
