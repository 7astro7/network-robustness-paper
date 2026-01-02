import networkx as nx
import numpy as np


class TargetedFailureModel:
    """
    Removes the highest-degree nodes first, implementing a classical
    targeted attack on hubs (Albert–Jeong–Barabási 2000).
    """

    def targeted_failure(self, G: nx.Graph, q: float) -> nx.Graph:
        """
        Remove the top q fraction of nodes by degree.

        Parameters
        ----------
        G : networkx.Graph
            Input graph.
        q : float
            Fraction of nodes to remove.

        Returns
        -------
        networkx.Graph
            Graph after targeted hub removal.
        """
        G_copy = G.copy()
        num_remove = int(q * G_copy.number_of_nodes())

        # sort nodes by degree descending
        degrees = G_copy.degree()
        nodes_sorted = sorted(degrees, key=lambda x: x[1], reverse=True)

        # take the top q fraction
        nodes_to_remove = [node for node, deg in nodes_sorted[:num_remove]]

        G_copy.remove_nodes_from(nodes_to_remove)
        return G_copy

