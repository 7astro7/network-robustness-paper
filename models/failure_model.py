import numpy as np


class FailureModel:
    """Implements node-removal processes (failures)."""

    def random_failure(self, G, q):
        """
        Apply uniform random node removal.

        Parameters
        ----------
        G : networkx.Graph
            The graph to damage.
        q : float
            Fraction of nodes removed.

        Returns
        -------
        networkx.Graph
            Damaged graph.
        """
        G2 = G.copy()
        n_remove = int(q * G.number_of_nodes())
        to_remove = np.random.choice(G.nodes(), size=n_remove, replace=False)
        G2.remove_nodes_from(to_remove)
        return G2

    def targeted_failure(self, G, q):
        """
        Remove the top q fraction of nodes ranked by degree (highest first).

        Parameters
        ----------
        q : float
            Fraction of nodes to remove.

        Returns
        -------
        networkx.Graph
            Damaged graph after targeted hub removal.
        """
        G2 = G.copy()
        n_remove = int(q * G.number_of_nodes())

        # sort nodes by degree descending
        degrees = dict(G.degree())
        sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)

        to_remove = sorted_nodes[:n_remove]
        G2.remove_nodes_from(to_remove)

        return G2
