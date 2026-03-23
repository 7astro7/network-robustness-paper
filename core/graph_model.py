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

    def __init__(self, n: int, gamma: float) -> None:
        self.n = n
        self.gamma = gamma
        self.G = self._generate_graph()
        self.P0 = self._degree_distribution(self.G)

    def _generate_graph(self) -> nx.Graph:
        """Generate a Chung–Lu expected-degree graph with a power-law degree sequence."""
        degrees = nx.utils.powerlaw_sequence(self.n, exponent=self.gamma)
        return nx.expected_degree_graph(degrees, selfloops=False)
        
    def _degree_distribution(self, G: nx.Graph, k_max: int | None = None, eps: float = 0.0) -> np.ndarray:
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


class ConfigurationModel(GraphModel):
    """
    Configuration-model network with power-law degree sequence.
    
    Generates a degree sequence from a power-law distribution, then constructs
    a simple graph via the configuration model (random stub pairing). Self-loops
    and multi-edges are removed, and the largest connected component is taken.
    
    This provides a degree-sequence-based alternative to the Chung-Lu model,
    using the same gamma parameter for consistency.
    
    Parameters
    ----------
    n : int
        Target number of nodes.
    gamma : float
        Power-law exponent for degree distribution.
    k_min : int, optional
        Minimum degree (default: 1).
    k_max_frac : float, optional
        Maximum degree as a fraction of n (default: 0.1).
    """
    
    def __init__(self, n: int, gamma: float, k_min: int = 1, k_max_frac: float = 0.1) -> None:
        self.k_min = k_min
        self.k_max_frac = k_max_frac
        super().__init__(n, gamma)
    
    def _generate_graph(self) -> nx.Graph:
        """Generate a configuration-model graph from a power-law degree sequence."""
        # Generate power-law degree sequence with bounds
        k_max = max(self.k_min + 1, int(self.n * self.k_max_frac))
        
        # Sample degrees from power-law with truncation
        degrees = []
        for _ in range(self.n):
            # Use inverse transform sampling for power-law
            u = np.random.uniform(0, 1)
            k = int((self.k_min ** (1 - self.gamma) - u * (self.k_min ** (1 - self.gamma) - k_max ** (1 - self.gamma))) ** (1 / (1 - self.gamma)))
            k = max(self.k_min, min(k, k_max))
            degrees.append(k)
        
        # Ensure sum is even (required for configuration model)
        degree_sum = sum(degrees)
        if degree_sum % 2 == 1:
            # Add 1 to a random degree that's below k_max
            candidates = [i for i, d in enumerate(degrees) if d < k_max]
            if candidates:
                degrees[np.random.choice(candidates)] += 1
            else:
                # If all at k_max, decrement one
                degrees[np.random.randint(0, len(degrees))] -= 1
        
        # Create configuration model graph
        G_multi = nx.configuration_model(degrees, create_using=nx.MultiGraph())
        
        # Convert to simple graph (removes self-loops and parallel edges)
        G = nx.Graph(G_multi)
        G.remove_edges_from(nx.selfloop_edges(G))
        
        # Take largest connected component for stability
        if G.number_of_nodes() > 0:
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
        
        return G
