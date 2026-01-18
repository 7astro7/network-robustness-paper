from __future__ import annotations

from pathlib import Path
import networkx as nx
import numpy as np


class EmpiricalGraphModel:
    """
    Lightweight wrapper to run the existing Experiment pipeline on an empirical graph.

    Mirrors the interface used by Experiment:
      - self.G : nx.Graph
      - self._degree_distribution(G, k_max=None, eps=0.0) -> np.ndarray
    """

    def __init__(self, G: nx.Graph):
        if not isinstance(G, nx.Graph):
            raise TypeError("EmpiricalGraphModel expects an undirected networkx.Graph")
        self.G = G

    @staticmethod
    def from_edge_list(path: str | Path) -> "EmpiricalGraphModel":
        """
        Load an undirected graph from a whitespace-separated edge list file (u v per line).
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        G = nx.read_edgelist(p, nodetype=int, data=False, create_using=nx.Graph())
        return EmpiricalGraphModel(G)

    def _degree_distribution(self, G, k_max=None, eps: float = 0.0) -> np.ndarray:
        """
        Return empirical degree PMF P(k) on fixed support k=0..k_max.

        Copied from GraphModel so synthetic/empirical experiments behave identically.
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


