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
    def successive_kl(distributions):
        """
        Compute successive KL divergences:
        D_KL(P_{i+1} || P_i)

        Parameters
        ----------
        distributions : list of dict
            Sequence of degree distributions indexed by q.

        Returns
        -------
        list of float
            Successive KL values (length = len(distributions) - 1)
        """
        kl_values = []

        for i in range(len(distributions) - 1):
            P_prev = distributions[i]
            P_next = distributions[i + 1]

            # Align supports
            all_k = set(P_prev.keys()) | set(P_next.keys())

            D = 0.0
            for k in all_k:
                p = P_next.get(k, 0.0)
                q = P_prev.get(k, 0.0)
                if p > 0 and q > 0:
                    D += p * np.log2(p / q)

            kl_values.append(D)

        return kl_values

    @staticmethod
    def detect_warning(signal, percentile=90):
        """
        Detect warning index using a simple percentile-based rule.
        """
        signal = np.asarray(signal)
        threshold = np.percentile(signal, percentile)

        for i, x in enumerate(signal):
            if x >= threshold:
                return i

        return None


    @staticmethod
    def detect_baseline_deviation(signal, baseline_frac=0.3, k=2.0):
        """
        Warn when signal departs from early baseline by k std deviations.

        Parameters
        ----------
        signal : array-like
        baseline_frac : float
            Fraction of early signal used as baseline.
        k : float
            Std deviation multiplier.

        Returns
        -------
        int or None
            Warning index.
        """
        signal = np.asarray(signal)
        n0 = max(2, int(len(signal) * baseline_frac))

        mu = signal[:n0].mean()
        sigma = signal[:n0].std()

        threshold = mu + k * sigma

        for i in range(n0, len(signal)):
            if signal[i] > threshold:
                return i

        return None

    @staticmethod
    def detect_persistence(signal, N=3):
        """
        Warn when signal increases for N consecutive steps.

        Parameters
        ----------
        signal : array-like
        N : int
            Number of consecutive increases required.

        Returns
        -------
        int or None
            Warning index.
        """
        signal = np.asarray(signal)
        count = 0

        for i in range(1, len(signal)):
            if signal[i] > signal[i - 1]:
                count += 1
                if count >= N:
                    return i
            else:
                count = 0

        return None

    @staticmethod
    def detect_slope(dKL, m=3):
        """
        Detect first index where dKL increases for m consecutive steps.
        """
        for i in range(len(dKL) - m):
            if all(dKL[i + j + 1] > dKL[i + j] for j in range(m)):
                return i + m
        return None

