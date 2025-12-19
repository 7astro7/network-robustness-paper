import numpy as np
import matplotlib.pyplot as plt
from models.metrics import Metrics


class Experiment:
    """
    Runs robustness experiments across a sweep of failure probabilities q.
    """

    def __init__(self, graph_model, failure_model):
        self.graph_model = graph_model
        self.failure_model = failure_model

    def sweep(self, qs):
        """
        Compute S(q) and H(q) for a sequence of q values.

        Returns
        -------
        tuple of lists
            S_values : list of giant component fractions
            H_values : list of entropies (in bits)
            DKL_values : list of KL divergences (in bits)
        """
        S_values, H_values, DKL_values = [], [], []

        for q in qs:
            if hasattr(self.failure_model, "random_failure"):
                Gq = self.failure_model.random_failure(self.graph_model.G, q)
            elif hasattr(self.failure_model, "targeted_failure"):
                Gq = self.failure_model.targeted_failure(self.graph_model.G, q)
            else:
                raise ValueError("Failure model must implement random_failure or targeted_failure.")

            S = Metrics.giant_component_fraction(Gq)
            H = Metrics.degree_entropy(Gq)

            S_values.append(S)
            H_values.append(H)

            Pq = self.graph_model._degree_distribution(Gq)
            DKL = Metrics.kl_divergence(Pq, self.graph_model.P0)
            DKL_values.append(DKL)

        return S_values, H_values, DKL_values

    def plot_results(self, qs, S_values, H_values):
        """Plot S(q) and H(q) side-by-side."""
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        # S(q)
        ax[0].plot(qs, S_values, marker='o')
        ax[0].set_title("Giant Component S(q)")
        ax[0].set_xlabel("Fraction Removed q")
        ax[0].set_ylabel("S(q)")
        ax[0].grid(True)

        # H(q)
        ax[1].plot(qs, H_values, marker='o', color='orange')
        ax[1].set_title("Degree Entropy H(q) (bits)")
        ax[1].set_xlabel("Fraction Removed q")
        ax[1].set_ylabel("Entropy (bits)")
        ax[1].grid(True)

        plt.tight_layout()
        plt.show()

    def compute_derivative(self, qs, H_values):
        """
        Compute numerical derivative dH/dq using finite differences.

        Returns
        -------
        np.ndarray
            Array of dH/dq values.
        """
        return np.gradient(H_values, qs)

    def plot_entropy_derivative(self, qs, dH):
        """
        Plot dH/dq and annotate the critical point q* = argmin(dH/dq).
        """
        # detect critical point
        idx = np.argmin(dH)   # index of most negative derivative
        qc = qs[idx]
        dH_qc = dH[idx]

        plt.figure(figsize=(6, 4))
        plt.plot(qs, dH, marker='o', color='green', label="dH/dq")

        # vertical line at q*
        plt.axvline(qc, color='red', linestyle='--', alpha=0.7, label=f"q* = {qc:.3f}")

        # highlight the point
        plt.scatter([qc], [dH_qc], color='red', zorder=5)

        # annotate the point with coordinates
        plt.text(
            qc, dH_qc,
            f"  q*={qc:.3f}\n  dH/dq={dH_qc:.2f}",
            fontsize=10,
            verticalalignment='top',
            color='red'
        )

        plt.xlabel("Fraction Removed q")
        plt.ylabel("dH/dq")
        plt.title("Derivative of Degree Entropy")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        return qc, dH_qc


    def plot_full_results(self, qs, S_values, H_values, DKL_values):
        """Plot S(q), H(q), and D_KL(q) together."""
        fig, ax = plt.subplots(1, 3, figsize=(16, 4))

        # S(q)
        ax[0].plot(qs, S_values, marker='o')
        ax[0].set_title("Giant Component S(q)")
        ax[0].set_xlabel("q")
        ax[0].set_ylabel("S(q)")
        ax[0].grid(True)

        # H(q)
        ax[1].plot(qs, H_values, marker='o', color='orange')
        ax[1].set_title("Degree Entropy H(q)")
        ax[1].set_xlabel("q")
        ax[1].set_ylabel("H(q) [bits]")
        ax[1].grid(True)

        # KL divergence
        ax[2].plot(qs, DKL_values, marker='o', color='red')
        ax[2].set_title("KL Divergence D_KL(Pq || P0)")
        ax[2].set_xlabel("q")
        ax[2].set_ylabel("D_KL [bits]")
        ax[2].grid(True)

        plt.tight_layout()
        plt.show()

