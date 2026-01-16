import numpy as np
import matplotlib
from models.graph_model import GraphModel
from models.metrics import Metrics
# Try to use an interactive backend for display
try:
    matplotlib.use('TkAgg')
except ImportError:
    try:
        matplotlib.use('Qt5Agg')
    except ImportError:
        pass  # Use default backend
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
        S_values, H_values = [], []
        Pq_values = []

        # Fixed degree support across the entire sweep (based on the initial graph)
        k_max0 = max(dict(self.graph_model.G.degree()).values())
        eps = 1e-12

        for q in qs:
            Gq = self.failure_model.apply(self.graph_model.G, q)

            S = Metrics.giant_component_fraction(Gq)
            H = Metrics.degree_entropy(Gq)

            S_values.append(S)
            H_values.append(H)

            # Fixed support + epsilon smoothing (requires updated GraphModel._degree_distribution)
            Pq = self.graph_model._degree_distribution(Gq, k_max=k_max0, eps=eps)
            Pq_values.append(Pq)

        return S_values, H_values, Pq_values


    def successive_kl(self, Pq_values: list) -> np.ndarray:
        """
        Compute D_KL(P_{q+1} || P_q) for successive damage levels.
        """
        dKL = []

        for i in range(len(Pq_values) - 1):
            d = Metrics.kl_divergence(Pq_values[i + 1], Pq_values[i])
            dKL.append(d)

        return np.array(dKL)

    def ewma(self, signal: np.ndarray, alpha: float = 0.2) -> np.ndarray:
        """
        Exponentially weighted moving average for 1D signals.
        """
        out = np.empty_like(signal)
        out[0] = signal[0]
        for i in range(1, len(signal)):
            out[i] = alpha * signal[i] + (1 - alpha) * out[i-1]
        return out

    def plot_successive_KL(
        self,
        qs: np.ndarray,
        dKL: np.ndarray, 
        q_warn, 
        q_collapse
    ) -> tuple[float, float]:
        """
        Plot successive KL divergence D_KL(P_{q+Δq} || P_q)
        and annotate the peak (maximum structural change rate).
        """
        qs_mid = 0.5 * (qs[:-1] + qs[1:])

        idx = np.argmax(dKL)
        qc = qs_mid[idx]
        dKL_qc = dKL[idx]

        plt.figure(figsize=(6, 4))
        plt.plot(qs_mid, dKL, marker='o', color='purple',
                label=r"$D_{KL}(P_{q+\Delta q}\|P_q)$")

        plt.axvline(qc, color='red', linestyle='--',
                    alpha=0.7, label=f"q* = {qc:.3f}")

        plt.scatter([qc], [dKL_qc], color='red', zorder=5)

        plt.text(
            qc, dKL_qc,
            f"  q*={qc:.3f}\n  ΔKL={dKL_qc:.3f}",
            fontsize=10,
            verticalalignment='top',
            color='red'
        )

        if q_warn is not None:
            plt.axvline(q_warn, color="orange", linestyle="--", label="warning")

        if q_collapse is not None:
            plt.axvline(q_collapse, color="black", linestyle=":", label="collapse")


        plt.xlabel("Fraction Removed q")
        plt.ylabel("Successive KL [bits]")
        plt.title("Structural Change Rate via Successive KL Divergence")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        return qc, dKL_qc


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

    def compute_derivative(
        self, 
        qs: np.ndarray, 
        Y_values: np.ndarray
    ) -> np.ndarray:
        """
        Compute numerical derivative dY/dq using finite differences.

        Returns
        -------
        np.ndarray
            Array of dY/dq values.
        """
        return np.gradient(Y_values, qs)

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

    def plot_KL_derivative(self, qs: np.ndarray, dDKL: np.ndarray) -> tuple[float, float]:
        """
        Plot dDKL/dq and annotate the critical point q* = argmin(dDKL/dq).
        """
        # detect critical point
        idx = np.argmin(dDKL)   # index of most negative derivative
        qc = qs[idx]
        dDKL_qc = dDKL[idx]
        
        plt.figure(figsize=(6, 4))
        plt.plot(qs, dDKL, marker='o', color='blue', label="dDKL/dq")
        
        # vertical line at q*
        plt.axvline(qc, color='red', linestyle='--', alpha=0.7, label=f"q* = {qc:.3f}")
        
        # highlight the point
        plt.scatter([qc], [dDKL_qc], color='red', zorder=5)
        
        # annotate the point with coordinates
        plt.text(
            qc, dDKL_qc,
            f"  q*={qc:.3f}\n  dDKL/dq={dDKL_qc:.2f}",
            fontsize=10,
            verticalalignment='top',
            color='red'
        )
        
        plt.xlabel("Fraction Removed q")
        plt.ylabel("dDKL/dq")
        plt.title("Derivative of KL Divergence")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        return qc, dDKL_qc

    def plot_S_derivative(self, qs: np.ndarray, dS: np.ndarray) -> tuple[float, float]:
        """
        Plot dS/dq and annotate the critical point q* = argmin(dS/dq).
        """
        # detect critical point
        idx = np.argmin(dS)   # index of most negative derivative
        qc = qs[idx]
        dS_qc = dS[idx]
        
        plt.figure(figsize=(6, 4))
        plt.plot(qs, dS, marker='o', color='green', label="dS/dq")
        
        # vertical line at q*
        plt.axvline(qc, color='red', linestyle='--', alpha=0.7, label=f"q* = {qc:.3f}")
        
        # highlight the point
        plt.scatter([qc], [dS_qc], color='red', zorder=5)
        
        # annotate the point with coordinates
        plt.text(
            qc, dS_qc,
            f"  q*={qc:.3f}\n  dS/dq={dS_qc:.2f}",
            fontsize=10,
            verticalalignment='top',
            color='red'
        )
        
        plt.xlabel("Fraction Removed q")
        plt.ylabel("dS/dq")
        plt.title("Derivative of Giant Component Fraction")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        return qc, dS_qc

    def plot_full_results(
        self, 
        qs: np.ndarray, 
        S_values: np.ndarray, 
        H_values: np.ndarray, 
        DKL_values: np.ndarray
    ) -> None:
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

    def plot_successive_KL_overlay(self, qs, seeds, alpha=0.2):
        """
        Overlay successive-KL curves across multiple seeds.
        """
        plt.figure(figsize=(6, 4))
        qs_mid = 0.5 * (qs[:-1] + qs[1:])

        for seed in seeds:
            np.random.seed(seed)

            graph = GraphModel(n=self.graph_model.n, gamma=self.graph_model.gamma)
            failure = self.failure_model.__class__()
            experiment = Experiment(graph, failure)

            _, _, _, Pq_values = experiment.sweep(qs)
            dKL = Metrics.successive_kl(Pq_values)
            dKL_smooth = experiment.ewma(dKL, alpha=alpha)

            plt.plot(qs_mid, dKL_smooth, alpha=0.6, label=f"seed {seed}")

        plt.xlabel("q")
        plt.ylabel("Successive KL (EWMA)")
        plt.title("Successive KL across seeds")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
