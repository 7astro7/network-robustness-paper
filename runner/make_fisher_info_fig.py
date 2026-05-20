"""
Fisher information I(q) of the residual degree distribution under binomial thinning.

Computes I(q) = sum_k [dP_tilde(k;q)/dq]^2 / P_tilde(k;q) numerically via central
finite differences on P_tilde(k;q) for gamma in {2.1, 2.5, 2.9}.

Plots I(q) vs q on a single panel with distinct line styles per gamma, a shaded
baseline window, and a vertical dotted line at q_c for gamma=2.9 (the only case
where q_c falls within the plotted range).

Output: paper/figures/fig_fisher_info.pdf + .png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom


_KMAX = 1000
_N_Q = 400
_Q_MAX = 0.90
_EPS = 1e-12
_Q0 = 0.15
_GAMMAS = [2.1, 2.5, 2.9]


class FisherInfoExperiment:
    """
    Computes I(q) = sum_k [dP_tilde(k;q)/dq]^2 / P_tilde(k;q) for a discrete
    power-law degree distribution under binomial thinning, using central finite
    differences for the derivative.
    """

    def __init__(
        self,
        gamma: float,
        kmax: int = _KMAX,
        n_q: int = _N_Q,
        q_max: float = _Q_MAX,
        eps: float = _EPS,
    ) -> None:
        self.gamma = gamma
        self.kmax = kmax
        self.n_q = n_q
        self.q_max = q_max
        self.eps = eps

        self.ks = np.arange(0, kmax + 1, dtype=int)
        # Start at dq rather than 0 to avoid the q=0 boundary where P_tilde has
        # a degenerate form (all mass on k=0); matches the analytical_kl.py convention.
        dq = q_max / n_q
        self.qs = np.linspace(dq, q_max, n_q)

        self._P0 = self._compute_P0()
        self._qc = self._compute_qc()

    def _compute_P0(self) -> np.ndarray:
        """Discrete power-law P0(k) = C * k^{-gamma}, k in [1, kmax], normalised."""
        P0 = np.zeros(self.kmax + 1, dtype=float)
        P0[1:] = self.ks[1:].astype(float) ** (-self.gamma)
        P0 /= P0.sum()
        return P0

    def _compute_qc(self) -> float:
        """Molloy-Reed percolation threshold qc = 1 - <k> / (<k^2> - <k>)."""
        k1 = float(np.dot(self.ks, self._P0))
        k2 = float(np.dot(self.ks.astype(float) ** 2, self._P0))
        denom = k2 - k1
        return 1.0 - k1 / denom if denom > 0 else float("nan")

    def _compute_P_tilde(self) -> np.ndarray:
        """
        P_tilde(k'; q) for all (q, k') pairs via binomial thinning (Eq. 1).

        P_tilde(k'; q) = sum_{k >= k'} P0(k) C(k,k') (1-q)^{k'} q^{k-k'}

        Returns shape (n_q, kmax+1).
        """
        P_tilde = np.zeros((self.n_q, self.kmax + 1), dtype=float)
        p_vals = 1.0 - self.qs

        for k_prime in range(self.kmax + 1):
            k_start = max(k_prime, 1)
            if k_start > self.kmax:
                break
            k_range = np.arange(k_start, self.kmax + 1)
            B = binom.pmf(
                k_prime,
                n=k_range[np.newaxis, :],
                p=p_vals[:, np.newaxis],
            )
            P_tilde[:, k_prime] = B @ self._P0[k_start:]

        return P_tilde

    def run(self) -> dict:
        """
        Compute I(q) via central finite differences and return results dict.

        Returns
        -------
        dict with keys: gamma, qc, qs, fisher_info
          qs          : q values at interior grid points (central diff domain)
          fisher_info : I(q) at those points
        """
        P_tilde = self._compute_P_tilde()
        dq = float(self.qs[1] - self.qs[0])

        # Central differences at interior indices 1 .. n_q-2
        dP_dq = (P_tilde[2:] - P_tilde[:-2]) / (2.0 * dq)   # (n_q-2, kmax+1)
        P_mid = P_tilde[1:-1]                                  # (n_q-2, kmax+1)

        fisher_info = np.sum(dP_dq ** 2 / (P_mid + self.eps), axis=1)  # (n_q-2,)
        qs_inner = self.qs[1:-1]

        return {
            "gamma": self.gamma,
            "qc": self._qc,
            "qs": qs_inner,
            "fisher_info": fisher_info,
        }


class FisherInfoFigure:
    """
    Single-panel figure of I(q) vs q for multiple gamma values.
    """

    LINE_STYLES: list[str] = ["-", "--", "-."]
    LINE_WIDTHS: list[float] = [2.5, 1.8, 1.2]
    # All black: line style + weight carry the distinction in grayscale print.
    # Red (#d62728) and green (#2ca02c) share near-identical luminance (~0.21)
    # and become indistinguishable when converted to grey.
    COLORS: list[str] = ["black", "black", "black"]

    def __init__(
        self,
        results: list[dict],
        outdir: str = "paper/figures",
        q0: float = _Q0,
        q_max: float = _Q_MAX,
    ) -> None:
        self.results = results
        self.outdir = Path(outdir)
        self.q0 = q0
        self.q_max = q_max

    def plot(self) -> None:
        fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True, dpi=150)

        ax.axvspan(0.0, self.q0, color="0.85", alpha=1.0, zorder=0,
                   label=rf"calibration window ($q \leq {self.q0}$)")
        ax.axvline(self.q0, color="0.5", lw=0.8, ls="-", alpha=0.5, zorder=1)

        for r, ls, color, lw in zip(self.results, self.LINE_STYLES, self.COLORS, self.LINE_WIDTHS):
            ax.plot(r["qs"], r["fisher_info"], ls=ls, color=color, lw=lw,
                    label=rf"$\gamma={r['gamma']}$", zorder=3)

        # q_c line only for gamma values where it falls within the plotted range
        for r in self.results:
            qc = r["qc"]
            if np.isfinite(qc) and qc < self.q_max:
                ax.axvline(qc, color="grey", lw=1.2, ls=":", zorder=4,
                           label=rf"$q_c={qc:.3f}$ ($\gamma={r['gamma']}$)")

        ax.set_xlabel(r"Removal fraction $q$", fontsize=11)
        ax.set_ylabel(r"$\mathcal{I}(q)$ (log scale)", fontsize=11)
        ax.set_yscale("log")
        ax.set_xlim(0.0, self.q_max)
        ax.grid(True, alpha=0.2, which="both")
        ax.legend(fontsize=8, loc="upper right")

        self.outdir.mkdir(parents=True, exist_ok=True)
        out_pdf = self.outdir / "fig_fisher_info.pdf"
        out_png = self.outdir / "fig_fisher_info.png"
        fig.savefig(out_pdf, bbox_inches="tight")
        fig.savefig(out_png, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved: {out_pdf}")
        print(f"Saved: {out_png}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fisher information I(q) figure.")
    parser.add_argument("--kmax",  type=int,   default=_KMAX)
    parser.add_argument("--n_q",   type=int,   default=_N_Q)
    parser.add_argument("--outdir", type=str,  default="paper/figures")
    args = parser.parse_args()

    results = []
    for gamma in _GAMMAS:
        print(f"  gamma={gamma:.1f} ...", end=" ", flush=True)
        exp = FisherInfoExperiment(gamma=gamma, kmax=args.kmax, n_q=args.n_q)
        r = exp.run()
        results.append(r)
        print(f"qc={r['qc']:.4f}  I_max={r['fisher_info'].max():.4f}  "
              f"I_at_q0={r['fisher_info'][r['qs'] <= 0.15][-1]:.4f}")

    fig = FisherInfoFigure(results, outdir=args.outdir)
    fig.plot()
