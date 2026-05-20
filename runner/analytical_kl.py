"""
Analytical KL divergence from binomial thinning of a power-law degree distribution.

Implements the theoretical residual degree distribution P̃(k'; q) from Eq. 1 in
the paper and applies the same baseline-deviation detection rule used in the main
pipeline. Serves as a population-level (noise-free) reference: P̃ is derived from
the exact power-law P0, averaged over all network realizations and removal sequences.

Outputs
-------
paper/figures/fig_analytical_kl.pdf   (3-panel: gamma = 2.1, 2.5, 2.9)
paper/figures/fig_analytical_kl.png
Summary table printed to stdout.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom


# ---------------------------------------------------------------------------
# Module-level defaults (match the main pipeline)
# ---------------------------------------------------------------------------
_KMAX = 1000
_N_Q = 100
_Q_MAX = 0.9
_ALPHA = 0.20
_EPS = 1e-12
_Q0 = 0.15
_Z = 2.0
_GAMMAS_ALL = [2.1, 2.3, 2.5, 2.7, 2.9]
_GAMMAS_PLOT = [2.1, 2.5, 2.9]


class AnalyticalKLExperiment:
    """
    Population-level successive KL divergence via binomial thinning of a
    discrete power-law degree distribution.

    Parameters
    ----------
    gamma   : power-law exponent (P0(k) ~ k^{-gamma}, kmin=1)
    kmax    : truncation of the degree distribution (default 1000)
    n_q     : number of q values on the damage grid
    q_max   : maximum removal fraction
    alpha   : EWMA smoothing parameter
    eps     : additive regularisation for KL computation
    q0      : baseline window upper boundary (q_mid <= q0)
    z       : detection threshold multiplier (mu0 + z*sigma0)
    """

    def __init__(
        self,
        gamma: float,
        kmax: int = _KMAX,
        n_q: int = _N_Q,
        q_max: float = _Q_MAX,
        alpha: float = _ALPHA,
        eps: float = _EPS,
        q0: float = _Q0,
        z: float = _Z,
    ) -> None:
        self.gamma = gamma
        self.kmax = kmax
        self.n_q = n_q
        self.q_max = q_max
        self.alpha = alpha
        self.eps = eps
        self.q0 = q0
        self.z = z

        self.ks = np.arange(0, kmax + 1, dtype=int)
        # Start at Δq rather than 0: at q=0, P̃(0;q)=0 exactly, so the
        # q=0→Δq step compares against eps-regularised zeros and produces a
        # spurious spike that dominates the baseline window.  Starting at Δq
        # keeps P̃(0;q)>0 for all grid points, matching the finite-graph
        # behaviour of the empirical pipeline.
        dq = q_max / n_q
        self.qs = np.linspace(dq, q_max, n_q)
        self.qs_mid = 0.5 * (self.qs[:-1] + self.qs[1:])

        self._P0 = self._compute_P0()
        self._qc = self._compute_qc()

    # ------------------------------------------------------------------
    # Distribution and threshold
    # ------------------------------------------------------------------

    def _compute_P0(self) -> np.ndarray:
        """Discrete power-law P0(k) = C * k^{-gamma}, k in [1, kmax], normalised."""
        P0 = np.zeros(self.kmax + 1, dtype=float)
        ks_pos = self.ks[1:].astype(float)
        P0[1:] = ks_pos ** (-self.gamma)
        P0 /= P0.sum()
        return P0

    def _compute_qc(self) -> float:
        """
        Molloy-Reed percolation threshold:
            qc = 1 - <k> / (<k^2> - <k>)
        computed from the moments of P0.
        """
        k1 = float(np.dot(self.ks, self._P0))
        k2 = float(np.dot(self.ks.astype(float) ** 2, self._P0))
        denom = k2 - k1
        return 1.0 - k1 / denom if denom > 0 else float("nan")

    def _compute_all_P_tilde(self) -> np.ndarray:
        """
        Compute P̃(k'; q) for all (q, k') pairs at once.

        P̃(k'; q) = sum_{k >= k'} P0(k) * C(k, k') * (1-q)^k' * q^{k-k'}

        Vectorises over q for each k', reducing Python-level calls to O(kmax).

        Returns
        -------
        P_tilde : np.ndarray, shape (n_q, kmax+1)
        """
        P_tilde = np.zeros((self.n_q, self.kmax + 1), dtype=float)
        p_vals = 1.0 - self.qs  # survival probabilities, shape (n_q,)

        for k_prime in range(self.kmax + 1):
            # P0[0] = 0, so effective k starts from max(k_prime, 1)
            k_start = max(k_prime, 1)
            if k_start > self.kmax:
                break
            k_range = np.arange(k_start, self.kmax + 1)
            # binom.pmf(k_prime, n=k_range, p=p_vals) — shape (n_q, kmax+1-k_start)
            B = binom.pmf(
                k_prime,
                n=k_range[np.newaxis, :],
                p=p_vals[:, np.newaxis],
            )
            # Dot with P0[k_start..kmax] — shape (n_q,)
            P_tilde[:, k_prime] = B @ self._P0[k_start:]

        return P_tilde

    # ------------------------------------------------------------------
    # Signal processing
    # ------------------------------------------------------------------

    @staticmethod
    def _ewma(x: np.ndarray, alpha: float) -> np.ndarray:
        """Exponentially weighted moving average (matches core/experiment.py)."""
        out = np.empty_like(x)
        out[0] = x[0]
        for i in range(1, len(x)):
            out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
        return out

    def _successive_kl(self, P_tilde: np.ndarray) -> np.ndarray:
        """
        Successive KL divergence D_KL(P̃_{i+1} || P̃_i) in bits.
        P_tilde has shape (n_q, kmax+1); returns shape (n_q-1,).
        """
        raw = np.empty(self.n_q - 1, dtype=float)
        for i in range(self.n_q - 1):
            P_new = P_tilde[i + 1] + self.eps
            P_old = P_tilde[i] + self.eps
            raw[i] = float(np.sum(P_new * np.log2(P_new / P_old)))
        return raw

    def _detect(
        self, signal: np.ndarray
    ) -> tuple[float, float, float, float]:
        """
        Baseline-deviation rule on midpoint grid.

        Returns (q_warn, mu0, sigma0, threshold); q_warn is nan if not detected.
        """
        baseline_mask = self.qs_mid <= self.q0
        baseline_vals = signal[baseline_mask]
        if baseline_mask.sum() < 2:
            return float("nan"), float("nan"), float("nan"), float("nan")
        mu0 = float(np.mean(baseline_vals))
        sigma0 = float(np.std(baseline_vals))
        threshold = mu0 + self.z * sigma0
        idx = np.where((self.qs_mid > self.q0) & (signal > threshold))[0]
        q_warn = float(self.qs_mid[idx[0]]) if idx.size else float("nan")
        return q_warn, mu0, sigma0, threshold

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Full pipeline: compute P̃, successive KL, EWMA, detection.

        Returns a results dict with all quantities needed for the figure
        and summary table.
        """
        P_tilde = self._compute_all_P_tilde()          # (n_q, kmax+1)
        raw_kl = self._successive_kl(P_tilde)          # (n_q-1,)
        dkl_smooth = self._ewma(raw_kl, self.alpha)    # (n_q-1,)
        q_warn, mu0, sigma0, threshold = self._detect(dkl_smooth)

        return {
            "gamma": self.gamma,
            "qc": self._qc,
            "qs_mid": self.qs_mid,
            "dkl_smooth": dkl_smooth,
            "q_warn": q_warn,
            "detected": bool(np.isfinite(q_warn)),
            "mu0": mu0,
            "sigma0": sigma0,
            "threshold": threshold,
        }


class AnalyticalKLFigure:
    """
    Three-panel figure of the analytical D̃_KL(q) for gamma in {2.1, 2.5, 2.9}.
    Style matches existing paper figures.
    """

    PANEL_GAMMAS: list[float] = [2.1, 2.5, 2.9]

    def __init__(
        self,
        results: dict[float, dict],
        outdir: str = "paper/figures",
        q0: float = _Q0,
        q_max: float = _Q_MAX,
    ) -> None:
        self.results = results
        self.outdir = Path(outdir)
        self.q0 = q0
        self.q_max = q_max

    def plot(self) -> None:
        fig, axes = plt.subplots(
            1, 3,
            figsize=(13.5, 4.2),
            constrained_layout=True,
            dpi=150,
        )

        for ax, gamma in zip(axes, self.PANEL_GAMMAS):
            r = self.results[gamma]
            qs_mid = r["qs_mid"]
            dkl = r["dkl_smooth"]
            q_warn = r["q_warn"]
            qc = r["qc"]
            threshold = r["threshold"]

            # Baseline calibration window (shaded)
            ax.axvspan(0.0, self.q0, color="0.85", alpha=1.0, zorder=0,
                       label=rf"calibration window ($q \leq {self.q0}$)")
            ax.axvline(self.q0, color="0.5", lw=0.8, ls="-", alpha=0.5, zorder=1)

            # Smoothed analytical signal
            ax.plot(qs_mid, dkl, color="black", lw=1.5, zorder=3,
                    label=r"$\tilde{D}_{\mathrm{KL}}(q)$ (theoretical)")

            # Detection threshold
            ax.axhline(threshold, color="#1f77b4", lw=1.0, ls="--", zorder=2,
                       label=rf"$\mu_0+2\sigma_0={threshold:.4f}$")

            # q_warn
            if np.isfinite(q_warn):
                ax.axvline(q_warn, color="#d62728", lw=1.5, ls="--", zorder=4,
                           label=rf"$q_{{\mathrm{{warn}}}}={q_warn:.3f}$")

            # qc (Molloy-Reed)
            if np.isfinite(qc):
                if qc < self.q_max:
                    ax.axvline(qc, color="#ff7f0e", lw=1.5, ls=":", zorder=4,
                               label=rf"$q_c={qc:.3f}$")
                else:
                    ax.plot([], [], color="grey", lw=1.5, ls=":",
                            label=rf"$q_c={qc:.3f}$ (outside range)")

            ax.set_xlabel(r"Removal fraction $q$")
            ax.set_ylabel(r"$\tilde{D}_{\mathrm{KL}}(q)$ (bits)")
            ax.set_title(rf"$\gamma = {gamma}$", fontsize=10)
            ax.set_xlim(0.0, self.q_max)
            ax.set_ylim(bottom=0.0)
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=7.5, loc="upper left")

        self.outdir.mkdir(parents=True, exist_ok=True)
        out_pdf = self.outdir / "fig_analytical_kl.pdf"
        out_png = self.outdir / "fig_analytical_kl.png"
        fig.savefig(out_pdf, bbox_inches="tight")
        fig.savefig(out_png, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved: {out_pdf}")
        print(f"Saved: {out_png}")


def print_summary(results: dict[float, dict]) -> None:
    """Print the summary table to stdout."""
    print()
    print(f"{'gamma':>6}  {'qc':>8}  {'q_warn (analytical)':>20}  {'detected':>10}")
    print("-" * 54)
    for gamma in sorted(results):
        r = results[gamma]
        qc_str = f"{r['qc']:.4f}" if np.isfinite(r["qc"]) else "     n/a"
        qw_str = f"{r['q_warn']:.4f}" if r["detected"] else "             n/a"
        det_str = "yes" if r["detected"] else "no"
        print(f"{gamma:>6.1f}  {qc_str:>8}  {qw_str:>20}  {det_str:>10}")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analytical KL divergence from binomial thinning."
    )
    parser.add_argument("--kmax",   type=int,   default=_KMAX)
    parser.add_argument("--alpha",  type=float, default=_ALPHA)
    parser.add_argument("--outdir", type=str,   default="paper/figures")
    args = parser.parse_args()

    print("Running analytical KL experiment (kmax={})...".format(args.kmax))

    results: dict[float, dict] = {}
    for gamma in _GAMMAS_ALL:
        print(f"  gamma={gamma:.1f} ...", end=" ", flush=True)
        exp = AnalyticalKLExperiment(
            gamma=gamma,
            kmax=args.kmax,
            alpha=args.alpha,
        )
        r = exp.run()
        results[gamma] = r
        det_str = "yes" if r["detected"] else "no"
        qw_str = f"{r['q_warn']:.4f}" if r["detected"] else "n/a"
        print(f"qc={r['qc']:.4f}  q_warn={qw_str}  detected={det_str}")

    print_summary(results)

    fig = AnalyticalKLFigure(results, outdir=args.outdir)
    fig.plot()
