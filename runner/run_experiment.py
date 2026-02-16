import numpy as np
from pathlib import Path
import argparse
import sys

# Allow running this file directly (so imports like `models.*` resolve from repo root).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random
from core.graph_model import GraphModel
from core.failure_model import RandomFailure, TargetedFailure
from core.experiment import Experiment
from runner.gamma_sweep import _detect_targeted_onset, _null_baseline_mu_sigma


def make_fig1_random(
    gamma: float = 2.5,
    seed: int = 0,
    outdir: str = "paper/figures",
    alpha: float = 0.2,
    n: int = 10_000,
):
    """
    Figure 1 (random failure; single representative gamma/seed).

    x-axis: q in [0, 0.9] (100 points)
    left y-axis: S(q) = GCC fraction
    right y-axis: \\tilde{D}_{KL}(q) = EWMA-smoothed successive KL on midpoint grid q_{i+1/2}
    vertical lines: q_warn (baseline break with q<=0.15 on qs_mid) and q_collapse (first S(q)<0.1)
    saves: PNG + PDF (for LaTeX inclusion)
    """
    # Ensure deterministic graph + failure realization (networkx uses Python's random in places)
    np.random.seed(seed)
    random.seed(seed)

    graph = GraphModel(n=n, gamma=gamma)
    experiment = Experiment(graph, RandomFailure())

    qs = np.linspace(0, 0.9, 100)
    S_values, _, Pq_values = experiment.sweep(qs)

    dKL_successive = experiment.successive_kl(Pq_values)
    dKL_smooth = experiment.ewma(dKL_successive, alpha=alpha)
    qs_mid = 0.5 * (qs[:-1] + qs[1:])
    # Sanity: successive KL lives on the midpoint grid between q_i and q_{i+1}
    assert len(qs_mid) == len(dKL_smooth)
    assert np.isclose(qs_mid[0], 0.5 * (qs[0] + qs[1]))
    # With qs = linspace(0, 0.9, 100), the first midpoint is ~0.0045 (not 0.0)
    assert qs_mid[0] > 0.0

    # --- q_warn baseline rule using q <= 0.15 on qs_mid ---
    baseline_mask = qs_mid <= 0.15
    if baseline_mask.sum() >= 2:
        mu0 = float(np.mean(dKL_smooth[baseline_mask]))
        sigma0 = float(np.std(dKL_smooth[baseline_mask]))
        threshold = mu0 + 2.0 * sigma0
        idx = np.where((qs_mid > 0.15) & (dKL_smooth > threshold))[0]
        # Report q_warn on the midpoint grid (same grid as successive KL)
        q_warn = float(qs_mid[idx[0]]) if len(idx) else None
    else:
        mu0 = float("nan")
        sigma0 = float("nan")
        threshold = float("nan")
        q_warn = None

    # --- q_collapse rule using S(q) < 0.1 on qs ---
    collapse_idx = next((i for i, s in enumerate(S_values) if s < 0.1), None)
    q_collapse = float(qs[collapse_idx]) if collapse_idx is not None else None

    # --- plot (twin y-axis) ---
    fig, ax1 = plt.subplots(constrained_layout=True, figsize=(7.0, 4.0), dpi=300)

    # baseline region (make criterion visible)
    ax1.axvspan(0.0, 0.15, color="0.5", alpha=0.15, zorder=0)
    # baseline boundary (q = 0.15): subtle but explicit
    ax1.axvline(0.15, color="0.5", lw=1.0, alpha=0.35, zorder=0, label="_nolegend_")

    (line_s,) = ax1.plot(qs, S_values, color="tab:blue", lw=2.5, label=r"$S(q)$")
    ax1.set_xlabel(r"$q$")
    ax1.set_ylabel(r"$S(q)$", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xlim(float(qs.min()), float(qs.max()))
    ax1.grid(True, alpha=0.2)

    ax2 = ax1.twinx()
    # successive KL is defined between q_i and q_{i+1}, so plot on midpoint grid
    (line_dkl,) = ax2.plot(
        qs_mid,
        dKL_smooth,
        color="tab:purple",
        lw=2,
        label=r"$\tilde{D}_{\mathrm{KL}}(q)$",
    )
    # Make it visually obvious the curve starts at q_{1/2} (~0.0045), not q=0
    ax2.scatter([qs_mid[0]], [dKL_smooth[0]], s=16, color="tab:purple", zorder=4)
    ax2.set_ylabel(r"$\tilde{D}_{\mathrm{KL}}(q)$", color="tab:purple")
    ax2.tick_params(axis="y", labelcolor="tab:purple")

    if q_warn is not None:
        ax1.axvline(q_warn, color="orange", linestyle="--", lw=1.5, label="_nolegend_")
    if q_collapse is not None:
        ax1.axvline(q_collapse, color="black", linestyle=":", lw=1.5, label="_nolegend_")

    # For paper figures, keep the plot itself minimal: no internal title (use LaTeX caption),
    # and keep the legend inside the axes so tight bounding boxes don't add huge whitespace.
    ax1.legend(
        handles=[line_s, line_dkl],
        labels=[line_s.get_label(), line_dkl.get_label()],
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=8,
        handlelength=2.0,
        columnspacing=1.6,
    )

    # No in-plot q_warn/q_collapse text: keep values in LaTeX caption for a cleaner figure.

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = f"fig1_random_gamma{gamma:.1f}_seed{seed}_alpha{alpha:.2f}"
    png_path = out_path / f"{stem}.png"
    pdf_path = out_path / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    return {
        "q_warn": q_warn,
        "q_collapse": q_collapse,
        "mu0": mu0,
        "sigma0": sigma0,
        "threshold": threshold,
        "png": str(png_path),
        "pdf": str(pdf_path),
    }


def make_fig2_targeted(
    gamma: float = 2.5,
    seed: int = 0,
    outdir: str = "paper/figures",
    alpha: float = 0.2,
    n: int = 10_000,
    n_baseline: int = 3,
    z: float = 2.0,
    x_max_cap: float = 0.3,
):
    """
    Figure 2 (targeted failure; single representative gamma/seed).

    Layout mirrors Figure 1:
      - x-axis: q in [0, 0.9] (100 points)
      - left y-axis: S(q) = GCC fraction
      - right y-axis: \\tilde{D}_{KL}(q) = EWMA-smoothed successive KL on midpoint grid q_{i+1/2}
      - vertical lines: q_warn (drift rule on midpoint grid) and q_collapse (first S(q)<0.1)
      - saves: PNG + PDF

    Targeted onset rule (attack-onset detection):
      Use a short initial baseline on the midpoint grid (first `n_baseline` midpoints) and
      trigger when the midpoint-indexed smoothed successive KL exceeds mu + z*sigma.
    """
    np.random.seed(seed)
    random.seed(seed)

    graph = GraphModel(n=n, gamma=gamma)
    experiment = Experiment(graph, TargetedFailure())

    qs = np.linspace(0, 0.9, 400)
    S_values, _, Pq_values = experiment.sweep(qs)

    dKL_successive = experiment.successive_kl(Pq_values)
    dKL_smooth = experiment.ewma(dKL_successive, alpha=alpha)
    qs_mid = 0.5 * (qs[:-1] + qs[1:])
    assert len(qs_mid) == len(dKL_smooth)
    assert qs_mid[0] > 0.0

    # --- q_collapse rule using S(q) < 0.1 on qs ---
    collapse_idx = next((i for i, s in enumerate(S_values) if s < 0.1), None)
    q_collapse = float(qs[collapse_idx]) if collapse_idx is not None else None

    # --- targeted onset detection on midpoint grid ---
    mu_null, sig_null = _null_baseline_mu_sigma(
        graph,
        qs,
        alpha=alpha,
        n_baseline=n_baseline,
    )
    q_warn_tgt, mu0, sigma0, threshold = _detect_targeted_onset(
        qs_mid,
        dKL_smooth,
        n_baseline=n_baseline,
        z=z,
        mu0=mu_null,
        sigma0=sig_null,
    )

    # Temporary debug print (safe to keep; remove if you don't want stdout noise).
    print(
        f"[targeted onset] q_warn_tgt={q_warn_tgt}  mu={mu0:.3e}  sigma={sigma0:.3e}  thr={threshold:.3e}"
    )

    # --- plot (twin y-axis) ---
    fig, ax1 = plt.subplots(constrained_layout=True, figsize=(7.0, 4.0), dpi=300)

    (line_s,) = ax1.plot(qs, S_values, color="tab:blue", lw=2, label=r"$S(q)$")
    ax1.set_xlabel(r"$q$")
    ax1.set_ylabel(r"$S(q)$", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    if q_collapse is not None:
        x_max = float(min(x_max_cap, q_collapse + 0.05))
        x_max = max(x_max, float(qs[1]))  # keep > 0
    else:
        x_max = float(x_max_cap)
    ax1.set_xlim(0.0, x_max)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    (line_dkl,) = ax2.plot(
        qs_mid,
        dKL_smooth,
        color="tab:purple",
        lw=2,
        label=r"$\tilde{D}_{\mathrm{KL}}(q)$",
    )
    ax2.scatter([qs_mid[0]], [dKL_smooth[0]], s=16, color="tab:purple", zorder=4)
    ax2.set_ylabel(r"$\tilde{D}_{\mathrm{KL}}(q)$", color="tab:purple")
    ax2.tick_params(axis="y", labelcolor="tab:purple")

    if q_warn_tgt is not None:
        ax1.axvline(q_warn_tgt, color="orange", linestyle="--", lw=1.5)

    if q_collapse is not None:
        ax1.axvline(q_collapse, color="black", linestyle=":", lw=1.5)
        # Label collapse line horizontally (avoid rotated/sideways text in the exported figure).
        y0, y1 = ax1.get_ylim()
        y_span = (y1 - y0) if (y1 > y0) else 1.0
        ax1.text(
            q_collapse,
            y0 + 0.50 * y_span,
            r"$q_{\mathrm{collapse}}$",
            rotation=0,
            ha="center",
            va="center",
            color="black",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5),
        )

    # Same styling as Fig1: no internal title; legend inside axes.
    ax1.legend(
        handles=[line_s, line_dkl],
        labels=[line_s.get_label(), line_dkl.get_label()],
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=8,
        handlelength=2.0,
        columnspacing=1.6,
    )

    # No in-plot q_warn/q_collapse text: keep values in LaTeX caption.

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = f"fig2_targeted_gamma{gamma:.1f}_seed{seed}_alpha{alpha:.2f}"
    png_path = out_path / f"{stem}.png"
    pdf_path = out_path / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    return {
        "q_warn_tgt": q_warn_tgt,
        "q_collapse": q_collapse,
        "png": str(png_path),
        "pdf": str(pdf_path),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run experiments and/or export paper figures.")
    ap.add_argument(
        "--make-fig1-random",
        action="store_true",
        help="Generate the random-failure representative figure (Fig 1) into --outdir.",
    )
    ap.add_argument(
        "--make-fig2-targeted",
        action="store_true",
        help="Generate the targeted (hub-first) representative figure (Fig 2) into --outdir.",
    )
    ap.add_argument("--gamma", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--outdir", type=str, default="paper/figures")
    args = ap.parse_args()

    if args.make_fig1_random:
        res = make_fig1_random(
            gamma=float(args.gamma),
            seed=int(args.seed),
            outdir=str(args.outdir),
            alpha=float(args.alpha),
            n=int(args.n),
        )
        print(res)

    if args.make_fig2_targeted:
        res = make_fig2_targeted(
            gamma=float(args.gamma),
            seed=int(args.seed),
            outdir=str(args.outdir),
            alpha=float(args.alpha),
            n=int(args.n),
        )
        print(res)

    if not args.make_fig1_random and not args.make_fig2_targeted:
        ap.print_help()
        raise SystemExit(2)

