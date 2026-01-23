import numpy as np
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random
from models.graph_model import GraphModel
from models.failure_model import RandomFailure, TargetedFailure
from experiment.experiment import Experiment
from models.metrics import Metrics


def run_default(seed=None, gamma_override=None):
    if seed is not None:
        np.random.seed(seed)

    n = 10_000
    gamma = gamma_override if gamma_override is not None else 2.3

    # --- setup ---
    graph = GraphModel(n=n, gamma=gamma)
    failure = RandomFailure()
    experiment = Experiment(graph, failure)

    qs = np.linspace(0, 0.9, 100)

    # --- run damage sweep ---
    S_values, H_values, Pq_values = experiment.sweep(qs)
    P0 = Pq_values[0]
    DKL_values = np.array([Metrics.kl_divergence(Pq, P0) for Pq in Pq_values], dtype=float)

    # --- reference plots only ---
    experiment.plot_full_results(qs, S_values, H_values, DKL_values)

    # --- successive KL signal ---
    dKL_successive = experiment.successive_kl(Pq_values)
    dKL_successive_smooth = experiment.ewma(dKL_successive, alpha=0.2)

    # --- EARLY WARNING RULE (baseline deviation) ---
    warn_idx = Metrics.detect_baseline_deviation(dKL_successive_smooth)

    q_warn = qs[warn_idx] if warn_idx is not None else None

    # --- COLLAPSE RULE (GCC threshold) ---
    collapse_idx = next(
        (i for i, s in enumerate(S_values) if s < 0.1),
        None
    )
    q_collapse = qs[collapse_idx] if collapse_idx is not None else None

    return q_warn, q_collapse


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


def make_fig1_random_baselines(
    gamma: float = 2.5,
    seed: int = 0,
    outdir: str = "paper/figures",
    alpha: float = 0.2,
    n: int = 10_000,
):
    """
    Random failure baseline comparison (single representative gamma/seed).

    Plots:
      - left axis: S(q)
      - right axis (midpoint support): EWMA-smoothed rate signals in bits:
          * successive KL  \\tilde D_KL(q)
          * successive JS  \\tilde JS(q)
          * smoothed |ΔH|(q) where ΔH_i = H_{i+1} - H_i (entropy in bits)

    Warning rule (all three signals; random failure):
      baseline window q <= 0.15 on midpoint grid; warn is first q>0.15 where
      signal(q) > mu0 + 2 sigma0 (computed on the baseline window).

    Saves PNG + PDF.
    """
    np.random.seed(seed)
    random.seed(seed)

    graph = GraphModel(n=n, gamma=gamma)
    experiment = Experiment(graph, RandomFailure())

    qs = np.linspace(0, 0.9, 100)
    S_values, H_values, Pq_values = experiment.sweep(qs)

    qs_mid = 0.5 * (qs[:-1] + qs[1:])
    baseline_mask = qs_mid <= 0.15

    def _warn_q(signal_mid: np.ndarray) -> float | None:
        signal_mid = np.asarray(signal_mid, dtype=float)
        if baseline_mask.sum() < 2:
            return None
        mu0 = float(np.mean(signal_mid[baseline_mask]))
        sigma0 = float(np.std(signal_mid[baseline_mask]))
        thr = mu0 + 2.0 * sigma0
        idx = np.where((qs_mid > 0.15) & (signal_mid > thr))[0]
        return float(qs_mid[idx[0]]) if len(idx) else None

    # successive KL (midpoints)
    dkl = experiment.ewma(experiment.successive_kl(Pq_values), alpha=alpha)
    # successive JS (midpoints)
    js = experiment.ewma(Metrics.successive_js(Pq_values), alpha=alpha)
    # entropy change magnitude (midpoints)
    dH = np.abs(np.diff(np.asarray(H_values, dtype=float)))
    dH = experiment.ewma(dH, alpha=alpha)

    assert len(dkl) == len(qs_mid) == len(js) == len(dH)

    q_warn_dkl = _warn_q(dkl)
    q_warn_js = _warn_q(js)
    q_warn_dh = _warn_q(dH)

    q_collapse = next((float(q) for q, s in zip(qs, S_values) if s < 0.1), None)

    # --- plot ---
    fig, ax1 = plt.subplots(constrained_layout=True, figsize=(7.0, 4.0), dpi=300)
    ax1.axvspan(0.0, 0.15, color="0.5", alpha=0.12, zorder=0)
    ax1.axvline(0.15, color="0.5", lw=1.0, alpha=0.30, zorder=0, label="_nolegend_")

    (line_s,) = ax1.plot(qs, S_values, color="tab:blue", lw=2.5, label=r"$S(q)$")
    ax1.set_xlabel(r"$q$")
    ax1.set_ylabel(r"$S(q)$", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xlim(float(qs.min()), float(qs.max()))
    ax1.grid(True, alpha=0.2)

    ax2 = ax1.twinx()
    (line_dkl,) = ax2.plot(qs_mid, dkl, color="tab:purple", lw=2.0, label=r"$\tilde{D}_{\mathrm{KL}}(q)$")
    (line_js,) = ax2.plot(qs_mid, js, color="tab:green", lw=1.8, label=r"$\widetilde{\mathrm{JS}}(q)$")
    (line_dh,) = ax2.plot(qs_mid, dH, color="tab:red", lw=1.8, label=r"$\widetilde{|\Delta H|}(q)$")
    ax2.set_ylabel(r"smoothed rate signal (bits)", color="black")

    # warning markers (color-matched to signal)
    if q_warn_dkl is not None:
        ax1.axvline(q_warn_dkl, color="tab:purple", linestyle="--", lw=1.2, alpha=0.7)
    if q_warn_js is not None:
        ax1.axvline(q_warn_js, color="tab:green", linestyle="--", lw=1.2, alpha=0.7)
    if q_warn_dh is not None:
        ax1.axvline(q_warn_dh, color="tab:red", linestyle="--", lw=1.2, alpha=0.7)
    if q_collapse is not None:
        ax1.axvline(q_collapse, color="black", linestyle=":", lw=1.5, alpha=0.9)

    ax1.legend(
        handles=[line_s, line_dkl, line_js, line_dh],
        labels=[line_s.get_label(), line_dkl.get_label(), line_js.get_label(), line_dh.get_label()],
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=8,
        handlelength=2.0,
        columnspacing=1.4,
    )

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = f"fig_random_baselines_gamma{gamma:.1f}_seed{seed}_alpha{alpha:.2f}"
    png_path = out_path / f"{stem}.png"
    pdf_path = out_path / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    return {
        "q_warn_dkl": q_warn_dkl,
        "q_warn_js": q_warn_js,
        "q_warn_dh": q_warn_dh,
        "q_collapse": q_collapse,
        "png": str(png_path),
        "pdf": str(pdf_path),
    }


def make_fig2_targeted(
    gamma: float = 2.5,
    seed: int = 0,
    outdir: str = "paper/figures",
    alpha: float = 0.2,
    n: int = 10_000,
    q0: float = 0.0,
    m: int = 3,
    max_violations: int = 1,
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

    Drift rule (paper-aligned):
      Smallest midpoint q_{i+1/2} >= q0 such that
        \\tilde D(q_{i+1/2+m}) - \\tilde D(q_{i+1/2}) > 0
      while allowing up to `max_violations` local downward steps within the window.
    """
    np.random.seed(seed)
    random.seed(seed)

    graph = GraphModel(n=n, gamma=gamma)
    experiment = Experiment(graph, TargetedFailure())

    qs = np.linspace(0, 0.9, 100)
    S_values, _, Pq_values = experiment.sweep(qs)

    dKL_successive = experiment.successive_kl(Pq_values)
    dKL_smooth = experiment.ewma(dKL_successive, alpha=alpha)
    qs_mid = 0.5 * (qs[:-1] + qs[1:])
    assert len(qs_mid) == len(dKL_smooth)
    assert qs_mid[0] > 0.0

    # --- q_collapse rule using S(q) < 0.1 on qs ---
    collapse_idx = next((i for i, s in enumerate(S_values) if s < 0.1), None)
    q_collapse = float(qs[collapse_idx]) if collapse_idx is not None else None

    # --- q_warn drift rule on midpoint grid (clamped to occur before collapse) ---
    q_warn = None
    dq = float(qs[1] - qs[0])
    # If collapse exists, ensure we can still warn before it (otherwise return None)
    q_end = (q_collapse - dq) if (q_collapse is not None) else float(qs_mid.max())
    if q_end > 0:
        # Allow early-q warning search; optionally clamp q0 to be < collapse
        q0_eff = float(max(0.0, min(q0, q_end)))
        start_idx = int(np.searchsorted(qs_mid, q0_eff, side="left"))
        end_idx = int(np.searchsorted(qs_mid, q_end, side="right"))
        max_i = min(end_idx - m - 1, len(dKL_smooth) - m - 1)
        for i in range(start_idx, max_i + 1):
            window = dKL_smooth[i:i + m + 1]
            diffs = np.diff(window)
            violations = int(np.sum(diffs <= 0))
            net_increase = float(window[-1] - window[0])
            if net_increase > 0 and violations <= max_violations:
                q_warn = float(qs_mid[i])
                break

    # Enforce "no warning after collapse"
    if q_collapse is not None and q_warn is not None and q_warn >= q_collapse:
        q_warn = None

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

    if q_warn is not None:
        ax1.axvline(q_warn, color="orange", linestyle="--", lw=1.5)

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
        "q_warn": q_warn,
        "q_collapse": q_collapse,
        "png": str(png_path),
        "pdf": str(pdf_path),
    }


def run_targeted():
    """
    Run robustness experiment under targeted failures (hub removal).
    """
    n = 10_000
    gamma = 2.3

    graph = GraphModel(n=n, gamma=gamma)
    failure = TargetedFailure()
    experiment = Experiment(graph, failure)

    qs = np.linspace(0, 0.5, 20)  # targeted failure collapses earlier

    _, _, _ = experiment.sweep(qs)


def run_targeted_warning(seed=None, gamma_override=None):
    if seed is not None:
        np.random.seed(seed)

    n = 10_000
    gamma = gamma_override if gamma_override is not None else 2.3

    graph = GraphModel(n=n, gamma=gamma)
    failure = TargetedFailure()
    experiment = Experiment(graph, failure)

    qs = np.linspace(0, 0.9, 100)

    S_values, _, Pq_values = experiment.sweep(qs)

    dKL_successive = experiment.successive_kl(Pq_values)
    dKL_successive_smooth = experiment.ewma(dKL_successive, alpha=0.2)

    warn_idx = Metrics.detect_slope(dKL_successive_smooth, m=3)
    q_warn = qs[warn_idx] if warn_idx is not None else None

    collapse_idx = next(
        (i for i, s in enumerate(S_values) if s < 0.1),
        None
    )
    q_collapse = qs[collapse_idx] if collapse_idx is not None else None

    experiment.plot_successive_KL(
        qs,
        dKL_successive_smooth,
        q_warn,
        q_collapse
    )

    return q_warn, q_collapse

def summarize_warnings(run_fn, seeds):
    q_warns = []

    for seed in seeds:
        q_warn, q_collapse = run_fn(seed=seed)
        if q_warn is not None:
            q_warns.append(q_warn)

    q_warns = np.array(q_warns)

    return q_warns.mean(), q_warns.std(), len(q_warns)

def gamma_sweep_table(gammas, seeds):
    rows = []

    for gamma in gammas:
        def run_random(seed=None):
            return run_default(seed=seed, gamma_override=gamma)

        def run_targeted(seed=None):
            return run_targeted_warning(seed=seed, gamma_override=gamma)

        mean_r, std_r, n_r = summarize_warnings(run_random, seeds)
        mean_t, std_t, n_t = summarize_warnings(run_targeted, seeds)

        rows.append((gamma, mean_r, std_r, mean_t, std_t))

    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run experiments and/or export paper figures.")
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

    if args.make_fig2_targeted:
        res = make_fig2_targeted(
            gamma=float(args.gamma),
            seed=int(args.seed),
            outdir=str(args.outdir),
            alpha=float(args.alpha),
            n=int(args.n),
        )
        print(res)
        raise SystemExit(0)

    # Default: quick console summary used during development
    seeds = [0, 1, 2, 3, 4]

    mean_r, std_r, n_r = summarize_warnings(run_default, seeds)
    mean_t, std_t, n_t = summarize_warnings(run_targeted_warning, seeds)

    print("\nRandom failure:")
    print(f"q_warn = {mean_r:.3f} ± {std_r:.3f} (n={n_r})")

    print("\nTargeted failure:")
    print(f"q_warn = {mean_t:.3f} ± {std_t:.3f} (n={n_t})")

