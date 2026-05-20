import argparse
from dataclasses import dataclass
from pathlib import Path
import csv
import random
import sys

import numpy as np

# Allow running this file directly (ensure repo root is on sys.path)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.graph_model import GraphModel
from core.metrics import Metrics
from core.experiment import Experiment

_ROLLING_W = 7


@dataclass
class NullControlResult:
    seed: int
    metric: str
    triggered: int
    q_warn: float


def detect_baseline_break_midpoint(
    qs: np.ndarray,
    signal_mid: np.ndarray,
    q0: float = 0.15,
    z: float = 2.0,
) -> float:
    """
    Baseline-deviation rule on midpoint support (random-failure rule, reused for null control).

    - baseline window is defined on midpoints: q_mid <= q0
    - warn at the earliest midpoint q_mid > q0 where signal > mu0 + z*sigma0
    - returns np.nan if no detection
    """
    qs = np.asarray(qs, dtype=float)
    signal_mid = np.asarray(signal_mid, dtype=float)
    qs_mid = 0.5 * (qs[:-1] + qs[1:])
    if len(signal_mid) != len(qs_mid):
        raise ValueError(f"signal length {len(signal_mid)} must equal len(qs_mid) {len(qs_mid)}")

    baseline_mask = qs_mid <= q0
    if baseline_mask.sum() < 2:
        return float("nan")

    mu0 = float(np.mean(signal_mid[baseline_mask]))
    sigma0 = float(np.std(signal_mid[baseline_mask]))
    thr = mu0 + z * sigma0

    idx = np.where((qs_mid > q0) & (signal_mid > thr))[0]
    if len(idx) == 0:
        return float("nan")
    return float(qs_mid[idx[0]])


def _rolling_std_signal(signal: np.ndarray, w: int = _ROLLING_W, min_periods: int = 3) -> np.ndarray:
    n = len(signal)
    out = np.full(n, np.nan)
    for i in range(n):
        start = max(0, i - w + 1)
        window = signal[start : i + 1]
        if len(window) >= min_periods and np.all(np.isfinite(window)):
            out[i] = float(np.std(window, ddof=1)) if len(window) > 1 else 0.0
    return out


def _rolling_ac1_signal(signal: np.ndarray, w: int = _ROLLING_W, min_periods: int = 5) -> np.ndarray:
    n = len(signal)
    out = np.full(n, np.nan)
    for i in range(n):
        start = max(0, i - w + 1)
        window = signal[start : i + 1]
        if len(window) >= min_periods and np.all(np.isfinite(window)):
            x = window[:-1]
            y = window[1:]
            sx, sy = float(np.std(x)), float(np.std(y))
            if sx > 1e-15 and sy > 1e-15:
                out[i] = float(np.corrcoef(x, y)[0, 1])
    return out


def _detect_nanfree(qs: np.ndarray, signal: np.ndarray, q0: float = 0.15, z: float = 2.0) -> float:
    """Baseline-deviation detection that skips NaN values in the baseline window."""
    qs = np.asarray(qs, dtype=float)
    signal = np.asarray(signal, dtype=float)
    qs_mid = 0.5 * (qs[:-1] + qs[1:])
    if signal.size != qs_mid.size:
        raise ValueError("signal length mismatch")
    baseline_vals = signal[qs_mid <= q0]
    baseline_vals = baseline_vals[np.isfinite(baseline_vals)]
    if len(baseline_vals) < 5:
        return np.nan
    threshold = float(np.mean(baseline_vals)) + z * float(np.std(baseline_vals))
    exceed = np.isfinite(signal) & (signal > threshold) & (qs_mid > q0)
    idx = np.where(exceed)[0]
    return float(qs_mid[idx[0]]) if idx.size else np.nan


def run_null_control(
    gamma: float,
    n: int,
    seeds: list[int],
    alpha: float,
    outdir: str,
) -> tuple[Path, Path]:
    """
    Null control under 'random failure' pipeline:
      - generate one Chung–Lu graph per seed
      - for each q_i, set G_q = G0 (no removals) but still recompute P_q(k), H(q)
      - compute midpoint signals: successive KL, successive JS, |ΔH|
      - EWMA smooth with alpha
      - apply baseline-deviation rule (q<=0.15 window) and report false triggers
    """
    qs = np.linspace(0.0, 0.9, 100)
    qs_mid = 0.5 * (qs[:-1] + qs[1:])

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / f"null_control_random_failure_gamma{gamma:.1f}_alpha{alpha:.2f}.csv"
    tex_path = out_path / f"null_control_random_failure_gamma{gamma:.1f}_alpha{alpha:.2f}.tex"

    rows: list[NullControlResult] = []

    for seed in seeds:
        np.random.seed(seed)
        random.seed(seed)

        graph = GraphModel(n=n, gamma=gamma)
        exp = Experiment(graph, failure_model=None)  # unused; we only use ewma()

        G0 = graph.G
        k_max0 = max(dict(G0.degree()).values()) if G0.number_of_nodes() else 0
        eps = 1e-12

        Pq_values: list[np.ndarray] = []
        H_values: list[float] = []

        # No-op "damage": always evaluate on the unchanged G0
        for _q in qs:
            Pq_values.append(graph._degree_distribution(G0, k_max=k_max0, eps=eps))
            H_values.append(Metrics.degree_entropy(G0))

        # Midpoint signals
        raw_dkl = np.asarray([Metrics.kl_divergence(Pq_values[i + 1], Pq_values[i]) for i in range(len(Pq_values) - 1)])
        dkl_mid = exp.ewma(raw_dkl, alpha=alpha)
        js_mid = exp.ewma(Metrics.successive_js(Pq_values), alpha=alpha)
        dh_mid = exp.ewma(np.abs(np.diff(np.asarray(H_values, dtype=float))), alpha=alpha)

        assert len(dkl_mid) == len(js_mid) == len(dh_mid) == len(qs_mid)

        qw_dkl = detect_baseline_break_midpoint(qs, dkl_mid)
        qw_js = detect_baseline_break_midpoint(qs, js_mid)
        qw_dh = detect_baseline_break_midpoint(qs, dh_mid)
        qw_rv = _detect_nanfree(qs, _rolling_std_signal(raw_dkl))
        qw_ac1 = _detect_nanfree(qs, _rolling_ac1_signal(raw_dkl))

        for metric, q_warn in [
            ("successive_KL", qw_dkl),
            ("successive_JS", qw_js),
            ("abs_delta_entropy", qw_dh),
            ("rolling_sd_KL", qw_rv),
            ("rolling_ac1_KL", qw_ac1),
        ]:
            triggered = int(np.isfinite(q_warn))
            rows.append(NullControlResult(seed=seed, metric=metric, triggered=triggered, q_warn=float(q_warn) if np.isfinite(q_warn) else float("nan")))

    # --- write CSV ---
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "metric", "triggered", "q_warn_or_nan"])
        for r in rows:
            w.writerow([r.seed, r.metric, r.triggered, f"{r.q_warn:.6f}" if np.isfinite(r.q_warn) else "nan"])

    # --- aggregate + write tiny LaTeX table ---
    def agg(metric: str) -> tuple[int, int, float]:
        rs = [r for r in rows if r.metric == metric]
        n_total = len(rs)
        n_trig = sum(r.triggered for r in rs)
        rate = (n_trig / n_total) if n_total else float("nan")
        return n_trig, n_total, rate

    trig_kl,  n_total, _ = agg("successive_KL")
    trig_js,  _,       _ = agg("successive_JS")
    trig_dh,  _,       _ = agg("abs_delta_entropy")
    trig_rv,  _,       _ = agg("rolling_sd_KL")
    trig_ac1, _,       _ = agg("rolling_ac1_KL")

    n_tex = f"{n:,}".replace(",", "{,}")

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{Null control (no removals): false-trigger rates under the random-failure baseline-deviation rule (baseline window $q\le 0.15$, EWMA $\alpha={alpha:.2f}$, rolling window $w={_ROLLING_W}$). Each metric is computed from adjacent $q$-levels (reported on the midpoint grid) with $N={n_tex}$ and $\gamma={gamma:.1f}$ over {len(seeds)} seeds.}}"
    )
    lines.append(r"\label{tab:null_control_random_failure}")
    lines.append(r"\begin{tabular}{l c}")
    lines.append(r"\toprule")
    lines.append(r"metric & false-trigger rate ($n_{\mathrm{trig}}/n$) \\")
    lines.append(r"\midrule")
    lines.append(rf"successive KL ($\tilde{{D}}_{{\mathrm{{KL}}}}$) & {trig_kl}/{n_total} \\")
    lines.append(rf"successive JS ($\widetilde{{\mathrm{{JS}}}}$) & {trig_js}/{n_total} \\")
    lines.append(rf"entropy change ($|\Delta H|$) & {trig_dh}/{n_total} \\")
    lines.append(rf"rolling $\mathrm{{sd}}(D_{{\mathrm{{KL}}}})$ & {trig_rv}/{n_total} \\")
    lines.append(rf"rolling $\mathrm{{AC}}(1)(D_{{\mathrm{{KL}}}})$ & {trig_ac1}/{n_total} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    tex_path.write_text("\n".join(lines))

    return csv_path, tex_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=2.5)
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(40)))
    ap.add_argument("--outdir", type=str, default="paper")
    args = ap.parse_args()

    csv_path, tex_path = run_null_control(
        gamma=args.gamma,
        n=args.n,
        seeds=list(args.seeds),
        alpha=args.alpha,
        outdir=args.outdir,
    )
    print({"csv": str(csv_path), "tex": str(tex_path)})


if __name__ == "__main__":
    main()


