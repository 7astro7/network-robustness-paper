import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import random
import sys

import numpy as np

# Allow running this file directly (ensure repo root is on sys.path)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.graph_model import GraphModel
from core.failure_model import RandomFailure
from core.metrics import Metrics
from core.experiment import Experiment


@dataclass
class ControlRow:
    seed: int
    metric: str
    triggered: int
    q_warn: float
    s_min: float


def detect_baseline_break_midpoint(
    qs: np.ndarray,
    signal_mid: np.ndarray,
    q0: float = 0.15,
    z: float = 2.0,
) -> float:
    """
    Baseline-deviation rule on midpoint support:
      baseline window: q_mid <= q0
      warn: first q_mid > q0 where signal > mu0 + z*sigma0
    Returns np.nan if no detection.
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


def run_low_damage_control(
    gamma: float,
    n: int,
    seeds: list[int],
    alpha: float,
    outdir: str,
    q_max: float,
) -> tuple[Path, Path]:
    """
    Damage-without-collapse control (random failure, but q capped at q_max):
      - run the same pipeline (Pq, H, successive KL/JS/|ΔH| on midpoints, EWMA alpha)
      - apply same baseline-deviation rule (baseline window q<=0.15 on midpoint support)
      - do NOT compute collapse; instead report S_min to confirm no collapse
    """
    qs = np.linspace(0.0, float(q_max), 100)
    qs_mid = 0.5 * (qs[:-1] + qs[1:])

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "tables").mkdir(parents=True, exist_ok=True)

    csv_path = out_path / "tables" / f"control_low_damage_gamma{gamma:.1f}_alpha{alpha:.2f}.csv"
    tex_path = out_path / "tables" / f"control_low_damage_gamma{gamma:.1f}_alpha{alpha:.2f}.tex"

    rows: list[ControlRow] = []

    for seed in seeds:
        np.random.seed(seed)
        random.seed(seed)

        graph = GraphModel(n=n, gamma=gamma)
        exp = Experiment(graph, RandomFailure())

        S_values, H_values, Pq_values = exp.sweep(qs)
        s_min = float(np.min(np.asarray(S_values, dtype=float))) if len(S_values) else float("nan")

        # Midpoint signals (rate-of-change), EWMA-smoothed
        dkl_mid = exp.ewma(exp.successive_kl(Pq_values), alpha=alpha)
        js_mid = exp.ewma(Metrics.successive_js(Pq_values), alpha=alpha)
        dh_mid = exp.ewma(np.abs(np.diff(np.asarray(H_values, dtype=float))), alpha=alpha)

        assert len(dkl_mid) == len(js_mid) == len(dh_mid) == len(qs_mid)

        # Important: baseline window is q_mid <= 0.15; triggers are only evaluated for q_mid > 0.15
        qw_dkl = detect_baseline_break_midpoint(qs, dkl_mid, q0=0.15, z=2.0)
        qw_js = detect_baseline_break_midpoint(qs, js_mid, q0=0.15, z=2.0)
        qw_dh = detect_baseline_break_midpoint(qs, dh_mid, q0=0.15, z=2.0)

        for metric, q_warn in [
            ("successive_KL", qw_dkl),
            ("successive_JS", qw_js),
            ("abs_delta_entropy", qw_dh),
        ]:
            trig = int(np.isfinite(q_warn))
            rows.append(
                ControlRow(
                    seed=seed,
                    metric=metric,
                    triggered=trig,
                    q_warn=float(q_warn) if np.isfinite(q_warn) else float("nan"),
                    s_min=s_min,
                )
            )

    # --- write CSV (per-seed transparency) ---
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "metric", "triggered", "q_warn_or_nan", "s_min"])
        for r in rows:
            w.writerow([
                r.seed,
                r.metric,
                r.triggered,
                f"{r.q_warn:.6f}" if np.isfinite(r.q_warn) else "nan",
                f"{r.s_min:.6f}" if np.isfinite(r.s_min) else "nan",
            ])

    # --- aggregate + write LaTeX table ---
    def agg(metric: str) -> tuple[int, int, float]:
        rs = [r for r in rows if r.metric == metric]
        n_total = len(rs)
        n_trig = sum(r.triggered for r in rs)
        mean_q = float(np.nanmean([r.q_warn for r in rs])) if n_trig > 0 else float("nan")
        return n_trig, n_total, mean_q

    trig_kl, n_total, mean_q_kl = agg("successive_KL")
    trig_js, _, mean_q_js = agg("successive_JS")
    trig_dh, _, mean_q_dh = agg("abs_delta_entropy")

    # Same S_min across metrics for a given seed; report mean across seeds.
    s_min_mean = float(np.mean([r.s_min for r in rows if r.metric == "successive_KL"]))

    n_tex = f"{n:,}".replace(",", "{,}")

    def fmt_mean_q(x: float) -> str:
        return "--" if not np.isfinite(x) else f"{x:.3f}"

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{Damage-without-collapse control (random failure, $q\le {q_max:.2f}$): trigger rates under the baseline-deviation rule (baseline window $q\le 0.15$) with EWMA $\alpha={alpha:.2f}$. Metrics are computed from adjacent $q$-levels (reported on the midpoint grid) with $N={n_tex}$ and $\gamma={gamma:.1f}$ over {len(seeds)} seeds. Collapse is not evaluated; instead we report the mean minimum GCC fraction $\min_{{q\le {q_max:.2f}}} S(q)$ over the run.}}"
    )
    lines.append(r"\label{tab:control_low_damage}")
    lines.append(r"\begin{tabular}{l c c c}")
    lines.append(r"\toprule")
    lines.append(r"metric & trigger rate ($n_{\mathrm{trig}}/n$) & mean $q_{\mathrm{warn}}$ (if triggered) & mean $\min_q S(q)$ \\")
    lines.append(r"\midrule")
    lines.append(rf"successive KL ($\tilde{{D}}_{{\mathrm{{KL}}}}$) & {trig_kl}/{n_total} & {fmt_mean_q(mean_q_kl)} & {s_min_mean:.3f} \\")
    lines.append(rf"successive JS ($\widetilde{{\mathrm{{JS}}}}$) & {trig_js}/{n_total} & {fmt_mean_q(mean_q_js)} & {s_min_mean:.3f} \\")
    lines.append(rf"entropy change ($\widetilde{{|\Delta H|}}$) & {trig_dh}/{n_total} & {fmt_mean_q(mean_q_dh)} & {s_min_mean:.3f} \\")
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
    ap.add_argument("--q-max", type=float, default=0.25)
    args = ap.parse_args()

    csv_path, tex_path = run_low_damage_control(
        gamma=args.gamma,
        n=args.n,
        seeds=list(args.seeds),
        alpha=args.alpha,
        outdir=args.outdir,
        q_max=args.q_max,
    )
    print({"csv": str(csv_path), "tex": str(tex_path)})


if __name__ == "__main__":
    main()


