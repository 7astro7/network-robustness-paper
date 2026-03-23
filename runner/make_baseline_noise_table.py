"""
runner/make_baseline_noise_table.py

Read paper/data/baseline_noise_chunglu.csv and paper/data/baseline_noise_config.csv,
compute per-gamma means of mu0/sigma0, and write paper/tables/baseline_noise_comparison.tex.
Also prints the global mechanistic check stat to stdout.

Run as:
    python -m runner.make_baseline_noise_table
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CL_CSV = Path("paper/data/baseline_noise_chunglu.csv")
CM_CSV = Path("paper/data/baseline_noise_config.csv")
OUT_TEX = Path("paper/tables/baseline_noise_comparison.tex")


def main() -> None:
    for p in (CL_CSV, CM_CSV):
        if not p.exists():
            raise FileNotFoundError(f"Missing input CSV: {p}")

    cl = pd.read_csv(CL_CSV)
    cm = pd.read_csv(CM_CSV)

    # Mechanistic check across all CM runs
    cm_failed = cm[~cm["detected"].astype(bool)]
    n_failed = len(cm_failed)
    n_thresh_exceeds = int(cm_failed["threshold_exceeds_max_post"].astype(bool).sum())
    print(
        f"Mechanistic check: {n_thresh_exceeds}/{n_failed} failed CM runs "
        f"had threshold > max post-baseline signal "
        f"(out of {len(cm)} total CM runs)"
    )

    # Per-gamma aggregates
    cl_agg = (
        cl.groupby("gamma")[["mu0", "sigma0"]]
        .mean()
        .rename(columns={"mu0": "cl_mu0", "sigma0": "cl_sigma0"})
    )
    cm_agg = (
        cm.groupby("gamma")
        .apply(lambda g: pd.Series({
            "cm_mu0": g["mu0"].mean(),
            "cm_sigma0": g["sigma0"].mean(),
            "cm_n_det": int(g["detected"].astype(bool).sum()),
            "cm_n_total": len(g),
        }))
    )
    df = cl_agg.join(cm_agg, how="inner").reset_index()
    df = df.sort_values("gamma").reset_index(drop=True)

    # Sanity check: CM sigma0 > CL sigma0 at every gamma
    violations = df[df["cm_sigma0"] <= df["cl_sigma0"]]
    if not violations.empty:
        print(
            f"WARNING: CM sigma0 <= CL sigma0 at gamma(s): "
            f"{violations['gamma'].tolist()}"
        )

    # Write LaTeX table
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(
        r"\caption{Baseline-window noise floor comparison between Chung--Lu (CL) and "
        r"configuration-model (CM) ensembles across the $\gamma$ sweep. "
        r"$\bar{\mu}_0$ and $\bar{\sigma}_0$ are means across 40 seeds of the baseline-window "
        r"mean and standard deviation of the EWMA-smoothed successive KL signal "
        r"($q \le 0.15$, $\alpha=0.20$). "
        r"CM $\bar{\sigma}_0$ is systematically elevated at every $\gamma$, "
        r"consistent with structural churn during the baseline window inflating "
        r"the estimated noise floor and detection threshold. "
        r"$n_{\mathrm{det}}$: number of seeds (out of 40) with a detection prior to collapse.}"
    )
    lines.append(r"\label{tab:baseline_noise_comparison}")
    lines.append(r"\begin{tabular}{c c c c c c}")
    lines.append(r"\toprule")
    lines.append(
        r"$\gamma$ & CL $\bar{\mu}_0$ & CL $\bar{\sigma}_0$ & "
        r"CM $\bar{\mu}_0$ & CM $\bar{\sigma}_0$ & CM $n_{\mathrm{det}}/40$ \\"
    )
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        n_det = int(row["cm_n_det"])
        n_tot = int(row["cm_n_total"])
        lines.append(
            f"{float(row['gamma']):.1f} & "
            f"{float(row['cl_mu0']):.4f} & {float(row['cl_sigma0']):.4f} & "
            f"{float(row['cm_mu0']):.4f} & {float(row['cm_sigma0']):.4f} & "
            f"{n_det}/{n_tot} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    OUT_TEX.write_text("\n".join(lines))
    print(f"Written: {OUT_TEX}")


if __name__ == "__main__":
    main()
