"""
runner/make_degree_matched_table.py

Read paper/data/degree_matched_config_random_failure.csv and write a LaTeX summary
table comparing detection rates across gamma values for the degree-matched CM check.

Run as:
    python -m runner.make_degree_matched_table
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

IN_CSV = Path("paper/data/degree_matched_config_random_failure.csv")
OUT_TEX = Path("paper/tables/degree_matched_config_random_failure.tex")


def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Missing input: {IN_CSV}")

    df = pd.read_csv(IN_CSV)
    df["detected"] = df["detected"].astype(bool)

    agg = (
        df.groupby("gamma")
        .apply(lambda g: pd.Series({
            "n_detected": int(g["detected"].sum()),
            "n_total": len(g),
            "delta_med": float(np.nanmedian(g.loc[g["detected"], "delta_warn"])) if g["detected"].any() else float("nan"),
            "delta_iqr": float(
                np.nanpercentile(g.loc[g["detected"], "delta_warn"], 75) -
                np.nanpercentile(g.loc[g["detected"], "delta_warn"], 25)
            ) if g["detected"].sum() >= 2 else float("nan"),
        }))
        .reset_index()
        .sort_values("gamma")
    )

    lines: list[str] = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(
        r"\caption{Degree-matched configuration-model diagnostic: detection rates under random failure. "
        r"For each $\gamma$, a Chung--Lu graph is generated per source seed and its realized degree "
        r"sequence is passed to the configuration model; multiple CM wiring replicates are constructed "
        r"from the same degree sequence. "
        r"$n_{\mathrm{det}}$: runs with a valid $q_{\mathrm{warn}}$ prior to collapse. "
        r"$\tilde{\Delta}_{\mathrm{warn}}$ [IQR]: median [interquartile range] lead time "
        r"($q_{\mathrm{collapse}} - q_{\mathrm{warn}}$) for detected runs. "
        r"All settings ($\alpha=0.20$, $z=2$, baseline $q\le 0.15$) are identical to the Chung--Lu analysis. "
        r"$\gamma=3.0$ is included as an out-of-range test point; it is absent from all primary results tables.}"
    )
    lines.append(r"\label{tab:degree_matched_config}")
    lines.append(r"\begin{tabular}{c c c}")
    lines.append(r"\toprule")
    lines.append(r"$\gamma$ & $n_{\mathrm{det}} / n_{\mathrm{total}}$ & $\tilde{\Delta}_{\mathrm{warn}}$ [IQR] \\")
    lines.append(r"\midrule")

    for _, row in agg.iterrows():
        n_det = int(row["n_detected"])
        n_tot = int(row["n_total"])
        if np.isfinite(row["delta_med"]):
            delta_str = f"{row['delta_med']:.3f} [{row['delta_iqr']:.3f}]"
        else:
            delta_str = r"\textemdash"
        lines.append(
            f"{float(row['gamma']):.1f} & {n_det}/{n_tot} & {delta_str} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(lines))
    print(f"Written: {OUT_TEX}")


if __name__ == "__main__":
    main()
