# runner/export.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def export_gamma_table(rows, out_path: str = "paper/gamma_sweep.tex") -> None:
    """
    Export the gamma sweep table to LaTeX.

    Expected row formats:
      - old: (gamma, mr, sr, mt, st)
      - new: (gamma, mr, sr, nr, mt, st, nt)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("export_gamma_table: rows is empty")

    k = len(rows[0])
    if k == 5:
        cols = ["gamma", "random_mean", "random_std", "targeted_mean", "targeted_std"]
        df = pd.DataFrame(rows, columns=cols)
        df["random_n"] = np.nan
        df["targeted_n"] = np.nan
    elif k == 7:
        cols = ["gamma", "random_mean", "random_std", "random_n",
                "targeted_mean", "targeted_std", "targeted_n"]
        df = pd.DataFrame(rows, columns=cols)
    else:
        raise ValueError(f"export_gamma_table: unsupported row length {k} (expected 5 or 7)")

    def fmt_cell(mean, std, n):
        if pd.isna(mean) or pd.isna(std):
            # if your detector returns NaN, show a dash
            return r"--"
        if pd.isna(n):
            return f"{mean:.3f} $\\pm$ {std:.3f}"
        return f"{mean:.3f} $\\pm$ {std:.3f} [{int(n)}]"

    df["random_cell"] = [
        fmt_cell(m, s, n) for m, s, n in zip(df["random_mean"], df["random_std"], df["random_n"])
    ]
    df["targeted_cell"] = [
        fmt_cell(m, s, n) for m, s, n in zip(df["targeted_mean"], df["targeted_std"], df["targeted_n"])
    ]

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Mean and standard deviation of detected warning points $q_{\mathrm{warn}}$ across $\gamma$ under random and targeted removal.}")
    lines.append(r"\label{tab:gamma_sweep}")
    lines.append(r"\begin{tabular}{c c c}")
    lines.append(r"\toprule")
    lines.append(r"$\gamma$ & Random $q_{\mathrm{warn}}$ (mean $\pm$ std) [n] & Targeted $q_{\mathrm{warn}}$ (mean $\pm$ std) [n] \\")
    lines.append(r"\midrule")

    for _, r in df.iterrows():
        lines.append(f"{r['gamma']:.1f} & {r['random_cell']} & {r['targeted_cell']} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    out_path.write_text("\n".join(lines))
