"""
Regenerate paper/tables/gamma_sweep_random.tex from the existing per-seed CSV
without re-running experiments.

Reads paper/data/gamma_sweep_random_long.csv and produces the 35-col summary
rows expected by export_gamma_table_random(), then writes the table.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow importing from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runner.export import export_gamma_table_random

IN_CSV = Path("paper/data/gamma_sweep_random_long.csv")
OUT_TEX = Path("paper/tables/gamma_sweep_random.tex")
N_TOTAL = 40


def _med_iqr_n(values: pd.Series) -> tuple[float, float, int]:
    """Median, IQR, and count of finite values."""
    finite = values.dropna()
    n = len(finite)
    if n == 0:
        return float("nan"), float("nan"), 0
    med = float(np.median(finite))
    q1, q3 = float(np.percentile(finite, 25)), float(np.percentile(finite, 75))
    return med, q3 - q1, n


def build_rows(df: pd.DataFrame) -> list[tuple]:
    """
    Produce one 35-element summary tuple per gamma.

    Targeted columns are filled with NaN/0 because the long CSV only contains
    random-failure data.  The export function uses only the random columns
    when rendering the random-removal table.
    """
    nan = float("nan")
    rows = []
    for gamma, grp in df[df["regime"] == "random"].groupby("gamma"):
        warn_med, warn_iqr, warn_n = _med_iqr_n(grp["q_warn"])

        delta = grp["q_collapse"] - grp["q_warn"]
        delta_finite = delta[grp["q_warn"].notna() & grp["q_collapse"].notna()]
        delta_med, delta_iqr, delta_n = _med_iqr_n(delta_finite)

        row = (
            float(gamma),
            # random warn mean/std/n (legacy mean-based fields; unused by current table)
            float(grp["q_warn"].mean()) if warn_n > 0 else nan,
            float(grp["q_warn"].std()) if warn_n > 1 else nan,
            warn_n,
            # random delta mean/std/n
            float(delta_finite.mean()) if delta_n > 0 else nan,
            float(delta_finite.std()) if delta_n > 1 else nan,
            delta_n,
            # JS and |ΔH| warn (not in long CSV — fill NaN)
            nan, nan, 0,
            nan, nan, 0,
            # targeted columns (not in long CSV — fill NaN/0)
            0, N_TOTAL, 0.0,
            nan, nan, 0,
            nan, nan, 0,
            nan, nan, 0,
            nan, nan,
            # median/IQR columns (used by the table renderer)
            warn_med, warn_iqr,
            delta_med, delta_iqr,
            nan, nan,
            nan, nan,
        )
        assert len(row) == 35, f"Expected 35 cols, got {len(row)}"
        rows.append(row)

    return rows


def main() -> None:
    if not IN_CSV.exists():
        sys.exit(f"Input not found: {IN_CSV}")

    df = pd.read_csv(IN_CSV)
    rows = build_rows(df)
    if not rows:
        sys.exit("No rows produced — check that the CSV contains regime='random' rows.")

    export_gamma_table_random(rows, n_total=N_TOTAL, out_path=str(OUT_TEX))
    print(f"Written: {OUT_TEX}")


if __name__ == "__main__":
    main()
