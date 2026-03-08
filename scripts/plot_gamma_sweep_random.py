import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def main():
    in_csv = Path("paper/data/gamma_sweep_random_long.csv")
    out_pdf = Path("paper/figures/fig_gamma_sweep_random.pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    if not in_csv.exists():
        raise FileNotFoundError(f"missing {in_csv} (run python main.py first)")

    by_gamma = defaultdict(lambda: {"q_warn": [], "q_collapse": []})

    with in_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("regime") != "random":
                continue
            g = _to_float(row.get("gamma", "nan"))
            by_gamma[g]["q_warn"].append(_to_float(row.get("q_warn", "nan")))
            by_gamma[g]["q_collapse"].append(_to_float(row.get("q_collapse", "nan")))

    gammas = np.array(sorted(k for k in by_gamma.keys() if np.isfinite(k)), dtype=float)
    if len(gammas) == 0:
        raise ValueError("no gamma rows found in CSV")

    def mean_std(arr):
        arr = np.asarray(arr, dtype=float)
        n = int(np.count_nonzero(np.isfinite(arr)))
        if n == 0:
            return float("nan"), float("nan")
        mean = float(np.nanmean(arr))
        std = float(np.nanstd(arr, ddof=1)) if n > 1 else 0.0
        return mean, std

    q_warn_mean, q_warn_std = [], []
    q_col_mean, q_col_std = [], []
    for g in gammas:
        m, s = mean_std(by_gamma[g]["q_warn"])
        q_warn_mean.append(m)
        q_warn_std.append(s)
        m, s = mean_std(by_gamma[g]["q_collapse"])
        q_col_mean.append(m)
        q_col_std.append(s)

    q_warn_mean = np.asarray(q_warn_mean, dtype=float)
    q_warn_std = np.asarray(q_warn_std, dtype=float)
    q_col_mean = np.asarray(q_col_mean, dtype=float)
    q_col_std = np.asarray(q_col_std, dtype=float)

    fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=300)

    ax.errorbar(
        gammas,
        q_warn_mean,
        yerr=q_warn_std,
        fmt="o-",
        color="tab:orange",
        lw=1.8,
        markersize=4,
        capsize=3,
        label=r"$q_{\mathrm{warn}}$",
    )
    ax.errorbar(
        gammas,
        q_col_mean,
        yerr=q_col_std,
        fmt="s-",
        color="black",
        lw=1.8,
        markersize=4,
        capsize=3,
        label=r"$q_{\mathrm{collapse}}$",
    )

    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$q$")
    ax.grid(True, alpha=0.25)

    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=2,
        frameon=False,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()


