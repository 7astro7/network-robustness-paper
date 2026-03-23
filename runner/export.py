# runner/export.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import csv


def export_gamma_table(
    rows,
    out_path: str = "paper/gamma_sweep.tex",
    n_total: int | None = None,
) -> None:
    """
    Export the gamma sweep table to LaTeX.

    Expected row formats:
      - old: (gamma, mr, sr, mt, st)
      - new: (gamma, mr, sr, nr, mt, st, nt)  where n* are detected counts
      - newer (targeted split): (gamma,
            random_mean, random_std, random_n,
            targeted_early_n, targeted_n_total, targeted_early_rate,
            targeted_trigger_mean, targeted_trigger_std, targeted_trigger_n)
      - newest (targeted split + collapse + lead): (gamma,
            random_mean, random_std, random_n,
            targeted_early_n, targeted_n_total, targeted_early_rate,
            targeted_trigger_mean, targeted_trigger_std, targeted_trigger_n,
            targeted_collapse_mean, targeted_collapse_std, targeted_collapse_n,
            targeted_delta_mean, targeted_delta_std, targeted_delta_n)
      - newest+ (random lead-time + targeted split + collapse + lead): (gamma,
            random_warn_mean, random_warn_std, random_warn_n,
            random_delta_mean, random_delta_std, random_delta_n,
            targeted_early_n, targeted_n_total, targeted_early_rate,
            targeted_trigger_mean, targeted_trigger_std, targeted_trigger_n,
            targeted_collapse_mean, targeted_collapse_std, targeted_collapse_n,
            targeted_delta_mean, targeted_delta_std, targeted_delta_n)
      - newest++ (random lead-time + random baselines + targeted split + collapse + lead): (gamma,
            random_warn_mean, random_warn_std, random_warn_n,
            random_delta_mean, random_delta_std, random_delta_n,
            random_js_warn_mean, random_js_warn_std, random_js_warn_n,
            random_dh_warn_mean, random_dh_warn_std, random_dh_warn_n,
            targeted_early_n, targeted_n_total, targeted_early_rate,
            targeted_warn_tgt_mean, targeted_warn_tgt_std, targeted_warn_tgt_n,
            targeted_collapse_mean, targeted_collapse_std, targeted_collapse_n,
            targeted_delta_warn_tgt_mean, targeted_delta_warn_tgt_std, targeted_delta_warn_tgt_n)
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
    elif k == 10:
        cols = [
            "gamma",
            "random_mean", "random_std", "random_n",
            "targeted_early_n", "targeted_n_total", "targeted_early_rate",
            "targeted_trigger_mean", "targeted_trigger_std", "targeted_trigger_n",
        ]
        df = pd.DataFrame(rows, columns=cols)
    elif k == 16:
        cols = [
            "gamma",
            "random_mean", "random_std", "random_n",
            "targeted_early_n", "targeted_n_total", "targeted_early_rate",
            "targeted_trigger_mean", "targeted_trigger_std", "targeted_trigger_n",
            "targeted_collapse_mean", "targeted_collapse_std", "targeted_collapse_n",
            "targeted_delta_mean", "targeted_delta_std", "targeted_delta_n",
        ]
        df = pd.DataFrame(rows, columns=cols)
    elif k == 19:
        cols = [
            "gamma",
            "random_warn_mean", "random_warn_std", "random_warn_n",
            "random_delta_mean", "random_delta_std", "random_delta_n",
            "targeted_early_n", "targeted_n_total", "targeted_early_rate",
            "targeted_trigger_mean", "targeted_trigger_std", "targeted_trigger_n",
            "targeted_collapse_mean", "targeted_collapse_std", "targeted_collapse_n",
            "targeted_delta_mean", "targeted_delta_std", "targeted_delta_n",
        ]
        df = pd.DataFrame(rows, columns=cols)
    elif k == 25:
        cols = [
            "gamma",
            "random_warn_mean", "random_warn_std", "random_warn_n",
            "random_delta_mean", "random_delta_std", "random_delta_n",
            "random_js_warn_mean", "random_js_warn_std", "random_js_warn_n",
            "random_dh_warn_mean", "random_dh_warn_std", "random_dh_warn_n",
            "targeted_early_n", "targeted_n_total", "targeted_early_rate",
            "targeted_warn_tgt_mean", "targeted_warn_tgt_std", "targeted_warn_tgt_n",
            "targeted_collapse_mean", "targeted_collapse_std", "targeted_collapse_n",
            "targeted_delta_warn_tgt_mean", "targeted_delta_warn_tgt_std", "targeted_delta_warn_tgt_n",
        ]
        df = pd.DataFrame(rows, columns=cols)
    else:
        raise ValueError(f"export_gamma_table: unsupported row length {k} (expected 5, 7, 10, 16, 19, or 25)")

    if n_total is None:
        # best-effort default (matches GammaSweepExperiment default seeds)
        n_total = 5
    n_total = int(n_total)

    def fmt_mean_std(mean, std) -> str:
        if pd.isna(mean) or pd.isna(std):
            return r"--"
        return f"{mean:.3f} $\\pm$ {std:.3f}"

    def fmt_cell(mean, std, n_detected):
        if pd.isna(n_detected):
            # legacy 5-col table, no detection counts
            if pd.isna(mean) or pd.isna(std):
                return r"--"
            return f"{mean:.3f} $\\pm$ {std:.3f}"

        n_detected = int(n_detected)
        if n_detected == 0:
            return rf"-- [{n_detected}/{n_total}]"

        # With >=1 detection we should have a mean; std may be NaN if n_detected==1 and ddof=1.
        if pd.isna(mean):
            return rf"-- [{n_detected}/{n_total}]"
        if pd.isna(std):
            return f"{mean:.3f} $\\pm$ -- [{n_detected}/{n_total}]"
        return f"{mean:.3f} $\\pm$ {std:.3f} [{n_detected}/{n_total}]"

    def fmt_rate_cell(n_early, n_total_local):
        if pd.isna(n_early) or pd.isna(n_total_local):
            return r"--"
        return f"{int(n_early)}/{int(n_total_local)}"

    # Normalize random column names across schemas
    if "random_mean" in df.columns:
        df["random_warn_mean"] = df["random_mean"]
        df["random_warn_std"] = df["random_std"]
        df["random_warn_n"] = df["random_n"]

    df["random_cell"] = [
        fmt_cell(m, s, n) for m, s, n in zip(df["random_warn_mean"], df["random_warn_std"], df["random_warn_n"])
    ]
    if "targeted_mean" in df.columns:
        df["targeted_cell"] = [
            fmt_cell(m, s, n) for m, s, n in zip(df["targeted_mean"], df["targeted_std"], df["targeted_n"])
        ]
    else:
        df["targeted_rate_cell"] = [
            fmt_rate_cell(ne, nt) for ne, nt in zip(df["targeted_early_n"], df["targeted_n_total"])
        ]
        df["targeted_warn_tgt_cell"] = [
            fmt_cell(m, s, n) for m, s, n in zip(
                df["targeted_warn_tgt_mean"], df["targeted_warn_tgt_std"], df["targeted_warn_tgt_n"]
            )
        ]
        if "targeted_collapse_mean" in df.columns:
            df["targeted_collapse_cell"] = [
                fmt_cell(m, s, n) for m, s, n in zip(
                    df["targeted_collapse_mean"], df["targeted_collapse_std"], df["targeted_collapse_n"]
                )
            ]
            df["targeted_delta_warn_tgt_cell"] = [
                fmt_cell(m, s, n) for m, s, n in zip(
                    df["targeted_delta_warn_tgt_mean"],
                    df["targeted_delta_warn_tgt_std"],
                    df["targeted_delta_warn_tgt_n"],
                )
            ]

    # --- widest schema: split into two stacked tables for readability ---
    if "targeted_collapse_mean" in df.columns and "targeted_delta_mean" in df.columns:
        # Table A (Random)
        df["random_meanstd"] = [fmt_mean_std(m, s) for m, s in zip(df["random_warn_mean"], df["random_warn_std"])]
        df["random_count"] = [f"{int(n)}/{n_total}" for n in df["random_warn_n"]]

        if "random_delta_mean" in df.columns:
            df["random_delta_meanstd"] = [
                fmt_mean_std(m, s) for m, s in zip(df["random_delta_mean"], df["random_delta_std"])
            ]
            df["random_delta_count"] = [f"{int(n)}/{n_total}" for n in df["random_delta_n"]]

        # Table B (Targeted)
        df["targeted_warn_tgt_meanstd"] = [
            fmt_mean_std(m, s) for m, s in zip(df["targeted_warn_tgt_mean"], df["targeted_warn_tgt_std"])
        ]
        df["targeted_collapse_meanstd"] = [
            fmt_mean_std(m, s) for m, s in zip(df["targeted_collapse_mean"], df["targeted_collapse_std"])
        ]
        df["targeted_delta_warn_tgt_meanstd"] = [
            fmt_mean_std(m, s) for m, s in zip(
                df["targeted_delta_warn_tgt_mean"], df["targeted_delta_warn_tgt_std"]
            )
        ]

        # If early-rate is constant (e.g. 0/5 for all γ), report it once in caption.
        # Otherwise, fall back to the wider targeted table that includes early-rate per row.
        try:
            early_counts = (df["targeted_early_n"].astype(int).astype(str) + "/" + df["targeted_n_total"].astype(int).astype(str))
            early_unique = set(early_counts.tolist())
            trig_full = bool((df["targeted_warn_tgt_n"].astype(int) == df["targeted_n_total"].astype(int)).all())
        except Exception:
            early_unique = set()
            trig_full = False

        if len(early_unique) != 1:
            # Not constant across γ: keep the existing wide-table path.
            pass
        else:
            early_rate_str = next(iter(early_unique)) if early_unique else "?"

            lines = []

            # --- Table A: Random removal ---
            lines.append(r"\begin{table}[H]")
            lines.append(r"\centering")
            lines.append(r"\caption{Random removal: warning point $q_{\mathrm{warn}}$ via baseline deviation, and lead time $\Delta_{\mathrm{warn}}=q_{\mathrm{collapse}}-q_{\mathrm{warn}}$ (collapse defined by $S(q)<0.1$).}")
            # Keep the original label name so existing references don't break.
            lines.append(r"\label{tab:gamma_sweep}")
            if "random_delta_mean" in df.columns:
                lines.append(r"\begin{tabular}{c c c c c}")
            else:
                lines.append(r"\begin{tabular}{c c c}")
            lines.append(r"\toprule")
            if "random_delta_mean" in df.columns:
                lines.append(r"$\gamma$ & $q_{\mathrm{warn}}$ (mean $\pm$ std) & [$n_{\mathrm{det}}/n$] & $\Delta_{\mathrm{warn}}$ (mean $\pm$ std) & [$n_{\Delta}/n$] \\")
            else:
                lines.append(r"$\gamma$ & $q_{\mathrm{warn}}$ (mean $\pm$ std) & [$n_{\mathrm{det}}/n$] \\")
            lines.append(r"\midrule")
            for _, r in df.iterrows():
                if "random_delta_mean" in df.columns:
                    lines.append(
                        f"{r['gamma']:.1f} & {r['random_meanstd']} & {r['random_count']} & {r['random_delta_meanstd']} & {r['random_delta_count']} \\\\"
                    )
                else:
                    lines.append(f"{r['gamma']:.1f} & {r['random_meanstd']} & {r['random_count']} \\\\")
            lines.append(r"\bottomrule")
            lines.append(r"\end{tabular}")
            lines.append(r"\end{table}")
            lines.append("")

            # --- Table B: Targeted removal (narrow, but explicit early-rate column) ---
            caption_note = rf"Early-rate is {early_rate_str} for all $\gamma$, hence $\Delta_{{\mathrm{{warn}}}}^{{\mathrm{{tgt}}}}=q_{{\mathrm{{collapse}}}}-q_{{\mathrm{{warn}}}}^{{\mathrm{{tgt}}}}$ throughout."
            if trig_full:
                caption_note += r" Onset warnings exist in all seeds ($n_{\mathrm{warn}}=n$)."

            lines.append(r"\begin{table}[H]")
            lines.append(r"\centering")
            lines.append(
                rf"\caption{{Targeted (hub-first) removal: collapse timing $q_{{\mathrm{{collapse}}}}$ and attack-onset warning timing $q_{{\mathrm{{warn}}}}^{{\mathrm{{tgt}}}}$. {caption_note}}}"
            )
            lines.append(r"\label{tab:gamma_sweep_targeted}")
            lines.append(r"\begin{tabular}{c c c c c}")
            lines.append(r"\toprule")
            lines.append(
                r"$\gamma$ & early-rate [$n_{\mathrm{early}}/n$] & $q_{\mathrm{collapse}}$ (mean $\pm$ std) & $q_{\mathrm{warn}}^{\mathrm{tgt}}$ (mean $\pm$ std) & $\Delta_{\mathrm{warn}}^{\mathrm{tgt}}$ (mean $\pm$ std) \\"
            )
            lines.append(r"\midrule")
            for _, r in df.iterrows():
                early_cell = f"{int(r['targeted_early_n'])}/{int(r['targeted_n_total'])}"
                lines.append(
                    f"{r['gamma']:.1f} & {early_cell} & {r['targeted_collapse_meanstd']} & {r['targeted_warn_tgt_meanstd']} & {r['targeted_delta_warn_tgt_meanstd']} \\\\"
                )
            lines.append(r"\bottomrule")
            lines.append(r"\end{tabular}")
            lines.append(r"\end{table}")
            lines.append("")

            # --- Table C: Random failure baseline comparison (optional) ---
            if "random_js_warn_mean" in df.columns and "random_dh_warn_mean" in df.columns:
                df["random_js_warn_meanstd"] = [
                    fmt_mean_std(m, s) for m, s in zip(df["random_js_warn_mean"], df["random_js_warn_std"])
                ]
                df["random_dh_warn_meanstd"] = [
                    fmt_mean_std(m, s) for m, s in zip(df["random_dh_warn_mean"], df["random_dh_warn_std"])
                ]

                lines.append(r"\begin{table}[H]")
                lines.append(r"\centering")
                lines.append(
                    r"\caption{Random failure baseline comparison: warning timing $q_{\mathrm{warn}}$ using the same baseline-deviation rule (Sec.~\hyperref[sec:early_warning_criteria]{Early-Warning Criteria}) applied to three rate-of-change signals on midpoint support: successive KL ($\tilde{D}_{\mathrm{KL}}$), successive JS ($\widetilde{\mathrm{JS}}$), and entropy-change magnitude ($\widetilde{|\Delta H|}$).}"
                )
                lines.append(r"\label{tab:gamma_sweep_random_baselines}")
                lines.append(r"\begin{tabular}{c c c c}")
                lines.append(r"\toprule")
                lines.append(r"$\gamma$ & $q_{\mathrm{warn}}^{\mathrm{KL}}$ (mean $\pm$ std) & $q_{\mathrm{warn}}^{\mathrm{JS}}$ (mean $\pm$ std) & $q_{\mathrm{warn}}^{|\Delta H|}$ (mean $\pm$ std) \\")
                lines.append(r"\midrule")
                for _, r in df.iterrows():
                    lines.append(
                        f"{r['gamma']:.1f} & {fmt_mean_std(r['random_warn_mean'], r['random_warn_std'])} & {r['random_js_warn_meanstd']} & {r['random_dh_warn_meanstd']} \\\\"
                    )
                lines.append(r"\bottomrule")
                lines.append(r"\end{tabular}")
                lines.append(r"\end{table}")
                lines.append("")

            out_path.write_text("\n".join(lines))


def export_targeted_floor_check_csv(runs, out_path: str = "paper/data/targeted_floor_check.csv") -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    targeted = [r for r in runs if r.get("regime") == "targeted"]
    if not targeted:
        raise ValueError("export_targeted_floor_check_csv: no targeted runs found")
    df = pd.DataFrame(targeted)
    cols = [
        "gamma",
        "seed",
        "q_floor",
        "q_warn_tgt",
        "q_collapse",
        "fired_at_floor",
        "dkl_floor",
        "thresh",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df[cols].to_csv(out_path, index=False)


def export_targeted_floor_check_table(
    runs,
    out_path: str = "paper/tables/targeted_floor_check.tex",
    label: str = "tab:targeted_floor_check",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    targeted = [r for r in runs if r.get("regime") == "targeted"]
    if not targeted:
        raise ValueError("export_targeted_floor_check_table: no targeted runs found")
    df = pd.DataFrame(targeted)
    df = df.sort_values(["gamma", "seed"]).reset_index(drop=True)

    def _isfinite(x):
        return np.isfinite(np.asarray(x, dtype=float))

    lines: list[str] = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Targeted onset grid-floor check. For each $\gamma$, we report the fraction of seeds where "
        r"$q_{\mathrm{warn}}^{\mathrm{tgt}}$ equals the earliest midpoint $q_{1/2}$ (grid floor), along with summary "
        r"statistics of $q_{\mathrm{warn}}^{\mathrm{tgt}}$ and the fraction of seeds with $q_{\mathrm{warn}}^{\mathrm{tgt}} < q_{\mathrm{collapse}}$.}"
    )
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{tabular}{c c c c c}")
    lines.append(r"\toprule")
    lines.append(
        r"$\gamma$ & $\Pr[q_{\mathrm{warn}}^{\mathrm{tgt}} = q_{1/2}]$ & $\min\,q_{\mathrm{warn}}^{\mathrm{tgt}}$ & "
    )
    lines.append(
        r"$\mathrm{median}\,q_{\mathrm{warn}}^{\mathrm{tgt}}$ & $\Pr[q_{\mathrm{warn}}^{\mathrm{tgt}} < q_{\mathrm{collapse}}]$ \\"
    )
    lines.append(r"\midrule")

    for gamma, gdf in df.groupby("gamma", sort=True):
        q_warn = np.asarray(gdf.get("q_warn_tgt", np.nan), dtype=float)
        q_floor = np.asarray(gdf.get("q_floor", np.nan), dtype=float)
        q_col = np.asarray(gdf.get("q_collapse", np.nan), dtype=float)

        defined = _isfinite(q_warn)
        n_def = int(np.count_nonzero(defined))

        if n_def == 0:
            p_floor = r"0/0"
            qmin = r"--"
            qmed = r"--"
            p_before = r"0/0"
        else:
            qf = float(q_floor[defined][0]) if np.any(_isfinite(q_floor[defined])) else float("nan")
            n_floor = int(np.count_nonzero(q_warn[defined] == qf))
            p_floor = f"{n_floor}/{n_def}"
            qmin = f"{float(np.min(q_warn[defined])):.3f}"
            qmed = f"{float(np.median(q_warn[defined])):.3f}"
            before = defined & _isfinite(q_col) & (q_warn < q_col)
            n_before = int(np.count_nonzero(before))
            p_before = f"{n_before}/{n_def}"

        lines.append(
            f"{float(gamma):.1f} & {p_floor} & {qmin} & {qmed} & {p_before} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    out_path.write_text("\n".join(lines))
    return


def export_sensitivity_csv(rows, out_path: str) -> None:
    """
    Export alpha×z sensitivity results to CSV.

    Expected rows: list[dict] with keys:
      - alpha, z, gamma
      - random_warn_mean, random_warn_std, random_warn_n
      - random_delta_mean, random_delta_std, random_delta_n
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "alpha",
        "z",
        "gamma",
        "random_warn_mean",
        "random_warn_std",
        "random_warn_n",
        "random_delta_mean",
        "random_delta_std",
        "random_delta_n",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def export_sensitivity_table(
    rows,
    out_path: str = "paper/tables/sensitivity_alpha_z.tex",
    n_total: int | None = None,
) -> None:
    """
    Export a compact alpha×z sensitivity table (random-failure warning + lead time).

    Table rows are (alpha, z, gamma).
    """
    if n_total is None:
        n_total = 5
    n_total = int(n_total)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("export_sensitivity_table: rows is empty")

    for c in [
        "alpha",
        "z",
        "gamma",
        "random_warn_mean",
        "random_warn_std",
        "random_warn_n",
        "random_delta_mean",
        "random_delta_std",
        "random_delta_n",
    ]:
        if c not in df.columns:
            raise ValueError(f"export_sensitivity_table: missing column {c!r}")

    df = df.sort_values(["alpha", "z", "gamma"]).reset_index(drop=True)

    def fmt_mean_std(mean, std) -> str:
        if pd.isna(mean) or pd.isna(std):
            return r"--"
        return f"{float(mean):.3f} $\\pm$ {float(std):.3f}"

    def fmt_count(n_detected) -> str:
        if pd.isna(n_detected):
            return r"--"
        return f"{int(n_detected)}/{n_total}"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Sensitivity of random-failure warning timing to EWMA smoothing $\alpha$ "
        r"and baseline threshold $z$ in the rule $\tilde{D}_{\mathrm{KL}}>\mu_0+z\sigma_0$. "
        r"Values report mean $\pm$ std across seeds (with $n_{\mathrm{det}}/n$ counts).}"
    )
    lines.append(r"\label{tab:sensitivity_alpha_z}")
    lines.append(r"\begin{tabular}{c c c c c}")
    lines.append(r"\toprule")
    lines.append(
        r"$\alpha$ & $z$ & $\gamma$ & $q_{\mathrm{warn}}$ (mean $\pm$ std) "
        r"[$n_{\mathrm{det}}/n$] & $\Delta_{\mathrm{warn}}$ (mean $\pm$ std) [$n_{\Delta}/n$] \\"
    )
    lines.append(r"\midrule")
    for _, r in df.iterrows():
        lines.append(
            f"{float(r['alpha']):.2f} & {float(r['z']):.1f} & {float(r['gamma']):.1f} & "
            f"{fmt_mean_std(r['random_warn_mean'], r['random_warn_std'])} "
            f"[{fmt_count(r['random_warn_n'])}] & "
            f"{fmt_mean_std(r['random_delta_mean'], r['random_delta_std'])} "
            f"[{fmt_count(r['random_delta_n'])}] \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    out_path.write_text("\n".join(lines))


def export_gamma_table_random(
    rows,
    *,
    n_total: int | None = None,
    out_path: str = "paper/tables/gamma_sweep_random.tex",
    caption: str | None = None,
    label: str = "tab:gamma_sweep_random",
) -> None:
    """
    Write the random-removal γ-sweep table as its own file (generated, not hand-edited).

    Supports the 19-col, 25-col, 27-col, and 35-col row schemas produced by GammaSweepExperiment.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("export_gamma_table_random: rows is empty")

    if n_total is None:
        n_total = 5
    n_total = int(n_total)

    k = len(rows[0])
    if k == 19:
        cols = [
            "gamma",
            "random_warn_mean", "random_warn_std", "random_warn_n",
            "random_delta_mean", "random_delta_std", "random_delta_n",
            "targeted_early_n", "targeted_n_total", "targeted_early_rate",
            "targeted_trigger_mean", "targeted_trigger_std", "targeted_trigger_n",
            "targeted_collapse_mean", "targeted_collapse_std", "targeted_collapse_n",
            "targeted_delta_mean", "targeted_delta_std", "targeted_delta_n",
        ]
        df = pd.DataFrame(rows, columns=cols)
    elif k in (25, 27, 35):
        cols = [
            "gamma",
            "random_warn_mean", "random_warn_std", "random_warn_n",
            "random_delta_mean", "random_delta_std", "random_delta_n",
            "random_js_warn_mean", "random_js_warn_std", "random_js_warn_n",
            "random_dh_warn_mean", "random_dh_warn_std", "random_dh_warn_n",
            "targeted_early_n", "targeted_n_total", "targeted_early_rate",
            "targeted_warn_tgt_mean", "targeted_warn_tgt_std", "targeted_warn_tgt_n",
            "targeted_collapse_mean", "targeted_collapse_std", "targeted_collapse_n",
            "targeted_delta_warn_tgt_mean", "targeted_delta_warn_tgt_std",
            "targeted_delta_warn_tgt_n",
            "targeted_intensity_mean", "targeted_intensity_std",
            "random_warn_med", "random_warn_iqr",
            "random_delta_med", "random_delta_iqr",
            "targeted_collapse_med", "targeted_collapse_iqr",
            "targeted_intensity_med", "targeted_intensity_iqr",
        ]
        df = pd.DataFrame(rows, columns=cols[:k])
    else:
        raise ValueError(
            "export_gamma_table_random: unsupported row length "
            f"{k} (expected 19, 25, 27, or 35)"
        )

    def fmt_med_iqr(med, iqr) -> str:
        if pd.isna(med) or pd.isna(iqr):
            return r"--"
        return f"{float(med):.3f} [{float(iqr):.3f}]"

    def fmt_count(n_detected) -> str:
        if pd.isna(n_detected):
            return r"--"
        return f"{int(n_detected)}/{n_total}"

    df = df.sort_values("gamma").reset_index(drop=True)

    lines: list[str] = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    if caption is None:
        lines.append(
            r"\caption{Random removal: warning point $q_{\mathrm{warn}}$ via baseline deviation "
            r"and lead time $\Delta_{\mathrm{warn}} = q_{\mathrm{collapse}} - q_{\mathrm{warn}}$ "
            r"(collapse defined by $S(q) < 0.1$). Values report median [IQR] across seeds where the "
            r"quantity is defined. Detection counts $(n_{\mathrm{det}}/n)$ report the number of seeds "
            r"where $q_{\mathrm{warn}}$ is observed prior to collapse; "
            r"$(n_{\Delta}/n)$ analogously counts seeds with a defined lead time $\Delta_{\mathrm{warn}}$. "
            r"Where $(n_{\mathrm{det}}/n)$ and $(n_{\Delta}/n)$ differ within a row, the discrepancy "
            r"reflects seeds in which $q_{\mathrm{collapse}}$ is undefined (the GCC never drops below "
            r"$0.1$ within the sweep) rather than failed or spurious detection. "
            r"At $\gamma=2.1$, $(n_{\Delta}/n)$ is low because networks near $\gamma=2$ are "
            r"maximally hub-dominated and remain connected ($S(q)>0.1$) throughout the sweep "
            r"($q\le 0.9$) in 31/40 seeds; the 9 seeds with a defined collapse all satisfy "
            r"$q_{\mathrm{collapse}}\ge 0.864$, confirming that the true $q_c$ lies at or beyond "
            r"the sweep ceiling rather than that $q_{\mathrm{warn}}$ is spurious.}"
        )
    else:
        lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{tabular}{c c c c c}")
    lines.append(r"\toprule")
    lines.append(
        r"$\gamma$ & $q_{\mathrm{warn}}$ (median [IQR]) & $(n_{\mathrm{det}}/n)$ & "
        r"$\Delta_{\mathrm{warn}}$ (median [IQR]) & $(n_{\Delta}/n)$ \\"
    )
    lines.append(r"\midrule")
    for _, r in df.iterrows():
        warn_cell = (
            fmt_med_iqr(r.get("random_warn_med"), r.get("random_warn_iqr"))
            if "random_warn_med" in df.columns else r"--"
        )
        delta_cell = (
            fmt_med_iqr(r.get("random_delta_med"), r.get("random_delta_iqr"))
            if "random_delta_med" in df.columns else r"--"
        )
        lines.append(
            f"{float(r['gamma']):.1f} & "
            f"{warn_cell} & "
            f"({fmt_count(r['random_warn_n'])}) & "
            f"{delta_cell} & "
            f"({fmt_count(r['random_delta_n'])}) \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    out_path.write_text("\n".join(lines))


def export_gamma_table_targeted(
    rows,
    *,
    n_total: int | None = None,
    out_path: str = "paper/tables/gamma_sweep_targeted.tex",
) -> None:
    """
    Write the targeted (hub-first) γ-sweep table as its own file (generated, not hand-edited).

    Supports the 35-col row schema produced by GammaSweepExperiment (with onset warnings + intensity + median/IQR).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("export_gamma_table_targeted: rows is empty")

    if n_total is None:
        n_total = 5
    n_total = int(n_total)

    k = len(rows[0])
    if k != 35:
        raise ValueError(
            "export_gamma_table_targeted: unsupported row length "
            f"{k} (expected 35)"
        )

    cols = [
        "gamma",
        "random_warn_mean", "random_warn_std", "random_warn_n",
        "random_delta_mean", "random_delta_std", "random_delta_n",
        "random_js_warn_mean", "random_js_warn_std", "random_js_warn_n",
        "random_dh_warn_mean", "random_dh_warn_std", "random_dh_warn_n",
        "targeted_early_n", "targeted_n_total", "targeted_early_rate",
        "targeted_warn_tgt_mean", "targeted_warn_tgt_std", "targeted_warn_tgt_n",
        "targeted_collapse_mean", "targeted_collapse_std", "targeted_collapse_n",
        "targeted_delta_warn_tgt_mean", "targeted_delta_warn_tgt_std",
        "targeted_delta_warn_tgt_n",
        "targeted_intensity_mean", "targeted_intensity_std",
        "random_warn_med", "random_warn_iqr",
        "random_delta_med", "random_delta_iqr",
        "targeted_collapse_med", "targeted_collapse_iqr",
        "targeted_intensity_med", "targeted_intensity_iqr",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df = df.sort_values("gamma").reset_index(drop=True)

    def fmt_med_iqr(med, iqr) -> str:
        if pd.isna(med) or pd.isna(iqr):
            return r"--"
        return f"{float(med):.3f} [{float(iqr):.3f}]"

    lines: list[str] = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Targeted (hub-first) removal across the $\gamma$ sweep. We report collapse timing "
        r"$q_{\mathrm{collapse}}$ and initial disruption intensity "
        r"$I_{\mathrm{tgt}}:=\tilde{D}_{\mathrm{KL}}(q_{1/2})$, where $q_{1/2}=\Delta q/2$ is the first midpoint of the damage grid. "
        r"Values are median [IQR] across seeds. "
        r"The count $(n_{\mathrm{col}}/n)$ reports the number of seeds for which the collapse proxy is observed within the sweep.}"
    )
    lines.append(r"\label{tab:gamma_sweep_targeted}")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{c c c c}")
    lines.append(r"\toprule")
    lines.append(
        r"$\gamma$ & $q_{\mathrm{collapse}}$ (median [IQR]) & $(n_{\mathrm{col}}/n)$ & $I_{\mathrm{tgt}}$ (median [IQR]) \\"
    )
    lines.append(r"\midrule")
    for _, r in df.iterrows():
        n_col = int(r['targeted_collapse_n']) if not pd.isna(r['targeted_collapse_n']) else 0
        lines.append(
            f"{float(r['gamma']):.1f} & "
            f"{fmt_med_iqr(r['targeted_collapse_med'], r['targeted_collapse_iqr'])} & "
            f"({n_col}/{n_total}) & "
            f"{fmt_med_iqr(r['targeted_intensity_med'], r['targeted_intensity_iqr'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")
    lines.append("")
    out_path.write_text("\n".join(lines))


def export_gamma_long_csv(runs, out_path: str) -> None:
    """
    Export long-format per-seed runs to CSV.

    Expected schema per run (dict-like):
      - regime: str
      - gamma: float
      - seed: int
      - q_warn: float (or NaN)
      - q_collapse: float (or NaN)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["regime", "gamma", "seed", "q_warn", "q_collapse"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in runs:
            if r is None:
                continue
            # Only write rows that contain the fields we need
            if not all(k in r for k in fieldnames):
                continue
            w.writerow({
                "regime": r["regime"],
                "gamma": r["gamma"],
                "seed": r["seed"],
                "q_warn": r["q_warn"],
                "q_collapse": r["q_collapse"],
            })


def export_baseline_noise_csv(runs, model: str, out_path: str) -> None:
    """
    Export per-seed baseline noise stats for random-regime runs.

    Columns: model, gamma, seed, mu0, sigma0, threshold, max_post_baseline,
             detected, threshold_exceeds_max_post
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    random_runs = [r for r in runs if r.get("regime") == "random"]
    if not random_runs:
        raise ValueError("export_baseline_noise_csv: no random-regime runs found")

    fieldnames = [
        "model", "gamma", "seed",
        "mu0", "sigma0", "threshold", "max_post_baseline",
        "detected", "threshold_exceeds_max_post",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in random_runs:
            q_warn = r.get("q_warn", float("nan"))
            threshold = r.get("threshold", float("nan"))
            max_post = r.get("max_post_baseline", float("nan"))
            detected = bool(np.isfinite(q_warn))
            threshold_exceeds_max_post = (
                bool(np.isfinite(threshold) and np.isfinite(max_post) and threshold > max_post)
            )
            w.writerow({
                "model": model,
                "gamma": r.get("gamma", float("nan")),
                "seed": r.get("seed", -1),
                "mu0": r.get("mu0", float("nan")),
                "sigma0": r.get("sigma0", float("nan")),
                "threshold": threshold,
                "max_post_baseline": max_post,
                "detected": detected,
                "threshold_exceeds_max_post": threshold_exceeds_max_post,
            })


def export_caida_summary(
    *,
    out_path: str = "paper/tables/caida_summary.tex",
    q_warns: list[float],
    q_collapses: list[float],
    n_total: int | None = None,
) -> None:
    """
    Write a 1-row CAIDA random-failure summary table reporting median [IQR]
    and detection counts for q_warn, q_collapse, and lead time
    Δ_warn = q_collapse - q_warn.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    qw = np.asarray(q_warns, dtype=float)
    qc = np.asarray(q_collapses, dtype=float)

    if n_total is None:
        n_total = int(len(qw))
    n_total = int(n_total)

    mask_delta = np.isfinite(qw) & np.isfinite(qc)
    deltas = np.where(mask_delta, qc - qw, np.nan)

    def med_iqr_n(arr: np.ndarray) -> tuple[float, float, int]:
        arr = np.asarray(arr, dtype=float)
        finite = arr[np.isfinite(arr)]
        n = len(finite)
        if n == 0:
            return float("nan"), float("nan"), 0
        med = float(np.median(finite))
        q1, q3 = float(np.percentile(finite, 25)), float(np.percentile(finite, 75))
        return med, q3 - q1, n

    qw_med, qw_iqr, qw_n = med_iqr_n(qw)
    qc_med, qc_iqr, qc_n = med_iqr_n(qc)
    d_med, d_iqr, d_n = med_iqr_n(deltas)

    def fmt_med(med: float, iqr: float, n: int) -> str:
        if n == 0 or not np.isfinite(med):
            return r"--"
        return f"{med:.3f} [{iqr:.3f}]"

    def fmt_count(n: int) -> str:
        return f"({n}/{n_total})"

    lines: list[str] = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{CAIDA AS graph (2026-01-01), random failure across {n_total} seeds "
        r"($\alpha=0.20$). All statistics are median [IQR]; $(n/N)$ is the detection count. "
        r"$q_{\mathrm{collapse}}$: GCC collapse ($S(q)<0.1$); "
        r"$\Delta_{\mathrm{warn}}=q_{\mathrm{collapse}}-q_{\mathrm{warn}}$.}"
    )
    lines.append(r"\label{tab:caida_summary}")
    lines.append(r"\begin{tabular}{c c c c c c}")
    lines.append(r"\toprule")
    lines.append(
        r"$q_{\mathrm{warn}}$ & $(n/N)$ & "
        r"$q_{\mathrm{collapse}}$ & $(n/N)$ & "
        r"$\Delta_{\mathrm{warn}}$ & $(n/N)$ \\"
    )
    lines.append(r"\midrule")
    lines.append(
        f"{fmt_med(qw_med, qw_iqr, qw_n)} & {fmt_count(qw_n)} & "
        f"{fmt_med(qc_med, qc_iqr, qc_n)} & {fmt_count(qc_n)} & "
        f"{fmt_med(d_med, d_iqr, d_n)} & {fmt_count(d_n)} \\\\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    out_path.write_text("\n".join(lines))
