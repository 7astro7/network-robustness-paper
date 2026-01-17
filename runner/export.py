# runner/export.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


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
    else:
        raise ValueError(f"export_gamma_table: unsupported row length {k} (expected 5, 7, 10, or 16)")

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

    df["random_cell"] = [
        fmt_cell(m, s, n) for m, s, n in zip(df["random_mean"], df["random_std"], df["random_n"])
    ]
    if "targeted_mean" in df.columns:
        df["targeted_cell"] = [
            fmt_cell(m, s, n) for m, s, n in zip(df["targeted_mean"], df["targeted_std"], df["targeted_n"])
        ]
    else:
        df["targeted_rate_cell"] = [
            fmt_rate_cell(ne, nt) for ne, nt in zip(df["targeted_early_n"], df["targeted_n_total"])
        ]
        df["targeted_trigger_cell"] = [
            fmt_cell(m, s, n) for m, s, n in zip(
                df["targeted_trigger_mean"], df["targeted_trigger_std"], df["targeted_trigger_n"]
            )
        ]
        if "targeted_collapse_mean" in df.columns:
            df["targeted_collapse_cell"] = [
                fmt_cell(m, s, n) for m, s, n in zip(
                    df["targeted_collapse_mean"], df["targeted_collapse_std"], df["targeted_collapse_n"]
                )
            ]
            df["targeted_delta_cell"] = [
                fmt_cell(m, s, n) for m, s, n in zip(
                    df["targeted_delta_mean"], df["targeted_delta_std"], df["targeted_delta_n"]
                )
            ]

    # --- widest schema: split into two stacked tables for readability ---
    if "targeted_collapse_mean" in df.columns and "targeted_delta_mean" in df.columns:
        # Table A (Random)
        df["random_meanstd"] = [fmt_mean_std(m, s) for m, s in zip(df["random_mean"], df["random_std"])]
        df["random_count"] = [f"{int(n)}/{n_total}" for n in df["random_n"]]

        # Table B (Targeted)
        df["targeted_trigger_meanstd"] = [
            fmt_mean_std(m, s) for m, s in zip(df["targeted_trigger_mean"], df["targeted_trigger_std"])
        ]
        df["targeted_collapse_meanstd"] = [
            fmt_mean_std(m, s) for m, s in zip(df["targeted_collapse_mean"], df["targeted_collapse_std"])
        ]

        # If early-rate is constant (e.g. 0/5 for all γ), report it once in caption.
        # Otherwise, fall back to the wider targeted table that includes early-rate per row.
        try:
            early_counts = (df["targeted_early_n"].astype(int).astype(str) + "/" + df["targeted_n_total"].astype(int).astype(str))
            early_unique = set(early_counts.tolist())
            trig_full = bool((df["targeted_trigger_n"].astype(int) == df["targeted_n_total"].astype(int)).all())
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
            lines.append(r"\caption{Random removal: warning point $q_{\mathrm{warn}}$ via baseline deviation.}")
            # Keep the original label name so existing references don't break.
            lines.append(r"\label{tab:gamma_sweep}")
            lines.append(r"\begin{tabular}{c c c}")
            lines.append(r"\toprule")
            lines.append(r"$\gamma$ & $q_{\mathrm{warn}}$ (mean $\pm$ std) & [$n_{\mathrm{det}}/n$] \\")
            lines.append(r"\midrule")
            for _, r in df.iterrows():
                lines.append(f"{r['gamma']:.1f} & {r['random_meanstd']} & {r['random_count']} \\\\")
            lines.append(r"\bottomrule")
            lines.append(r"\end{tabular}")
            lines.append(r"\end{table}")
            lines.append("")

            # --- Table B: Targeted removal (narrow) ---
            caption_note = rf"Early-trigger rate is {early_rate_str} for all $\gamma$, hence $\Delta=q_{{\mathrm{{collapse}}}}-q_{{\mathrm{{trigger}}}}<0$ throughout."
            if trig_full:
                caption_note += r" Drift triggers exist in all seeds ($n_{\mathrm{trig}}=n$)."

            lines.append(r"\begin{table}[H]")
            lines.append(r"\centering")
            lines.append(
                rf"\caption{{Targeted (hub-first) removal: collapse timing $q_{{\mathrm{{collapse}}}}$ and drift-trigger timing $q_{{\mathrm{{trigger}}}}$. {caption_note}}}"
            )
            lines.append(r"\label{tab:gamma_sweep_targeted}")
            lines.append(r"\begin{tabular}{c c c}")
            lines.append(r"\toprule")
            lines.append(r"$\gamma$ & $q_{\mathrm{collapse}}$ (mean $\pm$ std) & $q_{\mathrm{trigger}}$ (mean $\pm$ std) \\")
            lines.append(r"\midrule")
            for _, r in df.iterrows():
                lines.append(f"{r['gamma']:.1f} & {r['targeted_collapse_meanstd']} & {r['targeted_trigger_meanstd']} \\\\")
            lines.append(r"\bottomrule")
            lines.append(r"\end{tabular}")
            lines.append(r"\end{table}")
            lines.append("")

            out_path.write_text("\n".join(lines))
            return

        lines = []

        # --- Table A: Random removal ---
        lines.append(r"\begin{table}[H]")
        lines.append(r"\centering")
        lines.append(r"\caption{Random removal: warning point $q_{\mathrm{warn}}$ via baseline deviation.}")
        # Keep the original label name so existing references don't break.
        lines.append(r"\label{tab:gamma_sweep}")
        lines.append(r"\begin{tabular}{c c c}")
        lines.append(r"\toprule")
        lines.append(r"$\gamma$ & $q_{\mathrm{warn}}$ (mean $\pm$ std) & [$n_{\mathrm{det}}/n$] \\")
        lines.append(r"\midrule")
        for _, r in df.iterrows():
            lines.append(f"{r['gamma']:.1f} & {r['random_meanstd']} & {r['random_count']} \\\\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

        # --- Table B: Targeted removal ---
        lines.append(r"\begin{table}[H]")
        lines.append(r"\centering")
        lines.append(
            r"\caption{Targeted (hub-first) removal: drift trigger timing $q_{\mathrm{trigger}}$ and early-trigger rate ($q_{\mathrm{trigger}}<q_{\mathrm{collapse}}$), with collapse defined by $S(q)<0.1$.}"
        )
        lines.append(r"\label{tab:gamma_sweep_targeted}")
        lines.append(r"\begin{tabular}{c c c c c}")
        lines.append(r"\toprule")
        lines.append(
            r"$\gamma$ & early-rate [$n_{\mathrm{early}}/n$] & $q_{\mathrm{trigger}}$ (mean $\pm$ std) [$n_{\mathrm{trig}}/n$] & $q_{\mathrm{collapse}}$ (mean $\pm$ std) & $\Delta$ (mean $\pm$ std) [$n_{\mathrm{trig}}/n$] \\"
        )
        lines.append(r"\midrule")
        for _, r in df.iterrows():
            lines.append(
                f"{r['gamma']:.1f} & {r['targeted_rate_cell']} & {r['targeted_trigger_cell']} & {r['targeted_collapse_meanstd']} & {r['targeted_delta_cell']} \\\\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

        out_path.write_text("\n".join(lines))
        return

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    if "targeted_mean" in df.columns:
        lines.append(r"\caption{Mean and standard deviation of detected warning points $q_{\mathrm{warn}}$ across $\gamma$ under random and targeted removal.}")
    else:
        lines.append(
            r"\caption{Random removal reports the warning point $q_{\mathrm{warn}}$. Targeted removal reports the fraction of seeds with an \emph{early} drift trigger ($q_{\mathrm{trigger}}<q_{\mathrm{collapse}}$) and the drift trigger timing $q_{\mathrm{trigger}}$.}"
        )
    lines.append(r"\label{tab:gamma_sweep}")
    if "targeted_mean" in df.columns:
        lines.append(r"\begin{tabular}{c c c}")
    elif "targeted_collapse_mean" in df.columns:
        lines.append(r"\begin{tabular}{c c c c c c}")
    else:
        lines.append(r"\begin{tabular}{c c c c}")
    lines.append(r"\toprule")
    if "targeted_mean" in df.columns:
        lines.append(r"$\gamma$ & Random $q_{\mathrm{warn}}$ (mean $\pm$ std) [$n_{\mathrm{det}}/n$] & Targeted $q_{\mathrm{warn}}$ (mean $\pm$ std) [$n_{\mathrm{det}}/n$] \\")
    elif "targeted_collapse_mean" in df.columns:
        lines.append(
            r"$\gamma$ & Random $q_{\mathrm{warn}}$ (mean $\pm$ std) [$n_{\mathrm{det}}/n$] & Targeted early-rate [$n_{\mathrm{early}}/n$] & Targeted $q_{\mathrm{trigger}}$ (drift) (mean $\pm$ std) [$n_{\mathrm{trig}}/n$] & Targeted $q_{\mathrm{collapse}}$ (mean $\pm$ std) [$n/n$] & Targeted $\Delta=q_{\mathrm{collapse}}-q_{\mathrm{trigger}}$ (mean $\pm$ std) [$n_{\mathrm{trig}}/n$] \\"
        )
    else:
        lines.append(
            r"$\gamma$ & Random $q_{\mathrm{warn}}$ (mean $\pm$ std) [$n_{\mathrm{det}}/n$] & Targeted early-rate [$n_{\mathrm{early}}/n$] & Targeted $q_{\mathrm{trigger}}$ (mean $\pm$ std) [$n_{\mathrm{trig}}/n$] \\"
        )
    lines.append(r"\midrule")

    for _, r in df.iterrows():
        if "targeted_mean" in df.columns:
            lines.append(f"{r['gamma']:.1f} & {r['random_cell']} & {r['targeted_cell']} \\\\")
        elif "targeted_collapse_mean" in df.columns:
            lines.append(
                f"{r['gamma']:.1f} & {r['random_cell']} & {r['targeted_rate_cell']} & {r['targeted_trigger_cell']} & {r['targeted_collapse_cell']} & {r['targeted_delta_cell']} \\\\"
            )
        else:
            lines.append(
                f"{r['gamma']:.1f} & {r['random_cell']} & {r['targeted_rate_cell']} & {r['targeted_trigger_cell']} \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    out_path.write_text("\n".join(lines))
