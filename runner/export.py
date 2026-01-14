import pandas as pd
import numpy as np

def export_gamma_table(rows, path="paper/gamma_sweep.tex"):
    """
    rows: list of tuples
          (gamma, mean_random, std_random, mean_targeted, std_targeted)
    """
    df = pd.DataFrame(
        rows,
        columns=[
            r"$\gamma$",
            r"Random Failure $q_{\mathrm{warn}}$",
            r"Random Std",
            r"Targeted Failure $q_{\mathrm{warn}}$",
            r"Targeted Std",
        ]
    )

    # Format mean ± std as a single column
    df[r"Random Failure $q_{\mathrm{warn}}$"] = (
        df[r"Random Failure $q_{\mathrm{warn}}$"]
        .map("{:.3f}".format)
        + r" $\pm$ "
        + df["Random Std"].map("{:.3f}".format)
    )

    df[r"Targeted Failure $q_{\mathrm{warn}}$"] = (
        df[r"Targeted Failure $q_{\mathrm{warn}}$"]
        .map("{:.3f}".format)
        + r" $\pm$ "
        + df["Targeted Std"].map("{:.3f}".format)
    )

    df = df[[r"$\gamma$", 
             r"Random Failure $q_{\mathrm{warn}}$", 
             r"Targeted Failure $q_{\mathrm{warn}}$"]]

    latex = df.to_latex(
        index=False,
        escape=False,
        caption=(
            "Mean and standard deviation of detected warning points "
            "$q_{\\mathrm{warn}}$ across degree exponents $\\gamma$ "
            "under random and targeted failure."
        ),
        label="tab:gamma_sweep",
        column_format="c c c",
    )

    with open(path, "w") as f:
        f.write(latex)

