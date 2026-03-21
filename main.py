from runner.gamma_sweep import GammaSweepExperiment
from runner.export import (
    export_gamma_table_random,
    export_gamma_table_targeted,
    export_gamma_long_csv,
    export_targeted_floor_check_csv,
    export_targeted_floor_check_table,
    export_sensitivity_csv,
    export_sensitivity_table,
    export_baseline_noise_csv,
)
from runner.kappa_control import run_kappa_control_random_failure
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument(
        "--sensitivity-only",
        action="store_true",
        help="Skip the full gamma sweep and only generate the sensitivity table/CSV.",
    )
    parser.add_argument(
        "--graph-model",
        type=str,
        default="chunglu",
        choices=["chunglu", "config"],
        help="Graph model to use: 'chunglu' (default) or 'config' (configuration model).",
    )
    args = parser.parse_args()

    if args.sensitivity_only and not args.sensitivity:
        args.sensitivity = True

    if args.sensitivity_only:
        ALPHAS = [0.10, 0.20, 0.30]
        ZS = [1.5, 2.0, 2.5]
        SENS_GAMMAS = [2.5]
        sensitivity_rows = []

        # Keep it fast: one representative gamma; reuse default seed list.
        for alpha in ALPHAS:
            for z in ZS:
                exp = GammaSweepExperiment(alpha=alpha, z=z, gammas=SENS_GAMMAS)
                rows_az = exp.run_random_only()
                for (g, mr, sr, nr, mdr, sdr, ndr) in rows_az:
                    sensitivity_rows.append(
                        {
                            "alpha": float(alpha),
                            "z": float(z),
                            "gamma": float(g),
                            "random_warn_mean": float(mr),
                            "random_warn_std": float(sr),
                            "random_warn_n": int(nr),
                            "random_delta_mean": float(mdr),
                            "random_delta_std": float(sdr),
                            "random_delta_n": int(ndr),
                        }
                    )

        export_sensitivity_csv(
            sensitivity_rows, out_path="paper/data/sensitivity_alpha_z.csv"
        )
        export_sensitivity_table(
            sensitivity_rows,
            out_path="paper/tables/sensitivity_alpha_z.tex",
            n_total=len(exp.seeds),
        )
        raise SystemExit(0)

    experiment = GammaSweepExperiment(graph_model=args.graph_model)
    rows, runs = experiment.run()
    
    # Adjust output file names and captions based on graph model
    model_suffix = "_config" if args.graph_model == "config" else ""
    
    if args.graph_model == "config":
        config_caption = (
            r"Configuration-model random failure across the full $\gamma$ grid. "
            r"Detection counts $(n_{\mathrm{det}}/n)$ report the number of seeds where "
            r"$q_{\mathrm{warn}}$ is observed prior to collapse; $(n_{\Delta}/n)$ analogously "
            r"counts seeds with a defined lead time $\Delta_{\mathrm{warn}}$. Results use the "
            r"same baseline-deviation rule as the Chung--Lu experiments ($\alpha=0.20$, $z=2.0$, "
            r"baseline window $q \le 0.15$). "
            r"Where $(n_{\mathrm{det}}/n)$ and $(n_{\Delta}/n)$ differ within a row, the discrepancy "
            r"reflects seeds in which $q_{\mathrm{collapse}}$ is undefined (the GCC never drops below "
            r"$0.1$ within the sweep) rather than failed or spurious detection. "
            r"The low and $\gamma$-non-monotonic $(n_{\mathrm{det}}/n)$ counts reflect inconsistent "
            r"threshold inflation across seeds, as quantified in Table~\ref{tab:baseline_noise_comparison}, "
            r"rather than a systematic relationship between $\gamma$ and signal detectability."
        )
        export_gamma_table_random(
            rows, 
            n_total=len(experiment.seeds),
            out_path=f"paper/tables/gamma_sweep_random{model_suffix}.tex",
            caption=config_caption,
            label="tab:config_random"
        )
    else:
        export_gamma_table_random(rows, n_total=len(experiment.seeds))
    
    export_gamma_table_targeted(rows, n_total=len(experiment.seeds),
                               out_path=f"paper/tables/gamma_sweep_targeted{model_suffix}.tex")

    # Per-seed long CSV (random regime only)
    random_runs = [r for r in runs if r.get("regime") == "random"]
    export_gamma_long_csv(random_runs, out_path=f"paper/data/gamma_sweep_random_long{model_suffix}.csv")
    export_baseline_noise_csv(
        random_runs,
        model=args.graph_model,
        out_path=f"paper/data/baseline_noise_{args.graph_model}.csv",
    )

    targeted_runs = [r for r in runs if r.get("regime") == "targeted"]
    export_targeted_floor_check_csv(
        targeted_runs, out_path=f"paper/data/targeted_floor_check{model_suffix}.csv"
    )
    export_targeted_floor_check_table(
        targeted_runs, out_path=f"paper/tables/targeted_floor_check{model_suffix}.tex"
    )

    # Appendix comparator: baseline deviation on EWMA-smoothed kappa(q) under random failure.
    run_kappa_control_random_failure(outdir="paper")

    n_total = len(experiment.seeds)
    print("γ | Random q_warn(KL) (mean ± std) [n/n] | Random q_warn(JS) (mean ± std) [n/n] | Random q_warn(|ΔH|) (mean ± std) [n/n] | Random Δ_warn (mean ± std) [n/n] | Targeted early-rate [n/n] | Targeted q_warn_tgt (mean ± std) [n/n] | Targeted q_collapse (mean ± std) [n/n] | Targeted Δ_warn_tgt (mean ± std) [n/n] | Targeted I_tgt=~DKL(q_1/2) (mean ± std)")
    for row in rows:
        # Keep console output stable even if the gamma-sweep row schema expands.
        (
            g,
            mr, sr, nr,
            mdr, sdr, ndr,
            mjs, sjs, njs,
            mdh, sdh, ndh,
            ne, nt, er,
            mt, st, nw,
            mc, sc, nc,
            md, sd, nd,
            mi, si,
            *_,  # ignore any additional summary stats (e.g., median/IQR) in newer schemas
        ) = row

        r_cell = "--" if nr == 0 or mr != mr else f"{mr:.3f} ± {sr:.3f}"
        js_cell = "--" if njs == 0 or mjs != mjs else f"{mjs:.3f} ± {sjs:.3f}"
        dh_cell = "--" if ndh == 0 or mdh != mdh else f"{mdh:.3f} ± {sdh:.3f}"
        dr_cell = "--" if ndr == 0 or mdr != mdr else f"{mdr:.3f} ± {sdr:.3f}"
        trig_cell = "--" if nw == 0 or mt != mt else f"{mt:.3f} ± {st:.3f}"
        collapse_cell = "--" if nc == 0 or mc != mc else f"{mc:.3f} ± {sc:.3f}"
        delta_cell = "--" if nd == 0 or md != md else f"{md:.3f} ± {sd:.3f}"
        i_cell = "--" if mi != mi else f"{mi:.3e} ± {si:.3e}"
        print(
            f"{g:.1f} | {r_cell} [{nr}/{n_total}] | {js_cell} [{int(njs)}/{n_total}] | {dh_cell} [{int(ndh)}/{n_total}] | "
            f"{dr_cell} [{int(ndr)}/{n_total}] | {int(ne)}/{int(nt)} | {trig_cell} [{int(nw)}/{n_total}] | "
            f"{collapse_cell} [{int(nc)}/{n_total}] | {delta_cell} [{int(nd)}/{n_total}] | {i_cell}"
        )

    if args.sensitivity:
        ALPHAS = [0.10, 0.20, 0.30]
        ZS = [1.5, 2.0, 2.5]
        # Keep sensitivity cheap: one representative gamma is enough to show robustness of
        # the qualitative warning timing to (alpha, z).
        SENS_GAMMAS = [2.5]
        sensitivity_rows = []

        for alpha in ALPHAS:
            for z in ZS:
                exp = GammaSweepExperiment(alpha=alpha, z=z, gammas=SENS_GAMMAS)
                rows_az, _ = exp.run()
                for (
                    g,
                    mr, sr, nr,
                    mdr, sdr, ndr,
                    _mjs, _sjs, _njs,
                    _mdh, _sdh, _ndh,
                    *_rest,
                ) in rows_az:
                    sensitivity_rows.append(
                        {
                            "alpha": float(alpha),
                            "z": float(z),
                            "gamma": float(g),
                            "random_warn_mean": float(mr),
                            "random_warn_std": float(sr),
                            "random_warn_n": int(nr),
                            "random_delta_mean": float(mdr),
                            "random_delta_std": float(sdr),
                            "random_delta_n": int(ndr),
                        }
                    )

        export_sensitivity_csv(
            sensitivity_rows, out_path="paper/data/sensitivity_alpha_z.csv"
        )
        export_sensitivity_table(
            sensitivity_rows,
            out_path="paper/tables/sensitivity_alpha_z.tex",
            n_total=len(experiment.seeds),
        )
