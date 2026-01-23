from runner.gamma_sweep import GammaSweepExperiment
from runner.export import export_gamma_table

if __name__ == "__main__":
    experiment = GammaSweepExperiment()
    rows = experiment.run()
    export_gamma_table(rows, n_total=len(experiment.seeds))

    n_total = len(experiment.seeds)
    print("γ | Random q_warn(KL) (mean ± std) [n/n] | Random q_warn(JS) (mean ± std) [n/n] | Random q_warn(|ΔH|) (mean ± std) [n/n] | Random Δ_warn (mean ± std) [n/n] | Targeted early-rate [n/n] | Targeted q_trigger (mean ± std) [n/n] | Targeted q_collapse (mean ± std) [n/n] | Targeted Δ_trigger (mean ± std) [n/n]")
    for (
        g,
        mr, sr, nr,
        mdr, sdr, ndr,
        mjs, sjs, njs,
        mdh, sdh, ndh,
        ne, nt, er,
        mt, st, ntrig,
        mc, sc, nc,
        md, sd, nd,
    ) in rows:
        r_cell = "--" if nr == 0 or mr != mr else f"{mr:.3f} ± {sr:.3f}"
        js_cell = "--" if njs == 0 or mjs != mjs else f"{mjs:.3f} ± {sjs:.3f}"
        dh_cell = "--" if ndh == 0 or mdh != mdh else f"{mdh:.3f} ± {sdh:.3f}"
        dr_cell = "--" if ndr == 0 or mdr != mdr else f"{mdr:.3f} ± {sdr:.3f}"
        trig_cell = "--" if ntrig == 0 or mt != mt else f"{mt:.3f} ± {st:.3f}"
        collapse_cell = "--" if nc == 0 or mc != mc else f"{mc:.3f} ± {sc:.3f}"
        delta_cell = "--" if nd == 0 or md != md else f"{md:.3f} ± {sd:.3f}"
        print(
            f"{g:.1f} | {r_cell} [{nr}/{n_total}] | {js_cell} [{int(njs)}/{n_total}] | {dh_cell} [{int(ndh)}/{n_total}] | "
            f"{dr_cell} [{int(ndr)}/{n_total}] | {int(ne)}/{int(nt)} | {trig_cell} [{int(ntrig)}/{n_total}] | "
            f"{collapse_cell} [{int(nc)}/{n_total}] | {delta_cell} [{int(nd)}/{n_total}]"
        )
