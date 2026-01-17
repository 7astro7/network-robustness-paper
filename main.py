from runner.gamma_sweep import GammaSweepExperiment
from runner.export import export_gamma_table

if __name__ == "__main__":
    experiment = GammaSweepExperiment()
    rows = experiment.run()
    export_gamma_table(rows, n_total=len(experiment.seeds))

    n_total = len(experiment.seeds)
    print("γ | Random q_warn (mean ± std) [n_det/n] | Targeted early-rate [n_early/n] | Targeted q_trigger (mean ± std) [n_trig/n] | Targeted q_collapse (mean ± std) [n/n] | Targeted Δ (mean ± std) [n_trig/n]")
    for g, mr, sr, nr, ne, nt, er, mt, st, ntrig, mc, sc, nc, md, sd, nd in rows:
        r_cell = "--" if nr == 0 or mr != mr else f"{mr:.3f} ± {sr:.3f}"
        trig_cell = "--" if ntrig == 0 or mt != mt else f"{mt:.3f} ± {st:.3f}"
        collapse_cell = "--" if nc == 0 or mc != mc else f"{mc:.3f} ± {sc:.3f}"
        delta_cell = "--" if nd == 0 or md != md else f"{md:.3f} ± {sd:.3f}"
        print(f"{g:.1f} | {r_cell} [{nr}/{n_total}] | {int(ne)}/{int(nt)} | {trig_cell} [{int(ntrig)}/{n_total}] | {collapse_cell} [{int(nc)}/{n_total}] | {delta_cell} [{int(nd)}/{n_total}]")
