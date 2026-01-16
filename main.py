from runner.gamma_sweep import GammaSweepExperiment
from runner.export import export_gamma_table

if __name__ == "__main__":
    experiment = GammaSweepExperiment()
    rows = experiment.run()
    export_gamma_table(rows)

    print("γ | Random q_warn (mean ± std) [n] | Targeted q_warn (mean ± std) [n]")
    for g, mr, sr, mt, st, nr, nt in rows:
        print(f"{g:.1f} | {mr:.3f} ± {sr:.3f} [{nr}] | {mt:.3f} ± {st:.3f} [{nt}]")
