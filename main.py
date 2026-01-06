from runner.run_experiment import gamma_sweep_table

if __name__ == "__main__":
    gammas = [2.1, 2.3, 2.5, 2.7, 2.9]
    seeds = [0, 1, 2, 3, 4]

    rows = gamma_sweep_table(gammas, seeds)

    print("γ | Random q_warn (mean ± std) | Targeted q_warn (mean ± std)")
    for g, mr, sr, mt, st in rows:
        print(f"{g:.1f} | {mr:.3f} ± {sr:.3f} | {mt:.3f} ± {st:.3f}")
