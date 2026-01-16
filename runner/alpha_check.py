import numpy as np
from experiment.experiment import Experiment
from models.graph_model import GraphModel
from models.failure_model import RandomFailure, TargetedFailure
from runner.gamma_sweep import GammaSweepExperiment  # reuse detection methods
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))


def alpha_sanity_check(
    gamma: float = 2.5,
    seed: int = 0,
    alphas=(0.1, 0.2, 0.3),
    n: int = 10_000,
    qs: np.ndarray | None = None,
):
    qs = qs if qs is not None else np.linspace(0, 0.9, 100)

    np.random.seed(seed)
    graph = GraphModel(n=n, gamma=gamma)

    detector = GammaSweepExperiment(n=n, qs=qs, seeds=[seed])  # just for methods

    results = {"random": {}, "targeted": {}}

    # random
    exp_r = Experiment(graph, RandomFailure())
    _, _, Pq_r = exp_r.sweep(qs)
    raw_r = exp_r.successive_kl(Pq_r)
    assert np.all(np.isfinite(raw_r)), "non-finite raw KL (random)"

    for a in alphas:
        dkl = exp_r.ewma(raw_r, alpha=a)
        results["random"][a] = detector._detect_baseline_break(qs, dkl)

    # targeted
    exp_t = Experiment(graph, TargetedFailure())
    _, _, Pq_t = exp_t.sweep(qs)
    raw_t = exp_t.successive_kl(Pq_t)
    assert np.all(np.isfinite(raw_t)), "non-finite raw KL (targeted)"

    for a in alphas:
        dkl = exp_t.ewma(raw_t, alpha=a)
        results["targeted"][a] = detector._detect_positive_drift(qs, dkl)

    return results

if __name__ == "__main__":
    out = alpha_sanity_check()
    print("Alpha sanity check (gamma=2.5, seed=0)")
    for mode in ["random", "targeted"]:
        print(f"\n{mode}:")
        for a, qwarn in out[mode].items():
            print(f"  alpha={a:.1f} -> q_warn={qwarn}")

