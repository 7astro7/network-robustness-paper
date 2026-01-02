import numpy as np
from models.graph_model import GraphModel
from models.failure_model import FailureModel
from models.targeted_failure_model import TargetedFailureModel
from experiment.experiment import Experiment
from models.metrics import Metrics
import csv 


def run_default(seed=None):
    if seed is not None:
        np.random.seed(seed)

    n = 10_000
    gamma = 2.3

    # --- setup ---
    graph = GraphModel(n=n, gamma=gamma)
    failure = FailureModel()
    experiment = Experiment(graph, failure)

    qs = np.linspace(0, 0.9, 20)

    # --- run damage sweep ---
    S_values, H_values, DKL_values, Pq_values = experiment.sweep(qs)

    # --- reference plots only ---
    experiment.plot_full_results(qs, S_values, H_values, DKL_values)

    # --- successive KL signal ---
    dKL_successive = Metrics.successive_kl(Pq_values)
    dKL_successive_smooth = experiment.ewma(dKL_successive, alpha=0.2)

    # --- EARLY WARNING RULE (baseline deviation) ---
    warn_idx = Metrics.detect_baseline_deviation(dKL_successive_smooth)

    q_warn = qs[warn_idx] if warn_idx is not None else None

    # --- COLLAPSE RULE (GCC threshold) ---
    collapse_idx = next(
        (i for i, s in enumerate(S_values) if s < 0.1),
        None
    )
    q_collapse = qs[collapse_idx] if collapse_idx is not None else None

    # --- visualization only ---
    experiment.plot_successive_KL(
        qs,
        dKL_successive_smooth,
        q_warn,
        q_collapse
    )

    return q_warn, q_collapse


def run_targeted():
    """
    Run robustness experiment under targeted failures (hub removal).
    """
    n = 10_000
    gamma = 2.3

    graph = GraphModel(n=n, gamma=gamma)
    failure = TargetedFailureModel()
    experiment = Experiment(graph, failure)

    qs = np.linspace(0, 0.5, 20)  # targeted failure collapses earlier

    S_values, H_values, DKL_values, Pq_values = experiment.sweep(qs)
    experiment.plot_full_results(qs, S_values, H_values, DKL_values)

    dH = experiment.compute_derivative(qs, H_values)
    qc, _ = experiment.plot_entropy_derivative(qs, dH)

    print(f"Entropy critical point under targeted attack: q* = {qc:.4f}")


if __name__ == "__main__":
    for seed in [0, 1, 2, 3, 4]:
        print(f"\nRunning seed {seed}")
        q_warn, q_collapse = run_default(seed=seed)
        print(f"q_warn={q_warn}, q_collapse={q_collapse}")
