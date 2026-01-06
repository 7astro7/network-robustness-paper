import numpy as np
from models.graph_model import GraphModel
from models.failure_model import FailureModel
from models.targeted_failure_model import TargetedFailureModel
from experiment.experiment import Experiment
from models.metrics import Metrics


def run_default(seed=None, gamma_override=None):
    if seed is not None:
        np.random.seed(seed)

    n = 10_000
    gamma = gamma_override if gamma_override is not None else 2.3

    # --- setup ---
    graph = GraphModel(n=n, gamma=gamma)
    failure = FailureModel()
    experiment = Experiment(graph, failure)

    qs = np.linspace(0, 0.9, 100)

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
#    experiment.plot_successive_KL(
#        qs,
#        dKL_successive_smooth,
#        q_warn,
#        q_collapse
#    )

#    experiment.plot_successive_KL_overlay(qs, seeds=[0,1,2,3,4])

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
#    qc, _ = experiment.plot_entropy_derivative(qs, dH)

    print(f"Entropy critical point under targeted attack: q* = {qc:.4f}")

def run_targeted_warning(seed=None, gamma_override=None):
    if seed is not None:
        np.random.seed(seed)

    n = 10_000
    gamma = gamma_override if gamma_override is not None else 2.3

    graph = GraphModel(n=n, gamma=gamma)
    failure = TargetedFailureModel()
    experiment = Experiment(graph, failure)

    qs = np.linspace(0, 0.9, 100)

    S_values, H_values, DKL_values, Pq_values = experiment.sweep(qs)

    dKL_successive = Metrics.successive_kl(Pq_values)
    dKL_successive_smooth = experiment.ewma(dKL_successive, alpha=0.2)

    warn_idx = Metrics.detect_slope(dKL_successive_smooth, m=3)
    q_warn = qs[warn_idx] if warn_idx is not None else None

    collapse_idx = next(
        (i for i, s in enumerate(S_values) if s < 0.1),
        None
    )
    q_collapse = qs[collapse_idx] if collapse_idx is not None else None

    experiment.plot_successive_KL(
        qs,
        dKL_successive_smooth,
        q_warn,
        q_collapse
    )

    return q_warn, q_collapse

def summarize_warnings(run_fn, seeds):
    q_warns = []

    for seed in seeds:
        q_warn, q_collapse = run_fn(seed=seed)
        if q_warn is not None:
            q_warns.append(q_warn)

    q_warns = np.array(q_warns)

    return q_warns.mean(), q_warns.std(), len(q_warns)

def gamma_sweep_table(gammas, seeds):
    rows = []

    for gamma in gammas:
        def run_random(seed=None):
            return run_default(seed=seed, gamma_override=gamma)

        def run_targeted(seed=None):
            return run_targeted_warning(seed=seed, gamma_override=gamma)

        mean_r, std_r, n_r = summarize_warnings(run_random, seeds)
        mean_t, std_t, n_t = summarize_warnings(run_targeted, seeds)

        rows.append((gamma, mean_r, std_r, mean_t, std_t))

    return rows


if __name__ == "__main__":
    seeds = [0, 1, 2, 3, 4]

    mean_r, std_r, n_r = summarize_warnings(run_default, seeds)
    mean_t, std_t, n_t = summarize_warnings(run_targeted_warning, seeds)

    print("\nRandom failure:")
    print(f"q_warn = {mean_r:.3f} ± {std_r:.3f} (n={n_r})")

    print("\nTargeted failure:")
    print(f"q_warn = {mean_t:.3f} ± {std_t:.3f} (n={n_t})")

