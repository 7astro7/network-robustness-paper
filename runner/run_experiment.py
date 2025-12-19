import numpy as np
from models.graph_model import GraphModel
from models.failure_model import FailureModel
from models.failure_model import TargetedFailureModel
from experiment.experiment import Experiment


def run_default():
    """
    Run a standard robustness experiment on a Chung–Lu power-law graph
    and return S(q), H(q), D_KL(q), and dH/dq.
    """
    # graph parameters
    n = 10_000
    gamma = 2.3

    graph = GraphModel(n=n, gamma=gamma)
    failure = FailureModel()
    experiment = Experiment(graph, failure)

    # sweep q from 0 → 0.9
    qs = np.linspace(0, 0.9, 20)

    S_values, H_values, DKL_values = experiment.sweep(qs)

    # full tri-plot
    experiment.plot_full_results(qs, S_values, H_values, DKL_values)

    dH = experiment.compute_derivative(qs, H_values)
    qc, dH_qc = experiment.plot_entropy_derivative(qs, dH)
    print("Critical point q* detected at:", qc)

    return qs, S_values, H_values, DKL_values, dH


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

    S_values, H_values, DKL_values = experiment.sweep(qs)
    experiment.plot_full_results(qs, S_values, H_values, DKL_values)

    dH = experiment.compute_derivative(qs, H_values)
    qc, _ = experiment.plot_entropy_derivative(qs, dH)

    print(f"Entropy critical point under targeted attack: q* = {qc:.4f}")



if __name__ == "__main__":
    run_default()
