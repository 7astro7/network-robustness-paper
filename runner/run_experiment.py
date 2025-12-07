import numpy as np
from models.graph_model import GraphModel
from models.failure_model import FailureModel
from experiment.experiment import Experiment


def run_default():
    """
    Run a standard robustness experiment on a Chung–Lu power-law graph
    and return S(q), H(q), and D_KL(q).
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
    experiment.plot_full_results(qs, S_values, H_values, DKL_values)

    return qs, S_values, H_values, DKL_values


if __name__ == "__main__":
    run_default()
