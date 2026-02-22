import numpy as np
import networkx as nx
import pytest

from core.experiment import Experiment
from core.graph_model import GraphModel
from core.failure_model import RandomFailure, TargetedFailure


@pytest.fixture
def small_model():
    np.random.seed(0)
    return GraphModel(n=100, gamma=2.5)


@pytest.fixture
def random_experiment(small_model):
    return Experiment(small_model, RandomFailure())


@pytest.fixture
def targeted_experiment(small_model):
    return Experiment(small_model, TargetedFailure())


@pytest.fixture
def qs():
    return np.linspace(0, 0.9, 20)


# ---------------------------------------------------------------------------
# Experiment.sweep
# ---------------------------------------------------------------------------

class TestSweep:
    def test_output_lengths_match_qs(self, random_experiment, qs):
        S, H, Pq = random_experiment.sweep(qs)
        assert len(S) == len(qs)
        assert len(H) == len(qs)
        assert len(Pq) == len(qs)

    def test_S_values_in_unit_interval(self, random_experiment, qs):
        S, _, _ = random_experiment.sweep(qs)
        assert all(0.0 <= s <= 1.0 for s in S)

    def test_S_at_q0_geq_S_at_q_high(self, random_experiment):
        S, _, _ = random_experiment.sweep(np.array([0.0, 0.9]))
        assert S[0] >= S[1]

    def test_H_values_nonnegative(self, random_experiment, qs):
        _, H, _ = random_experiment.sweep(qs)
        assert all(h >= 0.0 for h in H)

    def test_Pq_each_sums_to_one(self, random_experiment, qs):
        _, _, Pq = random_experiment.sweep(qs)
        for P in Pq:
            assert P.sum() == pytest.approx(1.0, abs=1e-9)

    def test_Pq_fixed_support_length_constant(self, random_experiment, qs):
        _, _, Pq = random_experiment.sweep(qs)
        lengths = [len(P) for P in Pq]
        assert len(set(lengths)) == 1

    def test_Pq_all_positive_with_eps(self, random_experiment, qs):
        _, _, Pq = random_experiment.sweep(qs)
        for P in Pq:
            assert np.all(P > 0.0)

    def test_targeted_S_decreases_monotonically_on_average(self, targeted_experiment):
        qs = np.linspace(0, 0.9, 10)
        S, _, _ = targeted_experiment.sweep(qs)
        assert S[0] >= S[-1]

    def test_single_q_sweep(self, random_experiment):
        S, H, Pq = random_experiment.sweep(np.array([0.5]))
        assert len(S) == 1
        assert len(H) == 1
        assert len(Pq) == 1


# ---------------------------------------------------------------------------
# Experiment.successive_kl
# ---------------------------------------------------------------------------

class TestSuccessiveKL:
    def test_output_length(self, random_experiment, qs):
        _, _, Pq = random_experiment.sweep(qs)
        dkl = random_experiment.successive_kl(Pq)
        assert len(dkl) == len(qs) - 1

    def test_values_nonnegative(self, random_experiment, qs):
        _, _, Pq = random_experiment.sweep(qs)
        dkl = random_experiment.successive_kl(Pq)
        assert np.all(dkl >= 0.0)

    def test_identical_distributions_zero(self, random_experiment):
        P = np.array([0.25, 0.25, 0.25, 0.25])
        Pq = [P.copy() for _ in range(5)]
        dkl = random_experiment.successive_kl(Pq)
        np.testing.assert_allclose(dkl, 0.0, atol=1e-10)

    def test_returns_ndarray(self, random_experiment, qs):
        _, _, Pq = random_experiment.sweep(qs)
        dkl = random_experiment.successive_kl(Pq)
        assert isinstance(dkl, np.ndarray)


# ---------------------------------------------------------------------------
# Experiment.ewma
# ---------------------------------------------------------------------------

class TestEWMA:
    def test_output_length_matches_input(self, random_experiment):
        signal = np.arange(10, dtype=float)
        out = random_experiment.ewma(signal, alpha=0.2)
        assert len(out) == len(signal)

    def test_first_value_unchanged(self, random_experiment):
        signal = np.array([5.0, 1.0, 2.0, 3.0])
        out = random_experiment.ewma(signal, alpha=0.2)
        assert out[0] == pytest.approx(5.0)

    def test_constant_signal_unchanged(self, random_experiment):
        signal = np.ones(10) * 3.0
        out = random_experiment.ewma(signal, alpha=0.3)
        np.testing.assert_allclose(out, 3.0, atol=1e-10)

    def test_alpha_one_is_identity(self, random_experiment):
        signal = np.array([1.0, 2.0, 3.0, 4.0])
        out = random_experiment.ewma(signal, alpha=1.0)
        np.testing.assert_allclose(out, signal, atol=1e-10)

    def test_alpha_zero_holds_first_value(self, random_experiment):
        signal = np.array([7.0, 1.0, 2.0, 3.0])
        out = random_experiment.ewma(signal, alpha=0.0)
        np.testing.assert_allclose(out, 7.0, atol=1e-10)

    def test_empty_signal_returns_empty(self, random_experiment):
        out = random_experiment.ewma(np.array([]), alpha=0.2)
        assert len(out) == 0

    def test_smoothing_reduces_variance(self, random_experiment):
        rng = np.random.default_rng(42)
        signal = rng.normal(0, 1, 50)
        out = random_experiment.ewma(signal, alpha=0.1)
        assert out.std() < signal.std()

    def test_second_value_formula(self, random_experiment):
        signal = np.array([2.0, 4.0])
        alpha = 0.3
        out = random_experiment.ewma(signal, alpha=alpha)
        expected = alpha * 4.0 + (1 - alpha) * 2.0
        assert out[1] == pytest.approx(expected)

    def test_single_element_returns_unchanged(self, random_experiment):
        signal = np.array([3.14])
        out = random_experiment.ewma(signal, alpha=0.2)
        assert len(out) == 1
        assert out[0] == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# Edge cases: Pq support length and successive_kl known value
# ---------------------------------------------------------------------------

class TestSweepEdgeCases:
    def test_pq_support_length_equals_k_max0_plus_one(self, random_experiment, qs):
        k_max0 = max(d for _, d in random_experiment.graph_model.G.degree())
        _, _, Pq = random_experiment.sweep(qs)
        assert len(Pq[0]) == k_max0 + 1

    def test_successive_kl_known_value(self, random_experiment):
        P = np.array([0.5, 0.5])
        Q = np.array([0.25, 0.75])
        Pq = [Q, P]
        dkl = random_experiment.successive_kl(Pq)
        expected = 0.5 * np.log2(0.5 / 0.25) + 0.5 * np.log2(0.5 / 0.75)
        assert dkl[0] == pytest.approx(expected, rel=1e-6)

    def test_targeted_sweep_pq_sums_to_one(self, targeted_experiment):
        qs = np.linspace(0, 0.9, 10)
        _, _, Pq = targeted_experiment.sweep(qs)
        for P in Pq:
            assert P.sum() == pytest.approx(1.0, abs=1e-9)

    def test_targeted_sweep_pq_fixed_support(self, targeted_experiment):
        qs = np.linspace(0, 0.9, 10)
        _, _, Pq = targeted_experiment.sweep(qs)
        lengths = [len(P) for P in Pq]
        assert len(set(lengths)) == 1


# ---------------------------------------------------------------------------
# Integration test: full pipeline detects warning before collapse
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    def test_warning_before_collapse_random_failure(self):
        np.random.seed(0)
        model = GraphModel(n=500, gamma=2.5)
        experiment = Experiment(model, RandomFailure())
        qs = np.linspace(0, 0.9, 100)

        S, _, Pq = experiment.sweep(qs)
        dkl = experiment.successive_kl(Pq)
        smoothed = experiment.ewma(dkl, alpha=0.2)
        qs_mid = 0.5 * (qs[:-1] + qs[1:])

        baseline_mask = qs_mid <= 0.15
        mu0 = smoothed[baseline_mask].mean()
        sigma0 = smoothed[baseline_mask].std()
        threshold = mu0 + 2 * sigma0

        warn_idx = next(
            (i for i, v in enumerate(smoothed) if qs_mid[i] > 0.15 and v > threshold),
            None,
        )
        collapse_idx = next((i for i, s in enumerate(S) if s < 0.1), None)

        assert warn_idx is not None, "No warning detected"
        assert collapse_idx is not None, "No collapse detected"
        assert qs_mid[warn_idx] < qs[collapse_idx], (
            f"Warning q={qs_mid[warn_idx]:.3f} not before collapse q={qs[collapse_idx]:.3f}"
        )
