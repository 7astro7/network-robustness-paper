import numpy as np
import networkx as nx
import pytest

from core.metrics import Metrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def triangle():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    return G


@pytest.fixture
def path4():
    return nx.path_graph(4)


@pytest.fixture
def empty_graph():
    return nx.Graph()


@pytest.fixture
def uniform_pmf():
    P = np.array([0.25, 0.25, 0.25, 0.25])
    return P


# ---------------------------------------------------------------------------
# giant_component_fraction
# ---------------------------------------------------------------------------

class TestGiantComponentFraction:
    def test_empty_graph_returns_zero(self, empty_graph):
        assert Metrics.giant_component_fraction(empty_graph) == 0.0

    def test_fully_connected(self, triangle):
        assert Metrics.giant_component_fraction(triangle) == pytest.approx(1.0)

    def test_two_components(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (2, 3)])
        assert Metrics.giant_component_fraction(G) == pytest.approx(0.5)

    def test_single_node(self):
        G = nx.Graph()
        G.add_node(0)
        assert Metrics.giant_component_fraction(G) == pytest.approx(1.0)

    def test_path_graph(self, path4):
        assert Metrics.giant_component_fraction(path4) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# degree_entropy
# ---------------------------------------------------------------------------

class TestDegreeEntropy:
    def test_empty_graph_returns_zero(self, empty_graph):
        assert Metrics.degree_entropy(empty_graph) == 0.0

    def test_regular_graph_entropy(self):
        G = nx.cycle_graph(4)
        H = Metrics.degree_entropy(G)
        assert H == pytest.approx(0.0, abs=1e-10)

    def test_entropy_nonnegative(self, path4):
        assert Metrics.degree_entropy(path4) >= 0.0

    def test_entropy_positive_for_heterogeneous(self):
        G = nx.star_graph(5)
        assert Metrics.degree_entropy(G) > 0.0


# ---------------------------------------------------------------------------
# kl_divergence
# ---------------------------------------------------------------------------

class TestKLDivergence:
    def test_identical_distributions_zero(self, uniform_pmf):
        assert Metrics.kl_divergence(uniform_pmf, uniform_pmf) == pytest.approx(0.0, abs=1e-10)

    def test_shape_mismatch_raises(self):
        P = np.array([0.5, 0.5])
        Q = np.array([0.25, 0.25, 0.5])
        with pytest.raises(ValueError):
            Metrics.kl_divergence(P, Q)

    def test_known_value(self):
        P = np.array([1.0, 0.0])
        Q = np.array([0.5, 0.5])
        expected = 1.0 * np.log2(1.0 / 0.5)
        assert Metrics.kl_divergence(P, Q) == pytest.approx(expected, rel=1e-6)

    def test_nonnegative(self):
        rng = np.random.default_rng(0)
        P = rng.dirichlet(np.ones(5))
        Q = rng.dirichlet(np.ones(5))
        assert Metrics.kl_divergence(P, Q) >= 0.0


# ---------------------------------------------------------------------------
# js_divergence
# ---------------------------------------------------------------------------

class TestJSDivergence:
    def test_identical_distributions_zero(self, uniform_pmf):
        assert Metrics.js_divergence(uniform_pmf, uniform_pmf) == pytest.approx(0.0, abs=1e-10)

    def test_shape_mismatch_raises(self):
        P = np.array([0.5, 0.5])
        Q = np.array([1 / 3, 1 / 3, 1 / 3])
        with pytest.raises(ValueError):
            Metrics.js_divergence(P, Q)

    def test_symmetric(self):
        P = np.array([0.7, 0.3])
        Q = np.array([0.4, 0.6])
        assert Metrics.js_divergence(P, Q) == pytest.approx(Metrics.js_divergence(Q, P), rel=1e-10)

    def test_bounded_by_one_bit(self):
        P = np.array([1.0, 0.0])
        Q = np.array([0.0, 1.0])
        assert Metrics.js_divergence(P, Q) <= 1.0 + 1e-10

    def test_nonnegative(self):
        rng = np.random.default_rng(1)
        P = rng.dirichlet(np.ones(4))
        Q = rng.dirichlet(np.ones(4))
        assert Metrics.js_divergence(P, Q) >= 0.0


# ---------------------------------------------------------------------------
# successive_js
# ---------------------------------------------------------------------------

class TestSuccessiveJS:
    def test_length(self):
        Pqs = [np.array([0.5, 0.5]), np.array([0.6, 0.4]), np.array([0.8, 0.2])]
        result = Metrics.successive_js(Pqs)
        assert len(result) == 2

    def test_identical_sequence_zero(self):
        P = np.array([0.3, 0.4, 0.3])
        Pqs = [P.copy(), P.copy(), P.copy()]
        result = Metrics.successive_js(Pqs)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_nonnegative(self):
        rng = np.random.default_rng(2)
        Pqs = [rng.dirichlet(np.ones(5)) for _ in range(6)]
        result = Metrics.successive_js(Pqs)
        assert np.all(result >= 0.0)


# ---------------------------------------------------------------------------
# detect_warning
# ---------------------------------------------------------------------------

class TestDetectWarning:
    def test_returns_index(self):
        signal = np.array([0.1, 0.2, 0.3, 0.9, 0.4])
        idx = Metrics.detect_warning(signal, percentile=80)
        assert idx is not None
        assert signal[idx] >= np.percentile(signal, 80)

    def test_monotone_increasing_returns_early(self):
        signal = np.arange(10, dtype=float)
        idx = Metrics.detect_warning(signal, percentile=50)
        assert idx is not None
        assert idx < len(signal)

    def test_returns_first_crossing_not_max(self):
        signal = np.array([0.1, 0.9, 0.5, 1.0, 0.2])
        idx = Metrics.detect_warning(signal, percentile=70)
        assert idx == 1

    def test_all_below_percentile_still_returns(self):
        signal = np.array([1.0, 1.0, 1.0, 1.0])
        idx = Metrics.detect_warning(signal, percentile=90)
        assert idx is not None


# ---------------------------------------------------------------------------
# detect_baseline_deviation
# ---------------------------------------------------------------------------

class TestDetectBaselineDeviation:
    def test_flat_signal_no_warning(self):
        signal = np.ones(20)
        assert Metrics.detect_baseline_deviation(signal) is None

    def test_spike_triggers_warning(self):
        signal = np.concatenate([np.ones(10), [100.0], np.ones(9)])
        idx = Metrics.detect_baseline_deviation(signal, baseline_frac=0.3, k=2.0)
        assert idx is not None
        assert idx >= 3

    def test_warning_after_baseline_window(self):
        signal = np.concatenate([np.ones(10), np.linspace(1, 10, 10)])
        idx = Metrics.detect_baseline_deviation(signal, baseline_frac=0.3, k=2.0)
        n0 = max(2, int(len(signal) * 0.3))
        assert idx is None or idx >= n0

    def test_sigma_zero_flat_baseline_no_spurious_trigger(self):
        signal = np.concatenate([np.ones(10), np.ones(10)])
        idx = Metrics.detect_baseline_deviation(signal, baseline_frac=0.5, k=2.0)
        assert idx is None

    def test_sigma_zero_any_increase_triggers(self):
        signal = np.concatenate([np.ones(10), [1.001], np.ones(9)])
        idx = Metrics.detect_baseline_deviation(signal, baseline_frac=0.5, k=0.0)
        assert idx is not None


# ---------------------------------------------------------------------------
# detect_persistence
# ---------------------------------------------------------------------------

class TestDetectPersistence:
    def test_flat_signal_no_warning(self):
        signal = np.ones(10)
        assert Metrics.detect_persistence(signal, N=3) is None

    def test_monotone_increase_triggers(self):
        signal = np.arange(10, dtype=float)
        idx = Metrics.detect_persistence(signal, N=3)
        assert idx is not None
        assert idx >= 3

    def test_returns_correct_index(self):
        signal = np.array([1.0, 0.5, 2.0, 3.0, 4.0, 5.0])
        idx = Metrics.detect_persistence(signal, N=3)
        assert idx == 4

    def test_n1_triggers_on_first_increase(self):
        signal = np.array([1.0, 2.0, 0.5, 0.5])
        idx = Metrics.detect_persistence(signal, N=1)
        assert idx == 1

    def test_counter_resets_after_decrease(self):
        signal = np.array([1.0, 2.0, 1.0, 2.0, 3.0, 4.0])
        idx = Metrics.detect_persistence(signal, N=3)
        assert idx == 5

    def test_decrease_prevents_early_trigger(self):
        signal = np.array([1.0, 2.0, 3.0, 1.0, 5.0, 6.0])
        idx = Metrics.detect_persistence(signal, N=3)
        assert idx is None


# ---------------------------------------------------------------------------
# detect_slope
# ---------------------------------------------------------------------------

class TestDetectSlope:
    def test_flat_returns_none(self):
        signal = np.ones(10)
        assert Metrics.detect_slope(signal, m=3) is None

    def test_monotone_increase_triggers(self):
        signal = np.arange(10, dtype=float)
        idx = Metrics.detect_slope(signal, m=3)
        assert idx is not None
        assert idx >= 3

    def test_decreasing_returns_none(self):
        signal = np.arange(10, dtype=float)[::-1]
        assert Metrics.detect_slope(signal, m=3) is None

    def test_m1_triggers_on_first_increase(self):
        signal = np.array([1.0, 2.0, 0.5, 0.5])
        idx = Metrics.detect_slope(signal, m=1)
        assert idx == 1

    def test_interruption_resets(self):
        signal = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0])
        idx = Metrics.detect_slope(signal, m=3)
        assert idx == 6


# ---------------------------------------------------------------------------
# kl_divergence edge cases
# ---------------------------------------------------------------------------

class TestKLDivergenceEdgeCases:
    def test_p_zero_q_zero_skipped(self):
        P = np.array([1.0, 0.0])
        Q = np.array([0.5, 0.0])
        result = Metrics.kl_divergence(P, Q)
        assert np.isfinite(result)
        assert result >= 0.0

    def test_p_zero_q_nonzero_skipped(self):
        P = np.array([1.0, 0.0])
        Q = np.array([0.5, 0.5])
        result = Metrics.kl_divergence(P, Q)
        expected = 1.0 * np.log2(1.0 / 0.5)
        assert result == pytest.approx(expected, rel=1e-6)
