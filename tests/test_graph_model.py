import numpy as np
import networkx as nx
import pytest

from core.graph_model import GraphModel, ConfigurationModel


# ---------------------------------------------------------------------------
# GraphModel
# ---------------------------------------------------------------------------

class TestGraphModel:
    @pytest.fixture
    def model(self):
        np.random.seed(0)
        return GraphModel(n=200, gamma=2.5)

    def test_graph_has_correct_node_count(self, model):
        assert model.G.number_of_nodes() == 200

    def test_graph_is_undirected(self, model):
        assert isinstance(model.G, nx.Graph)

    def test_graph_has_no_selfloops(self, model):
        assert len(list(nx.selfloop_edges(model.G))) == 0

    def test_p0_is_valid_pmf(self, model):
        assert model.P0.sum() == pytest.approx(1.0, abs=1e-10)
        assert np.all(model.P0 >= 0.0)

    def test_p0_length_matches_k_max(self, model):
        k_max = max(dict(model.G.degree()).values())
        assert len(model.P0) == k_max + 1

    # _degree_distribution ---------------------------------------------------

    def test_degree_distribution_sums_to_one(self, model):
        P = model._degree_distribution(model.G)
        assert P.sum() == pytest.approx(1.0, abs=1e-10)

    def test_degree_distribution_fixed_support(self, model):
        k_max_actual = int(max(d for _, d in model.G.degree()))
        k_max = k_max_actual + 20
        P = model._degree_distribution(model.G, k_max=k_max)
        assert len(P) == k_max + 1

    def test_degree_distribution_eps_smoothing_no_zeros(self, model):
        k_max = int(max(d for _, d in model.G.degree())) + 20
        P = model._degree_distribution(model.G, k_max=k_max, eps=1e-12)
        assert np.all(P > 0.0)

    def test_degree_distribution_eps_still_sums_to_one(self, model):
        k_max = int(max(d for _, d in model.G.degree())) + 20
        P = model._degree_distribution(model.G, k_max=k_max, eps=1e-12)
        assert P.sum() == pytest.approx(1.0, abs=1e-10)

    def test_degree_distribution_empty_graph(self, model):
        G_empty = nx.Graph()
        P = model._degree_distribution(G_empty, k_max=5)
        assert len(P) == 6
        assert P[0] == pytest.approx(1.0)

    def test_degree_distribution_no_eps_may_have_zeros(self, model):
        k_max = int(max(d for _, d in model.G.degree())) + 100
        P = model._degree_distribution(model.G, k_max=k_max, eps=0.0)
        assert np.any(P == 0.0)

    def test_degree_distribution_k_max_below_actual_max_grows(self, model):
        k_max_actual = int(max(d for _, d in model.G.degree()))
        k_max_requested = max(0, k_max_actual - 10)
        P = model._degree_distribution(model.G, k_max=k_max_requested, eps=0.0)
        assert len(P) >= k_max_requested + 1
        assert P.sum() == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# ConfigurationModel
# ---------------------------------------------------------------------------

class TestConfigurationModel:
    @pytest.fixture
    def cm(self):
        np.random.seed(42)
        return ConfigurationModel(n=200, gamma=2.5)

    def test_graph_is_simple(self, cm):
        assert isinstance(cm.G, nx.Graph)
        assert len(list(nx.selfloop_edges(cm.G))) == 0

    def test_graph_is_connected(self, cm):
        assert nx.is_connected(cm.G)

    def test_graph_nonempty(self, cm):
        assert cm.G.number_of_nodes() > 0

    def test_all_degrees_at_least_k_min(self, cm):
        degrees = [d for _, d in cm.G.degree()]
        assert all(d >= cm.k_min for d in degrees)

    def test_p0_is_valid_pmf(self, cm):
        assert cm.P0.sum() == pytest.approx(1.0, abs=1e-10)
        assert np.all(cm.P0 >= 0.0)

    def test_degree_distribution_inherits_correctly(self, cm):
        P = cm._degree_distribution(cm.G, k_max=20, eps=1e-12)
        assert P.sum() == pytest.approx(1.0, abs=1e-10)
        assert len(P) == 21

    def test_degree_heterogeneity(self, cm):
        degrees = [d for _, d in cm.G.degree()]
        assert max(degrees) > min(degrees)

    def test_different_seeds_produce_different_graphs(self):
        np.random.seed(0)
        cm1 = ConfigurationModel(n=100, gamma=2.5)
        np.random.seed(99)
        cm2 = ConfigurationModel(n=100, gamma=2.5)
        assert set(cm1.G.edges()) != set(cm2.G.edges())

    def test_lcc_extraction_n_actual_leq_n_requested(self, cm):
        """After LCC extraction, actual node count may be <= requested n."""
        assert cm.G.number_of_nodes() <= cm.n

    def test_self_n_attribute_reflects_requested_not_actual(self, cm):
        """self.n stores the requested n, not the post-LCC count."""
        assert cm.n == 200

    def test_degree_sum_is_even(self):
        """Configuration model requires even degree sum; parity fix must guarantee this."""
        for seed in range(10):
            np.random.seed(seed)
            cm = ConfigurationModel(n=50, gamma=2.5)
            degrees = [d for _, d in cm.G.degree()]
            assert sum(degrees) % 2 == 0

    def test_no_selfloops_after_construction(self):
        for seed in range(5):
            np.random.seed(seed)
            cm = ConfigurationModel(n=100, gamma=2.5)
            assert len(list(nx.selfloop_edges(cm.G))) == 0

    def test_no_parallel_edges_after_construction(self):
        for seed in range(5):
            np.random.seed(seed)
            cm = ConfigurationModel(n=100, gamma=2.5)
            assert isinstance(cm.G, nx.Graph)
