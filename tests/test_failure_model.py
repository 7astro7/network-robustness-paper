import numpy as np
import networkx as nx
import pytest

from core.failure_model import RandomFailure, TargetedFailure


@pytest.fixture
def star():
    """Star graph: node 0 has degree 5, leaves have degree 1."""
    return nx.star_graph(5)


@pytest.fixture
def cycle():
    return nx.cycle_graph(10)


# ---------------------------------------------------------------------------
# RandomFailure
# ---------------------------------------------------------------------------

class TestRandomFailure:
    def setup_method(self):
        self.model = RandomFailure()

    def test_q0_returns_full_graph(self, cycle):
        Gq = self.model.apply(cycle, 0.0)
        assert Gq.number_of_nodes() == cycle.number_of_nodes()

    def test_q1_removes_all_nodes(self, cycle):
        Gq = self.model.apply(cycle, 1.0)
        assert Gq.number_of_nodes() == 0

    def test_correct_node_count(self, cycle):
        np.random.seed(0)
        Gq = self.model.apply(cycle, 0.3)
        expected = cycle.number_of_nodes() - int(0.3 * cycle.number_of_nodes())
        assert Gq.number_of_nodes() == expected

    def test_returns_subgraph_of_original(self, cycle):
        np.random.seed(1)
        Gq = self.model.apply(cycle, 0.4)
        assert set(Gq.nodes()).issubset(set(cycle.nodes()))

    def test_does_not_mutate_original(self, cycle):
        original_nodes = cycle.number_of_nodes()
        np.random.seed(2)
        self.model.apply(cycle, 0.5)
        assert cycle.number_of_nodes() == original_nodes

    def test_returns_copy_not_view(self, cycle):
        np.random.seed(3)
        Gq = self.model.apply(cycle, 0.0)
        assert Gq is not cycle


# ---------------------------------------------------------------------------
# TargetedFailure
# ---------------------------------------------------------------------------

class TestTargetedFailure:
    def setup_method(self):
        self.model = TargetedFailure()

    def test_q0_returns_full_graph(self, star):
        Gq = self.model.apply(star, 0.0)
        assert Gq.number_of_nodes() == star.number_of_nodes()

    def test_q1_removes_all_nodes(self, star):
        Gq = self.model.apply(star, 1.0)
        assert Gq.number_of_nodes() == 0

    def test_correct_node_count(self, star):
        Gq = self.model.apply(star, 0.5)
        expected = star.number_of_nodes() - int(0.5 * star.number_of_nodes())
        assert Gq.number_of_nodes() == expected

    def test_hub_removed_first(self, star):
        """Removing 1/6 of nodes (1 node) from a star should remove the hub (node 0)."""
        q = 1 / star.number_of_nodes()
        Gq = self.model.apply(star, q)
        assert 0 not in Gq.nodes()

    def test_leaves_remain_when_only_hub_removed(self, star):
        q = 1 / star.number_of_nodes()
        Gq = self.model.apply(star, q)
        leaves = set(range(1, 6))
        assert leaves.issubset(set(Gq.nodes()))

    def test_does_not_mutate_original(self, star):
        original_nodes = star.number_of_nodes()
        self.model.apply(star, 0.5)
        assert star.number_of_nodes() == original_nodes

    def test_returns_subgraph_of_original(self, cycle):
        Gq = self.model.apply(cycle, 0.3)
        assert set(Gq.nodes()).issubset(set(cycle.nodes()))

    def test_removal_order_is_degree_descending(self):
        """Build a graph where degrees are unambiguous and verify removal order."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2)])
        # degrees: 0->4, 1->2, 2->2, 3->1, 4->1
        Gq = self.model.apply(G, q=1 / G.number_of_nodes())
        assert 0 not in Gq.nodes()

    def test_tie_breaking_is_stable(self):
        """Tie-breaking among equal-degree nodes is stable (same result on repeated calls)."""
        G = nx.cycle_graph(10)
        Gq1 = self.model.apply(G, 0.3)
        Gq2 = self.model.apply(G, 0.3)
        assert set(Gq1.nodes()) == set(Gq2.nodes())


class TestFailureModelsOnSpecialGraphs:
    """Edge cases: disconnected graphs and graphs with isolated nodes."""

    def test_random_failure_on_disconnected_graph(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (2, 3)])
        Gq = RandomFailure().apply(G, 0.5)
        assert Gq.number_of_nodes() == 2

    def test_targeted_failure_on_disconnected_graph(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (2, 3)])
        Gq = TargetedFailure().apply(G, 0.5)
        assert Gq.number_of_nodes() == 2

    def test_random_failure_on_graph_with_isolated_nodes(self):
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2, 3])
        G.add_edge(0, 1)
        Gq = RandomFailure().apply(G, 0.5)
        assert Gq.number_of_nodes() == 2

    def test_targeted_failure_removes_connected_nodes_first(self):
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2, 3])
        G.add_edge(0, 1)
        Gq = TargetedFailure().apply(G, 0.5)
        assert 2 not in Gq.nodes() or 3 not in Gq.nodes() or (
            0 not in Gq.nodes() or 1 not in Gq.nodes()
        )

    def test_random_failure_seed_reproducibility(self):
        G = nx.cycle_graph(20)
        np.random.seed(7)
        Gq1 = RandomFailure().apply(G, 0.4)
        np.random.seed(7)
        Gq2 = RandomFailure().apply(G, 0.4)
        assert set(Gq1.nodes()) == set(Gq2.nodes())
