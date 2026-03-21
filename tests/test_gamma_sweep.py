import numpy as np
import pytest

from runner.gamma_sweep import (
    GammaSweepExperiment,
    _detect_targeted_onset,
    _null_baseline_mu_sigma,
)
from core.graph_model import GraphModel


# ---------------------------------------------------------------------------
# GammaSweepExperiment._detect_baseline_break
# ---------------------------------------------------------------------------

class TestDetectBaselineBreak:
    @pytest.fixture
    def sweep(self):
        return GammaSweepExperiment(n=100, gammas=[2.5], seeds=[0])

    def test_returns_nan_when_baseline_too_sparse(self, sweep):
        qs = np.linspace(0, 0.9, 6)
        dkl = np.ones(5)
        result = sweep._detect_baseline_break(qs, dkl, q0=0.15)
        assert np.isnan(result)

    def test_flat_signal_no_trigger(self, sweep):
        qs = np.linspace(0, 0.9, 100)
        dkl = np.ones(99)
        result = sweep._detect_baseline_break(qs, dkl, q0=0.15)
        assert np.isnan(result)

    def test_spike_beyond_window_triggers(self, sweep):
        qs = np.linspace(0, 0.9, 100)
        dkl = np.ones(99) * 0.01
        dkl[80] = 100.0
        result = sweep._detect_baseline_break(qs, dkl, q0=0.15)
        assert np.isfinite(result)
        assert result > 0.15

    def test_trigger_is_first_exceedance_not_max(self, sweep):
        qs = np.linspace(0, 0.9, 100)
        dkl = np.ones(99) * 0.01
        dkl[50] = 50.0
        dkl[70] = 100.0
        result = sweep._detect_baseline_break(qs, dkl, q0=0.15)
        qs_mid = 0.5 * (qs[:-1] + qs[1:])
        assert result == pytest.approx(qs_mid[50], rel=1e-6)

    def test_sigma_zero_flat_baseline_no_spurious_trigger(self, sweep):
        """Flat baseline → sigma=0 → threshold=mu. Values equal to mu should NOT trigger."""
        qs = np.linspace(0, 0.9, 100)
        dkl = np.ones(99) * 1.0
        result = sweep._detect_baseline_break(qs, dkl, q0=0.15, z=2.0)
        assert np.isnan(result)

    def test_sigma_zero_any_increase_triggers(self, sweep):
        """Flat baseline → sigma=0 → threshold=mu. Any value > mu beyond window triggers."""
        qs = np.linspace(0, 0.9, 100)
        dkl = np.ones(99) * 1.0
        dkl[60] = 1.001
        result = sweep._detect_baseline_break(qs, dkl, q0=0.15, z=0.0)
        assert np.isfinite(result)

    def test_no_trigger_inside_baseline_window(self, sweep):
        """Exceedance inside the baseline window (q_mid <= q0) must not trigger."""
        qs = np.linspace(0, 0.9, 100)
        dkl = np.ones(99) * 0.01
        qs_mid = 0.5 * (qs[:-1] + qs[1:])
        inside = np.where(qs_mid <= 0.15)[0]
        dkl[inside[-1]] = 999.0
        result = sweep._detect_baseline_break(qs, dkl, q0=0.15)
        assert np.isnan(result)

    def test_dkl_length_mismatch_raises_value_error(self, sweep):
        """_detect_baseline_break must raise ValueError when dkl length != len(qs)-1."""
        qs = np.linspace(0, 0.9, 100)
        dkl_wrong = np.ones(50)
        with pytest.raises(ValueError, match="dkl length"):
            sweep._detect_baseline_break(qs, dkl_wrong, q0=0.15)

    def test_returns_nan_when_no_exceedance_beyond_window(self, sweep):
        qs = np.linspace(0, 0.9, 100)
        dkl = np.zeros(99)
        result = sweep._detect_baseline_break(qs, dkl, q0=0.15)
        assert np.isnan(result)


# ---------------------------------------------------------------------------
# _detect_targeted_onset
# ---------------------------------------------------------------------------

class TestDetectTargetedOnset:
    def test_empty_inputs_return_none(self):
        q_warn, mu0, sigma0, thresh = _detect_targeted_onset(
            np.array([]), np.array([])
        )
        assert q_warn is None
        assert np.isnan(mu0)

    def test_size_mismatch_raises(self):
        with pytest.raises(ValueError):
            _detect_targeted_onset(np.array([0.1, 0.2]), np.array([0.5]))

    def test_sigma_zero_threshold_equals_mu(self):
        """When baseline is flat, sigma=0 → threshold=mu. First point > mu triggers."""
        qs_mid = np.linspace(0.01, 0.9, 50)
        signal = np.ones(50) * 0.5
        signal[5] = 0.6
        q_warn, mu0, sigma0, thresh = _detect_targeted_onset(
            qs_mid, signal, n_baseline=3, z=2.0
        )
        assert sigma0 == pytest.approx(0.0, abs=1e-10)
        assert thresh == pytest.approx(mu0, abs=1e-10)
        assert q_warn is not None

    def test_no_exceedance_returns_none(self):
        qs_mid = np.linspace(0.01, 0.9, 50)
        signal = np.ones(50) * 0.5
        q_warn, _, _, _ = _detect_targeted_onset(qs_mid, signal, n_baseline=3, z=2.0)
        assert q_warn is None

    def test_exceedance_returns_first_crossing(self):
        qs_mid = np.linspace(0.01, 0.9, 50)
        signal = np.zeros(50)
        signal[10] = 10.0
        signal[20] = 20.0
        q_warn, _, _, _ = _detect_targeted_onset(qs_mid, signal, n_baseline=3, z=2.0)
        assert q_warn == pytest.approx(qs_mid[10], rel=1e-6)

    def test_precomputed_mu_sigma_used(self):
        qs_mid = np.linspace(0.01, 0.9, 50)
        signal = np.ones(50) * 5.0
        q_warn, mu0, sigma0, thresh = _detect_targeted_onset(
            qs_mid, signal, n_baseline=3, z=2.0, mu0=0.0, sigma0=1.0
        )
        assert mu0 == pytest.approx(0.0)
        assert sigma0 == pytest.approx(1.0)
        assert thresh == pytest.approx(2.0)
        assert q_warn is not None

    def test_n_baseline_larger_than_signal_clamped(self):
        qs_mid = np.linspace(0.01, 0.9, 5)
        signal = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        q_warn, _, _, _ = _detect_targeted_onset(qs_mid, signal, n_baseline=100)
        assert q_warn is None


# ---------------------------------------------------------------------------
# _null_baseline_mu_sigma
# ---------------------------------------------------------------------------

class TestNullBaselineMuSigma:
    def test_returns_nan_for_empty_graph(self):
        import networkx as nx
        from core.graph_model import GraphModel
        model = GraphModel.__new__(GraphModel)
        model.G = nx.Graph()
        model.n = 0
        model.gamma = 2.5
        model.P0 = np.array([1.0])
        qs = np.linspace(0, 0.9, 10)
        mu, sigma = _null_baseline_mu_sigma(model, qs, alpha=0.2)
        assert np.isnan(mu)
        assert np.isnan(sigma)

    def test_returns_nan_for_single_q(self):
        np.random.seed(0)
        model = GraphModel(n=100, gamma=2.5)
        qs = np.array([0.0])
        mu, sigma = _null_baseline_mu_sigma(model, qs, alpha=0.2)
        assert np.isnan(mu)
        assert np.isnan(sigma)

    def test_no_damage_signal_is_zero(self):
        """No-damage control: all Pq are identical → successive KL = 0 → mu=sigma=0."""
        np.random.seed(0)
        model = GraphModel(n=200, gamma=2.5)
        qs = np.linspace(0, 0.9, 20)
        mu, sigma = _null_baseline_mu_sigma(model, qs, alpha=0.2, n_baseline=3)
        assert mu == pytest.approx(0.0, abs=1e-10)
        assert sigma == pytest.approx(0.0, abs=1e-10)

    def test_returns_finite_values_for_valid_graph(self):
        np.random.seed(1)
        model = GraphModel(n=200, gamma=2.5)
        qs = np.linspace(0, 0.9, 20)
        mu, sigma = _null_baseline_mu_sigma(model, qs, alpha=0.2)
        assert np.isfinite(mu)
        assert np.isfinite(sigma)


# ---------------------------------------------------------------------------
# GammaSweepExperiment initialisation
# ---------------------------------------------------------------------------

class TestGammaSweepInit:
    def test_invalid_graph_model_raises(self):
        with pytest.raises(ValueError, match="graph_model"):
            GammaSweepExperiment(graph_model="badmodel")

    def test_valid_chunglu(self):
        sweep = GammaSweepExperiment(graph_model="chunglu")
        assert sweep.graph_model == "chunglu"

    def test_valid_config(self):
        sweep = GammaSweepExperiment(graph_model="config")
        assert sweep.graph_model == "config"

    def test_default_seeds(self):
        sweep = GammaSweepExperiment()
        assert sweep.seeds == list(range(40))

    def test_custom_gammas(self):
        sweep = GammaSweepExperiment(gammas=[2.5, 2.7])
        assert list(sweep.gammas) == pytest.approx([2.5, 2.7])

    def test_alpha_stored(self):
        sweep = GammaSweepExperiment(alpha=0.3)
        assert sweep.alpha == pytest.approx(0.3)

    def test_z_stored(self):
        sweep = GammaSweepExperiment(z=3.0)
        assert sweep.z == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# _median_iqr (internal helper via run output)
# ---------------------------------------------------------------------------

class TestMedianIQR:
    """Test _median_iqr indirectly by running a minimal sweep and checking output shape."""

    def test_single_seed_iqr_is_zero(self):
        np.random.seed(0)
        sweep = GammaSweepExperiment(
            n=500, gammas=[2.5], seeds=[0], alpha=0.2, z=2.0
        )
        rows, _ = sweep.run()
        assert len(rows) == 1
        row = rows[0]
        targeted_collapse_iqr = row[32]
        assert targeted_collapse_iqr == pytest.approx(0.0, abs=1e-10)

    def test_all_nan_seeds_produce_nan_median(self):
        """If no warnings detected, median/IQR should be nan."""
        np.random.seed(0)
        sweep = GammaSweepExperiment(
            n=500, gammas=[2.9], seeds=[0], alpha=0.2, z=100.0
        )
        rows, _ = sweep.run()
        row = rows[0]
        random_warn_med = row[27]
        assert np.isnan(random_warn_med)
