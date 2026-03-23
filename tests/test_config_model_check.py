import numpy as np
import pytest

from runner.config_model_check import detect_baseline_break, run_config_model_experiment


# ---------------------------------------------------------------------------
# detect_baseline_break (config_model_check version)
# ---------------------------------------------------------------------------

class TestDetectBaselineBreakConfigModel:
    def test_empty_qs_returns_none(self):
        result = detect_baseline_break(np.array([]), np.array([]))
        assert result is None

    def test_empty_signal_returns_none(self):
        result = detect_baseline_break(np.array([0.0, 0.1]), np.array([]))
        assert result is None

    def test_size_mismatch_raises(self):
        qs = np.linspace(0, 0.9, 10)
        signal = np.ones(5)
        with pytest.raises(ValueError):
            detect_baseline_break(qs, signal)

    def test_flat_signal_returns_none(self):
        qs = np.linspace(0, 0.9, 100)
        signal = np.ones(99)
        result = detect_baseline_break(qs, signal, z=2.0, baseline_frac=0.15)
        assert result is None

    def test_spike_beyond_window_triggers(self):
        qs = np.linspace(0, 0.9, 100)
        signal = np.ones(99) * 0.01
        signal[80] = 100.0
        result = detect_baseline_break(qs, signal, z=2.0, baseline_frac=0.15)
        assert result is not None
        assert result > 0.15

    def test_returns_first_exceedance_not_max(self):
        qs = np.linspace(0, 0.9, 100)
        signal = np.ones(99) * 0.01
        signal[50] = 50.0
        signal[70] = 100.0
        result = detect_baseline_break(qs, signal, z=2.0, baseline_frac=0.15)
        qs_mid = (qs[:-1] + qs[1:]) / 2.0
        search_mask = qs_mid > 0.15
        search_qs = qs_mid[search_mask]
        search_vals = signal[search_mask]
        mu0 = signal[qs_mid <= 0.15].mean()
        sigma0 = signal[qs_mid <= 0.15].std(ddof=0)
        thresh = mu0 + 2.0 * sigma0
        first_idx = np.where(search_vals > thresh)[0][0]
        assert result == pytest.approx(float(search_qs[first_idx]), rel=1e-6)

    def test_sigma_zero_flat_baseline_no_trigger(self):
        qs = np.linspace(0, 0.9, 100)
        signal = np.ones(99)
        result = detect_baseline_break(qs, signal, z=2.0, baseline_frac=0.15)
        assert result is None

    def test_sigma_zero_any_increase_triggers_with_z0(self):
        qs = np.linspace(0, 0.9, 100)
        signal = np.ones(99)
        signal[60] = 1.001
        result = detect_baseline_break(qs, signal, z=0.0, baseline_frac=0.15)
        assert result is not None

    def test_exceedance_inside_baseline_window_does_not_trigger(self):
        qs = np.linspace(0, 0.9, 100)
        signal = np.ones(99) * 0.01
        qs_mid = (qs[:-1] + qs[1:]) / 2.0
        inside = np.where(qs_mid <= 0.15)[0]
        signal[inside[-1]] = 999.0
        result = detect_baseline_break(qs, signal, z=2.0, baseline_frac=0.15)
        assert result is None

    def test_fewer_than_two_baseline_points_returns_none(self):
        qs = np.linspace(0.5, 0.9, 10)
        signal = np.ones(9)
        result = detect_baseline_break(qs, signal, z=2.0, baseline_frac=0.15)
        assert result is None


# ---------------------------------------------------------------------------
# run_config_model_experiment
# ---------------------------------------------------------------------------

class TestRunConfigModelExperiment:
    @pytest.fixture
    def result(self):
        return run_config_model_experiment(gamma=2.5, seed=0, n=500, alpha=0.2, z=2.0, num_q=50)

    def test_output_has_required_keys(self, result):
        for key in ["seed", "gamma", "q_warn", "q_collapse", "delta_warn", "S_final"]:
            assert key in result

    def test_gamma_stored_correctly(self, result):
        assert result["gamma"] == pytest.approx(2.5)

    def test_seed_stored_correctly(self, result):
        assert result["seed"] == 0

    def test_S_final_in_unit_interval(self, result):
        assert 0.0 <= result["S_final"] <= 1.0

    def test_q_warn_before_q_collapse_when_both_finite(self, result):
        q_warn = result["q_warn"]
        q_collapse = result["q_collapse"]
        if np.isfinite(q_warn) and np.isfinite(q_collapse):
            assert q_warn < q_collapse

    def test_delta_warn_consistent(self, result):
        q_warn = result["q_warn"]
        q_collapse = result["q_collapse"]
        delta = result["delta_warn"]
        if np.isfinite(q_warn) and np.isfinite(q_collapse):
            assert delta == pytest.approx(q_collapse - q_warn, rel=1e-6)
        else:
            assert np.isnan(delta)

    def test_reproducible_with_same_seed(self):
        r1 = run_config_model_experiment(gamma=2.5, seed=42, n=300, num_q=30)
        r2 = run_config_model_experiment(gamma=2.5, seed=42, n=300, num_q=30)
        assert r1["q_warn"] == r2["q_warn"] or (np.isnan(r1["q_warn"]) and np.isnan(r2["q_warn"]))
        assert r1["q_collapse"] == r2["q_collapse"] or (
            np.isnan(r1["q_collapse"]) and np.isnan(r2["q_collapse"])
        )

    def test_different_seeds_may_differ(self):
        r1 = run_config_model_experiment(gamma=2.5, seed=0, n=500, num_q=50)
        r2 = run_config_model_experiment(gamma=2.5, seed=99, n=500, num_q=50)
        assert r1["S_final"] != r2["S_final"] or r1["q_collapse"] != r2["q_collapse"]
