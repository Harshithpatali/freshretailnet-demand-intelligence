"""Unit tests for leakage-safe Fourier/STL/spectral research features."""
import numpy as np
import pandas as pd
import pytest

from experiments.forecast_research.forecast_research_benchmark import (
    _spectral_summary,
    origin_features,
    select_ensemble_weights,
)


def test_spectral_diagnostics_detect_weekly_periodicity():
    t = np.arange(56, dtype=float)
    values = 10.0 + np.sin(2 * np.pi * t / 7.0)
    result = _spectral_summary(values)
    assert abs(result["spectral_dominant_period_days"] - 7.0) < 1.0
    assert 0.0 <= result["spectral_weekly_energy_share"] <= 1.0
    assert 0.0 <= result["spectral_entropy"] <= 1.0
    assert result["spectral_weekly_energy_share"] > 0.5


def test_origin_features_ignore_target_and_future_observations():
    dates = pd.date_range("2025-01-01", periods=63, freq="D")
    history = pd.DataFrame({
        "store_id": ["s1"] * 63,
        "product_id": ["p1"] * 63,
        "date": dates,
        "sale_amount": 10.0 + np.sin(2 * np.pi * np.arange(63) / 7.0),
    })
    origin = dates[56]
    targets = pd.DataFrame({
        "store_id": ["s1", "s1"],
        "product_id": ["p1", "p1"],
        "date": [dates[57], dates[58]],
    })
    first = origin_features(history, targets, origin)
    mutated = history.copy()
    mutated.loc[mutated["date"] >= origin, "sale_amount"] = 10000.0
    second = origin_features(mutated, targets, origin)
    feature_columns = [
        "stl_trend_level", "stl_trend_slope_14", "stl_residual_std_28",
        "stl_seasonal_for_weekday", "spectral_dominant_period_days",
        "spectral_weekly_energy_share", "spectral_entropy",
    ]
    np.testing.assert_allclose(
        first[feature_columns].to_numpy(),
        second[feature_columns].to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    )


def test_ensemble_weights_are_nonnegative_and_sum_to_one():
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    candidates = {
        "a": np.array([1.1, 2.1, 3.1, 4.1]),
        "b": np.array([5.0, 5.0, 5.0, 5.0]),
        "c": np.array([0.8, 1.8, 2.8, 3.8]),
    }
    weights = select_ensemble_weights(actual, candidates)
    assert set(weights) == set(candidates)
    assert all(value >= 0 for value in weights.values())
    assert max(weights.values()) <= 0.65 + 1e-6
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-7)


def test_fixed_seed_series_sample_is_independent_of_input_row_order():
    from experiments.forecast_research.forecast_research_benchmark import (
        select_deterministic_series_sample,
    )

    pairs = pd.DataFrame({
        "store_id": ["s3", "s1", "s2", "s1", "s3", "s2"],
        "product_id": ["p1", "p2", "p1", "p1", "p2", "p2"],
    })
    first = select_deterministic_series_sample(pairs, sample_size=4)
    shuffled = select_deterministic_series_sample(
        pairs.sample(frac=1, random_state=91), sample_size=4
    )
    pd.testing.assert_frame_equal(first, shuffled)
    assert len(first) == 4
    assert not first.duplicated(["store_id", "product_id"]).any()



def test_ensemble_weight_cap_forces_a_multi_model_candidate():
    actual = np.arange(1.0, 21.0)
    candidates = {
        "near_perfect": actual.copy(),
        "biased": actual + 4.0,
        "noisy": actual + np.sin(actual) * 3.0,
    }
    weights = select_ensemble_weights(actual, candidates, max_weight=0.65)
    assert max(weights.values()) <= 0.65 + 1e-6
    assert sum(value > 1e-6 for value in weights.values()) >= 2
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-7)
