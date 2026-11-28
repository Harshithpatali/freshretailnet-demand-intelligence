"""Unit tests for the stochastic inventory optimizer and time-series research tools."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.optimize_inventory import (
    candidate_order_levels,
    expected_normal_components,
    optimize_store,
)
from experiments.time_series.research_benchmark import (
    add_fourier_features,
    build_stl_spectral_features,
    select_ensemble_weights,
)


def test_expected_normal_components_are_bounded_and_additive():
    result = expected_normal_components(order_up_to=12.0, mean=10.0, std=2.0)
    assert 0.0 <= result["cycle_service_level"] <= 1.0
    assert 0.0 <= result["expected_fill_rate"] <= 1.0
    assert result["expected_fulfilled_units"] + result["expected_shortage_units"] == pytest.approx(10.0)


def test_inventory_candidates_respect_pack_and_order_cap():
    levels = candidate_order_levels(
        mean=5.0, std=1.0, service_level=0.9,
        max_order_qty=8.0, min_order_qty=1.0, pack_size=0.5,
    )
    assert levels[0] == 0.0
    assert max(levels) <= 8.0
    assert all(round(value / 0.5) == pytest.approx(value / 0.5) for value in levels)


def test_store_optimizer_enforces_capacity_and_service_target_when_feasible():
    dates = pd.date_range("2025-01-01", periods=5, freq="D")
    rows = []
    for product, center in (("p1", 1.0), ("p2", 1.5), ("p3", 0.8)):
        for index, day in enumerate(dates):
            rows.append({
                "date": day, "store_id": "s1", "product_id": product,
                "actual_sale_amount": center,
                "p10": center * 0.8, "p50": center, "p90": center * 1.2,
            })
    plan, summary = optimize_store(
        pd.DataFrame(rows),
        service_level=0.85,
        capacity_days=20,
        lead_time_days=2,
        review_period_days=1,
        holding_cost_per_unit_day=0.01,
        shortage_penalty_per_unit=10.0,
        ordering_cost_per_order=0.1,
        min_order_qty=0.1,
        pack_size=0.1,
        max_order_qty_days=10,
    )
    assert len(plan) == 3
    assert summary["solver_status"] == 0
    assert summary["service_level_met"] is True
    assert summary["expected_store_fill_rate_pct"] >= 84.99
    assert summary["total_order_up_to_units"] <= summary["store_capacity_units"] + 1e-7
    assert (plan["recommended_order_up_to_units"] >= 0).all()


def test_store_optimizer_reports_infeasible_service_capacity_gap():
    dates = pd.date_range("2025-01-01", periods=3, freq="D")
    rows = []
    for product in ("p1", "p2"):
        for day in dates:
            rows.append({
                "date": day, "store_id": "s1", "product_id": product,
                "actual_sale_amount": 1.0, "p10": 0.9, "p50": 1.0, "p90": 1.1,
            })
    _, summary = optimize_store(
        pd.DataFrame(rows), service_level=0.95, capacity_days=0.1,
        lead_time_days=1, review_period_days=1, pack_size=0.1,
        min_order_qty=0.1, max_order_qty_days=10,
    )
    assert summary["service_level_met"] is False
    assert summary["service_level_shortfall_units"] > 0
    assert summary["capacity_utilization_pct"] <= 100.0001


def test_fourier_features_are_deterministic_and_bounded():
    frame = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=15)})
    result = add_fourier_features(frame)
    columns = [column for column in result if column.startswith("fourier_")]
    assert len(columns) == 10
    assert np.isfinite(result[columns].to_numpy()).all()
    assert result[columns].abs().to_numpy().max() <= 1.0


def test_stl_and_spectral_features_use_only_prior_history():
    dates = pd.date_range("2025-01-01", periods=45, freq="D")
    values = 2.0 + np.sin(2 * np.pi * np.arange(45) / 7.0) + np.arange(45) * 0.01
    raw = pd.DataFrame({
        "store_id": ["s1"] * 45, "product_id": ["p1"] * 45,
        "date": dates, "sale_amount": values,
    })
    features_a, diag_a = build_stl_spectral_features(raw, window=28, period=7)
    changed = raw.copy()
    changed.loc[changed["date"] == dates[40], "sale_amount"] = 9999.0
    features_b, _ = build_stl_spectral_features(changed, window=28, period=7)
    # Feature at date 40 is based on observations through date 39, never its own target.
    row_a = features_a.loc[features_a["date"] == dates[40]].reset_index(drop=True)
    row_b = features_b.loc[features_b["date"] == dates[40]].reset_index(drop=True)
    pd.testing.assert_frame_equal(row_a, row_b)
    assert len(diag_a) == 1
    assert {"stl_seasonal_strength", "spectral_weekly_power_share", "spectral_dominant_period"}.issubset(features_a.columns)


def test_ensemble_weights_are_nonnegative_and_sum_to_one():
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    predictions = {
        "a": np.array([1.0, 2.0, 3.0, 4.0]),
        "b": np.array([1.5, 2.5, 3.5, 4.5]),
        "c": np.array([0.5, 1.5, 2.5, 3.5]),
    }
    weights = select_ensemble_weights(actual, predictions)
    assert all(value >= -1e-8 for value in weights.values())
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
