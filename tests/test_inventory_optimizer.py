"""Tests for service-level constrained replenishment optimization."""
import numpy as np
import pandas as pd
import pytest

from src.inventory_optimizer import (
    lognormal_cycle_service,
    lognormal_expected_leftover,
    lognormal_expected_shortage,
    optimize_store_orders,
)
from scripts.optimized_inventory_decision_engine import (
    attach_inventory_segments,
    build_demand_distribution,
    score_distribution_candidates,
)


def _one_sku(mean=10.0, std=2.0):
    return pd.DataFrame([{
        "store_id": "store-1",
        "product_id": "sku-1",
        "demand_mean": mean,
        "demand_std": std,
        "inventory_position": 0.0,
        "unit_holding_cost": 0.05,
        "unit_shortage_cost": 2.0,
        "fixed_order_cost": 1.0,
        "pack_size": 1.0,
        "min_order_qty": 1.0,
        "max_order_qty": 25.0,
        "volume_per_unit": 1.0,
    }])


def test_lognormal_shortage_and_leftover_are_nonnegative():
    assert lognormal_expected_shortage(10.0, 2.0, 0.0) == pytest.approx(10.0)
    assert lognormal_expected_leftover(10.0, 2.0, 0.0) == pytest.approx(0.0)
    for stock in (5.0, 10.0, 15.0, 20.0):
        assert lognormal_expected_shortage(10.0, 2.0, stock) >= 0
        assert lognormal_expected_leftover(10.0, 2.0, stock) >= 0


def test_cycle_service_is_monotone_in_stock():
    levels = [lognormal_cycle_service(10.0, 3.0, stock) for stock in (0, 5, 10, 15, 20)]
    assert np.all(np.diff(levels) >= 0)


def test_optimizer_meets_feasible_store_fill_rate_and_pack_constraints():
    decisions, summary = optimize_store_orders(
        _one_sku(),
        target_fill_rate=0.90,
        storage_capacity=25.0,
        replenishment_capacity=25.0,
    )
    assert summary["service_target_met"] is True
    assert summary["achieved_fill_rate"] >= 0.90
    assert summary["storage_used"] <= 25.0 + 1e-8
    assert summary["replenishment_used"] <= 25.0 + 1e-8
    assert decisions.loc[0, "order_qty"] % 1.0 == pytest.approx(0.0)
    assert decisions.loc[0, "order_qty"] >= 1.0


def test_optimizer_reports_infeasible_sla_when_capacity_is_too_small():
    decisions, summary = optimize_store_orders(
        _one_sku(mean=10.0, std=3.0),
        target_fill_rate=0.95,
        storage_capacity=2.0,
        replenishment_capacity=2.0,
    )
    assert summary["service_target_met"] is False
    assert summary["status"] == "CAPACITY_LIMITED_SLA_INFEASIBLE"
    assert summary["storage_used"] <= 2.0 + 1e-8
    assert summary["replenishment_used"] <= 2.0 + 1e-8
    assert decisions.loc[0, "order_qty"] <= 2.0 + 1e-8


def test_optimizer_rejects_invalid_pack_size_and_duplicate_products():
    invalid = _one_sku()
    invalid.loc[0, "pack_size"] = 0.0
    with pytest.raises(ValueError, match="pack_size"):
        optimize_store_orders(
            invalid, target_fill_rate=0.9,
            storage_capacity=20.0, replenishment_capacity=20.0,
        )
    duplicate = pd.concat([_one_sku(), _one_sku()], ignore_index=True)
    with pytest.raises(ValueError, match="one row per product"):
        optimize_store_orders(
            duplicate, target_fill_rate=0.9,
            storage_capacity=20.0, replenishment_capacity=20.0,
        )



def test_seven_day_residual_calibration_uses_robust_tier_shrinkage():
    from scripts.optimized_inventory_decision_engine import GROUP

    rows = []
    for sku, forecast, errors in (
        ("p1", 8.0, [1.0, 1.2, 0.8, 1.1, 25.0]),
        ("p2", 18.0, [2.0, 2.2, 1.8, 2.1, 2.4]),
        ("p3", 35.0, [3.0, 3.2, 2.8, 3.1, 3.4]),
    ):
        for error in errors:
            rows.append({
                "store_id": "s1",
                "product_id": sku,
                "actual_protection_demand": forecast + error,
                "forecast_protection_demand": forecast,
            })
    calibrated = attach_inventory_segments(build_demand_distribution(pd.DataFrame(rows)))
    assert calibrated[GROUP].duplicated().sum() == 0
    assert calibrated["demand_mean"].gt(0).all()
    assert calibrated["demand_std"].gt(0).all()
    assert set(calibrated["demand_segment"]) == {"low", "medium", "high"}
    outlier_row = calibrated.loc[calibrated["product_id"] == "p1"].iloc[0]
    assert abs(outlier_row["shrunk_residual_bias_7d"]) < 25.0
    assert outlier_row["store_size_segment"] == "large"


def test_lognormal_empirical_and_conformal_distribution_diagnostics_are_scored():
    calibration_rows, evaluation_rows = [], []
    for sku, forecast in (("p1", 8.0), ("p2", 18.0), ("p3", 35.0)):
        for error in (-1.0, 0.5, 1.0, -0.5, 2.0):
            calibration_rows.append({
                "store_id": "s1", "product_id": sku,
                "actual_protection_demand": forecast + error,
                "forecast_protection_demand": forecast,
            })
        for error in (0.0, 1.0, -1.0, 0.5, 1.5):
            evaluation_rows.append({
                "store_id": "s1", "product_id": sku,
                "actual_protection_demand": forecast + error,
            })
    calibrated = attach_inventory_segments(build_demand_distribution(pd.DataFrame(calibration_rows)))
    scored = score_distribution_candidates(pd.DataFrame(evaluation_rows), calibrated)
    assert set(scored["distribution"]) == {
        "moment_lognormal", "empirical_bootstrap", "conformal_residual"
    }
    assert {"overall", "demand_segment"}.issubset(set(scored["segment_type"]))
    assert scored["central_80_coverage_pct"].between(0, 100).all()
    assert scored["central_80_mean_width"].ge(0).all()
