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
