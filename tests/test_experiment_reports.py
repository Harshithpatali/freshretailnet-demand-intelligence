"""Tests for demand censoring, forecast monitoring, bootstrap and scenario simulation."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.demand_intelligence.cluster_bootstrap_report import paired_cluster_bootstrap
from scripts.demand_censoring_sensitivity import apply_sensitivity
from scripts.forecast_monitoring import build_monitoring_report, population_stability_index
from scripts.inventory_scenario_backtest import run_scenarios
from src.simulate_inventory import simulate


def test_censoring_sensitivity_bounds_and_zero_exposure():
    frame = pd.DataFrame({
        "observed_sales": [10.0, 4.0],
        "expected_lost_sales_proxy": [8.0, 0.0],
    })
    out = apply_sensitivity(frame)
    assert out.loc[0, "counterfactual_sales_lower"] == pytest.approx(14.0)
    assert out.loc[0, "counterfactual_sales_central"] == pytest.approx(18.0)
    assert out.loc[0, "counterfactual_sales_upper"] == pytest.approx(22.0)
    assert out.loc[1, "counterfactual_sales_central"] == pytest.approx(4.0)


def test_censoring_sensitivity_rejects_invalid_multipliers():
    with pytest.raises(ValueError):
        apply_sensitivity(pd.DataFrame({
            "observed_sales": [1.0], "expected_lost_sales_proxy": [1.0]
        }), low_multiplier=1.2)


def test_population_stability_index_zero_for_identical_data():
    values = np.arange(100, dtype=float)
    assert population_stability_index(values, values) == pytest.approx(0.0)


def test_monitoring_emits_error_coverage_drift_and_alerts():
    frame = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=4).repeat(5),
        "actual_sale_amount": [10, 11, 9, 10, 10, 20, 22, 18, 20, 19, 40, 42, 38, 40, 39, 50, 55, 45, 48, 49],
        "p10": [8.0] * 20, "p50": [9.0] * 20, "p90": [12.0] * 20,
        "p10_calibrated": [8.0] * 20, "p90_calibrated": [12.0] * 20,
        "stock_hour6_22_cnt": [0] * 10 + [8] * 10,
    })
    report, daily, drift, exposure, time_exposure = build_monitoring_report(frame)
    assert report["overall"]["rows"] == 20
    assert report["overall"]["mae"] > 0
    assert len(daily) == 4
    assert "stockout_hours_06_22" in set(drift["variable"])
    assert report["alert_count"] >= 1
    assert {"stockout_exposure_bucket", "calibrated_coverage_pct", "rows"}.issubset(exposure.columns)
    assert {"date", "stockout_exposure_bucket", "calibrated_coverage_pct"}.issubset(time_exposure.columns)
    assert len(exposure) == 6


def test_bootstrap_returns_paired_cluster_intervals():
    rows = []
    for cluster in range(12):
        for fold in range(2):
            actual = 10.0 + cluster + fold
            rows.append({
                "store_id": f"s{cluster % 3}", "product_id": f"p{cluster}",
                "horizon_days": 1, "fold": fold + 1,
                "target_date": f"2025-01-{fold + 1:02d}",
                "actual": actual, "baseline_prediction": actual + 2.0,
                "calibrated_prediction": actual + 1.0,
            })
    result = paired_cluster_bootstrap(pd.DataFrame(rows), reps=100, seed=7, batch_size=16)
    assert len(result) == 4
    mae = next(x for x in result if x["estimand"] == "mae_improvement_pct")
    assert mae["point_estimate"] > 0
    assert mae["ci_95_lower"] <= mae["ci_95_upper"]
    assert mae["store_product_clusters"] == 12


def test_inventory_simulation_respects_capacity_moq_and_pack_size():
    frame = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=3),
        "store_id": ["s1"] * 3, "product_id": ["p1"] * 3,
        "actual_sale_amount": [1.0, 1.0, 1.0],
        "forecast": [5.0, 5.0, 5.0],
    })
    result = simulate(
        frame, "forecast", "constrained", lead_time=1, review_period=1,
        holding_cost=0.0, shortage_cost=1.0, order_cost=1.0,
        initial_stock=0.0, storage_capacity=8.0, min_order_qty=6.0,
        pack_size=6.0, max_order_qty=6.0,
    )
    row = result.iloc[0]
    assert row["units_ordered"] <= 6.0
    assert row["units_ordered"] % 6.0 == pytest.approx(0.0)
    assert row["constraint_limited_orders"] >= 0


def test_inventory_scenario_runner_compares_structural_assumptions():
    dates = pd.date_range("2025-01-01", periods=4)
    frame = pd.DataFrame({
        "date": dates, "store_id": ["s1"] * 4, "product_id": ["p1"] * 4,
        "actual_sale_amount": [2.0, 3.0, 1.0, 2.0],
        "seasonal_naive_lag7": [2.0] * 4,
        "p50": [3.0] * 4, "p90_calibrated": [4.0] * 4,
    })
    out = run_scenarios(
        frame,
        scenarios=({"name": "test", "lead_time": 1, "review_period": 1,
                    "storage_capacity_days": 3, "min_order_qty": 2, "pack_size": 2},),
    )
    assert set(out["policy"]) == {"seasonal_naive_lag7", "p50", "p90_calibrated"}
    assert out["scenario"].nunique() == 3
    assert out["total_simulated_cost"].ge(0).all()
