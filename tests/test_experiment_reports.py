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
from scripts.freshretailnet_lt_inventory_backtest import select_continuous_h1_predictions
from src.train_quantile_models import (
    conditional_conformal_adjustments,
    calibrated_interval_by_group,
    risk_score_edges,
    risk_score_groups,
    risk_conformal_adjustments,
    calibrated_interval_by_risk,
)
from scripts.later_holdout_readiness import validate_later_holdout
import polars as pl


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



def test_forecast_monitoring_flags_nominal_coverage_miss_and_exposure_groups():
    frame = pd.DataFrame({
        "date": pd.date_range("2025-02-01", periods=2).repeat(4),
        "actual_sale_amount": [10.0] * 8,
        "p10": [9.5] * 8, "p50": [10.0] * 8, "p90": [10.5] * 8,
        "p10_calibrated": [9.5, 9.5, 0.0, 0.0] * 2,
        "p90_calibrated": [10.5, 10.5, 1.0, 1.0] * 2,
        "stock_hour6_22_cnt": [0, 0, 1, 1, 0, 0, 1, 1],
    })
    report, daily, drift, exposure, time_exposure = build_monitoring_report(
        frame, min_exposure_group_rows=1
    )
    alert_codes = {alert["code"] for alert in report["alerts"]}
    assert "INTERVAL_COVERAGE_LOW" in alert_codes
    assert "STOCKOUT_SEGMENT_COVERAGE_LOW" in alert_codes
    assert len(time_exposure) > 0


def test_artifact_provenance_hashes_inputs_and_versions_output(tmp_path, monkeypatch):
    from scripts.artifact_provenance import build_provenance, versioned_key

    snapshot = tmp_path / "snapshot.txt"
    snapshot.write_text("fresh-retailnet-snapshot")
    monkeypatch.setenv("GITHUB_SHA", "abc123def4567890")
    monkeypatch.setenv("GITHUB_RUN_ID", "123456")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    manifest = build_provenance(
        "test-workflow", {"raw/train.parquet": snapshot},
        "2024-07-01", "2024-07-14", {"folds": 5},
        outputs=["reports/report.json"],
    )
    key = versioned_key("reports/report.json", manifest)
    assert manifest["data_snapshot"]["sha256"]
    assert manifest["data_snapshot"]["inputs"]["raw/train.parquet"]["sha256"]
    assert "abc123def456" in key
    assert "123456-attempt-2" in key


def test_target_date_feature_audit_excludes_uncertified_dataset_flags():
    from scripts.feature_availability_audit import target_calendar_feature_audit

    audit = target_calendar_feature_audit(["holiday_flag", "activity_flag", "sale_amount"])
    assert audit["dataset_target_date_flags_excluded"] == ["activity_flag", "holiday_flag"]
    assert "target_day_of_week" in audit["deterministic_target_date_features_used"]
    assert "publication time" in audit["excluded_reason"]



def test_conditional_conformal_uses_prior_exposure_and_shrinks_small_groups():
    y = np.concatenate([np.ones(100), np.full(100, 5.0)])
    lower = np.zeros(200)
    upper = np.zeros(200)
    context = np.concatenate([np.zeros(100), np.full(100, 16.0)])
    adjustments, audit = conditional_conformal_adjustments(
        y, lower, upper, context, global_qhat=5.0,
        min_group_rows=20, prior_strength_rows=100,
    )
    assert adjustments["0_hours"] == pytest.approx(3.0)
    assert adjustments["16_hours"] == pytest.approx(5.0)
    assert len(audit) == 6
    assert next(row for row in audit if row["prior_exposure_group"] == "16_hours")["validation_rows"] == 100


def test_conditional_interval_applies_historical_exposure_bucket_only():
    predictions = {
        0.10: np.zeros(2),
        0.50: np.ones(2),
        0.90: np.full(2, 2.0),
    }
    y = np.array([1.0, 5.0])
    context = np.array([0.0, 16.0])
    adjustments = {
        "0_hours": 1.0, "1_2_hours": 1.0, "3_5_hours": 1.0,
        "6_9_hours": 1.0, "10_15_hours": 1.0, "16_hours": 4.0,
    }
    summary, lower, upper = calibrated_interval_by_group(y, predictions, context, adjustments)
    assert lower[0] == pytest.approx(0.0)
    assert upper[0] == pytest.approx(3.0)
    assert lower[1] == pytest.approx(0.0)
    assert upper[1] == pytest.approx(6.0)
    assert summary["context_feature"] == "stockout_hours_roll_max_7"



def _later_holdout_frame(start="2024-07-03", days=3):
    dates = pd.date_range(start, periods=days, freq="D")
    rows = []
    for date in dates:
        for store, product in (("s1", "p1"), ("s2", "p2")):
            rows.append({
                "store_id": store, "product_id": product, "dt": date.strftime("%Y-%m-%d"),
                "sale_amount": 1.0, "hours_sale": [0.0] * 24,
                "hours_stock_status": [0] * 24,
            })
    return pl.DataFrame(rows)


def test_later_holdout_gate_accepts_only_a_later_contiguous_window(monkeypatch):
    monkeypatch.setattr("scripts.later_holdout_readiness.MIN_HOLDOUT_DAYS", 3)
    monkeypatch.setattr("scripts.later_holdout_readiness.MIN_SERIES_COVERAGE_PCT", 100.0)
    result = validate_later_holdout("2024-07-02", 2, _later_holdout_frame())
    assert result["status"] == "READY_FOR_FROZEN_MODEL_EVALUATION"
    assert result["later_holdout_start"] == "2024-07-03"
    assert result["later_holdout_days"] == 3
    assert result["model_scoring_performed"] is False


def test_later_holdout_gate_rejects_overlap_with_used_evaluation(monkeypatch):
    monkeypatch.setattr("scripts.later_holdout_readiness.MIN_HOLDOUT_DAYS", 3)
    monkeypatch.setattr("scripts.later_holdout_readiness.MIN_SERIES_COVERAGE_PCT", 100.0)
    with pytest.raises(ValueError, match="not strictly after"):
        validate_later_holdout("2024-07-02", 2, _later_holdout_frame(start="2024-07-02"))


def test_later_holdout_gate_rejects_date_gaps(monkeypatch):
    monkeypatch.setattr("scripts.later_holdout_readiness.MIN_HOLDOUT_DAYS", 3)
    monkeypatch.setattr("scripts.later_holdout_readiness.MIN_SERIES_COVERAGE_PCT", 100.0)
    frame = _later_holdout_frame(days=4)
    frame = frame.filter(pl.col("dt") != "2024-07-04")
    with pytest.raises(ValueError, match="contiguous daily window"):
        validate_later_holdout("2024-07-02", 2, frame)



def test_freshretailnet_lt_inventory_uses_only_complete_continuous_series():
    dates = pd.date_range("2025-05-01", periods=70, freq="D")
    rows = []
    for date in dates:
        rows.append({
            "store_id": "complete", "product_id": "p1", "horizon_days": 1, "fold": 1,
            "target_date": date, "actual": 1.0, "baseline_prediction": 1.0,
            "selected_prediction": 1.1, "calibrated_prediction": 1.05,
        })
    for date in dates[:-1]:
        rows.append({
            "store_id": "incomplete", "product_id": "p2", "horizon_days": 1, "fold": 1,
            "target_date": date, "actual": 1.0, "baseline_prediction": 1.0,
            "selected_prediction": 1.1, "calibrated_prediction": 1.05,
        })
    frame, audit = select_continuous_h1_predictions(pd.DataFrame(rows), min_complete_series=1)
    assert len(frame) == 70
    assert frame["store_id"].nunique() == 1
    assert frame["store_id"].iloc[0] == "complete"
    assert audit["target_dates"] == 70
    assert audit["complete_continuous_series"] == 1
    assert audit["excluded_incomplete_series"] == 1


def test_freshretailnet_lt_inventory_rejects_noncontiguous_target_dates():
    dates = list(pd.date_range("2025-05-01", periods=71, freq="D"))
    dates.remove(pd.Timestamp("2025-05-10"))
    rows = [{
        "store_id": "s1", "product_id": "p1", "horizon_days": 1, "fold": 1,
        "target_date": date, "actual": 1.0, "baseline_prediction": 1.0,
        "selected_prediction": 1.1, "calibrated_prediction": 1.05,
    } for date in dates]
    with pytest.raises(RuntimeError, match="not contiguous"):
        select_continuous_h1_predictions(pd.DataFrame(rows))



def test_risk_score_edges_are_fitted_on_validation_and_reused_for_eval():
    validation_prob = np.linspace(0.0, 1.0, 1000)
    edges = risk_score_edges(validation_prob, bins=10)
    groups = risk_score_groups(validation_prob, edges)
    assert len(edges) == 11
    assert groups.min() == 0
    assert groups.max() == 9
    assert np.all(np.diff(groups) >= 0)


def test_risk_conditional_conformal_uses_validation_deciles():
    n = 2000
    validation_prob = np.linspace(0.001, 0.999, n)
    y = np.where(validation_prob >= 0.8, 4.0, 1.0)
    lower = np.zeros(n)
    upper = np.zeros(n)
    edges, adjustments, audit = risk_conformal_adjustments(
        y, lower, upper, validation_prob, global_qhat=1.0,
        min_group_rows=100, prior_strength_rows=100,
    )
    assert len(edges) == 11
    assert len(audit) == 10
    assert adjustments[9] > adjustments[0]
    summary, lo, hi = calibrated_interval_by_risk(
        y,
        {0.10: lower, 0.50: np.ones(n), 0.90: upper + 1.0},
        validation_prob, edges, adjustments,
    )
    assert summary["context_feature"] == "predicted_high_stockout_risk_decile"
    assert hi[-1] > hi[0]
    assert len(lo) == n
