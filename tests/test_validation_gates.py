"""Regression tests for temporal promotion gates and feature availability rules."""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.feature_availability_audit import target_calendar_feature_audit
from scripts.later_holdout_readiness import validate_later_holdout


def make_later_frame(start="2024-07-03", days=14, series=3):
    dates = pl.date_range(
        pl.date(2024, 7, 3),
        pl.date(2024, 7, 3) + pl.duration(days=days - 1),
        interval="1d",
        eager=True,
    ).cast(pl.String).to_list()
    rows = [
        {"store_id": f"s{idx}", "product_id": "p1", "dt": date, "sale_amount": 1.0,
         "hours_sale": [0.0] * 24, "hours_stock_status": [0] * 24}
        for idx in range(series) for date in dates
    ]
    return pl.DataFrame(rows)


def test_later_holdout_gate_accepts_strictly_later_contiguous_14_day_snapshot():
    frame = make_later_frame(days=14, series=3)
    result = validate_later_holdout("2024-07-02", base_series=3, later=frame)
    assert result["status"] == "READY_FOR_FROZEN_MODEL_EVALUATION"
    assert result["later_holdout_days"] == 14
    assert result["series_coverage_pct"] == pytest.approx(100.0)
    assert result["model_scoring_performed"] is False


def test_later_holdout_gate_rejects_short_window():
    frame = make_later_frame(days=7, series=3)
    with pytest.raises(ValueError, match="at least 14"):
        validate_later_holdout("2024-07-02", base_series=3, later=frame)


def test_later_holdout_gate_rejects_not_later_data():
    frame = make_later_frame(days=14, series=3).with_columns(
        pl.lit("2024-07-02").alias("dt")
    )
    with pytest.raises(ValueError, match="not strictly after"):
        validate_later_holdout("2024-07-02", base_series=3, later=frame)


def test_later_holdout_gate_rejects_series_coverage_below_95_pct():
    frame = make_later_frame(days=14, series=2)
    with pytest.raises(ValueError, match="series"):
        validate_later_holdout("2024-07-02", base_series=3, later=frame)


def test_future_holiday_and_activity_flags_are_not_admitted_without_contract():
    audit = target_calendar_feature_audit(["holiday_flag", "activity_flag", "sale_amount"])
    assert audit["dataset_target_date_flags_excluded"] == ["activity_flag", "holiday_flag"]
    assert audit["deterministic_target_date_features_used"]
    assert "publication time" in audit["excluded_reason"]
