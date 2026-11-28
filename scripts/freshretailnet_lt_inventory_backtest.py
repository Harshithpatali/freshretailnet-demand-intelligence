"""Seven-day-lead inventory stress test using external FreshRetailNet-LT rolling predictions.

This script uses only the H1 out-of-time predictions from the LT backtest. It
requires a continuous, complete 70-day window per included store-product series,
so the 7-day lead policies have multiple replenishment cycles and post-arrival
observations. All cost and capacity settings remain illustrative assumptions.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from botocore.config import Config

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.artifact_provenance import build_provenance, upload_versioned_artifact
from src.simulate_inventory import aggregate, simulate

PREFIX = os.getenv("SEGMENT_BACKTEST_PREFIX", "freshretailnet-lt-segment-backtest-artifacts").strip("/")
PREDICTIONS_KEY = f"{PREFIX}/segment_rolling_backtest_predictions.csv"
OUT = "freshretailnet-lt-inventory-artifacts"
GROUP = ["store_id", "product_id"]
POLICIES = (
    ("baseline_prediction", "current_lightgbm_baseline"),
    ("selected_prediction", "validation_gated_specialists"),
    ("calibrated_prediction", "calibrated_specialists"),
)
SCENARIOS = (
    {"name": "lead1_review1_unconstrained", "lead_time": 1, "review_period": 1},
    {"name": "lead3_review1_unconstrained", "lead_time": 3, "review_period": 1},
    {"name": "lead7_review1_unconstrained", "lead_time": 7, "review_period": 1},
    {"name": "lead7_review7_capacity14d_pack6", "lead_time": 7, "review_period": 7,
     "storage_capacity_days": 14, "max_order_qty_days": 7, "min_order_qty": 6, "pack_size": 6},
)


def r2_client():
    required = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"Missing R2 environment variables: {missing}")
    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
    )


def select_continuous_h1_predictions(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = GROUP + [
        "horizon_days", "fold", "target_date", "actual", "baseline_prediction",
        "selected_prediction", "calibrated_prediction",
    ]
    missing = set(required) - set(frame.columns)
    if missing:
        raise RuntimeError(f"Rolling predictions missing columns: {sorted(missing)}")
    data = frame.loc[frame["horizon_days"].astype(int) == 1, required].copy()
    data["target_date"] = pd.to_datetime(data["target_date"], errors="raise").dt.normalize()
    if data.duplicated(GROUP + ["target_date"]).any():
        raise RuntimeError("Duplicate H1 store-product-date rows in rolling predictions")
    dates = pd.DatetimeIndex(sorted(data["target_date"].unique()))
    if len(dates) < 70:
        raise RuntimeError(f"Expected at least 70 H1 target dates for the LT inventory test, got {len(dates)}")
    if not dates.equals(pd.date_range(dates.min(), dates.max(), freq="D")):
        raise RuntimeError("H1 target-date window is not contiguous; refusing lead-time simulation")
    n_dates = len(dates)
    counts = data.groupby(GROUP, sort=False)["target_date"].nunique()
    complete = counts[counts == n_dates].index
    complete_keys = pd.MultiIndex.from_frame(data[GROUP])
    data = data.loc[complete_keys.isin(complete)].copy()
    complete_series = int(data[GROUP].drop_duplicates().shape[0])
    total_series = int(counts.shape[0])
    if complete_series < 1000:
        raise RuntimeError(
            f"Only {complete_series:,}/{total_series:,} series have a complete {n_dates}-day window; "
            "at least 1,000 are required"
        )
    data = data.rename(columns={
        "target_date": "date",
        "actual": "actual_sale_amount",
    })
    prediction_values = data[[
        "actual_sale_amount", "baseline_prediction", "selected_prediction", "calibrated_prediction"
    ]].to_numpy(dtype=float)
    if not np.isfinite(prediction_values).all():
        raise RuntimeError("Non-finite H1 target or prediction values")
    audit = {
        "target_date_start": str(dates.min().date()),
        "target_date_end": str(dates.max().date()),
        "target_dates": n_dates,
        "total_series_with_any_prediction": total_series,
        "complete_continuous_series": complete_series,
        "complete_series_rate_pct": 100 * complete_series / max(total_series, 1),
        "excluded_incomplete_series": total_series - complete_series,
        "forecast_horizon": 1,
    }
    return data.sort_values(GROUP + ["date"]), audit


def run_inventory(frame: pd.DataFrame) -> pd.DataFrame:
    output = []
    for scenario in SCENARIOS:
        config = {key: value for key, value in scenario.items()
                  if key not in {"name", "lead_time", "review_period"}}
        for forecast_col, policy in POLICIES:
            detail = simulate(
                frame,
                forecast_col,
                policy,
                int(scenario["lead_time"]),
                int(scenario["review_period"]),
                holding_cost=0.05,
                shortage_cost=2.0,
                order_cost=5.0,
                initial_stock=0.0,
                **config,
            )
            params = {
                "lead_time_days": int(scenario["lead_time"]),
                "review_period_days": int(scenario["review_period"]),
                "holding_cost_per_unit_day": 0.05,
                "shortage_penalty_per_unit": 2.0,
                "ordering_cost_per_order": 5.0,
                "initial_stock_units_per_series": 0.0,
                "storage_capacity_basis": (
                    f"{config['storage_capacity_days']}x mean daily forecast"
                    if config.get("storage_capacity_days") is not None else "unlimited"
                ),
                "max_order_basis": (
                    f"{config['max_order_qty_days']}x mean daily forecast"
                    if config.get("max_order_qty_days") is not None else "unlimited"
                ),
                "min_order_qty": config.get("min_order_qty", 0.0),
                "pack_size": config.get("pack_size", 1.0),
                "continuous_h1_forecast_horizon_days": int(frame["date"].nunique()),
            }
            output.append(aggregate(detail, scenario["name"], params))
    return pd.DataFrame(output)


def json_records(frame: pd.DataFrame) -> list[dict]:
    safe = frame.replace([float("inf"), float("-inf")], float("nan")).astype(object)
    safe = safe.where(pd.notna(safe), None)
    return safe.to_dict(orient="records")


def main():
    bucket = os.getenv("R2_BUCKET")
    if not bucket:
        raise RuntimeError("Missing R2_BUCKET")
    client = r2_client()
    with tempfile.TemporaryDirectory(prefix="frn-lt-inventory-") as td:
        root = Path(td)
        source = root / "segment_rolling_backtest_predictions.csv"
        client.download_file(bucket, PREDICTIONS_KEY, str(source))
        predictions = pd.read_csv(source)
        frame, audit = select_continuous_h1_predictions(predictions)
        results = run_inventory(frame)
        output_keys = [
            f"{OUT}/inventory_scenario_comparison.csv",
            f"{OUT}/inventory_scenario_summary.json",
        ]
        configuration = {
            "external_dataset": "Dingdong-Inc/FreshRetailNet-LT v1.0",
            "forecast_source": PREDICTIONS_KEY,
            "policies": [policy for _, policy in POLICIES],
            "scenarios": list(SCENARIOS),
            "base_costs": {
                "holding_cost_per_unit_day": 0.05,
                "shortage_penalty_per_unit": 2.0,
                "ordering_cost_per_order": 5.0,
            },
            "series_filter": "complete daily H1 prediction window only",
            "demand_proxy": "observed sale_amount; may be censored by stockouts",
        }
        provenance = build_provenance(
            "FreshRetailNet-LT External Inventory Lead-Time Backtest",
            {PREDICTIONS_KEY: source},
            audit["target_date_start"],
            audit["target_date_end"],
            configuration,
            outputs=output_keys + [f"{OUT}/inventory_scenario_manifest.json"],
        )
        summary = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "provenance": provenance,
            "evaluation": audit,
            "scenario_count": int(results["scenario"].nunique()),
            "policy_scenario_rows": int(len(results)),
            "results": json_records(results),
            "demand_proxy": "observed sale_amount; may be censored by availability",
            "simulation_type": "rolling out-of-time H1 predictions; illustrative inventory cost assumptions",
            "interpretation": (
                "The H1 rolling predictions cover a continuous 70-day block. This permits multiple "
                "seven-day replenishment cycles, unlike the original seven-day evaluation period. "
                "Compare forecast policies on the same complete series and dates."
            ),
            "limitations": [
                "FreshRetailNet-LT is a different external store-product population from FreshRetailNet-50K.",
                "Observed sales may be censored and are not identified latent demand.",
                "Costs, initial inventory, capacity, MOQ and pack size are illustrative assumptions.",
                "Lead times are deterministic; expiry, perishable waste, supplier calendars and stochastic delivery are not modelled.",
                "Rolling predictions are from expanding-window test folds; the final official July 14-20 benchmark split is included only according to the source split and fold chronology.",
            ],
        }
        csv_path = root / "inventory_scenario_comparison.csv"
        json_path = root / "inventory_scenario_summary.json"
        results.to_csv(csv_path, index=False)
        json_path.write_text(json.dumps(summary, indent=2, default=str, allow_nan=False))
        for path, key, content_type in (
            (csv_path, output_keys[0], "text/csv"),
            (json_path, output_keys[1], "application/json"),
        ):
            destinations = upload_versioned_artifact(client, bucket, path, key, content_type, provenance)
            print(f"Verified and published: {destinations[0]} ; {destinations[1]}")
        manifest_path = root / "inventory_scenario_manifest.json"
        manifest_path.write_text(json.dumps(provenance, indent=2, sort_keys=True, default=str))
        upload_versioned_artifact(
            client, bucket, manifest_path, f"{OUT}/inventory_scenario_manifest.json",
            "application/json", provenance,
        )
        print(results.to_string(index=False))
        print(json.dumps({"evaluation": audit}, indent=2))


if __name__ == "__main__":
    main()
