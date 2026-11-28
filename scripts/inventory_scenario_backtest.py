"""Lead-time, review-cycle, storage and pack-size scenario backtest."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import boto3
import pandas as pd
from botocore.config import Config

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulate_inventory import simulate, aggregate
from scripts.artifact_provenance import build_provenance, upload_versioned_artifact

POLICIES = (
    ("seasonal_naive_lag7", "seasonal_naive_lag7"),
    ("p50", "p50"),
    ("p90_calibrated", "p90_calibrated"),
)
SCENARIOS = (
    {"name": "lead1_review1_unconstrained", "lead_time": 1, "review_period": 1},
    {"name": "lead3_review1_unconstrained", "lead_time": 3, "review_period": 1},
    {"name": "lead7_review1_unconstrained", "lead_time": 7, "review_period": 1},
    {"name": "lead1_review7_unconstrained", "lead_time": 1, "review_period": 7},
    {"name": "lead7_review7_capacity14d_pack6", "lead_time": 7, "review_period": 7,
     "storage_capacity_days": 14, "max_order_qty_days": 7,
     "min_order_qty": 6, "pack_size": 6},
)


def run_scenarios(frame: pd.DataFrame, scenarios=SCENARIOS, holding=0.05,
                  shortage=2.0, ordering=5.0, initial_stock=0.0) -> pd.DataFrame:
    """Run comparable policy scenarios on the exact same held-out rows."""
    output = []
    for scenario in scenarios:
        config = {**scenario}
        name = config.pop("name")
        lead = int(config.pop("lead_time"))
        review = int(config.pop("review_period"))
        for forecast_col, policy in POLICIES:
            detail = simulate(
                frame, forecast_col, policy, lead, review, holding, shortage,
                ordering, initial_stock, **config
            )
            params = {
                "lead_time_days": lead, "review_period_days": review,
                "holding_cost_per_unit_day": holding,
                "shortage_penalty_per_unit": shortage,
                "ordering_cost_per_order": ordering,
                "storage_capacity_basis": f"{config['storage_capacity_days']}x mean daily forecast"
                    if config.get("storage_capacity_days") is not None else "unlimited",
                "max_order_basis": f"{config['max_order_qty_days']}x mean daily forecast"
                    if config.get("max_order_qty_days") is not None else "unlimited",
                "min_order_qty": config.get("min_order_qty", 0.0),
                "pack_size": config.get("pack_size", 1.0),
            }
            output.append(aggregate(detail, name, params))
    # Cost sensitivity is run on two structural settings only to keep the
    # scenario grid bounded while retaining identical demand and forecasts.
    for scenario_name in ("lead1_review1_unconstrained", "lead7_review7_capacity14d_pack6"):
        scenario = next(x for x in SCENARIOS if x["name"] == scenario_name)
        config = {k: v for k, v in scenario.items() if k not in {"name", "lead_time", "review_period"}}
        for multiplier, cost_label in ((0.5, "shortage_cost_low"), (2.0, "shortage_cost_high")):
            for forecast_col, policy in POLICIES:
                detail = simulate(
                    frame, forecast_col, policy, scenario["lead_time"], scenario["review_period"],
                    holding, shortage * multiplier, ordering, initial_stock, **config
                )
                params = {
                    "lead_time_days": scenario["lead_time"],
                    "review_period_days": scenario["review_period"],
                    "holding_cost_per_unit_day": holding,
                    "shortage_penalty_per_unit": shortage * multiplier,
                    "ordering_cost_per_order": ordering,
                    "storage_capacity_basis": f"{config['storage_capacity_days']}x mean daily forecast"
                        if config.get("storage_capacity_days") is not None else "unlimited",
                    "max_order_basis": f"{config['max_order_qty_days']}x mean daily forecast"
                        if config.get("max_order_qty_days") is not None else "unlimited",
                    "min_order_qty": config.get("min_order_qty", 0.0),
                    "pack_size": config.get("pack_size", 1.0),
                }
                output.append(aggregate(detail, cost_label, params) | {
                    "structural_scenario": scenario_name,
                })
    return pd.DataFrame(output)


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


def main():
    bucket = os.getenv("R2_BUCKET")
    if not bucket:
        raise RuntimeError("Missing R2_BUCKET")
    s3 = r2_client()
    with tempfile.TemporaryDirectory(prefix="frn-inventory-scenarios-") as td:
        root = Path(td)
        source = root / "evaluation_quantile_predictions.csv"
        s3.download_file(bucket, "quantile-model-artifacts/evaluation_quantile_predictions.csv", str(source))
        frame = pd.read_csv(source, parse_dates=["date"])
        required = {"date", "store_id", "product_id", "actual_sale_amount", "p50",
                    "p90_calibrated", "seasonal_naive_lag7"}
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"Prediction artifact missing columns: {sorted(missing)}")
        if frame.duplicated(["store_id", "product_id", "date"]).any():
            raise RuntimeError("Duplicate store-product-date rows in quantile predictions")
        frame = frame.sort_values(["store_id", "product_id", "date"])
        results = run_scenarios(frame)
        output_keys = [
            "inventory-scenario-artifacts/inventory_scenario_comparison.csv",
            "inventory-scenario-artifacts/inventory_scenario_summary.json",
        ]
        configuration = {
            "lead_review_scenarios": list(SCENARIOS),
            "policies": [policy for _, policy in POLICIES],
            "base_costs": {"holding_per_unit_day": 0.05, "shortage_per_unit": 2.0, "order_per_order": 5.0},
            "shortage_penalty_multipliers": [0.5, 1.0, 2.0],
            "demand_proxy": "observed sale_amount; may be stockout censored",
        }
        provenance = build_provenance(
            "Inventory Lead-Time and Constraint Scenario Backtest",
            {"quantile-model-artifacts/evaluation_quantile_predictions.csv": source},
            str(frame["date"].min().date()), str(frame["date"].max().date()),
            configuration, outputs=output_keys,
        )
        results["demand_proxy"] = "observed sale_amount; may be censored by stock availability"
        results["simulation_type"] = "assumption-driven backtest; not measured operating savings"
        json_results = results.replace(
            [float("inf"), float("-inf")], float("nan")
        ).astype(object)
        json_results = json_results.where(pd.notna(json_results), None)
        summary = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "provenance": provenance,
            "source_key": "quantile-model-artifacts/evaluation_quantile_predictions.csv",
            "evaluation_rows": int(len(frame)),
            "store_product_series": int(frame[["store_id", "product_id"]].drop_duplicates().shape[0]),
            "evaluation_start": str(frame["date"].min()),
            "evaluation_end": str(frame["date"].max()),
            "scenario_count": int(results["scenario"].nunique()),
            "policy_scenario_rows": int(len(results)),
            "scenarios": list(SCENARIOS),
            "results": json_results.to_dict(orient="records"),
            "interpretation": (
                "Compare fill rate and simulated cost under the same observed-sales proxy. "
                "Capacity is scaled to each series' mean forecast; pack size and minimum order "
                "quantity are illustrative. These assumptions require retailer/supplier data "
                "before operational use."
            ),
            "limitations": [
                "Observed sales may be stockout-censored and do not identify latent demand.",
                "Lead time is deterministic; supplier reliability and transit-time variance are not modelled.",
                "Capacity and order caps are scaled by mean forecast rather than measured shelf/backroom capacity.",
                "Pack size and MOQ are illustrative and uniform across products.",
                "No shelf-life, perishability, minimum presentation stock, case rounding by SKU, or supplier calendar is modelled.",
                "The evaluation horizon is short; compare multiple later time windows before promotion.",
            ],
        }
        csv_path = root / "inventory_scenario_comparison.csv"
        json_path = root / "inventory_scenario_summary.json"
        results.to_csv(csv_path, index=False)
        json_path.write_text(json.dumps(summary, indent=2, default=str, allow_nan=False))
        uploads = [
            (csv_path, "inventory-scenario-artifacts/inventory_scenario_comparison.csv", "text/csv"),
            (json_path, "inventory-scenario-artifacts/inventory_scenario_summary.json", "application/json"),
        ]
        for path, key, content_type in uploads:
            destinations = upload_versioned_artifact(s3, bucket, path, key, content_type, provenance)
            print(f"Published stable and immutable copies: {destinations[0]} ; {destinations[1]}")
        manifest_path = root / "inventory_scenario_manifest.json"
        manifest_path.write_text(json.dumps(provenance, indent=2, sort_keys=True))
        upload_versioned_artifact(
            s3, bucket, manifest_path,
            "inventory-scenario-artifacts/inventory_scenario_manifest.json",
            "application/json", provenance,
        )
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
