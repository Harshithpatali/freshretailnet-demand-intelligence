"""Optimize store replenishment under fill-rate and capacity constraints.

Uses 70-day continuous H1 rolling predictions from the separate FreshRetailNet-LT
population. The first five non-overlapping seven-day windows calibrate demand
bias/uncertainty; the next five non-overlapping windows evaluate the optimized
protection-period stock targets. The external cohort is not pooled with the
original FreshRetailNet-50K results.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import boto3
import numpy as np
import pandas as pd
from botocore.config import Config

from scripts.artifact_provenance import build_provenance, upload_versioned_artifact
from scripts.freshretailnet_lt_inventory_backtest import select_continuous_h1_predictions
from src.inventory_optimizer import optimize_network


SOURCE_PREFIX = os.getenv(
    "SEGMENT_BACKTEST_PREFIX", "freshretailnet-lt-segment-backtest-artifacts"
).strip("/")
SOURCE_KEY = f"{SOURCE_PREFIX}/segment_rolling_backtest_predictions.csv"
OUT = "inventory-optimizer-artifacts"
GROUP = ["store_id", "product_id"]
PROTECTION_DAYS = 7
CALIBRATION_STARTS = (0, 7, 14, 21, 28)
EVALUATION_STARTS = (35, 42, 49, 56, 63)


SCENARIOS = (
    {
        "name": "balanced_90pct",
        "target_fill_rate": 0.90,
        "storage_capacity_days": 14.0,
        "replenishment_capacity_days": 7.0,
        "max_order_days_per_sku": 14.0,
    },
    {
        "name": "constrained_90pct",
        "target_fill_rate": 0.90,
        "storage_capacity_days": 10.0,
        "replenishment_capacity_days": 5.0,
        "max_order_days_per_sku": 10.0,
    },
    {
        "name": "tight_95pct",
        "target_fill_rate": 0.95,
        "storage_capacity_days": 7.0,
        "replenishment_capacity_days": 3.5,
        "max_order_days_per_sku": 7.0,
    },
)


def r2_client():
    required = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
    missing = [name for name in required if not os.getenv(name)]
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


def build_non_overlapping_windows(
    frame: pd.DataFrame,
    starts: tuple[int, ...],
    protection_days: int = PROTECTION_DAYS,
) -> pd.DataFrame:
    rows = []
    for (store_id, product_id), group in frame.groupby(GROUP, sort=False, observed=True):
        group = group.sort_values("date").reset_index(drop=True)
        dates = pd.to_datetime(group["date"]).dt.normalize()
        actual = group["actual_sale_amount"].to_numpy(dtype=float)
        prediction = np.maximum(group["calibrated_prediction"].to_numpy(dtype=float), 0.0)
        for start in starts:
            end = start + protection_days
            if end > len(group):
                continue
            rows.append({
                "store_id": str(store_id),
                "product_id": str(product_id),
                "window_start": str(dates.iloc[start].date()),
                "window_end": str(dates.iloc[end - 1].date()),
                "actual_protection_demand": float(actual[start:end].sum()),
                "forecast_protection_demand": float(prediction[start:end].sum()),
                "window_days": protection_days,
            })
    return pd.DataFrame(rows)


def build_demand_distribution(
    calibration_windows: pd.DataFrame,
    prior_strength: float = 3.0,
) -> pd.DataFrame:
    """Estimate protection demand and shrink noisy SKU residual scales."""
    if calibration_windows.empty:
        raise RuntimeError("No calibration windows were generated")
    rows = []
    for key, group in calibration_windows.groupby(GROUP, sort=False, observed=True):
        errors = (
            group["actual_protection_demand"].to_numpy(float)
            - group["forecast_protection_demand"].to_numpy(float)
        )
        forecast_mean = float(group["forecast_protection_demand"].mean())
        bias = float(np.mean(errors))
        demand_mean = max(0.0, forecast_mean + bias)
        raw_std = float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0
        rows.append({
            "store_id": str(key[0]),
            "product_id": str(key[1]),
            "forecast_mean_7d": forecast_mean,
            "forecast_error_bias_7d": bias,
            "demand_mean": demand_mean,
            "raw_residual_std": raw_std,
            "calibration_windows": int(len(group)),
        })
    result = pd.DataFrame(rows)
    global_std = float(result["raw_residual_std"].median())
    n = result["calibration_windows"].astype(float)
    # Empirical-Bayes variance shrinkage toward the cohort median; the five
    # calibration windows per SKU are too few for an unshrunk tail estimate.
    numerator = (n - 1.0) * np.square(result["raw_residual_std"]) + prior_strength * global_std**2
    denominator = np.maximum(n - 1.0 + prior_strength, 1.0)
    result["demand_std"] = np.sqrt(np.maximum(numerator / denominator, 0.0))
    result["demand_std"] = np.maximum(result["demand_std"], 0.05 * result["demand_mean"])
    result["inventory_position"] = 0.0
    result["volume_per_unit"] = 1.0
    return result


def evaluate_decisions(
    decisions: pd.DataFrame,
    evaluation_windows: pd.DataFrame,
    scenario_name: str,
    target_fill_rate: float,
) -> tuple[pd.DataFrame, dict]:
    joined = evaluation_windows.merge(
        decisions[GROUP + ["order_qty", "stock_position", "expected_cost"]],
        on=GROUP,
        how="inner",
        validate="many_to_one",
    )
    if joined.empty:
        raise RuntimeError(f"No evaluation windows align with decisions for {scenario_name}")
    stock = joined["stock_position"].to_numpy(float)
    demand = joined["actual_protection_demand"].to_numpy(float)
    fulfilled = np.minimum(stock, demand)
    shortage = np.maximum(demand - stock, 0.0)
    leftover = np.maximum(stock - demand, 0.0)
    holding_cost = 0.05 * PROTECTION_DAYS * leftover
    shortage_cost = 2.0 * shortage
    ordering_cost = np.where(joined["order_qty"].to_numpy(float) > 0, 5.0, 0.0)
    joined["fulfilled_units"] = fulfilled
    joined["shortage_units"] = shortage
    joined["leftover_units"] = leftover
    joined["realized_cycle_service"] = (demand <= stock + 1e-12).astype(float)
    joined["simulated_cost"] = holding_cost + shortage_cost + ordering_cost
    total_actual = float(demand.sum())
    total_fulfilled = float(fulfilled.sum())
    metrics = {
        "scenario": scenario_name,
        "target_fill_rate": target_fill_rate,
        "empirical_fill_rate": total_fulfilled / max(total_actual, 1e-12),
        "empirical_cycle_service_level": float(joined["realized_cycle_service"].mean()),
        "actual_protection_demand": total_actual,
        "fulfilled_protection_demand": total_fulfilled,
        "unfulfilled_protection_demand": float(shortage.sum()),
        "leftover_stock": float(leftover.sum()),
        "holding_cost": float(holding_cost.sum()),
        "shortage_cost": float(shortage_cost.sum()),
        "ordering_cost": float(ordering_cost.sum()),
        "total_simulated_cost": float(joined["simulated_cost"].sum()),
        "store_count": int(joined["store_id"].nunique()),
        "store_product_series": int(joined[GROUP].drop_duplicates().shape[0]),
        "evaluation_windows_per_series": int(len(EVALUATION_STARTS)),
        "evaluation_windows": int(len(joined)),
        "service_target_met_empirically": bool(
            total_fulfilled / max(total_actual, 1e-12) >= target_fill_rate
        ),
    }
    joined["scenario"] = scenario_name
    return joined, metrics


def _json_records(frame: pd.DataFrame) -> list[dict]:
    safe = frame.replace([np.inf, -np.inf], np.nan).astype(object)
    safe = safe.where(pd.notna(safe), None)
    return safe.to_dict(orient="records")


def main():
    bucket = os.getenv("R2_BUCKET")
    if not bucket:
        raise RuntimeError("Missing R2_BUCKET")
    s3 = r2_client()
    with tempfile.TemporaryDirectory(prefix="frn-inventory-optimizer-") as temporary:
        root = Path(temporary)
        source = root / "segment_rolling_backtest_predictions.csv"
        print(f"Downloading s3://{bucket}/{SOURCE_KEY}", flush=True)
        s3.download_file(bucket, SOURCE_KEY, str(source))
        raw_predictions = pd.read_csv(source)
        continuous, audit = select_continuous_h1_predictions(
            raw_predictions, min_complete_series=1000
        )
        if audit["target_dates"] != 70:
            raise RuntimeError(
                f"Expected exactly 70 continuous H1 dates for calibration/evaluation; got {audit['target_dates']}"
            )
        calibration = build_non_overlapping_windows(continuous, CALIBRATION_STARTS)
        evaluation = build_non_overlapping_windows(continuous, EVALUATION_STARTS)
        demand_inputs = build_demand_distribution(calibration)
        if demand_inputs[GROUP].duplicated().any():
            raise RuntimeError("Demand calibration created duplicate store-product rows")
        print(
            f"Calibration: {len(calibration):,} seven-day windows; "
            f"evaluation: {len(evaluation):,} non-overlapping seven-day windows; "
            f"series: {len(demand_inputs):,}",
            flush=True,
        )

        scenario_decisions = []
        scenario_stores = []
        scenario_evaluation = []
        scenario_metrics = []
        for scenario in SCENARIOS:
            config = {key: value for key, value in scenario.items() if key != "name"}
            print(f"Optimizing scenario {scenario['name']}: {config}", flush=True)
            decisions, stores = optimize_network(
                demand_inputs,
                target_fill_rate=float(scenario["target_fill_rate"]),
                protection_days=PROTECTION_DAYS,
                storage_capacity_days=float(scenario["storage_capacity_days"]),
                replenishment_capacity_days=float(scenario["replenishment_capacity_days"]),
                max_order_days_per_sku=float(scenario["max_order_days_per_sku"]),
                pack_size=0.10,
                min_order_qty=0.20,
                unit_holding_cost=0.05,
                unit_shortage_cost=2.0,
                fixed_order_cost=5.0,
                max_options_per_sku=60,
            )
            decisions["scenario"] = scenario["name"]
            stores["scenario"] = scenario["name"]
            evaluated, metrics = evaluate_decisions(
                decisions, evaluation, scenario["name"], float(scenario["target_fill_rate"])
            )
            metrics.update({
                "capacity_limited_store_count": int(
                    (stores["status"] == "CAPACITY_LIMITED_SLA_INFEASIBLE").sum()
                ),
                "service_target_met_store_count": int(stores["service_target_met"].sum()),
                "store_count": int(len(stores)),
                "mean_storage_utilization_pct": float(stores["storage_utilization_pct"].mean()),
                "mean_replenishment_utilization_pct": float(
                    stores["replenishment_utilization_pct"].mean()
                ),
                "theoretical_expected_fill_rate": float(
                    decisions.expected_fulfilled.sum() / max(decisions.demand_mean.sum(), 1e-12)
                ),
                "theoretical_expected_cost_per_protection_period": float(
                    decisions.expected_cost.sum()
                ),
            })
            scenario_decisions.append(decisions)
            scenario_stores.append(stores)
            scenario_evaluation.append(evaluated)
            scenario_metrics.append(metrics)

        decision_frame = pd.concat(scenario_decisions, ignore_index=True)
        store_frame = pd.concat(scenario_stores, ignore_index=True)
        evaluation_frame = pd.concat(scenario_evaluation, ignore_index=True)
        configuration = {
            "source_dataset": "Dingdong-Inc/FreshRetailNet-LT",
            "source_population": "external population; keep results separate from FreshRetailNet-50K",
            "source_key": SOURCE_KEY,
            "calibration_periods": list(CALIBRATION_STARTS),
            "evaluation_periods": list(EVALUATION_STARTS),
            "protection_days": PROTECTION_DAYS,
            "target_service_definition": "store-wide volume-weighted expected fill rate",
            "demand_distribution": "moment-matched lognormal from seven-day forecast bias and residual scale",
            "residual_scale_shrinkage": "empirical-Bayes variance shrinkage toward cohort median; prior strength 3",
            "solver": "scipy.optimize.milp (HiGHS), multiple-choice mixed-integer program",
            "objective": "expected holding cost + expected shortage cost + fixed ordering cost",
            "constraints": [
                "store-level expected fill rate >= scenario target; if infeasible, maximize expected fulfilled demand and report SLA gap",
                "sum of SKU stock positions * volume <= store storage capacity",
                "sum of SKU replenishment quantities * volume <= store replenishment capacity",
                "each SKU order is zero or a minimum-order/pack-size multiple",
                "SKU maximum order is bounded by its scenario-specific forecast-day cap",
            ],
            "cost_assumptions": {
                "holding_cost_per_unit_day": 0.05,
                "shortage_penalty_per_unit": 2.0,
                "fixed_order_cost": 5.0,
                "pack_size": 0.10,
                "minimum_order_quantity": 0.20,
                "volume_per_unit": 1.0,
            },
            "scenarios": list(SCENARIOS),
        }
        output_keys = [
            f"{OUT}/inventory_optimizer_summary.json",
            f"{OUT}/inventory_optimizer_store_results.csv",
            f"{OUT}/inventory_optimizer_item_decisions.csv",
            f"{OUT}/inventory_optimizer_rolling_evaluation.csv",
            f"{OUT}/inventory_optimizer_demand_calibration.csv",
            f"{OUT}/inventory_optimizer_manifest.json",
        ]
        provenance = build_provenance(
            "Service-Level Constrained Inventory Optimizer",
            {SOURCE_KEY: source},
            audit["target_date_start"],
            audit["target_date_end"],
            configuration,
            outputs=output_keys,
        )
        summary = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "provenance": provenance,
            "evaluation_population": audit,
            "calibration_design": {
                "non_overlapping_seven_day_windows": list(CALIBRATION_STARTS),
                "windows_per_series": len(CALIBRATION_STARTS),
                "purpose": "estimate per-SKU forecast bias and residual variability before inventory optimization",
            },
            "evaluation_design": {
                "non_overlapping_seven_day_windows": list(EVALUATION_STARTS),
                "windows_per_series": len(EVALUATION_STARTS),
                "purpose": "evaluate selected protection-period stock levels on a later disjoint block",
            },
            "scenario_results": scenario_metrics,
            "scenario_decision": (
                "A store is reported SLA-feasible only when the MILP reaches the expected fill-rate "
                "target under both capacity constraints. Empirical evaluation fill rate is reported separately."
            ),
            "limitations": [
                "FreshRetailNet-LT is a separate population; do not pool its results with FreshRetailNet-50K.",
                "Observed sales may be censored by stockouts and are not identified latent demand.",
                "Only five calibration and five evaluation protection-period blocks are available per series; empirical service estimates are noisy.",
                "Seven-day windows are non-overlapping in the reported calibration/evaluation blocks, but the external rolling model predictions are historical rolling forecasts.",
                "The lognormal distribution is a positive-support approximation fitted by two moments; tail shape is not independently validated.",
                "Holding/shortage/order costs, pack size, MOQ, capacity and volume factors are illustrative normalized assumptions, not measured retailer parameters.",
                "The MILP solves a single protection-period allocation, not a multi-period stochastic dynamic program with supplier lead-time uncertainty, expiry or substitution.",
            ],
        }
        manifest_path = root / "inventory_optimizer_manifest.json"
        manifest_path.write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
        summary_path = root / "inventory_optimizer_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str, allow_nan=False))
        artifacts = [
            (summary_path, output_keys[0], "application/json"),
        ]
        csvs = [
            (store_frame, "inventory_optimizer_store_results.csv", output_keys[1]),
            (decision_frame, "inventory_optimizer_item_decisions.csv", output_keys[2]),
            (evaluation_frame, "inventory_optimizer_rolling_evaluation.csv", output_keys[3]),
            (demand_inputs, "inventory_optimizer_demand_calibration.csv", output_keys[4]),
        ]
        for frame, filename, key in csvs:
            path = root / filename
            frame.to_csv(path, index=False)
            artifacts.append((path, key, "text/csv"))
        artifacts.append((manifest_path, output_keys[5], "application/json"))
        for path, key, content_type in artifacts:
            destinations = upload_versioned_artifact(
                s3, bucket, path, key, content_type, provenance
            )
            print(f"Published {key}: {destinations}", flush=True)

        print(json.dumps({
            "evaluation_population": audit,
            "scenarios": scenario_metrics,
            "artifacts": output_keys,
        }, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
