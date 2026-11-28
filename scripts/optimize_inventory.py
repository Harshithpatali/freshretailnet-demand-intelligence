"""Constrained stochastic inventory allocation using forecast uncertainty.

The optimizer selects one order-up-to quantity per store-product series using
mixed-integer linear programming. It enforces a store-level expected fill-rate
target and shared storage capacity, while respecting SKU-level MOQ, pack-size,
and maximum-order assumptions.

Demand over the lead-time + review protection window is approximated as Normal:
daily P50 values are summed and daily standard deviations are inferred from
(P90-P10)/2.563, with independent daily errors. These are planning assumptions,
not an identified demand distribution. The target remains observed sales.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from botocore.config import Config
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.artifact_provenance import build_provenance, upload_versioned_artifact


SCENARIOS = (
    {"name": "lead1_review1", "lead_time_days": 1, "review_period_days": 1},
    {"name": "lead3_review2", "lead_time_days": 3, "review_period_days": 2},
    {"name": "lead5_review2", "lead_time_days": 5, "review_period_days": 2},
)
SERVICE_LEVELS = (0.85, 0.90, 0.95)
CAPACITY_DAYS = (2.0, 4.0, 7.0)
Z_CANDIDATES = (0.50, 0.65, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98)
NORMAL_Q10_Q90_SPAN = 2.5631031311


def env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


def r2_client():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{env('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
        aws_access_key_id=env("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=env("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
    )


def expected_normal_components(order_up_to: float, mean: float, std: float) -> dict[str, float]:
    """Expected cycle service, fulfilled quantity, shortage and overage for Normal demand."""
    q = max(0.0, float(order_up_to))
    mu = max(0.0, float(mean))
    sigma = max(float(std), 1e-8)
    z = (q - mu) / sigma
    cdf = float(norm.cdf(z))
    density = float(norm.pdf(z))
    expected_shortage = max(0.0, sigma * density + (mu - q) * (1.0 - cdf))
    expected_overage = max(0.0, (q - mu) * cdf + sigma * density)
    expected_fulfilled = max(0.0, min(mu, mu - expected_shortage))
    return {
        "cycle_service_level": cdf,
        "expected_fulfilled_units": expected_fulfilled,
        "expected_shortage_units": expected_shortage,
        "expected_overage_units": expected_overage,
        "expected_fill_rate": expected_fulfilled / max(mu, 1e-12),
    }


def estimate_protection_demand(
    group: pd.DataFrame, lead_time_days: int, review_period_days: int
) -> dict[str, float]:
    """Approximate lead+review demand distribution from daily P10/P50/P90 forecasts."""
    horizon = int(lead_time_days + review_period_days)
    g = group.sort_values("date").head(horizon)
    if len(g) < horizon:
        raise ValueError(f"Need {horizon} consecutive forecast days, received {len(g)}")
    dates = pd.to_datetime(g["date"]).sort_values()
    expected_dates = pd.date_range(dates.iloc[0], periods=horizon, freq="D")
    if not np.array_equal(dates.to_numpy(dtype="datetime64[ns]"), expected_dates.to_numpy(dtype="datetime64[ns]")):
        raise ValueError("Protection-period forecast dates must be contiguous")
    p10 = np.maximum(g["p10"].to_numpy(dtype=float), 0.0)
    p50 = np.maximum(g["p50"].to_numpy(dtype=float), 0.0)
    p90 = np.maximum(g["p90"].to_numpy(dtype=float), 0.0)
    daily_sigma = np.maximum((p90 - p10) / NORMAL_Q10_Q90_SPAN, 1e-6)
    mean = float(p50.sum())
    std = float(np.sqrt(np.square(daily_sigma).sum()))
    return {
        "protection_horizon_days": float(horizon),
        "expected_demand": mean,
        "demand_std": std,
        "mean_daily_demand": mean / horizon,
    }


def candidate_order_levels(
    mean: float,
    std: float,
    service_level: float,
    max_order_qty: float,
    min_order_qty: float,
    pack_size: float,
) -> list[float]:
    """Create a compact discrete set of operationally feasible order-up-to levels."""
    if pack_size <= 0 or min_order_qty < 0 or max_order_qty < 0:
        raise ValueError("pack_size must be positive; order bounds must be non-negative")
    levels = {0.0}
    quantiles = set(Z_CANDIDATES) | {float(service_level)}
    for probability in quantiles:
        z = float(norm.ppf(probability))
        raw = max(0.0, mean + z * max(std, 1e-8))
        rounded = float(np.ceil(raw / pack_size - 1e-12) * pack_size)
        if rounded > 0 and rounded < min_order_qty:
            rounded = float(np.ceil(min_order_qty / pack_size - 1e-12) * pack_size)
        if rounded <= max_order_qty + 1e-9:
            levels.add(min(rounded, max_order_qty))
    max_feasible = float(np.floor(max_order_qty / pack_size + 1e-12) * pack_size)
    if max_feasible > 0:
        levels.add(max_feasible)
    return sorted(levels)


def _sku_options(
    demand: dict[str, float],
    service_level: float,
    holding_cost_per_unit_day: float,
    shortage_penalty_per_unit: float,
    ordering_cost_per_order: float,
    review_period_days: int,
    min_order_qty: float,
    pack_size: float,
    max_order_qty_days: float,
) -> list[dict[str, float]]:
    max_order = max_order_qty_days * demand["mean_daily_demand"]
    levels = candidate_order_levels(
        demand["expected_demand"], demand["demand_std"], service_level,
        max_order, min_order_qty, pack_size,
    )
    options = []
    holding_days = max(0.5, review_period_days / 2.0)
    for q in levels:
        component = expected_normal_components(q, demand["expected_demand"], demand["demand_std"])
        cost = (
            holding_cost_per_unit_day * holding_days * component["expected_overage_units"]
            + shortage_penalty_per_unit * component["expected_shortage_units"]
            + (ordering_cost_per_order if q > 0 else 0.0)
        )
        options.append({
            "order_qty": float(q),
            "expected_fulfilled_units": component["expected_fulfilled_units"],
            "expected_shortage_units": component["expected_shortage_units"],
            "expected_overage_units": component["expected_overage_units"],
            "cycle_service_level": component["cycle_service_level"],
            "expected_fill_rate": component["expected_fill_rate"],
            "expected_cost": float(cost),
        })
    return options


def optimize_store(
    frame: pd.DataFrame,
    service_level: float = 0.90,
    capacity_days: float = 4.0,
    lead_time_days: int = 1,
    review_period_days: int = 1,
    holding_cost_per_unit_day: float = 0.05,
    shortage_penalty_per_unit: float = 2.0,
    ordering_cost_per_order: float = 5.0,
    min_order_qty: float = 0.2,
    pack_size: float = 0.1,
    max_order_qty_days: float = 14.0,
    service_shortfall_penalty: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Solve a store-level multiple-choice MILP under shared capacity and fill-rate constraints."""
    if not 0 < service_level < 1:
        raise ValueError("service_level must be between 0 and 1")
    if capacity_days <= 0 or lead_time_days < 1 or review_period_days < 1:
        raise ValueError("capacity_days, lead time and review period must be positive")
    if min(holding_cost_per_unit_day, shortage_penalty_per_unit, ordering_cost_per_order) < 0:
        raise ValueError("Cost assumptions must be non-negative")

    products = []
    total_daily_mean = 0.0
    for (store, product), group in frame.groupby(["store_id", "product_id"], sort=True):
        demand = estimate_protection_demand(group, lead_time_days, review_period_days)
        total_daily_mean += demand["mean_daily_demand"]
        options = _sku_options(
            demand, service_level, holding_cost_per_unit_day, shortage_penalty_per_unit,
            ordering_cost_per_order, review_period_days, min_order_qty, pack_size,
            max_order_qty_days,
        )
        products.append({"store_id": str(store), "product_id": str(product), "demand": demand, "options": options})

    if not products:
        raise ValueError("No complete store-product forecast series supplied")
    capacity_units = float(capacity_days * total_daily_mean)
    total_expected_demand = float(sum(item["demand"]["expected_demand"] for item in products))
    n_choice = sum(len(item["options"]) for item in products)
    slack_idx = n_choice
    n_variables = n_choice + 1
    n_constraints = len(products) + 2
    rows, cols, values = [], [], []
    lower = np.full(n_constraints, -np.inf, dtype=float)
    upper = np.full(n_constraints, np.inf, dtype=float)
    objective = np.zeros(n_variables, dtype=float)
    variable_cursor = 0

    for row_idx, item in enumerate(products):
        for option in item["options"]:
            rows.append(row_idx); cols.append(variable_cursor); values.append(1.0)
            objective[variable_cursor] = option["expected_cost"]
            variable_cursor += 1
        lower[row_idx] = 1.0
        upper[row_idx] = 1.0

    capacity_row = len(products)
    service_row = capacity_row + 1
    variable_cursor = 0
    for item in products:
        for option in item["options"]:
            rows.append(capacity_row); cols.append(variable_cursor); values.append(option["order_qty"])
            rows.append(service_row); cols.append(variable_cursor); values.append(option["expected_fulfilled_units"])
            variable_cursor += 1
    rows.append(service_row); cols.append(slack_idx); values.append(1.0)
    upper[capacity_row] = capacity_units
    lower[service_row] = service_level * total_expected_demand

    matrix = coo_matrix((values, (rows, cols)), shape=(n_constraints, n_variables)).tocsr()
    objective[slack_idx] = service_shortfall_penalty or max(1000.0, shortage_penalty_per_unit * 100.0)
    integrality = np.zeros(n_variables, dtype=int)
    integrality[:n_choice] = 1
    lb = np.zeros(n_variables, dtype=float)
    ub = np.ones(n_variables, dtype=float)
    ub[slack_idx] = total_expected_demand
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"mip_rel_gap": 0.001},
    )
    if result.x is None or not result.success:
        raise RuntimeError(f"Inventory optimization failed for store {frame['store_id'].iloc[0]}: {result.message}")

    selected = []
    variable_cursor = 0
    store_expected_fulfilled = 0.0
    store_expected_shortage = 0.0
    store_expected_cost = 0.0
    order_units = 0.0
    cycle_levels = []
    for item in products:
        option_values = result.x[variable_cursor:variable_cursor + len(item["options"])]
        option_idx = int(np.argmax(option_values))
        option = item["options"][option_idx]
        variable_cursor += len(item["options"])
        demand = item["demand"]
        selected.append({
            "store_id": item["store_id"],
            "product_id": item["product_id"],
            "lead_time_days": lead_time_days,
            "review_period_days": review_period_days,
            "target_service_level": service_level,
            "expected_demand_protection_period": demand["expected_demand"],
            "demand_std_protection_period": demand["demand_std"],
            "recommended_order_up_to_units": option["order_qty"],
            "expected_fulfilled_units": option["expected_fulfilled_units"],
            "expected_shortage_units": option["expected_shortage_units"],
            "expected_overage_units": option["expected_overage_units"],
            "expected_fill_rate_pct": option["expected_fill_rate"] * 100,
            "cycle_service_level_pct": option["cycle_service_level"] * 100,
            "expected_cost": option["expected_cost"],
            "store_capacity_units": capacity_units,
            "capacity_days": capacity_days,
            "pack_size_units": pack_size,
            "minimum_order_units": min_order_qty,
            "max_order_qty_days": max_order_qty_days,
        })
        store_expected_fulfilled += option["expected_fulfilled_units"]
        store_expected_shortage += option["expected_shortage_units"]
        store_expected_cost += option["expected_cost"]
        order_units += option["order_qty"]
        cycle_levels.append(option["cycle_service_level"])

    slack_units = float(result.x[slack_idx])
    summary = {
        "store_id": str(frame["store_id"].iloc[0]),
        "products": len(products),
        "service_level_target_pct": service_level * 100,
        "expected_store_fill_rate_pct": store_expected_fulfilled / max(total_expected_demand, 1e-12) * 100,
        "service_level_shortfall_units": slack_units,
        "service_level_met": slack_units <= max(1e-5, total_expected_demand * 1e-7),
        "expected_demand_units": total_expected_demand,
        "expected_fulfilled_units": store_expected_fulfilled,
        "expected_shortage_units": store_expected_shortage,
        "total_order_up_to_units": order_units,
        "store_capacity_units": capacity_units,
        "capacity_utilization_pct": order_units / max(capacity_units, 1e-12) * 100,
        "expected_total_cost": store_expected_cost,
        "mean_sku_cycle_service_level_pct": float(np.mean(cycle_levels) * 100),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "optimization_gap": float(getattr(result, "mip_gap", 0.0) or 0.0),
    }
    return pd.DataFrame(selected), summary


def run_grid(
    forecasts: pd.DataFrame,
    service_levels=SERVICE_LEVELS,
    capacity_days_grid=CAPACITY_DAYS,
    scenarios=SCENARIOS,
    **costs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run comparable store-level optimizations on the exact same forecast artifact."""
    required = {"date", "store_id", "product_id", "actual_sale_amount", "p10", "p50", "p90"}
    missing = required - set(forecasts.columns)
    if missing:
        raise ValueError(f"Quantile forecast artifact missing columns: {sorted(missing)}")
    if forecasts.duplicated(["store_id", "product_id", "date"]).any():
        raise ValueError("Duplicate store-product-date forecasts")
    forecasts = forecasts.copy()
    forecasts["date"] = pd.to_datetime(forecasts["date"])
    forecasts = forecasts.sort_values(["store_id", "product_id", "date"])
    scenario_summaries, store_summaries = [], []
    reference_plan = pd.DataFrame()
    total_jobs = len(scenarios) * len(service_levels) * len(capacity_days_grid)
    job = 0
    for scenario in scenarios:
        lead, review = int(scenario["lead_time_days"]), int(scenario["review_period_days"])
        for target_service in service_levels:
            for capacity_days in capacity_days_grid:
                job += 1
                print(
                    f"Optimization {job}/{total_jobs}: {scenario['name']} · "
                    f"service={target_service:.0%} · capacity={capacity_days:g} demand-days"
                )
                plans, summaries = [], []
                for _, store_frame in forecasts.groupby("store_id", sort=True):
                    store_frame = store_frame.sort_values(["product_id", "date"])
                    # Run each SKU only once per store; the optimizer groups by product.
                    plan, summary = optimize_store(
                        store_frame, service_level=float(target_service),
                        capacity_days=float(capacity_days), lead_time_days=lead,
                        review_period_days=review, **costs,
                    )
                    plans.append(plan)
                    summaries.append(summary)
                stores = pd.DataFrame(summaries)
                plans_df = pd.concat(plans, ignore_index=True)
                scenario_key = f"{scenario['name']}_service{int(target_service*100)}_capacity{int(capacity_days)}d"
                stores.insert(0, "scenario", scenario_key)
                plans_df.insert(0, "scenario", scenario_key)
                store_summaries.append(stores)
                # Aggregate across stores using demand weights, not unweighted mean fill.
                demand = stores["expected_demand_units"].sum()
                scenario_summary = {
                    "scenario": scenario_key,
                    "lead_time_days": lead,
                    "review_period_days": review,
                    "target_service_level_pct": float(target_service * 100),
                    "capacity_days": float(capacity_days),
                    "stores": int(len(stores)),
                    "store_product_series": int(stores["products"].sum()),
                    "expected_demand_units": float(demand),
                    "expected_fulfilled_units": float(stores["expected_fulfilled_units"].sum()),
                    "expected_fill_rate_pct": float(stores["expected_fulfilled_units"].sum() / max(demand, 1e-12) * 100),
                    "service_level_shortfall_units": float(stores["service_level_shortfall_units"].sum()),
                    "stores_meeting_service_level_pct": float(stores["service_level_met"].mean() * 100),
                    "total_order_up_to_units": float(stores["total_order_up_to_units"].sum()),
                    "total_capacity_units": float(stores["store_capacity_units"].sum()),
                    "capacity_utilization_pct": float(
                        stores["total_order_up_to_units"].sum() / max(stores["store_capacity_units"].sum(), 1e-12) * 100
                    ),
                    "expected_shortage_units": float(stores["expected_shortage_units"].sum()),
                    "expected_total_cost": float(stores["expected_total_cost"].sum()),
                    "solver_success_pct": float((stores["solver_status"] == 0).mean() * 100),
                }
                scenario_summaries.append(scenario_summary)
                if (
                    scenario["name"] == "lead3_review2"
                    and abs(target_service - 0.90) < 1e-9
                    and abs(capacity_days - 4.0) < 1e-9
                ):
                    reference_plan = plans_df
                print(
                    f"  achieved fill={scenario_summary['expected_fill_rate_pct']:.2f}% "
                    f"(target {target_service:.0%}); capacity={scenario_summary['capacity_utilization_pct']:.1f}%; "
                    f"service-feasible stores={scenario_summary['stores_meeting_service_level_pct']:.1f}%"
                )
    return pd.DataFrame(scenario_summaries), pd.concat(store_summaries, ignore_index=True), reference_plan


def main() -> None:
    bucket = env("R2_BUCKET")
    s3 = r2_client()
    with tempfile.TemporaryDirectory(prefix="frn-inventory-optimizer-") as td:
        root = Path(td)
        source = root / "evaluation_quantile_predictions.csv"
        key = "quantile-model-artifacts/evaluation_quantile_predictions.csv"
        s3.download_file(bucket, key, str(source))
        forecasts = pd.read_csv(source, parse_dates=["date"])
        config = {
            "service_levels": list(SERVICE_LEVELS),
            "capacity_days": list(CAPACITY_DAYS),
            "scenarios": list(SCENARIOS),
            "costs": {
                "holding_cost_per_unit_day": 0.05,
                "shortage_penalty_per_unit": 2.0,
                "ordering_cost_per_order": 5.0,
                "min_order_qty": 0.2,
                "pack_size": 0.1,
                "max_order_qty_days": 14.0,
            },
            "protection_period_demand_distribution": "Normal approximation; mean is sum of daily P50; standard deviation combines daily P10-P90 spans assuming independent daily errors.",
            "service_constraint": "Store-level demand-weighted expected fill rate >= target; a heavily penalized continuous slack variable records infeasible capacity/service combinations.",
            "capacity_constraint": "Store shared capacity in normalized observed-sales units = capacity_days * sum of per-SKU mean daily P50 demand.",
            "target_semantics": "Observed sales; may be censored during stockouts, not guaranteed latent demand.",
        }
        summaries, stores, reference_plan = run_grid(forecasts, **config["costs"])
        if reference_plan.empty:
            raise RuntimeError("Reference plan (lead 3/review 2, 90% service, 4 capacity days) was not produced")
        output_keys = [
            "inventory-optimization-artifacts/inventory_optimization_summary.json",
            "inventory-optimization-artifacts/inventory_optimization_scenarios.csv",
            "inventory-optimization-artifacts/inventory_optimization_store_summary.csv",
            "inventory-optimization-artifacts/inventory_optimization_reference_plan.csv",
            "inventory-optimization-artifacts/inventory_optimization_manifest.json",
        ]
        provenance = build_provenance(
            "Stochastic Inventory Optimization Benchmark",
            {key: source},
            str(forecasts["date"].min().date()),
            str(forecasts["date"].max().date()),
            config,
            outputs=output_keys,
        )
        summary = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "provenance": provenance,
            "configuration": config,
            "forecast_rows": int(len(forecasts)),
            "store_product_series": int(forecasts[["store_id", "product_id"]].drop_duplicates().shape[0]),
            "evaluation_period": {
                "start": str(forecasts["date"].min().date()),
                "end": str(forecasts["date"].max().date()),
            },
            "scenario_count": int(len(summaries)),
            "scenario_results": summaries.to_dict(orient="records"),
            "limitations": [
                "Demand is proxied by observed sales and may be stockout-censored.",
                "Daily forecast errors are treated as independent when aggregating protection-period variance.",
                "Normal demand approximation is a planning approximation and may be poor for intermittent or highly skewed series.",
                "Capacity, MOQ, pack size, lead time, review cadence and costs are illustrative assumptions, not retailer measurements.",
                "The output is a replenishment decision-support plan, not automatic purchase-order execution.",
                "The seven-day evaluation horizon cannot validate service outcomes after a replenishment lead time longer than the available forecast window.",
            ],
        }
        payloads = [
            (root / "inventory_optimization_summary.json", json.dumps(summary, indent=2, allow_nan=False), "application/json"),
            (root / "inventory_optimization_scenarios.csv", summaries.to_csv(index=False), "text/csv"),
            (root / "inventory_optimization_store_summary.csv", stores.to_csv(index=False), "text/csv"),
            (root / "inventory_optimization_reference_plan.csv", reference_plan.to_csv(index=False), "text/csv"),
            (root / "inventory_optimization_manifest.json", json.dumps(provenance, indent=2), "application/json"),
        ]
        for path, text, content_type in payloads:
            path.write_text(text, encoding="utf-8")
            base_key = f"inventory-optimization-artifacts/{path.name}"
            destinations = upload_versioned_artifact(s3, bucket, path, base_key, content_type, provenance)
            verified = s3.head_object(Bucket=bucket, Key=destinations[0])
            if int(verified["ContentLength"]) != path.stat().st_size:
                raise RuntimeError(f"R2 size verification failed: {destinations[0]}")
            print(f"Verified {destinations[0]} ({verified['ContentLength']:,} bytes)")
        print("INVENTORY_OPTIMIZATION_BENCHMARK")
        print(summaries.to_string(index=False))
        print(json.dumps({"scenario_count": len(summaries), "reference_plan_rows": len(reference_plan)}, indent=2))


if __name__ == "__main__":
    main()
