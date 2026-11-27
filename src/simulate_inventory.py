"""Backtest P50/P90 order-up-to inventory policies on held-out observed-sales forecasts.

All costs and starting inventory are configurable assumptions, not dataset facts.
Observed sales are used as a demand proxy; stockout-censored demand is not recovered.
"""
import json, os, tempfile
from pathlib import Path
import boto3
import numpy as np
import pandas as pd
from botocore.config import Config

def env_float(name, default):
    return float(os.getenv(name, default))

def env_int(name, default):
    return int(os.getenv(name, default))

def r2_client():
    def required(k):
        v=os.getenv(k)
        if not v: raise RuntimeError(f"Missing secret {k}")
        return v
    return boto3.client(
        "s3",
        endpoint_url=f"https://{required('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
        aws_access_key_id=required("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=required("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts":5,"mode":"standard"}),
    )

def simulate(frame, forecast_col, policy_label, lead_time, review_period, holding_cost,
             shortage_cost, order_cost, initial_stock):
    """Periodic-review order-up-to simulation with explicit warm-up and horizon handling."""
    rows = []
    for (store, product), g in frame.groupby(["store_id", "product_id"], sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        n = len(g)
        actual = g["actual_sale_amount"].to_numpy(dtype=float)
        forecast = np.maximum(g[forecast_col].to_numpy(dtype=float), 0.0)
        inventory = float(initial_stock)
        pipeline = {}
        totals = {"units_ordered": 0.0, "orders_placed": 0, "orders_suppressed": 0,
                  "units_suppressed": 0.0, "shortage": 0.0, "holding": 0.0,
                  "ordering": 0.0, "fulfilled": 0.0}
        steady = {k: 0.0 for k in totals}
        warmup_days = min(lead_time, n)
        for i in range(n):
            inventory += pipeline.pop(i, 0.0)
            if i % review_period == 0:
                horizon_end = min(n, i + lead_time + review_period)
                protection_demand = float(forecast[i:horizon_end].sum())
                inventory_position = inventory + float(sum(pipeline.values()))
                qty = float(np.ceil(max(0.0, protection_demand - inventory_position)))
                if qty > 0:
                    arrival_day = i + lead_time
                    if arrival_day < n:
                        pipeline[arrival_day] = pipeline.get(arrival_day, 0.0) + qty
                        totals["units_ordered"] += qty
                        totals["orders_placed"] += 1
                        totals["ordering"] += order_cost
                        if i >= warmup_days:
                            steady["units_ordered"] += qty
                            steady["orders_placed"] += 1
                            steady["ordering"] += order_cost
                    else:
                        totals["orders_suppressed"] += 1
                        totals["units_suppressed"] += qty
                        if i >= warmup_days:
                            steady["orders_suppressed"] += 1
                            steady["units_suppressed"] += qty
            demand = max(0.0, float(actual[i]))
            sold = min(inventory, demand)
            shortage = max(0.0, demand - sold)
            inventory = max(0.0, inventory - sold)
            holding = inventory * holding_cost
            totals["fulfilled"] += sold
            totals["shortage"] += shortage
            totals["holding"] += holding
            if i >= warmup_days:
                steady["fulfilled"] += sold
                steady["shortage"] += shortage
                steady["holding"] += holding
        total_shortage_cost = totals["shortage"] * shortage_cost
        total_cost = totals["holding"] + total_shortage_cost + totals["ordering"]
        steady_shortage_cost = steady["shortage"] * shortage_cost
        steady_cost = steady["holding"] + steady_shortage_cost + steady["ordering"]
        observed = float(actual.sum())
        steady_actual = float(actual[warmup_days:].sum())
        rows.append({
            "store_id": store, "product_id": product, "policy": policy_label,
            "observed_sales": observed, "units_fulfilled": totals["fulfilled"],
            "unfulfilled_observed_sales": totals["shortage"],
            "fill_rate_pct": totals["fulfilled"] / max(observed, 1e-9) * 100,
            "orders_placed": totals["orders_placed"], "units_ordered": totals["units_ordered"],
            "orders_suppressed_at_horizon": totals["orders_suppressed"],
            "units_suppressed_at_horizon": totals["units_suppressed"],
            "ending_inventory": inventory, "holding_cost": totals["holding"],
            "shortage_cost": total_shortage_cost, "ordering_cost": totals["ordering"],
            "total_simulated_cost": total_cost, "warmup_days_excluded": warmup_days,
            "steady_state_observed_sales": steady_actual,
            "steady_state_fulfilled_units": steady["fulfilled"],
            "steady_state_unfulfilled_units": steady["shortage"],
            "steady_state_fill_rate_pct": steady["fulfilled"] / max(steady_actual, 1e-9) * 100,
            "steady_state_orders_placed": steady["orders_placed"],
            "steady_state_units_ordered": steady["units_ordered"],
            "steady_state_holding_cost": steady["holding"],
            "steady_state_shortage_cost": steady_shortage_cost,
            "steady_state_ordering_cost": steady["ordering"],
            "steady_state_total_cost": steady_cost,
        })
    return pd.DataFrame(rows)

def aggregate(detail, label, params):
    sums=detail[["observed_sales","units_fulfilled","unfulfilled_observed_sales","orders_placed",
                 "units_ordered","holding_cost","shortage_cost","ordering_cost","total_simulated_cost"]].sum()
    return {
        "scenario":label,"policy":str(detail["policy"].iloc[0]),
        "store_product_series":int(len(detail)),
        "observed_sales_proxy_units":float(sums["observed_sales"]),
        "fulfilled_units":float(sums["units_fulfilled"]),
        "unfulfilled_observed_sales_units":float(sums["unfulfilled_observed_sales"]),
        "fill_rate_pct":float(sums["units_fulfilled"]/max(sums["observed_sales"],1e-9)*100),
        "orders_placed":int(sums["orders_placed"]),
        "units_ordered":float(sums["units_ordered"]),
        "holding_cost":float(sums["holding_cost"]),
        "shortage_cost":float(sums["shortage_cost"]),
        "ordering_cost":float(sums["ordering_cost"]),
        "total_simulated_cost":float(sums["total_simulated_cost"]),
        "orders_suppressed_at_horizon":int(detail["orders_suppressed_at_horizon"].sum()),
        "units_suppressed_at_horizon":float(detail["units_suppressed_at_horizon"].sum()),
        "steady_state_observed_sales_proxy_units":float(detail["steady_state_observed_sales"].sum()),
        "steady_state_fulfilled_units":float(detail["steady_state_fulfilled_units"].sum()),
        "steady_state_unfulfilled_units":float(detail["steady_state_unfulfilled_units"].sum()),
        "steady_state_fill_rate_pct":float(detail["steady_state_fulfilled_units"].sum() /
            max(detail["steady_state_observed_sales"].sum(),1e-9)*100),
        "steady_state_orders_placed":int(detail["steady_state_orders_placed"].sum()),
        "steady_state_units_ordered":float(detail["steady_state_units_ordered"].sum()),
        "steady_state_holding_cost":float(detail["steady_state_holding_cost"].sum()),
        "steady_state_shortage_cost":float(detail["steady_state_shortage_cost"].sum()),
        "steady_state_ordering_cost":float(detail["steady_state_ordering_cost"].sum()),
        "steady_state_total_cost":float(detail["steady_state_total_cost"].sum()),
        **params,
    }

def main():
    bucket=os.getenv("R2_BUCKET")
    if not bucket: raise RuntimeError("Missing secret R2_BUCKET")
    lead=env_int("LEAD_TIME_DAYS",1)
    review=env_int("REVIEW_PERIOD_DAYS",1)
    initial=env_float("INITIAL_STOCK_UNITS",0)
    holding=env_float("HOLDING_COST_PER_UNIT_DAY",0.05)
    shortage=env_float("SHORTAGE_PENALTY_PER_UNIT",2.0)
    ordering=env_float("ORDERING_COST_PER_ORDER",5.0)
    if lead<1 or review<1 or min(initial,holding,shortage,ordering)<0:
        raise ValueError("Lead time/review period must be >=1 and costs/stock must be non-negative")
    s3=r2_client()
    with tempfile.TemporaryDirectory(prefix="freshretail-inventory-") as td:
        root=Path(td)
        src=root/"evaluation_quantile_predictions.csv"
        print("Downloading quantile predictions from R2")
        s3.download_file(bucket,"quantile-model-artifacts/evaluation_quantile_predictions.csv",str(src))
        df=pd.read_csv(src,parse_dates=["date"])
        required={"date","store_id","product_id","actual_sale_amount","p50","p90_calibrated","seasonal_naive_lag7"}
        missing=required-set(df.columns)
        if missing: raise ValueError(f"Prediction file missing columns: {sorted(missing)}")
        if df.duplicated(["store_id","product_id","date"]).any():
            raise ValueError("Duplicate store-product-date forecast rows")
        if df[list(required-{"date","store_id","product_id"})].isna().any().any():
            raise ValueError("Null actual/forecast values found")
        df=df.sort_values(["store_id","product_id","date"])
        print(f"Loaded {len(df):,} daily forecasts across {df[['store_id','product_id']].drop_duplicates().shape[0]:,} store-product series")
        params={"lead_time_days":lead,"review_period_days":review,"initial_stock_units_per_series":initial,
                "holding_cost_per_unit_day":holding,"shortage_penalty_per_unit":shortage,
                "ordering_cost_per_order":ordering,
                "demand_proxy":"observed sale_amount; may be censored by stock availability",
                "policy_definition":"order-up-to forecast sum over lead-time plus review protection period; orders placed only on scheduled review days",
                "review_schedule":"day index modulo review period equals zero",
                "warmup_days_excluded_from_steady_state":lead,
                "horizon_rule":"orders that would arrive on or after the final evaluation day are suppressed and reported, not charged",
                "benchmark":"seasonal naive forecast using leakage-safe sales_lag_7"}
        details=[]
        policies=(("seasonal_naive_lag7","seasonal_naive_lag7"),
                  ("p50","p50"),("p90_calibrated","p90_calibrated"))
        for forecast_col, label in policies:
            d=simulate(df,forecast_col,label,lead,review,holding,shortage,ordering,initial)
            details.append(d)
        detail=pd.concat(details,ignore_index=True)
        summaries=[aggregate(d, "base_cost_assumptions",params) for d in details]
        # One-way cost sensitivity: low/base/high shortage penalty, keeping other assumptions fixed.
        sensitivity=[]
        for multiplier,label in ((0.5,"shortage_penalty_low"),(1.0,"shortage_penalty_base"),(2.0,"shortage_penalty_high")):
            for forecast_col, policy_label in policies:
                d=simulate(df,forecast_col,policy_label,lead,review,holding,shortage*multiplier,ordering,initial)
                p={**params,"shortage_penalty_per_unit":shortage*multiplier}
                sensitivity.append(aggregate(d,label,p))
        summary={"assumptions":params,"base_policy_comparison":summaries,
                 "shortage_penalty_sensitivity":sensitivity,
                 "limitations":[
                    "Observed sales are a proxy for demand and may be censored during stockouts.",
                    "Starting stock, costs, lead time and order-up-to policy are user-configurable simulation assumptions.",
                    "The held-out prediction horizon is short; results are a controlled backtest, not measured operational savings.",
                    "Steady-state metrics exclude the first lead-time days per series; full-horizon metrics retain cold-start effects.",
                    "Orders that cannot arrive before the evaluation horizon ends are suppressed and counted separately.",
                    "The seasonal-naive benchmark uses the lag-7 observed-sales feature available at each forecast date.",
                    "The simulator assumes deterministic lead time and does not model shelf life, capacity, pack sizes or supplier constraints."
                 ]}
        out=root/"inventory_simulation_summary.json"
        out.write_text(json.dumps(summary,indent=2))
        detail_path=root/"inventory_simulation_by_series.csv"
        detail.to_csv(detail_path,index=False)
        sens_path=root/"inventory_cost_sensitivity.csv"
        pd.DataFrame(sensitivity).to_csv(sens_path,index=False)
        for f in (out,detail_path,sens_path):
            key=f"inventory-simulation-artifacts/{f.name}"
            content_type="application/json" if f.suffix==".json" else "text/csv"
            s3.upload_file(str(f),bucket,key,ExtraArgs={"ContentType":content_type})
            size=s3.head_object(Bucket=bucket,Key=key)["ContentLength"]
            if size!=f.stat().st_size: raise RuntimeError(f"R2 verification failed for {key}")
            print(f"Verified s3://{bucket}/{key} ({size:,} bytes)")
        print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
