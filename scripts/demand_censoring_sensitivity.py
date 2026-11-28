"""Availability-exposure demand-censoring sensitivity diagnostic.

Estimates a deliberately assumption-driven range for sales potentially missed
during observed out-of-stock hours. This is NOT causal lost-sales estimation.
Hourly rates are learned only from the historical training split.
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


def apply_sensitivity(daily: pd.DataFrame, low_multiplier: float = 0.5,
                      high_multiplier: float = 1.5) -> pd.DataFrame:
    """Build transparent lower/central/upper scenarios around a missing-sales proxy."""
    if not 0 <= low_multiplier <= 1 or high_multiplier < 1:
        raise ValueError("Require 0 <= low_multiplier <= 1 and high_multiplier >= 1")
    result = daily.copy()
    observed = result["observed_sales"].clip(lower=0)
    missing_proxy = result["expected_lost_sales_proxy"].clip(lower=0)
    result["counterfactual_sales_lower"] = observed + low_multiplier * missing_proxy
    result["counterfactual_sales_central"] = observed + missing_proxy
    result["counterfactual_sales_upper"] = observed + high_multiplier * missing_proxy
    result["imputed_share_central_pct"] = (
        100 * missing_proxy / result["counterfactual_sales_central"].clip(lower=1e-9)
    )
    return result


def exposure_bucket(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=[-1, 0, 2, 5, 9, 15, 16],
        labels=["no_stockout", "1-2_hours", "3-5_hours", "6-9_hours", "10-15_hours", "16_hours"],
        include_lowest=True,
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


def main():
    import polars as pl

    bucket = os.getenv("R2_BUCKET")
    if not bucket:
        raise RuntimeError("Missing R2_BUCKET")
    s3 = r2_client()
    with tempfile.TemporaryDirectory(prefix="frn-censoring-") as td:
        root = Path(td)
        train_path = root / "train.parquet"
        eval_path = root / "eval.parquet"
        s3.download_file(bucket, "raw/train.parquet", str(train_path))
        s3.download_file(bucket, "raw/eval.parquet", str(eval_path))
        train = pl.read_parquet(train_path)
        evaluation = pl.read_parquet(eval_path)
        required = {"store_id", "product_id", "dt", "sale_amount", "hours_sale", "hours_stock_status"}
        for label, frame in (("train", train), ("eval", evaluation)):
            missing = required - set(frame.columns)
            if missing:
                raise RuntimeError(f"{label} parquet missing columns: {sorted(missing)}")

        def expand(frame):
            return (
                frame.select(
                    "store_id", "product_id", "dt", "sale_amount",
                    "hours_sale", "hours_stock_status",
                )
                .with_columns(
                    pl.int_ranges(0, pl.col("hours_sale").list.len()).alias("hour_of_day")
                )
                .explode(["hours_sale", "hours_stock_status", "hour_of_day"])
                .rename({"hours_sale": "hourly_sales", "hours_stock_status": "stock_status"})
                .with_columns(
                    pl.col("hour_of_day").cast(pl.Int16),
                    pl.col("stock_status").cast(pl.Int8),
                    pl.col("hourly_sales").cast(pl.Float64),
                )
            )

        train_hourly = expand(train)
        eval_hourly = expand(evaluation)
        invalid = train_hourly.filter(~pl.col("stock_status").is_in([0, 1])).height
        invalid += eval_hourly.filter(~pl.col("stock_status").is_in([0, 1])).height
        if invalid:
            raise RuntimeError(f"Found {invalid:,} stock-status slots outside documented 0/1 encoding")
        # Status 0 = available, status 1 = out of stock, following the dataset
        # convention already used by scripts/stockout_diagnostics.py.
        series_rates = (
            train_hourly.filter(pl.col("stock_status") == 0)
            .group_by(["store_id", "product_id", "hour_of_day"])
            .agg(pl.col("hourly_sales").mean().alias("series_hour_rate"))
        )
        global_rates = (
            train_hourly.filter(pl.col("stock_status") == 0)
            .group_by("hour_of_day")
            .agg(pl.col("hourly_sales").mean().alias("global_hour_rate"))
        )
        scored = (
            eval_hourly.join(series_rates, on=["store_id", "product_id", "hour_of_day"], how="left")
            .join(global_rates, on="hour_of_day", how="left")
            .with_columns([
                pl.coalesce(["series_hour_rate", "global_hour_rate", pl.lit(0.0)]).alias("hour_rate"),
                (pl.col("series_hour_rate").is_null() & pl.col("global_hour_rate").is_not_null()).alias("global_fallback"),
            ])
            .with_columns(
                pl.when(pl.col("stock_status") == 1)
                .then(pl.col("hour_rate"))
                .otherwise(0.0)
                .alias("expected_lost_sales_hour")
            )
        )
        daily_hourly = (
            scored.group_by(["store_id", "product_id", "dt"])
            .agg([
                pl.len().alias("hourly_slots"),
                (pl.col("stock_status") == 1).sum().alias("stockout_hours_24"),
                ((pl.col("stock_status") == 1) & pl.col("hour_of_day").is_between(6, 21)).sum()
                    .alias("stockout_hours_06_21"),
                pl.col("expected_lost_sales_hour").sum().alias("expected_lost_sales_proxy"),
                (pl.col("global_fallback") & (pl.col("stock_status") == 1)).sum()
                    .alias("global_fallback_stockout_slots"),
            ])
        )
        observed = evaluation.select(
            "store_id", "product_id", "dt",
            pl.col("sale_amount").cast(pl.Float64).alias("observed_sales"),
        )
        daily = daily_hourly.join(observed, on=["store_id", "product_id", "dt"], how="inner").to_pandas()
        if not (daily["hourly_slots"] == 24).all():
            raise RuntimeError("Expected 24 hourly slots per store-product-date; inspect source vector lengths")
        daily = apply_sensitivity(daily)
        daily["stockout_bucket"] = exposure_bucket(daily["stockout_hours_06_21"])
        bucket_summary = (
            daily.groupby("stockout_bucket", observed=True)
            .agg(
                rows=("observed_sales", "size"),
                mean_stockout_hours_06_21=("stockout_hours_06_21", "mean"),
                observed_sales_units=("observed_sales", "sum"),
                expected_lost_sales_proxy_units=("expected_lost_sales_proxy", "sum"),
                counterfactual_sales_lower_units=("counterfactual_sales_lower", "sum"),
                counterfactual_sales_central_units=("counterfactual_sales_central", "sum"),
                counterfactual_sales_upper_units=("counterfactual_sales_upper", "sum"),
                mean_imputed_share_central_pct=("imputed_share_central_pct", "mean"),
                global_fallback_slots=("global_fallback_stockout_slots", "sum"),
            )
            .reset_index()
        )
        summary = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "reference_split": "raw/train.parquet, in-stock hourly observations only",
            "evaluation_split": "raw/eval.parquet",
            "evaluation_rows": int(len(daily)),
            "evaluation_start": str(evaluation["dt"].min()),
            "evaluation_end": str(evaluation["dt"].max()),
            "observed_sales_units": float(daily["observed_sales"].sum()),
            "expected_lost_sales_proxy_units": float(daily["expected_lost_sales_proxy"].sum()),
            "stockout_affected_row_rate_pct": float(100 * (daily["stockout_hours_06_21"] > 0).mean()),
            "global_fallback_stockout_slots": int(daily["global_fallback_stockout_slots"].sum()),
            "sensitivity_multipliers": {"lower": 0.5, "central": 1.0, "upper": 1.5},
            "interpretation": (
                "Counterfactual sales are sensitivity scenarios, not identified demand. "
                "The central proxy imputes each out-of-stock hour using the historical "
                "training-split mean hourly sales for the same store-product-hour; when "
                "unavailable it falls back to the global hour mean. It assumes in-stock "
                "hourly rates are representative of out-of-stock hours."
            ),
            "limitations": [
                "In-stock sales rates may differ systematically from demand during stockouts.",
                "Within-hour demand, substitutions, promotions and price responses are not observed.",
                "A missing-sales proxy is not a causal lost-sales estimate or a validated demand label.",
                "The diagnostic is retrospective and must not be used as a training target without independent validation.",
                "Status encoding is assumed to follow the documented 0=available, 1=out-of-stock convention.",
            ],
        }
        daily_path = root / "demand_censoring_sensitivity_daily.csv"
        bucket_path = root / "demand_censoring_sensitivity_by_bucket.csv"
        summary_path = root / "demand_censoring_sensitivity.json"
        daily.to_csv(daily_path, index=False)
        bucket_summary.to_csv(bucket_path, index=False)
        summary_path.write_text(json.dumps(summary, indent=2, default=str))
        uploads = [
            (daily_path, "diagnostics/demand-censoring/demand_censoring_sensitivity_daily.csv", "text/csv"),
            (bucket_path, "diagnostics/demand-censoring/demand_censoring_sensitivity_by_bucket.csv", "text/csv"),
            (summary_path, "diagnostics/demand-censoring/demand_censoring_sensitivity.json", "application/json"),
        ]
        for path, key, content_type in uploads:
            s3.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": content_type})
            size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
            if size != path.stat().st_size:
                raise RuntimeError(f"R2 verification failed for {key}")
            print(f"Verified s3://{bucket}/{key} ({size:,} bytes)")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
