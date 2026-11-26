"""Profile hourly stockouts and their association with observed sales.

Uses the dataset's documented convention: hours_stock_status marks hourly
out-of-stock status; stock_hour6_22_cnt counts out-of-stock hours from 06:00
through 21:00 (16 operating-hour slots). Sales remain observed sales, not
uncensored demand. This script is descriptive, not a lost-sales estimator.
"""
from __future__ import annotations
import json, os, tempfile
from pathlib import Path
from datetime import datetime, timezone

import boto3
import polars as pl
from botocore.config import Config

RAW_KEY = "raw/train.parquet"
OUT_PREFIX = "diagnostics/stockout"
GROUPS = ["no_stockout", "1-2_hours", "3-5_hours", "6-9_hours", "10-15_hours", "16_hours"]

def env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing secret {name}")
    return value

def client():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{env('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
        aws_access_key_id=env("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=env("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
    )

def main():
    s3 = client()
    with tempfile.TemporaryDirectory(prefix="frn-stockout-") as td:
        root = Path(td)
        local = root / "train.parquet"
        print(f"Downloading s3://{env('R2_BUCKET')}/{RAW_KEY}")
        s3.download_file(env("R2_BUCKET"), RAW_KEY, str(local))
        df = pl.read_parquet(local)
        required = {"store_id", "product_id", "dt", "sale_amount",
                    "hours_sale", "hours_stock_status", "stock_hour6_22_cnt"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        # Validate the documented 06:00-22:00 operating window against the
        # supplied count. Keep a mismatch report rather than silently remapping.
        df = df.with_columns([
            pl.col("dt").str.to_date(strict=False).alias("date"),
            pl.col("hours_stock_status").list.len().alias("status_vector_length"),
            pl.col("hours_stock_status").list.slice(6, 16).list.sum().alias("status_06_21_sum"),
            pl.col("hours_stock_status").list.sum().alias("status_all_hours_sum"),
            pl.col("hours_sale").list.sum().alias("hourly_sales_sum"),
        ]).with_columns([
            (pl.col("status_06_21_sum") == pl.col("stock_hour6_22_cnt")).alias("window_count_matches"),
            pl.when(pl.col("stock_hour6_22_cnt") == 0).then(pl.lit("no_stockout"))
              .when(pl.col("stock_hour6_22_cnt") <= 2).then(pl.lit("1-2_hours"))
              .when(pl.col("stock_hour6_22_cnt") <= 5).then(pl.lit("3-5_hours"))
              .when(pl.col("stock_hour6_22_cnt") <= 9).then(pl.lit("6-9_hours"))
              .when(pl.col("stock_hour6_22_cnt") <= 15).then(pl.lit("10-15_hours"))
              .otherwise(pl.lit("16_hours")).alias("stockout_bucket"),
        ])

        # One row per stockout exposure bucket; sales are not interpreted as demand.
        bucket = (df.group_by("stockout_bucket").agg([
            pl.len().alias("row_count"),
            pl.col("sale_amount").mean().alias("mean_observed_sales"),
            pl.col("sale_amount").median().alias("median_observed_sales"),
            pl.col("sale_amount").sum().alias("total_observed_sales"),
            (pl.col("sale_amount") == 0).mean().alias("zero_sales_rate"),
            pl.col("stock_hour6_22_cnt").mean().alias("mean_stockout_hours"),
        ]).sort("stockout_bucket"))
        bucket_path = root / "sales_by_stockout_bucket.csv"
        bucket.write_csv(bucket_path)

        # Expand the 24-hour vectors into hourly status and sales profiles.
        hourly = (df.select([
            pl.col("hours_stock_status").alias("status"),
            pl.col("hours_sale").alias("sales"),
        ]).explode(["status", "sales"])
          .with_row_index("exploded_index")
          .with_columns((pl.col("exploded_index") % 24).alias("hour_of_day"))
          .group_by("hour_of_day").agg([
              pl.len().alias("observations"),
              pl.col("status").mean().alias("out_of_stock_rate"),
              pl.col("sales").mean().alias("mean_observed_hourly_sales"),
              pl.col("sales").median().alias("median_observed_hourly_sales"),
              (pl.col("sales") == 0).mean().alias("zero_hourly_sales_rate"),
          ]).sort("hour_of_day"))
        hourly_path = root / "hourly_stock_status_profile.csv"
        hourly.write_csv(hourly_path)

        # Data integrity and high-level diagnostics.
        integrity = df.select([
            pl.len().alias("rows"),
            pl.col("store_id").n_unique().alias("stores"),
            pl.col("product_id").n_unique().alias("products"),
            pl.col("date").min().cast(pl.String).alias("date_start"),
            pl.col("date").max().cast(pl.String).alias("date_end"),
            (~pl.col("window_count_matches")).sum().alias("window_count_mismatches"),
            pl.col("stock_hour6_22_cnt").mean().alias("mean_stockout_hours_06_22"),
            (pl.col("stock_hour6_22_cnt") > 0).mean().alias("stockout_affected_row_rate"),
            (pl.col("stock_hour6_22_cnt") == 16).mean().alias("full_window_stockout_rate"),
        ]).to_dicts()[0]
        integrity["vector_lengths"] = df.group_by("status_vector_length").len().sort("status_vector_length").to_dicts()
        integrity["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        integrity["interpretation"] = (
            "Association with observed sales only. Stockout-affected sales are censored; "
            "bucket differences are descriptive and are not causal lost-sales estimates."
        )
        integrity["hour_index_note"] = (
            "Hourly vector positions are reported as indices 0-23. Confirm clock-time "
            "alignment against the technical report before assigning local-time labels."
        )
        integrity_path = root / "stockout_diagnostics.json"
        integrity_path.write_text(json.dumps(integrity, indent=2, default=str))
        for path in (bucket_path, hourly_path, integrity_path):
            key = f"{OUT_PREFIX}/{path.name}"
            content_type = "application/json" if path.suffix == ".json" else "text/csv"
            s3.upload_file(str(path), env("R2_BUCKET"), key,
                           ExtraArgs={"ContentType": content_type})
            head = s3.head_object(Bucket=env("R2_BUCKET"), Key=key)
            if head["ContentLength"] != path.stat().st_size:
                raise RuntimeError(f"R2 verification failed for {key}")
            print(f"Verified s3://{env('R2_BUCKET')}/{key} ({path.stat().st_size:,} bytes)")
        print(json.dumps(integrity, indent=2, default=str))
        print(bucket)

if __name__ == "__main__":
    main()
