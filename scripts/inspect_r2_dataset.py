"""Inspect FreshRetailNet data stored in private Cloudflare R2.

Prints the actual Parquet schema, representative rows, date range, cardinalities,
target distribution, and key stock/sales diagnostics to the GitHub Actions log.
"""
from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path

import boto3
import polars as pl
from botocore.config import Config

def env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing secret: {name}")
    return value

def main():
    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{env('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
        aws_access_key_id=env("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=env("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )
    bucket = env("R2_BUCKET")
    with tempfile.TemporaryDirectory() as td:
        for split in ("train", "eval"):
            key = f"raw/{split}.parquet"
            path = Path(td) / f"{split}.parquet"
            s3.download_file(bucket, key, str(path))
            lf = pl.scan_parquet(path)
            schema = lf.collect_schema()
            print(f"\n===== {split.upper()} SCHEMA =====")
            print(json.dumps({c: str(schema[c]) for c in schema.names()}, indent=2))
            print(f"\n===== {split.upper()} SAMPLE (first 5 rows) =====")
            print(lf.head(5).collect().to_dicts())
            print(f"\n===== {split.upper()} SHAPE =====")
            print(lf.select(pl.len().alias("rows")).collect())
            print(f"\n===== {split.upper()} NULL COUNTS =====")
            print(lf.select([pl.col(c).null_count().alias(c) for c in schema.names()]).collect())
            # Candidate semantic diagnostics, only for fields actually present.
            names = set(schema.names())
            print(f"\n===== {split.upper()} FIELD CANDIDATES =====")
            for c in ("date", "dt", "hour", "store_id", "product_id", "sales", "sale", "stock", "stock_status", "price", "discount", "promotion"):
                if c in names:
                    print(f"{c}: {lf.select(pl.col(c).n_unique().alias('n_unique'), pl.col(c).min().alias('min'), pl.col(c).max().alias('max')).collect().to_dicts()}")
if __name__ == "__main__":
    main()
