"""Validate, profile, and re-encode FreshRetailNet Parquet files in Cloudflare R2.

This first processing stage is intentionally schema-preserving: it does not drop
rows, impute values, or rename columns. It writes Zstandard-compressed Parquet
copies and data-quality metadata back to the same R2 bucket.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import boto3
import polars as pl
from botocore.config import Config

INPUTS = {
    "train": "raw/train.parquet",
    "eval": "raw/eval.parquet",
}
OUTPUT_PREFIX = "processed"


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_and_reencode(name: str, source_path: Path, workdir: Path) -> tuple[Path, dict, list[dict]]:
    lazy = pl.scan_parquet(source_path)
    schema = lazy.collect_schema()
    columns = list(schema.names())

    # One streaming aggregation for row count and per-column null counts.
    expressions = [pl.len().alias("__row_count__")]
    expressions.extend(pl.col(col).null_count().alias(col) for col in columns)
    summary = lazy.select(expressions).collect(engine="streaming").row(0, named=True)
    row_count = int(summary.pop("__row_count__"))

    null_profile = []
    for col in columns:
        nulls = int(summary[col])
        null_profile.append({
            "dataset": name,
            "column": col,
            "dtype": str(schema[col]),
            "null_count": nulls,
            "null_fraction": (nulls / row_count) if row_count else None,
        })

    output_path = workdir / f"{name}.parquet"
    # Re-encode without changing values, columns, or row order.
    lazy.sink_parquet(
        output_path,
        compression="zstd",
        compression_level=6,
        statistics=True,
        maintain_order=True,
        engine="streaming",
    )

    metadata = {
        "dataset": name,
        "source_object": INPUTS[name],
        "processed_object": f"{OUTPUT_PREFIX}/{name}.parquet",
        "rows": row_count,
        "columns": len(columns),
        "source_bytes": source_path.stat().st_size,
        "processed_bytes": output_path.stat().st_size,
        "source_sha256": sha256_file(source_path),
        "processed_sha256": sha256_file(output_path),
        "schema": {col: str(schema[col]) for col in columns},
        "null_cells": sum(int(summary[col]) for col in columns),
    }
    return output_path, metadata, null_profile


def main() -> None:
    account_id = required_env("R2_ACCOUNT_ID")
    access_key = required_env("R2_ACCESS_KEY_ID")
    secret_key = required_env("R2_SECRET_ACCESS_KEY")
    bucket = required_env("R2_BUCKET")

    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
    )

    with tempfile.TemporaryDirectory(prefix="freshretailnet-") as temp:
        workdir = Path(temp)
        datasets = []
        all_null_profiles = []

        for name, key in INPUTS.items():
            local_input = workdir / f"source_{name}.parquet"
            print(f"Downloading s3://{bucket}/{key}")
            s3.download_file(bucket, key, str(local_input))
            if not local_input.exists() or local_input.stat().st_size == 0:
                raise RuntimeError(f"Missing or empty input object: {key}")

            output_path, metadata, null_profile = profile_and_reencode(name, local_input, workdir)
            datasets.append(metadata)
            all_null_profiles.extend(null_profile)

            output_key = f"{OUTPUT_PREFIX}/{name}.parquet"
            print(f"Uploading processed Parquet to s3://{bucket}/{output_key}")
            s3.upload_file(
                str(output_path),
                bucket,
                output_key,
                ExtraArgs={"ContentType": "application/vnd.apache.parquet"},
            )
            head = s3.head_object(Bucket=bucket, Key=output_key)
            if head["ContentLength"] != output_path.stat().st_size:
                raise RuntimeError(f"R2 size verification failed for {output_key}")
            print(f"Verified {output_key}: {head['ContentLength']:,} bytes; rows={metadata['rows']}")

        # The two splits should have the same ordered column names for consistent modelling.
        train_cols = list(datasets[0]["schema"].keys())
        eval_cols = list(datasets[1]["schema"].keys())
        if train_cols != eval_cols:
            raise RuntimeError(
                "Train/eval column names or order differ. Processed Parquet files were uploaded; "
                "inspect the schema report before modelling."
            )

        report = {
            "project": "FreshRetailNet-50K",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "processing_policy": "Schema-preserving Zstandard re-encoding; no rows dropped or values imputed.",
            "bucket": bucket,
            "datasets": datasets,
        }
        report_path = workdir / "data_quality_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        profile_path = workdir / "null_profile.csv"
        pl.DataFrame(all_null_profiles).write_csv(profile_path)

        for local_path, key, content_type in [
            (report_path, f"{OUTPUT_PREFIX}/data_quality_report.json", "application/json"),
            (profile_path, f"{OUTPUT_PREFIX}/null_profile.csv", "text/csv"),
        ]:
            s3.upload_file(
                str(local_path),
                bucket,
                key,
                ExtraArgs={"ContentType": content_type},
            )
            s3.head_object(Bucket=bucket, Key=key)
            print(f"Uploaded report: s3://{bucket}/{key}")

    print("Processing complete. Processed Parquet and profiling reports are in the same R2 bucket.")


if __name__ == "__main__":
    main()
