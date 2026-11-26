"""Validate, profile, and export FreshRetailNet datasets in Cloudflare R2.

The pipeline preserves the source values, columns, and row order. It writes
Zstandard Parquet plus plain CSV copies and data-quality metadata to the same
private R2 bucket. CSV export uses Polars' streaming sink.
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


def upload_and_verify(s3, bucket: str, path: Path, key: str, content_type: str) -> int:
    print(f"Uploading {path.name} to s3://{bucket}/{key}")
    s3.upload_file(
        str(path),
        bucket,
        key,
        ExtraArgs={"ContentType": content_type},
    )
    head = s3.head_object(Bucket=bucket, Key=key)
    actual_size = int(head["ContentLength"])
    expected_size = path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"R2 size verification failed for {key}: expected {expected_size}, got {actual_size}"
        )
    print(f"Verified {key}: {actual_size:,} bytes")
    return actual_size


def profile_reencode_and_export(
    name: str, source_path: Path, workdir: Path
) -> tuple[Path, Path, dict, list[dict]]:
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

    parquet_path = workdir / f"{name}.parquet"
    lazy.sink_parquet(
        parquet_path,
        compression="zstd",
        compression_level=6,
        statistics=True,
        maintain_order=True,
        engine="streaming",
    )

    # Plain CSV, streamed to disk rather than collecting the full table in memory.
    csv_path = workdir / f"{name}.csv"
    pl.scan_parquet(parquet_path).sink_csv(
        csv_path,
        include_header=True,
        maintain_order=True,
        engine="streaming",
    )

    metadata = {
        "dataset": name,
        "source_object": INPUTS[name],
        "processed_parquet_object": f"{OUTPUT_PREFIX}/{name}.parquet",
        "processed_csv_object": f"{OUTPUT_PREFIX}/{name}.csv",
        "rows": row_count,
        "columns": len(columns),
        "source_bytes": source_path.stat().st_size,
        "processed_parquet_bytes": parquet_path.stat().st_size,
        "processed_csv_bytes": csv_path.stat().st_size,
        "source_sha256": sha256_file(source_path),
        "processed_parquet_sha256": sha256_file(parquet_path),
        "processed_csv_sha256": sha256_file(csv_path),
        "schema": {col: str(schema[col]) for col in columns},
        "null_cells": sum(int(summary[col]) for col in columns),
    }
    return parquet_path, csv_path, metadata, null_profile


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

            parquet_path, csv_path, metadata, null_profile = profile_reencode_and_export(
                name, local_input, workdir
            )
            datasets.append(metadata)
            all_null_profiles.extend(null_profile)

            upload_and_verify(
                s3, bucket, parquet_path, f"{OUTPUT_PREFIX}/{name}.parquet",
                "application/vnd.apache.parquet",
            )
            upload_and_verify(
                s3, bucket, csv_path, f"{OUTPUT_PREFIX}/{name}.csv", "text/csv",
            )
            print(
                f"Completed {name}: rows={metadata['rows']:,}, "
                f"columns={metadata['columns']}, "
                f"CSV={metadata['processed_csv_bytes']:,} bytes"
            )

        train_cols = list(datasets[0]["schema"].keys())
        eval_cols = list(datasets[1]["schema"].keys())
        if train_cols != eval_cols:
            raise RuntimeError(
                "Train/eval column names or order differ. Files were uploaded; "
                "inspect the schema report before modelling."
            )

        report = {
            "project": "FreshRetailNet-50K",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "processing_policy": (
                "Schema-preserving Parquet re-encoding and CSV export; "
                "no rows dropped, columns renamed, or values imputed."
            ),
            "bucket": bucket,
            "datasets": datasets,
        }
        report_path = workdir / "data_quality_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        profile_path = workdir / "null_profile.csv"
        pl.DataFrame(all_null_profiles).write_csv(profile_path)

        upload_and_verify(
            s3, bucket, report_path, f"{OUTPUT_PREFIX}/data_quality_report.json",
            "application/json",
        )
        upload_and_verify(
            s3, bucket, profile_path, f"{OUTPUT_PREFIX}/null_profile.csv", "text/csv",
        )

    print("Processing complete. Parquet, CSV, and profiling reports are in the same R2 bucket.")


if __name__ == "__main__":
    main()
