"""Fetch the pinned FreshRetailNet-LT v1.0 release and stage it in R2.

Source: Dingdong-Inc/FreshRetailNet-LT, immutable Hugging Face dataset revision.
The source is a separate, later benchmark population; it is not row-compatible
with the original FreshRetailNet-50K population and must be reported as an
external validation dataset.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import boto3
import polars as pl
from botocore.config import Config

DATASET = "Dingdong-Inc/FreshRetailNet-LT"
REVISION = "01c3bb306415e09d3ccd2fe762f33eb3aeb2affd"
BASE_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data"
PREFIX = "external/freshretailnet-lt/v1.0"
EXPECTED = {
    "train": {
        "rows": 7_869_549,
        "sha256": "e744c537ed16a8254c50f8456de7c5614866d4d8b6787e3cc3b67be02ea2f1e3",
        "key": f"{PREFIX}/train.parquet",
    },
    "eval": {
        "rows": 70_000,
        "sha256": "370ca7f995d2a059744e1659eb681e60db694bec2f80a84252426afdd2b6d626",
        "key": f"{PREFIX}/eval.parquet",
    },
}
REQUIRED = {
    "city_id", "store_id", "management_group_id", "first_category_id",
    "second_category_id", "third_category_id", "product_id", "dt",
    "sale_amount", "hours_sale", "stock_hour6_22_cnt", "hours_stock_status",
    "activity_flag", "discount", "holiday_flag", "precpt",
    "avg_temperature", "avg_humidity", "avg_wind_level",
}


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


def download_and_hash(split: str, destination: Path) -> dict:
    spec = EXPECTED[split]
    url = f"{BASE_URL}/{split}.parquet?download=true"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "FreshRetailNet-Demand-Intelligence/1.0"},
    )
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(request, timeout=90) as response, destination.open("wb") as handle:
        while True:
            block = response.read(8 * 1024 * 1024)
            if not block:
                break
            handle.write(block)
            digest.update(block)
            size += len(block)
    actual = digest.hexdigest()
    if actual != spec["sha256"]:
        raise RuntimeError(
            f"{split} checksum mismatch: expected={spec['sha256']}, actual={actual}"
        )
    return {
        "url": url,
        "revision": REVISION,
        "sha256": actual,
        "bytes": size,
        "expected_rows": spec["rows"],
        "r2_key": spec["key"],
    }


def validate_split(path: Path, split: str) -> dict:
    frame = pl.scan_parquet(path)
    columns = set(frame.collect_schema().names())
    missing = sorted(REQUIRED - columns)
    if missing:
        raise RuntimeError(f"{split} is missing required columns: {missing}")
    metadata = frame.select(
        pl.len().alias("rows"),
        pl.col("dt").cast(pl.String).str.to_date(strict=False).min().alias("date_min"),
        pl.col("dt").cast(pl.String).str.to_date(strict=False).max().alias("date_max"),
        pl.struct(["store_id", "product_id"]).n_unique().alias("series"),
        pl.col("sale_amount").is_null().sum().alias("null_sales"),
    ).collect().row(0, named=True)
    if int(metadata["rows"]) != EXPECTED[split]["rows"]:
        raise RuntimeError(
            f"{split} row count mismatch: expected {EXPECTED[split]['rows']}, got {metadata['rows']}"
        )
    if int(metadata["series"]) < 10_000:
        raise RuntimeError(f"{split} contains implausibly few store-product series: {metadata['series']}")
    if int(metadata["null_sales"]) != 0:
        raise RuntimeError(f"{split} contains null sale_amount values")
    duplicate_count = (
        frame.group_by(["store_id", "product_id", "dt"])
        .len()
        .filter(pl.col("len") > 1)
        .select(pl.len())
        .collect()
        .item()
    )
    if duplicate_count:
        raise RuntimeError(f"{split} contains {duplicate_count:,} duplicate store-product-date keys")
    return {
        "rows": int(metadata["rows"]),
        "series": int(metadata["series"]),
        "date_min": str(metadata["date_min"]),
        "date_max": str(metadata["date_max"]),
        "schema_columns": sorted(columns),
    }


def main():
    bucket = os.getenv("R2_BUCKET")
    if not bucket:
        raise RuntimeError("Missing R2_BUCKET")
    client = r2_client()
    with tempfile.TemporaryDirectory(prefix="frn-lt-ingest-") as temp_dir:
        root = Path(temp_dir)
        report = {
            "dataset": DATASET,
            "revision": REVISION,
            "license": "CC BY 4.0",
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_release_date": "2026-02-02",
            "official_labeled_evaluation_dates": ["2025-07-14", "2025-07-20"],
            "population_note": (
                "FreshRetailNet-LT is a distinct external population, with over 20K store-product "
                "series, 1,057 stores and 576 SKUs; do not pool its metrics with FreshRetailNet-50K."
            ),
            "files": {},
        }
        for split in ("train", "eval"):
            path = root / f"{split}.parquet"
            print(f"Downloading pinned {DATASET} {split} split at {REVISION}")
            download_info = download_and_hash(split, path)
            split_info = validate_split(path, split)
            download_info.update(split_info)
            report["files"][split] = download_info
            client.upload_file(
                str(path),
                bucket,
                EXPECTED[split]["key"],
                ExtraArgs={
                    "ContentType": "application/vnd.apache.parquet",
                    "Metadata": {
                        "dataset": DATASET,
                        "revision": REVISION,
                        "sha256": download_info["sha256"],
                        "split": split,
                    },
                },
            )
            head = client.head_object(Bucket=bucket, Key=EXPECTED[split]["key"])
            if int(head["ContentLength"]) != download_info["bytes"]:
                raise RuntimeError(f"R2 length verification failed for {split}")
            print(
                f"Verified R2 {EXPECTED[split]['key']}: {download_info['rows']:,} rows, "
                f"{download_info['series']:,} series, {download_info['date_min']}..{download_info['date_max']}"
            )
        train_end = report["files"]["train"]["date_max"]
        eval_start = report["files"]["eval"]["date_min"]
        eval_end = report["files"]["eval"]["date_max"]
        if not train_end < eval_start:
            raise RuntimeError(
                f"Official LT split is not strictly chronological: train_end={train_end}, eval_start={eval_start}"
            )
        if eval_start != "2025-07-14" or eval_end != "2025-07-20":
            raise RuntimeError(
                f"Unexpected official evaluation dates: {eval_start}..{eval_end}"
            )
        report["train_end_before_eval_start"] = True
        report_path = root / "freshretailnet_lt_ingestion_manifest.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        client.upload_file(
            str(report_path),
            bucket,
            f"{PREFIX}/ingestion_manifest.json",
            ExtraArgs={"ContentType": "application/json"},
        )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
