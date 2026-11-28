"""Fail-closed readiness gate for a later untouched FreshRetailNet snapshot.

The released benchmark snapshot ends at 2024-07-02 in this project. A later
holdout must be supplied separately as raw/later_holdout.parquet. This gate
checks temporal separation, minimum duration, schema and population coverage;
it does not manufacture a later test from already-used benchmark dates.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import boto3
import polars as pl
from botocore.config import Config
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.artifact_provenance import build_provenance, upload_versioned_artifact

REQUIRED = {"store_id", "product_id", "dt", "sale_amount", "hours_sale", "hours_stock_status"}
MIN_HOLDOUT_DAYS = 14
MIN_SERIES_COVERAGE_PCT = 95.0


def validate_later_holdout(base_eval_max: str, base_series: int, later: pl.DataFrame) -> dict:
    missing = REQUIRED - set(later.columns)
    if missing:
        raise ValueError(f"Later holdout missing required columns: {sorted(missing)}")
    dates = sorted(later.select(pl.col("dt").str.to_date(strict=False).unique()).to_series().to_list())
    latest_base = datetime.strptime(base_eval_max, "%Y-%m-%d").date()
    if dates and dates[0] <= latest_base:
        raise ValueError(
            f"Later holdout starts {dates[0]}, which is not strictly after the benchmark "
            f"evaluation end {latest_base}"
        )
    if len(dates) < MIN_HOLDOUT_DAYS:
        raise ValueError(
            f"Later holdout has {len(dates)} unique dates; at least {MIN_HOLDOUT_DAYS} are required"
        )
    duplicate_rows = later.select(
        pl.struct(["store_id", "product_id", "dt"]).is_duplicated().sum()
    ).item()
    if duplicate_rows:
        raise ValueError(f"Later holdout contains {duplicate_rows:,} duplicate store-product-date rows")
    series_count = later.select(["store_id", "product_id"]).unique().height
    coverage = 100.0 * series_count / max(base_series, 1)
    if coverage < MIN_SERIES_COVERAGE_PCT:
        raise ValueError(
            f"Later holdout has {series_count:,}/{base_series:,} store-product series "
            f"({coverage:.2f}%); minimum is {MIN_SERIES_COVERAGE_PCT:.1f}%"
        )
    gaps = []
    for previous, current in zip(dates, dates[1:]):
        if (current - previous).days != 1:
            gaps.append({"from": str(previous), "to": str(current), "gap_days": (current - previous).days - 1})
    if gaps:
        raise ValueError(f"Later holdout must be a contiguous daily window; gaps found: {gaps[:5]}")
    return {
        "status": "READY_FOR_FROZEN_MODEL_EVALUATION",
        "benchmark_evaluation_end": str(latest_base),
        "later_holdout_start": str(dates[0]),
        "later_holdout_end": str(dates[-1]),
        "later_holdout_days": len(dates),
        "later_holdout_rows": int(later.height),
        "later_holdout_store_product_series": int(series_count),
        "benchmark_store_product_series": int(base_series),
        "series_coverage_pct": coverage,
        "model_scoring_performed": False,
        "next_step": "Score this frozen data snapshot with the frozen model artifacts; do not tune on these results.",
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


def main():
    bucket = os.getenv("R2_BUCKET")
    if not bucket:
        raise RuntimeError("Missing R2_BUCKET")
    s3 = r2_client()
    with tempfile.TemporaryDirectory(prefix="frn-later-holdout-gate-") as td:
        root = Path(td)
        eval_path = root / "eval.parquet"
        later_path = root / "later_holdout.parquet"
        s3.download_file(bucket, "raw/eval.parquet", str(eval_path))
        try:
            s3.download_file(bucket, "raw/later_holdout.parquet", str(later_path))
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in {"404", "NoSuchKey", "NotFound"}:
                raise RuntimeError(
                    "BLOCKED: raw/later_holdout.parquet is not present in R2. "
                    "The released benchmark snapshot ends at the existing evaluation period; "
                    "upload a newer untouched daily snapshot before requesting a promotion check."
                ) from exc
            raise
        evaluation = pl.read_parquet(eval_path)
        later = pl.read_parquet(later_path)
        base_eval_max = str(evaluation.select(pl.col("dt").str.to_date(strict=False).max()).item())
        base_series = evaluation.select(["store_id", "product_id"]).unique().height
        gate = validate_later_holdout(base_eval_max, base_series, later)
        provenance = build_provenance(
            "Later Untouched Holdout Readiness Gate",
            {"raw/eval.parquet": eval_path, "raw/later_holdout.parquet": later_path},
            gate["later_holdout_start"], gate["later_holdout_end"],
            {
                "minimum_holdout_days": MIN_HOLDOUT_DAYS,
                "minimum_series_coverage_pct": MIN_SERIES_COVERAGE_PCT,
                "strictly_after_benchmark_evaluation": True,
                "contiguous_daily_window": True,
                "model_scoring_performed": False,
            },
            outputs=["validation-gates/later-holdout/later_holdout_readiness.json"],
        )
        gate["provenance"] = provenance
        report = root / "later_holdout_readiness.json"
        report.write_text(json.dumps(gate, indent=2, default=str))
        upload_versioned_artifact(
            s3, bucket, report,
            "validation-gates/later-holdout/later_holdout_readiness.json",
            "application/json", provenance,
        )
        print(json.dumps(gate, indent=2, default=str))


if __name__ == "__main__":
    main()
