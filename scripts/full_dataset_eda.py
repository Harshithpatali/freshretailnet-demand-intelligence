"""Build a complete, descriptive EDA report for the original FreshRetailNet-50K data.

The report profiles every source field across the train and evaluation splits.
It deliberately describes sale_amount as normalized observed sales, not
unconstrained customer demand.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import duckdb
from botocore.config import Config

SOURCE_KEYS = {"train": "raw/train.parquet", "evaluation": "raw/eval.parquet"}
OUTPUT_KEY = "dataset-eda/full_dataset_eda.json"
HISTOGRAM_BINS = 20

# This catalog mirrors the 19 columns in the published source dataset.
FEATURE_CATALOG = [
    {"name": "city_id", "label": "City ID", "group": "Store hierarchy", "kind": "identifier",
     "description": "Encoded city identifier."},
    {"name": "store_id", "label": "Store ID", "group": "Store hierarchy", "kind": "identifier",
     "description": "Encoded retail store identifier."},
    {"name": "management_group_id", "label": "Management group", "group": "Product hierarchy", "kind": "identifier",
     "description": "Encoded product management group."},
    {"name": "first_category_id", "label": "First-level category", "group": "Product hierarchy", "kind": "identifier",
     "description": "First level of the product-category hierarchy."},
    {"name": "second_category_id", "label": "Second-level category", "group": "Product hierarchy", "kind": "identifier",
     "description": "Second level of the product-category hierarchy."},
    {"name": "third_category_id", "label": "Third-level category", "group": "Product hierarchy", "kind": "identifier",
     "description": "Third level of the product-category hierarchy."},
    {"name": "product_id", "label": "Product ID", "group": "Product hierarchy", "kind": "identifier",
     "description": "Encoded product identifier."},
    {"name": "dt", "label": "Date", "group": "Time", "kind": "date",
     "description": "Calendar date for the store-product observation."},
    {"name": "sale_amount", "label": "Daily observed sales", "group": "Sales", "kind": "target",
     "description": "Daily sales quantity after global normalization. This is observed sales and can be censored by stockouts."},
    {"name": "hours_sale", "label": "Hourly sales", "group": "Hourly signals", "kind": "vector",
     "description": "24-element sequence of globally normalized hourly observed sales."},
    {"name": "stock_hour6_22_cnt", "label": "Stockout hours (06:00–22:00)", "group": "Availability", "kind": "numeric",
     "description": "Number of out-of-stock hours in the 06:00–22:00 operating window."},
    {"name": "hours_stock_status", "label": "Hourly stock status", "group": "Hourly signals", "kind": "vector",
     "description": "24-element sequence recording hourly out-of-stock status."},
    {"name": "discount", "label": "Discount factor", "group": "Commercial context", "kind": "numeric",
     "description": "Discount factor: 1.0 means no discount and 0.9 means 10% off."},
    {"name": "holiday_flag", "label": "Holiday flag", "group": "Calendar context", "kind": "binary",
     "description": "Indicator for a holiday."},
    {"name": "activity_flag", "label": "Activity flag", "group": "Calendar context", "kind": "binary",
     "description": "Indicator for a special activity."},
    {"name": "precpt", "label": "Precipitation", "group": "Weather", "kind": "numeric",
     "description": "Total precipitation for the observation date."},
    {"name": "avg_temperature", "label": "Average temperature", "group": "Weather", "kind": "numeric",
     "description": "Average temperature for the observation date."},
    {"name": "avg_humidity", "label": "Average humidity", "group": "Weather", "kind": "numeric",
     "description": "Average humidity for the observation date."},
    {"name": "avg_wind_level", "label": "Average wind level", "group": "Weather", "kind": "numeric",
     "description": "Average wind force for the observation date."},
]

NUMERIC_FIELDS = [
    "sale_amount", "stock_hour6_22_cnt", "discount", "precpt",
    "avg_temperature", "avg_humidity", "avg_wind_level",
]
CATEGORICAL_FIELDS = [
    "city_id", "store_id", "management_group_id", "first_category_id",
    "second_category_id", "third_category_id", "product_id", "holiday_flag", "activity_flag",
]
CORRELATION_FIELDS = [
    "sale_amount", "stock_hour6_22_cnt", "discount", "precpt",
    "avg_temperature", "avg_humidity", "avg_wind_level",
]


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def _records(connection: duckdb.DuckDBPyConnection, sql: str) -> list[dict[str, Any]]:
    cursor = connection.execute(sql)
    names = [column[0] for column in cursor.description]
    output = []
    for row in cursor.fetchall():
        record = {}
        for name, value in zip(names, row):
            if isinstance(value, (date, datetime)):
                value = value.isoformat()
            elif hasattr(value, "item"):
                value = value.item()
            record[name] = value
        output.append(record)
    return output


def _one(connection: duckdb.DuckDBPyConnection, sql: str) -> dict[str, Any]:
    rows = _records(connection, sql)
    return rows[0] if rows else {}


def _split_summary(connection: duckdb.DuckDBPyConnection, view: str) -> dict[str, Any]:
    return _one(connection, f"""
        SELECT
            COUNT(*) AS row_count,
            COUNT(DISTINCT city_id) AS city_count,
            COUNT(DISTINCT store_id) AS store_count,
            COUNT(DISTINCT product_id) AS product_count,
            COUNT(DISTINCT (store_id, product_id)) AS store_product_series,
            COUNT(DISTINCT CAST(dt AS DATE)) AS date_count,
            MIN(CAST(dt AS DATE)) AS date_start,
            MAX(CAST(dt AS DATE)) AS date_end,
            SUM(CASE WHEN sale_amount = 0 THEN 1 ELSE 0 END) AS zero_sales_rows,
            AVG(sale_amount) AS mean_observed_sales,
            MEDIAN(sale_amount) AS median_observed_sales,
            SUM(sale_amount) AS total_observed_sales,
            COUNT(*) - COUNT(DISTINCT (store_id, product_id, CAST(dt AS DATE))) AS duplicate_key_excess
        FROM {view}
    """)


def _numeric_profile(
    connection: duckdb.DuckDBPyConnection, field: str, row_count: int
) -> dict[str, Any]:
    stats = _one(connection, f"""
        SELECT
            COUNT("{field}") AS non_null_count,
            COUNT(DISTINCT "{field}") AS distinct_count,
            MIN("{field}") AS min,
            MAX("{field}") AS max,
            AVG("{field}") AS mean,
            STDDEV_POP("{field}") AS std,
            MEDIAN("{field}") AS median,
            QUANTILE_CONT("{field}", 0.01) AS p01,
            QUANTILE_CONT("{field}", 0.25) AS p25,
            QUANTILE_CONT("{field}", 0.75) AS p75,
            QUANTILE_CONT("{field}", 0.99) AS p99,
            SUM(CASE WHEN "{field}" = 0 THEN 1 ELSE 0 END) AS zero_count,
            SUM(CASE WHEN "{field}" < 0 THEN 1 ELSE 0 END) AS negative_count
        FROM all_data
    """)
    non_null = int(stats.get("non_null_count") or 0)
    stats["null_count"] = int(row_count - non_null)
    stats["null_pct"] = (stats["null_count"] / row_count * 100) if row_count else 0.0

    minimum, maximum = stats.get("min"), stats.get("max")
    histogram: list[dict[str, Any]] = []
    if minimum is not None and maximum is not None:
        if float(maximum) == float(minimum):
            histogram = [{"bin_start": float(minimum), "bin_end": float(maximum), "count": non_null}]
        else:
            span = float(maximum) - float(minimum)
            min_sql = format(float(minimum), ".17g")
            step_sql = format(span / HISTOGRAM_BINS, ".17g")
            bucket_expr = (
                f"LEAST({HISTOGRAM_BINS - 1}, GREATEST(0, "
                f"CAST(FLOOR((\"{field}\" - ({min_sql})) / ({step_sql})) AS INTEGER)))"
            )
            counts = _records(connection, f"""
                SELECT {bucket_expr} AS bucket, COUNT(*) AS count
                FROM all_data
                WHERE "{field}" IS NOT NULL
                GROUP BY bucket ORDER BY bucket
            """)
            for item in counts:
                index = int(item["bucket"])
                histogram.append({
                    "bin_start": float(minimum) + index * span / HISTOGRAM_BINS,
                    "bin_end": float(minimum) + (index + 1) * span / HISTOGRAM_BINS,
                    "count": int(item["count"]),
                })
    stats["histogram"] = histogram
    return stats


def _categorical_profile(
    connection: duckdb.DuckDBPyConnection, field: str, row_count: int
) -> dict[str, Any]:
    stats = _one(connection, f"""
        SELECT COUNT("{field}") AS non_null_count,
               COUNT(DISTINCT "{field}") AS distinct_count
        FROM all_data
    """)
    stats["null_count"] = int(row_count - int(stats.get("non_null_count") or 0))
    stats["null_pct"] = (stats["null_count"] / row_count * 100) if row_count else 0.0
    stats["top_values"] = _records(connection, f"""
        SELECT CAST("{field}" AS VARCHAR) AS value, COUNT(*) AS count
        FROM all_data
        WHERE "{field}" IS NOT NULL
        GROUP BY "{field}"
        ORDER BY count DESC, value
        LIMIT 15
    """)
    return stats


def _vector_profile(
    connection: duckdb.DuckDBPyConnection, field: str, row_count: int
) -> dict[str, Any]:
    stats = _one(connection, f"""
        SELECT
            COUNT("{field}") AS non_null_count,
            MIN(LEN("{field}")) AS min_length,
            MAX(LEN("{field}")) AS max_length,
            AVG(LEN("{field}")) AS mean_length,
            SUM(CASE WHEN LIST_SUM("{field}") = 0 THEN 1 ELSE 0 END) AS all_zero_vectors
        FROM all_data
    """)
    non_null = int(stats.get("non_null_count") or 0)
    stats["null_count"] = int(row_count - non_null)
    stats["null_pct"] = (stats["null_count"] / row_count * 100) if row_count else 0.0
    return stats


def _correlations(connection: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    expressions = []
    for i, left in enumerate(CORRELATION_FIELDS):
        for right in CORRELATION_FIELDS[i:]:
            expressions.append(f'CORR("{left}", "{right}") AS "{left}__{right}"')
    row = _one(connection, "SELECT " + ", ".join(expressions) + " FROM all_data")
    matrix = []
    for left in CORRELATION_FIELDS:
        matrix_row = []
        for right in CORRELATION_FIELDS:
            key = f"{left}__{right}" if f"{left}__{right}" in row else f"{right}__{left}"
            matrix_row.append(row.get(key))
        matrix.append(matrix_row)
    return {"fields": CORRELATION_FIELDS, "matrix": matrix}


def _build_report(connection: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    split_overview = {
        "train": _split_summary(connection, "train_data"),
        "evaluation": _split_summary(connection, "evaluation_data"),
        "combined": _split_summary(connection, "all_data"),
    }
    combined_rows = int(split_overview["combined"]["row_count"])

    profiles = {}
    for feature in FEATURE_CATALOG:
        name = feature["name"]
        kind = feature["kind"]
        if kind in ("target", "numeric"):
            profile = _numeric_profile(connection, name, combined_rows)
        elif kind in ("identifier", "binary"):
            profile = _categorical_profile(connection, name, combined_rows)
        elif kind == "date":
            profile = _one(connection, f"""
                SELECT COUNT("{name}") AS non_null_count,
                       COUNT(DISTINCT CAST("{name}" AS DATE)) AS distinct_count,
                       MIN(CAST("{name}" AS DATE)) AS min,
                       MAX(CAST("{name}" AS DATE)) AS max
                FROM all_data
            """)
            profile["null_count"] = combined_rows - int(profile.get("non_null_count") or 0)
            profile["null_pct"] = profile["null_count"] / combined_rows * 100 if combined_rows else 0.0
        elif kind == "vector":
            profile = _vector_profile(connection, name, combined_rows)
        else:
            raise ValueError(f"Unsupported feature profile type: {kind}")
        profiles[name] = profile

    daily_trend = _records(connection, """
        SELECT split, CAST(dt AS DATE) AS date, COUNT(*) AS store_product_rows,
               SUM(sale_amount) AS total_observed_sales,
               AVG(sale_amount) AS mean_observed_sales,
               AVG(CASE WHEN sale_amount = 0 THEN 1.0 ELSE 0.0 END) * 100 AS zero_sales_rate_pct
        FROM all_data
        GROUP BY split, CAST(dt AS DATE)
        ORDER BY date, split
    """)
    weekday_profile = _records(connection, """
        SELECT CAST(DATE_PART('dow', CAST(dt AS DATE)) AS INTEGER) AS weekday_number,
               STRFTIME(CAST(dt AS DATE), '%A') AS weekday,
               COUNT(*) AS observations,
               AVG(sale_amount) AS mean_observed_sales,
               MEDIAN(sale_amount) AS median_observed_sales,
               AVG(CASE WHEN sale_amount = 0 THEN 1.0 ELSE 0.0 END) * 100 AS zero_sales_rate_pct
        FROM all_data
        GROUP BY 1, 2 ORDER BY 1
    """)
    calendar_profile = _records(connection, """
        SELECT holiday_flag, activity_flag, COUNT(*) AS observations,
               AVG(sale_amount) AS mean_observed_sales,
               MEDIAN(sale_amount) AS median_observed_sales,
               AVG(CASE WHEN sale_amount = 0 THEN 1.0 ELSE 0.0 END) * 100 AS zero_sales_rate_pct
        FROM all_data
        GROUP BY holiday_flag, activity_flag
        ORDER BY holiday_flag, activity_flag
    """)
    discount_profile = _records(connection, """
        SELECT CASE
                   WHEN discount IS NULL THEN 'Missing'
                   WHEN discount >= 0.999 THEN 'No discount'
                   WHEN discount >= 0.90 THEN '1–10% off'
                   WHEN discount >= 0.80 THEN '10–20% off'
                   WHEN discount >= 0.70 THEN '20–30% off'
                   WHEN discount >= 0.50 THEN '30–50% off'
                   ELSE '50%+ off'
               END AS discount_band,
               COUNT(*) AS observations,
               AVG(discount) AS mean_discount_factor,
               AVG(sale_amount) AS mean_observed_sales,
               MEDIAN(sale_amount) AS median_observed_sales
        FROM all_data
        GROUP BY 1
        ORDER BY MIN(discount)
    """)
    stockout_profile = _records(connection, """
        SELECT CASE
                   WHEN stock_hour6_22_cnt = 0 THEN '0 hours'
                   WHEN stock_hour6_22_cnt <= 2 THEN '1–2 hours'
                   WHEN stock_hour6_22_cnt <= 5 THEN '3–5 hours'
                   WHEN stock_hour6_22_cnt <= 9 THEN '6–9 hours'
                   WHEN stock_hour6_22_cnt <= 15 THEN '10–15 hours'
                   ELSE '16 hours'
               END AS exposure_band,
               COUNT(*) AS observations,
               AVG(stock_hour6_22_cnt) AS mean_stockout_hours,
               AVG(sale_amount) AS mean_observed_sales,
               AVG(CASE WHEN sale_amount = 0 THEN 1.0 ELSE 0.0 END) * 100 AS zero_sales_rate_pct
        FROM all_data
        GROUP BY 1
        ORDER BY MIN(stock_hour6_22_cnt)
    """)
    top_stores = _records(connection, """
        SELECT store_id, COUNT(*) AS observations,
               SUM(sale_amount) AS total_observed_sales,
               AVG(sale_amount) AS mean_observed_sales,
               COUNT(DISTINCT product_id) AS distinct_products
        FROM all_data GROUP BY store_id
        ORDER BY total_observed_sales DESC LIMIT 20
    """)
    top_products = _records(connection, """
        SELECT product_id, COUNT(*) AS observations,
               SUM(sale_amount) AS total_observed_sales,
               AVG(sale_amount) AS mean_observed_sales,
               COUNT(DISTINCT store_id) AS distinct_stores
        FROM all_data GROUP BY product_id
        ORDER BY total_observed_sales DESC LIMIT 20
    """)
    category_profile = _records(connection, """
        SELECT first_category_id, second_category_id, third_category_id,
               COUNT(*) AS observations, SUM(sale_amount) AS total_observed_sales,
               AVG(sale_amount) AS mean_observed_sales,
               COUNT(DISTINCT product_id) AS distinct_products
        FROM all_data
        GROUP BY first_category_id, second_category_id, third_category_id
        ORDER BY total_observed_sales DESC LIMIT 30
    """)
    hourly_sales = _records(connection, """
        SELECT split, CAST(position - 1 AS INTEGER) AS hour_index,
               COUNT(*) AS observations, AVG(value) AS mean_hourly_sales,
               MEDIAN(value) AS median_hourly_sales,
               SUM(value) AS total_hourly_sales,
               AVG(CASE WHEN value = 0 THEN 1.0 ELSE 0.0 END) * 100 AS zero_hour_rate_pct
        FROM all_data
        CROSS JOIN UNNEST(hours_sale) WITH ORDINALITY AS h(value, position)
        GROUP BY split, hour_index
        ORDER BY split, hour_index
    """)
    hourly_status = _records(connection, """
        SELECT split, CAST(position - 1 AS INTEGER) AS hour_index,
               COUNT(*) AS observations, AVG(value) AS mean_status_value,
               AVG(CASE WHEN value = 1 THEN 1.0 ELSE 0.0 END) * 100 AS value_one_pct
        FROM all_data
        CROSS JOIN UNNEST(hours_stock_status) WITH ORDINALITY AS h(value, position)
        GROUP BY split, hour_index
        ORDER BY split, hour_index
    """)
    quality = [
        {
            "feature": item["name"],
            "label": item["label"],
            "kind": item["kind"],
            "null_count": profiles[item["name"]].get("null_count", 0),
            "null_pct": profiles[item["name"]].get("null_pct", 0.0),
            "distinct_count": profiles[item["name"]].get("distinct_count"),
        }
        for item in FEATURE_CATALOG
    ]
    return {
        "dataset": "FreshRetailNet-50K",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_objects": SOURCE_KEYS,
        "source_semantics": (
            "The original source contains 19 fields. sale_amount and hours_sale are globally "
            "normalized observed sales; stockouts can censor observed sales. Descriptive associations "
            "do not identify latent demand or causal effects."
        ),
        "feature_count": len(FEATURE_CATALOG),
        "feature_catalog": FEATURE_CATALOG,
        "split_overview": split_overview,
        "profiles": profiles,
        "data_quality": {
            "features": quality,
            "duplicate_key_excess": split_overview["combined"]["duplicate_key_excess"],
        },
        "daily_trend": daily_trend,
        "weekday_profile": weekday_profile,
        "calendar_profile": calendar_profile,
        "discount_profile": discount_profile,
        "stockout_profile": stockout_profile,
        "top_stores": top_stores,
        "top_products": top_products,
        "category_profile": category_profile,
        "hourly_sales": hourly_sales,
        "hourly_status": hourly_status,
        "correlations": _correlations(connection),
    }


def main() -> None:
    account_id = required_env("R2_ACCOUNT_ID")
    bucket = required_env("R2_BUCKET")
    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=required_env("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=required_env("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
    )

    with tempfile.TemporaryDirectory(prefix="freshretailnet-full-eda-") as temp_dir:
        root = Path(temp_dir)
        local_paths = {}
        input_hashes = {}
        for split, key in SOURCE_KEYS.items():
            path = root / f"{split}.parquet"
            print(f"Downloading s3://{bucket}/{key}")
            s3.download_file(bucket, key, str(path))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            input_hashes[split] = digest
            local_paths[split] = path

        connection = duckdb.connect(database=":memory:")
        connection.execute("SET threads = 4")
        connection.execute(f"CREATE VIEW train_data AS SELECT * FROM read_parquet('{_sql_path(local_paths['train'])}')")
        connection.execute(f"CREATE VIEW evaluation_data AS SELECT * FROM read_parquet('{_sql_path(local_paths['evaluation'])}')")
        connection.execute("""
            CREATE VIEW all_data AS
            SELECT 'train' AS split, * FROM train_data
            UNION ALL
            SELECT 'evaluation' AS split, * FROM evaluation_data
        """)
        actual_columns = [row[0] for row in connection.execute("DESCRIBE SELECT * FROM train_data").fetchall()]
        expected_columns = [item["name"] for item in FEATURE_CATALOG]
        if set(actual_columns) != set(expected_columns) or len(actual_columns) != len(expected_columns):
            raise RuntimeError(
                "Source schema differs from the documented 19-field dataset contract. "
                f"Expected {expected_columns}; received {actual_columns}"
            )
        eval_columns = [row[0] for row in connection.execute("DESCRIBE SELECT * FROM evaluation_data").fetchall()]
        if actual_columns != eval_columns:
            raise RuntimeError("Train/evaluation source schemas do not match exactly.")

        report = _build_report(connection)
        report["input_sha256"] = input_hashes
        report["source_row_count"] = int(report["split_overview"]["combined"]["row_count"])
        connection.close()

        output_path = root / "full_dataset_eda.json"
        output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        s3.upload_file(
            str(output_path), bucket, OUTPUT_KEY,
            ExtraArgs={"ContentType": "application/json"},
        )
        verified = s3.head_object(Bucket=bucket, Key=OUTPUT_KEY)
        if int(verified["ContentLength"]) != output_path.stat().st_size:
            raise RuntimeError("R2 upload verification failed for full dataset EDA report.")
        print(
            f"Published complete {report['feature_count']}-field EDA for "
            f"{report['source_row_count']:,} rows to s3://{bucket}/{OUTPUT_KEY} "
            f"({verified['ContentLength']:,} bytes)."
        )
        print(json.dumps({
            "generated_at_utc": report["generated_at_utc"],
            "feature_count": report["feature_count"],
            "split_overview": report["split_overview"],
            "input_sha256": input_hashes,
            "output_bytes": verified["ContentLength"],
        }, indent=2, default=str))


if __name__ == "__main__":
    main()
