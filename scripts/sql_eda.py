"""Run reproducible SQL EDA over the FreshRetailNet training Parquet in R2."""
from __future__ import annotations
import json, os, tempfile
from datetime import datetime, timezone
from pathlib import Path
import boto3
import duckdb
from botocore.config import Config

RAW_KEY = "raw/train.parquet"
OUTPUT_KEY = "sql-eda/sql_eda_results.json"

QUERIES = [
    [
        "dataset_overview",
        "Dataset overview",
        "SELECT COUNT(*) AS row_count, COUNT(DISTINCT store_id) AS store_count, COUNT(DISTINCT product_id) AS product_count, MIN(TRY_CAST(dt AS DATE)) AS first_date, MAX(TRY_CAST(dt AS DATE)) AS last_date, AVG(sale_amount) AS mean_observed_sales, MEDIAN(sale_amount) AS median_observed_sales, SUM(sale_amount) AS total_observed_sales FROM read_parquet(?)"
    ],
    [
        "daily_sales_trend",
        "Daily observed-sales trend",
        "SELECT TRY_CAST(dt AS DATE) AS date, COUNT(*) AS store_product_rows, SUM(sale_amount) AS total_observed_sales, AVG(sale_amount) AS mean_observed_sales, AVG(CASE WHEN sale_amount = 0 THEN 1.0 ELSE 0.0 END) * 100 AS zero_sales_rate_pct FROM read_parquet(?) GROUP BY 1 ORDER BY 1"
    ],
    [
        "weekday_pattern",
        "Day-of-week pattern",
        "SELECT DAYOFWEEK(TRY_CAST(dt AS DATE)) AS weekday_number, STRFTIME(TRY_CAST(dt AS DATE), '%A') AS weekday, COUNT(*) AS observations, AVG(sale_amount) AS mean_observed_sales, MEDIAN(sale_amount) AS median_observed_sales, AVG(CASE WHEN sale_amount = 0 THEN 1.0 ELSE 0.0 END) * 100 AS zero_sales_rate_pct FROM read_parquet(?) GROUP BY 1,2 ORDER BY 1"
    ],
    [
        "top_stores",
        "Top 20 stores by observed sales",
        "SELECT store_id, COUNT(*) AS observations, SUM(sale_amount) AS total_observed_sales, AVG(sale_amount) AS mean_observed_sales, COUNT(DISTINCT product_id) AS distinct_products FROM read_parquet(?) GROUP BY store_id ORDER BY total_observed_sales DESC LIMIT 20"
    ],
    [
        "top_products",
        "Top 20 products by observed sales",
        "SELECT product_id, COUNT(*) AS observations, SUM(sale_amount) AS total_observed_sales, AVG(sale_amount) AS mean_observed_sales, COUNT(DISTINCT store_id) AS distinct_stores FROM read_parquet(?) GROUP BY product_id ORDER BY total_observed_sales DESC LIMIT 20"
    ],
    [
        "discount_profile",
        "Discount band analysis",
        "SELECT CASE WHEN discount IS NULL THEN 'missing' WHEN discount = 0 THEN '0' WHEN discount < 0.1 THEN '(0, 0.1)' WHEN discount < 0.3 THEN '[0.1, 0.3)' WHEN discount < 0.5 THEN '[0.3, 0.5)' ELSE '0.5+' END AS discount_band, COUNT(*) AS observations, AVG(discount) AS mean_discount, AVG(sale_amount) AS mean_observed_sales, MEDIAN(sale_amount) AS median_observed_sales FROM read_parquet(?) GROUP BY 1 ORDER BY 1"
    ],
    [
        "stockout_exposure",
        "Stockout exposure vs observed sales",
        "SELECT CASE WHEN stock_hour6_22_cnt = 0 THEN '0 hours' WHEN stock_hour6_22_cnt <= 2 THEN '1-2 hours' WHEN stock_hour6_22_cnt <= 5 THEN '3-5 hours' WHEN stock_hour6_22_cnt <= 9 THEN '6-9 hours' WHEN stock_hour6_22_cnt <= 15 THEN '10-15 hours' ELSE '16 hours' END AS stockout_bucket, COUNT(*) AS observations, AVG(stock_hour6_22_cnt) AS mean_stockout_hours, AVG(sale_amount) AS mean_observed_sales, AVG(CASE WHEN sale_amount = 0 THEN 1.0 ELSE 0.0 END) * 100 AS zero_sales_rate_pct FROM read_parquet(?) GROUP BY 1 ORDER BY MIN(stock_hour6_22_cnt)"
    ],
    [
        "data_quality",
        "Data completeness and range checks",
        "SELECT COUNT(*) AS row_count, COUNT(*) - COUNT(store_id) AS missing_store_id, COUNT(*) - COUNT(product_id) AS missing_product_id, COUNT(*) - COUNT(dt) AS missing_date, COUNT(*) - COUNT(sale_amount) AS missing_sale_amount, SUM(CASE WHEN sale_amount < 0 THEN 1 ELSE 0 END) AS negative_sales_rows, SUM(CASE WHEN discount < 0 THEN 1 ELSE 0 END) AS negative_discount_rows, SUM(CASE WHEN discount > 1 THEN 1 ELSE 0 END) AS discount_over_one_rows FROM read_parquet(?)"
    ]
]

def env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing secret {name}")
    return value

def main():
    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{env('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
        aws_access_key_id=env("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=env("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
    )
    with tempfile.TemporaryDirectory(prefix="freshretailnet-sql-eda-") as td:
        parquet_path = Path(td) / "train.parquet"
        print(f"Downloading s3://{env('R2_BUCKET')}/{RAW_KEY}")
        s3.download_file(env("R2_BUCKET"), RAW_KEY, str(parquet_path))
        con = duckdb.connect(database=":memory:")
        results = []
        for query_id, title, sql in QUERIES:
            print(f"Running SQL EDA: {query_id}")
            cursor = con.execute(sql, [str(parquet_path)])
            columns = [item[0] for item in cursor.description]
            rows = cursor.fetchmany(1000)
            records = []
            for row in rows:
                records.append({key: (value.isoformat() if hasattr(value, "isoformat") else value) for key, value in zip(columns, row)})
            results.append({"id": query_id, "title": title, "sql": sql, "columns": columns, "rows": records, "row_count_returned": len(records)})
        con.close()
        payload = {
            "source": RAW_KEY,
            "source_semantics": "sale_amount is observed sales and may be censored by stock availability; this is descriptive EDA, not uncensored demand estimation.",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "engine": f"DuckDB {duckdb.__version__}",
            "query_count": len(results),
            "queries": results,
        }
        output = Path(td) / "sql_eda_results.json"
        output.write_text(json.dumps(payload, indent=2, default=str))
        s3.upload_file(str(output), env("R2_BUCKET"), OUTPUT_KEY, ExtraArgs={"ContentType": "application/json"})
        head = s3.head_object(Bucket=env("R2_BUCKET"), Key=OUTPUT_KEY)
        if head["ContentLength"] != output.stat().st_size:
            raise RuntimeError("R2 upload verification failed for SQL EDA results")
        print(f"Verified s3://{env('R2_BUCKET')}/{OUTPUT_KEY} ({head['ContentLength']:,} bytes)")
        print(f"Published {len(results)} SQL EDA result sets at {payload['generated_at_utc']}")

if __name__ == "__main__":
    main()
