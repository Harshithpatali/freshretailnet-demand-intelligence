# R2 data processing

The private R2 bucket is the source of truth for raw and processed data. The manual GitHub Actions workflow reads the raw Parquet files, profiles them, re-encodes Parquet with Zstandard, exports plain CSV files, and writes all outputs back to the **same bucket**.

## Workflow

Open **Actions → Profile and Process FreshRetailNet-50K in R2 → Run workflow**.

Required repository secrets:

- `R2_ACCOUNT_ID`
- `R2_ACCESS_KEY_ID`
- `R2_SECRET_ACCESS_KEY`
- `R2_BUCKET`

The R2 token should have Object Read & Write access scoped to the target bucket.

## Input objects

- `raw/train.parquet`
- `raw/eval.parquet`

## Output objects in the same bucket

- `processed/train.parquet`
- `processed/eval.parquet`
- `processed/train.csv`
- `processed/eval.csv`
- `processed/data_quality_report.json`
- `processed/null_profile.csv`

CSV files are plain UTF-8 CSV with headers. They can be substantially larger than Parquet; the workflow streams CSV output to local disk rather than collecting the entire dataset in memory, then uploads the files to R2. Check the reported CSV object sizes and available R2 storage before running.

The processing pass preserves values, columns, and row order. It does not drop rows, impute missing values, or rename columns. The report records row counts, column types, null counts, file sizes, and SHA-256 checksums.

Review the report and schema before implementing domain-specific cleaning, stockout-aware demand reconstruction, or feature engineering. Those steps must be based on the actual dataset fields and a documented validation strategy.
