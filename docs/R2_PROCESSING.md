# R2 data processing

The raw dataset remains in the private R2 bucket. The manual GitHub Actions workflow reads it, performs schema and null profiling, and writes processed outputs back to the **same bucket**.

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
- `processed/data_quality_report.json`
- `processed/null_profile.csv`

The first processing pass is deliberately schema-preserving. It re-encodes the Parquet files using Zstandard compression and preserves values, columns, and row order. It does not drop rows, impute missing values, or alter business meaning. The report records row counts, column types, null counts, file sizes, and SHA-256 checksums.

Review the report and schema before implementing domain-specific cleaning, stockout-aware demand reconstruction, or feature engineering. Those steps must be based on the actual dataset fields and a documented validation strategy.
