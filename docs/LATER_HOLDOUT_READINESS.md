# Later untouched evaluation: data requirement and promotion gate

The currently published benchmark data has an evaluation end date of **2024-07-02**. The existing model and segment experiments have already used dates through that boundary. Reusing those dates or carving another split from the same used evaluation period would not be an independent later holdout.

## Required input

Upload a separately sourced and previously unused daily Parquet snapshot to the configured R2 bucket at:

`raw/later_holdout.parquet`

It must contain the same store-product-day grain and at least these fields:

- `store_id`
- `product_id`
- `dt`
- `sale_amount`
- `hours_sale`
- `hours_stock_status`

The readiness gate also requires:

- every target date strictly after 2024-07-02;
- at least 14 contiguous daily dates (to include seven-day replenishment arrivals and a post-arrival observation period);
- no duplicate store-product-date rows;
- at least 95% of the benchmark store-product population;
- compatible hourly arrays and status encoding.

The snapshot should be obtained independently, have a documented extraction timestamp and source/version, and must not be used for model or calibration tuning. Do not fill missing future observations with synthetic records.

## Run the gate

After uploading the new object, run **Later Untouched Holdout Readiness Gate** from GitHub Actions. The workflow downloads the existing `raw/eval.parquet` and the separate later snapshot, validates strict date separation, duration, continuity, duplicates and population coverage, then publishes a provenance manifest and readiness report to:

`validation-gates/later-holdout/later_holdout_readiness.json`

A successful readiness check does **not** score any model. The next stage must freeze the current production and challenger artifacts, score the later file once, evaluate H7/H14 forecasts and rerun inventory policies with at least seven-day lead times. Do not tune using this final holdout. If the R2 object is absent, the workflow intentionally fails closed rather than creating a misleading test window.

## Current status

No later score is published by this repository yet. The later-snapshot object is a required external data dependency; the project source does not contain a later date range beyond the existing benchmark.
