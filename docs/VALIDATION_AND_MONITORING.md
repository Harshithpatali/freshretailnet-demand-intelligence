# Validation, censoring, inventory scenarios and monitoring

This document describes the extended validation work added to the repository. The workflows publish results only after they complete and verify their R2 uploads; source code or a workflow being present is not evidence that a result has completed.

## 1. Temporal validation

The segment-specialist backtest uses five disjoint chronological test windows (ROLLING_FOLDS=5 by default), an expanding training history and a preceding seven-day validation block. Each horizon is paired only when the target date is exactly 1, 7 or 14 calendar days ahead. The test block length is constrained by the available date history; the fold audit records actual train, validation and test dates and row counts.

Selection and segment calibration use validation data only. The prospective specialist gate uses previous validation decisions and never test outcomes. A specialist is fitted once per segment/fold and the same fitted model produces both validation and test predictions.

Run: **Segment Specialist Multi-Window Rolling Backtest**.

## 2. Paired cluster-bootstrap uncertainty

Confidence intervals are computed after model predictions have been saved. The post-hoc script reads segment_rolling_backtest_predictions.csv from R2 and does not train models again.

It first aggregates sufficient statistics by store-product cluster, then performs batched multinomial resampling (500 replicates by default). The paired bootstrap retains all held-out rows for each sampled cluster and reports 95% percentile intervals for relative MAE/RMSE improvement, WAPE improvement in percentage points and bias change.

This quantifies cross-sectional cluster uncertainty conditional on the tested time windows. It does **not** quantify variation across future time periods and cannot replace additional temporal windows.

## 3. Demand-censoring sensitivity

scripts/demand_censoring_sensitivity.py estimates a transparent range of sales potentially missed during out-of-stock hours.

- Hourly rates are calculated from in-stock observations in raw/train.parquet, at store-product-hour level.
- Missing store-product-hour rates fall back to the global in-stock mean for that hour.
- The historical hourly rates are applied only to out-of-stock slots in raw/eval.parquet.
- The report publishes observed sales plus 0.5x, 1.0x and 1.5x the missing-sales proxy.

The central estimate assumes the historical in-stock sales rate is representative of demand in hours that were out of stock. That assumption may be false because demand can vary by hour, promotion, price and stockout mechanism. The resulting figures are **sensitivity scenarios, not identified latent demand or causal lost sales**. Do not use them as a training target without independent validation.

Run: **Demand Censoring Sensitivity**.

## 4. Inventory lead-time and replenishment constraints

The simulator now supports optional storage capacity, minimum order quantity, pack-size rounding, maximum order quantity and capacity/order limits expressed as multiples of average series forecast.

The scenario workflow compares:
- 1-day lead / 1-day review, unconstrained
- 3-day lead / 1-day review, unconstrained
- 7-day lead / 1-day review, unconstrained
- 1-day lead / 7-day review, unconstrained
- 7-day lead / 7-day review with 14 forecast-days of storage capacity, a 7 forecast-day order cap, MOQ 6 and pack size 6

Shortage penalties are also varied to 0.5x and 2.0x base assumptions for the 1-day/1-day and constrained 7-day/7-day cases.

The scaled capacity and order limits, MOQ and pack size are illustrative uniform assumptions, not measurements of actual retailer or supplier constraints. The simulator still uses observed sales as a demand proxy, deterministic lead times and does not model expiry, perishability, supplier calendars or stochastic replenishment.

Run: **Inventory Lead-Time and Constraint Scenarios**.

## 4. Forecast error, interval and drift monitoring

scripts/forecast_monitoring.py reads the quantile prediction artifact and reports:
- daily and aggregate MAE, RMSE, WAPE and signed bias;
- calibrated P10-P90 coverage and mean interval width;
- raw quantile crossing rate;
- PSI for observed sales, median forecasts, calibrated interval width and stockout exposure (when available);
- warning alerts for low coverage, high absolute bias, quantile crossing and high PSI.

PSI compares the first and second chronological halves of the current evaluation window. The default PSI labels (<0.10 low, 0.10-0.25 moderate, >=0.25 high) and monitoring alert thresholds are heuristics, not statistically calibrated service-level objectives.

The initial report is a **retrospective batch monitor over the held-out evaluation artifact**. It is not yet continuous production monitoring: live forecast/actual records must be appended with consistent identifiers and dates, and metrics can only be computed after actual outcomes mature.

Run: **Forecast Quality and Drift Monitoring**.

## 5. Conditional interval calibration by prior exposure

The quantile training pipeline now contains an experimental conditional split-conformal alternative to the original global interval adjustment.

- A prior seven-day maximum stockout-hours feature is computed from historical daily rows, shifted so the current target day's stock status is never used to select its interval.
- Validation residual scores are grouped using that prior-exposure feature.
- Groups with fewer than 1,000 validation rows fall back to the global adjustment.
- Other group adjustments are shrunk toward the global validation adjustment with a prior strength of 2,000 rows.
- Both global and conditional intervals are scored on the held-out evaluation rows; coverage and mean width are compared by contemporaneous evaluation exposure for diagnosis only.

This is a candidate calibration method, not a production promotion. A higher 16+ hour coverage rate is useful only if interval width remains operationally reasonable. The evaluation stratification is retrospective; the calibration decision itself uses only prior exposure context and validation residuals.

Run: **Train FreshRetailNet Quantile Forecasts**. Review `conditional_conformal_calibration` in the R2 quantile metrics and the conditional coverage/width comparison in the dashboard.

## 6. Later untouched evaluation

The current benchmark evaluation ends on 2024-07-02. A separate `raw/later_holdout.parquet` snapshot is required before a genuinely later test can be scored. The fail-closed readiness gate requires at least 14 strictly later, contiguous dates, no duplicate store-product-date rows and at least 95% of the benchmark series population. It only validates data readiness; it does not score models.

See [Later Untouched Holdout Readiness](LATER_HOLDOUT_READINESS.md). Do not recycle dates already used in the existing benchmark as a later test.

## API and dashboard

After the workflows publish their outputs, the dashboard exposes:
- Forecast monitoring, daily metrics, drift and alerts
- Demand-censoring sensitivity by stockout exposure
- Inventory scenario fill-rate versus simulated-cost comparisons

API routes:
- /api/v1/monitoring/forecast-quality
- /api/v1/monitoring/forecast-quality/daily
- /api/v1/monitoring/forecast-quality/drift
- /api/v1/demand-censoring/sensitivity
- /api/v1/demand-censoring/sensitivity/buckets
- /api/v1/inventory/scenarios
- /api/v1/inventory/scenarios/comparison

## Validation before promotion

Do not promote a challenger or operational policy solely because a pooled metric improves. Review fold-level consistency, interval performance by exposure, the censoring assumptions and inventory costs under retailer-provided constraints. Preserve a later untouched period for final confirmation and retain data snapshot, commit, configuration and workflow identifiers with each published report.
