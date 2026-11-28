# Final validation and promotion protocol

## Snapshot boundary: why a later holdout is currently blocked

The project currently uses the official FreshRetailNet-50K release split stored as:

- `raw/train.parquet`
- `raw/eval.parquet`

The held-out evaluation partition used by the published results runs from **2024-06-26 through 2024-07-02** (seven days). The published source release does not provide a later partition in the configured R2 objects. We must not call a random slice, a later fold from this same used evaluation period, or a re-labelled row from the training history a new untouched test.

The new `Later Untouched Holdout Readiness Gate` checks for a separate R2 object:

`raw/later_holdout.parquet`

The gate requires:
- at least 14 contiguous daily dates;
- every date strictly after the current benchmark evaluation end (2024-07-02);
- no duplicate store-product-date rows;
- at least 95% of the original evaluation store-product population;
- required fields and a valid daily panel.

It publishes a readiness manifest only. It **does not score the model** and does not tune on the later data. Once the object exists and passes the gate, score the frozen point and quantile model artifacts against it once. Keep the model, configuration, feature logic and conformal adjustment frozen. Do not use the later period for model selection; after reporting it, a different future period is needed for repeated promotion decisions.

Run: **Later Untouched Holdout Readiness Gate**.

Current gate status for this project: **blocked by data availability until a newer untouched snapshot is supplied**.

## Stockout-censored demand: reconstruction validation

Observed daily sales are not a ground-truth label for unconstrained demand during stockouts. The reconstruction script therefore does not claim to identify lost sales.

The diagnostic now proceeds as follows:

1. Reconcile the hourly stock-status coding with the independent `stock_hour6_22_cnt` count for every training and evaluation store-product-day. Code 1 is treated as out of stock and indices 6 through 21 are counted; the workflow fails if the derived count does not reconcile.
2. Estimate mean hourly sales using **training-only** in-stock observations, grouped by store, product and hour. When a group has no in-stock history, use the global in-stock mean for that hour.
3. Perform an out-of-time *pseudo-censoring* validation on evaluation hours known to be in stock. Treat their observed hourly sales as hidden, predict them using the training-only rates, and report MAE, RMSE, WAPE, bias and global fallback rate.
4. For genuinely out-of-stock hours, continue to publish 0.5x / 1.0x / 1.5x sensitivity scenarios around the rate-imputation proxy, clearly labelled as a sensitivity estimate.

The pseudo-censoring check tests whether this rate estimator predicts observed sales during hours when the item was available. It **does not validate the unobservable counterfactual during actual stockouts**. In-stock and stockout hours can have different demand, promotions, substitutions and inventory selection mechanisms. No imputed value is promoted to a supervised demand label, and no descriptive difference is presented as causal lost sales.

Run: **Demand Censoring Sensitivity**.

## Quantile coverage by time and stockout exposure

The batch monitor now reports:
- daily calibrated P10-P90 coverage and daily P50 error;
- overall coverage against the 80% nominal target;
- coverage, mean width and row count by six stockout-exposure groups (0, 1-2, 3-5, 6-9, 10-15 and 16+ hours);
- date-by-exposure coverage, so changes in time and exposure mix can be inspected together;
- warnings when overall coverage is below 80%, or when an exposure group with at least 100 rows is below 75%;
- distribution-shift PSI and forecast-bias alerts.

Exposure-specific coverage is a diagnostic, not a causal estimate. Small date-by-exposure cells can be noisy; inspect row counts. The evaluation set is only seven days, so this is a short retrospective view—not a calibrated production service-level guarantee. A live monitor requires forecasts and matured actuals from a sequence of future periods.

Run: **Forecast Quality and Drift Monitoring**.

## Future calendar fields

The model can safely derive weekday, day-of-month, month and trigonometric weekday encodings once the target date is chosen. The dataset's `holiday_flag` and `activity_flag` values do not include an operational publication-time contract in the source documentation. The segment experiment therefore excludes these flags when joined from the **future target date**. Historical origin-date flags can remain part of the historical row.

Do not re-enable target-date holiday/activity fields until a timestamped upstream calendar source and a service-level commitment show that each value is published before the model's forecast issue time. The report contains a machine-readable availability audit.

## Inventory-policy validation

The inventory scenario grid compares seasonal-naive, P50 and calibrated P90 policies across 1-, 3- and 7-day lead times, 1- and 7-day review intervals, shortage-penalty sensitivity and an illustrative storage/order/MOQ/pack-size constrained case.

The current prediction window is seven days. Orders due on or after the horizon end are suppressed; consequently, the 7-day-lead scenarios have no replenishment arrival during this test and show 0% fill. These are **horizon-boundary stress results**, not valid steady-state service estimates for a 7-day lead time. A meaningful lead-7 evaluation needs a longer untouched actual-demand window (at least lead time plus review and a warm-up/evaluation tail), preferably 28-56 days, plus retailer-provided initial inventory, supplier lead-time variability, SKU case packs/MOQs, storage capacity and cost inputs.

The current policy grid is a controlled simulation using observed sales as a demand proxy. It is not a measured savings calculation. No policy is promoted on the current seven-day data.

Run: **Inventory Lead-Time and Constraint Scenarios**.

## Artifact provenance and immutable versions

The shared `scripts/artifact_provenance.py` helper creates a manifest containing:
- repository, ref, commit SHA, workflow run ID/attempt and run URL;
- SHA-256 and byte size for each local input snapshot;
- a content-addressed aggregate data-snapshot SHA-256;
- evaluation start/end dates;
- experiment configuration and output object names.

Each publication keeps the existing stable R2 alias used by the dashboard and also writes an immutable copy under:

`experiment-runs/<commit-prefix>/<run-id>-attempt-<n>/<workflow>/<artifact>`

The manifest is stored beside the stable output. New outputs are never just labelled with a commit without hashing the actual raw inputs. Re-running code against different input snapshots produces a different data snapshot ID.

## Release decision

The project has repeatable data, experiment, test and publication workflows, but the evidence is not yet sufficient for production promotion. Do not promote until:
1. a later untouched dataset passes the readiness gate and is scored once with frozen models;
2. the pseudo-censoring test and stock-status reconciliation complete successfully, with limitations retained;
3. coverage by time/exposure has been reviewed on a substantially longer period;
4. target-date calendar flags remain excluded until a forecast-time contract exists;
5. inventory policies have been re-evaluated on a longer horizon with measured replenishment/cost inputs.

A successful code-quality CI run is necessary but is not a substitute for these data-dependent gates.
