# Demand-segment specialists: five-fold rolling backtest

**Status:** successful  
**Workflow:** [Five-fold Segment Specialist Rolling Backtest #46124737202](https://github.com/Harshithpatali/freshretailnet-demand-intelligence/actions/runs/46124737202)  
**Commit:** `774cdda8ce0ed3409a0f6048a7861de45865c2bd`  
**R2 publication:** report, fold audit, selection audit, calibration audit, 249.8 MB prediction file and paired-bootstrap comparison were verified in both stable and immutable run-specific R2 paths.

## Design and evaluation population

The experiment compares the current LightGBM baseline with validation-gated demand-segment specialists, with an additional segment-wise calibration step estimated only from each fold's preceding validation block.

| Horizon | Test folds | Days per fold | Target dates | Evaluation rows | Store-product clusters |
|---|---:|---:|---:|---:|---:|
| 1 day | 5 | 5 | 25 | 1,250,000 | 50,000 |
| 7 days | 5 | 4 | 20 | 1,000,000 | 50,000 |
| 14 days | 5 | 2 | 10 | 500,000 | 50,000 |

The overall evaluation period covered 2024-06-08 to 2024-07-02. Each fold uses expanding historical training, a preceding 7-day validation block and a disjoint chronological test block. The model-selection gate requires validation improvement and does not use test outcomes. Calendar features admitted as future-known are deterministic date fields only; target-date holiday/activity flags remain excluded because forecast-time publication is not established.

## Pooled held-out results

Bias is defined as signed aggregate prediction error divided by observed target sum; negative values indicate under-forecasting. The calibration multiplier is learned from validation predictions and applied to untouched test predictions.

| Horizon | Model | MAE | RMSE | WAPE | Bias |
|---|---|---:|---:|---:|---:|
| 1 day | Current LightGBM baseline | 0.392972 | 0.676288 | 34.2066% | -1.1296% |
| 1 day | Validation-gated specialists | 0.392508 | 0.675560 | 34.1663% | -1.3353% |
| 1 day | Gated + calibrated specialists | 0.392764 | 0.678987 | 34.1886% | -1.9365% |
| 7 days | Current LightGBM baseline | 0.426586 | 0.728392 | 36.7708% | -4.4836% |
| 7 days | Validation-gated specialists | 0.425700 | 0.727407 | 36.6944% | -5.0174% |
| 7 days | Gated + calibrated specialists | 0.419909 | 0.706687 | 36.1953% | -1.3638% |
| 14 days | Current LightGBM baseline | 0.459067 | 0.785205 | 38.3640% | -4.5787% |
| 14 days | Validation-gated specialists | 0.457991 | 0.783558 | 38.2742% | -4.8670% |
| 14 days | Gated + calibrated specialists | 0.452444 | 0.759886 | 37.8106% | -0.8068% |

Relative to the baseline, the calibrated challenger improves pooled MAE by **0.053% at H1**, **1.565% at H7**, and **1.443% at H14**. H7 and H14 also improve pooled RMSE by 2.980% and 3.225%, respectively. The H1 calibrated model has 0.399% worse RMSE than baseline.

## Paired store-product cluster bootstrap

The post-hoc job resampled the 50,000 store-product clusters with replacement, retaining all dates and folds belonging to each sampled cluster. It used 500 paired replicates and 95% percentile intervals. Positive MAE/RMSE improvement means the challenger is better. WAPE improvement is expressed in percentage points; positive is better. Bias change is challenger bias minus baseline bias, so positive means less negative bias in this run.

| Horizon | Estimand | Point estimate | 95% CI |
|---|---|---:|---:|
| 1 day | MAE improvement | +0.053% | -0.014% to +0.112% |
| 1 day | RMSE improvement | -0.399% | -0.541% to -0.280% |
| 1 day | WAPE improvement | +0.018 pp | -0.0048 to +0.0385 pp |
| 1 day | Bias change | -0.807 pp | -0.837 to -0.778 pp |
| 7 days | MAE improvement | +1.565% | +1.453% to +1.673% |
| 7 days | RMSE improvement | +2.980% | +2.703% to +3.270% |
| 7 days | WAPE improvement | +0.576 pp | +0.536 to +0.615 pp |
| 7 days | Bias change | +3.120 pp | +3.079 to +3.153 pp |
| 14 days | MAE improvement | +1.443% | +1.318% to +1.561% |
| 14 days | RMSE improvement | +3.225% | +2.858% to +3.590% |
| 14 days | WAPE improvement | +0.553 pp | +0.508 to +0.596 pp |
| 14 days | Bias change | +3.772 pp | +3.742 to +3.803 pp |

**Interpretation:** H7 and H14 have positive MAE, RMSE and WAPE intervals excluding zero. H1 has an MAE interval spanning zero and a negative RMSE interval; it does not establish a meaningful improvement. H1 calibration also worsens the signed bias.

These are cross-sectional paired cluster-bootstrap intervals, conditional on the five tested chronological windows. They do **not** quantify uncertainty over future time windows or distribution shifts. Five folds improve temporal assessment but the tested date span is still short; a separate later untouched dataset period is required before model promotion.

## Segment diagnostics

The pooled gains are not uniform across demand segments.

- **Erratic:** at H7, specialists were selected in all five validation folds and calibrated segment MAE improved 2.34%. At H14, two validation folds selected the specialist and calibrated segment MAE improved 3.23%.
- **Smooth:** no specialists were selected at H7/H14, but the validation-derived calibration reduced pooled segment MAE by 1.52% and 1.26%, respectively.
- **Lumpy:** the calibration step slightly worsened segment MAE at H7 (0.45%) and H14 (0.83%) relative to baseline.
- **Intermittent:** held-out samples were only 38 rows at H1, 4 at H7 and 1 at H14. No reliable segment conclusion is possible.

The erratic H7 specialist is the clearest segment-specific candidate for another evaluation. The lumpy segment should not be promoted based on this run.

## Decision

| Horizon | Recommendation |
|---|---|
| H1 | Retain LightGBM baseline; the challenger MAE gain is inconclusive and RMSE worsens |
| H7 | Keep calibrated specialists as a validated challenger; strongest consistent evidence |
| H14 | Keep calibrated specialists as a validated challenger; consistent pooled improvement |
| All horizons | Do not promote to production solely on this evaluation; test a later untouched period |

## Reproducibility and artifacts

The workflow recorded the repository commit, run ID, raw train/evaluation SHA-256 hashes, evaluation window, model settings and forecast-time feature policy. The stable R2 prefix is `segment-rolling-backtest-artifacts/`; the run-specific immutable prefix is `experiment-runs/774cdda8ce0e/46124737202-attempt-1/`.

Verified outputs include:
- `segment_rolling_backtest_report.json`
- `segment_rolling_backtest_comparison.csv` and `.md`
- `segment_rolling_backtest_folds.csv`
- `segment_rolling_backtest_selection.csv`
- `segment_rolling_backtest_predictions.csv` (249,794,789 bytes)
- `segment_rolling_backtest_calibration.csv`
- run and bootstrap manifests

The target is observed daily sales, which may be censored by stock availability. The 1-, 7- and 14-day experiments are separate from the production one-day-ahead quantile forecasting pipeline.
