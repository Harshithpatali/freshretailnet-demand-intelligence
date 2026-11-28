# Conditional interval calibration by prior stockout exposure

**Workflow:** [Train FreshRetailNet Quantile Forecasts #46130544925](https://github.com/Harshithpatali/freshretailnet-demand-intelligence/actions/runs/46130544925)  
**Run status:** successful; quantile models and comparison metrics published to R2  
**Evaluation window:** 2024-06-26 to 2024-07-02  
**Evaluation rows:** 350,000  
**Nominal interval coverage:** 80%

## Method

The experiment adds a leakage-safe prior-exposure feature: the maximum daily stockout-hours count over the preceding seven days for the same store-product series. It uses that feature, not contemporaneous evaluation-day stock status, to select a group-specific split-conformal adjustment.

Group calibration adjustments are estimated from the validation period only. A group needs at least 1,000 validation rows; otherwise it falls back to the global adjustment. Eligible group adjustments are shrunk toward the global validation adjustment with a prior strength of 2,000 rows.

This is an experimental candidate alongside the original global split-conformal interval. The global interval remains the active reference.

## Overall interval results

| Method | Validation coverage | Evaluation coverage | Evaluation mean width |
|---|---:|---:|---:|
| Global split conformal | 80.00% | 77.46%* | 1.10751* |
| Prior-exposure conditional conformal | 80.01% | 77.61% | 1.11334 |

*The global figures above are from the earlier published quantile evaluation workflow. The conditional method's training run reports its own held-out evaluation coverage of 77.61%. Compare exposure-stratum values below from the same run when assessing the candidate.*

The conditional method is only about 0.15 percentage points higher in overall evaluation coverage than the earlier global benchmark, while mean interval width is about 0.5% wider. The comparison is not evidence of reliable recalibration.

## Coverage by actual evaluation-day stockout exposure

The actual exposure bucket is used only to stratify evaluation outcomes. The adjustment selection itself uses preceding seven-day exposure.

| Actual exposure | Rows | Global calibrated coverage | Conditional coverage | Global width | Conditional width |
|---|---:|---:|---:|---:|---:|
| 0 hours | 206,562 | 80.48% | 80.40% | 1.10987 | 1.10818 |
| 1–2 hours | 21,019 | 74.60% | 74.67% | 1.15197 | 1.15269 |
| 3–5 hours | 42,823 | 75.42% | 75.52% | 1.11443 | 1.11588 |
| 6–9 hours | 38,841 | 80.18% | 80.26% | 1.07727 | 1.07897 |
| 10–15 hours | 28,214 | 74.94% | 74.95% | 1.05585 | 1.05701 |
| 16+ hours | 12,541 | 42.66% | **41.62%** | 1.36076 | 1.35691 |

## Decision

**Do not promote the conditional calibration candidate.** It does not improve the 16+ hour group: coverage falls by about 1.04 percentage points and mean width decreases by about 0.28%. The main weakness remains severe under-coverage for high contemporaneous stockout exposure.

This is useful negative evidence. Prior seven-day maximum exposure is not sufficiently informative for the actual high-exposure evaluation stratum. The next experiment should estimate high-exposure risk using only forecast-time information and test whether risk-conditioned interval expansion improves 16+ coverage without excessive width inflation. Do not use current-day stock status to choose the forecast interval.

## Reproducibility and limitations

- The adjustment is fitted on validation residuals only.
- The prior exposure context is shifted and available before the target date.
- Current-day stockout hours are used only for post-hoc coverage stratification.
- The evaluation period is seven dates, so temporal uncertainty remains poorly characterized.
- The target is observed sales, not guaranteed uncensored customer demand.
- Stockout-exposure coverage is descriptive and does not establish a causal stockout effect.

Raw predictions and calibration audit details remain in private R2 under `quantile-model-artifacts/`.
