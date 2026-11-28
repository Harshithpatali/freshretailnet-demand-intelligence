# Advanced forecasting research benchmark: STL, Fourier, spectral, wavelet, residual and ensemble

**Workflow:** [Forecast Research Benchmark, run #46143243242](https://github.com/Harshithpatali/freshretailnet-demand-intelligence/actions/runs/46143243242)  
**Status:** successful; five R2 artifacts uploaded and size-verified  
**Commit:** `a87f8269`  
**Sample:** fixed-seed sample of 3,000 complete store-product series from FreshRetailNet-50K  
**Evaluation:** original seven-day split, 2024-06-26 to 2024-07-02; this period was used in earlier repository benchmarks and is not an untouched holdout

## Reproducible design

The panel sample is selected after sorting distinct (store_id, product_id) keys and applying seed `20261128`. A regression test verifies that permuting the upstream distinct-key row order does not change the selected sample.

The source train/evaluation Parquet SHA-256 hashes, commit, workflow run ID, model settings and seed are stored in the summary JSON. The benchmark reports three adjacent seven-day expanding-window test folds within the training period and a separate comparison on the original seven-day evaluation split.

Features are built with forecast-time information only:
- **Fourier:** three weekly harmonics and monthly/yearly calendar basis functions.
- **STL:** robust weekly decomposition of the 56 observations before the forecast origin; trend level/slope, weekday seasonal component and residual summaries.
- **Spectral:** periodogram dominant 2–28 day period, weekly-band energy share, normalized spectral entropy and peak share.
- **Wavelet:** six causal db2 trailing-32-day coefficient energy/entropy/ratio features.
- **Residual correction:** LightGBM correction trained on the first validation part and checked on a later validation part, then refit on validation history.
- **Ensemble:** non-negative simplex weights selected from validation predictions only and frozen for evaluation.

Contemporaneous discount, holiday/activity, weather, hourly sales, and current stock-status fields are excluded because their availability at forecast time is not established. STL/spectral statistics are fixed-origin history-only features. Wavelet windows end before the target date.

## Rolling-window results

The following metrics pool the three chronological seven-day rolling test windows over the deterministic 3,000-series sample.

| Candidate | MAE | RMSE | WAPE | Bias |
|---|---:|---:|---:|---:|
| Baseline | 0.380479 | 0.673677 | 34.3425% | -2.5661% |
| Fourier + STL | 0.379841 | 0.673234 | 34.2849% | -2.5078% |
| Spectral | 0.379886 | 0.675295 | 34.2890% | -2.6215% |
| Wavelet augmented | 0.380891 | 0.680387 | 34.3797% | -2.8531% |
| Residual corrected | 0.378056 | 0.662401 | 34.1238% | -3.4062% |
| Validation-selected ensemble | **0.377091** | **0.662286** | **34.0367%** | -2.9228% |

These are research-sample estimates, not metrics for all 50,000 series. The residual/ensemble methods improve pooled rolling metrics modestly; fold-to-fold performance still matters.

## Final comparison on the original seven-day evaluation split

| Candidate | MAE | RMSE | WAPE | Bias |
|---|---:|---:|---:|---:|
| Baseline | 0.427515 | 0.788904 | 35.5233% | -0.0548% |
| Fourier + STL | 0.420314 | 0.784210 | 34.9250% | -2.6439% |
| Spectral | 0.422132 | 0.797830 | 35.0761% | -2.7260% |
| Wavelet augmented | 0.421903 | 0.796267 | 35.0571% | -3.1877% |
| Residual corrected | **0.404923** | **0.711997** | **33.6461%** | -0.6434% |
| Validation-selected ensemble | **0.404923** | **0.711997** | **33.6461%** | -0.6434% |

The residual-corrected model reduces final comparison MAE by approximately 5.12% and RMSE by approximately 9.79% versus baseline on this research sample and previously used seven-day test window. This is a promising challenger result, not a production-promotion result.

The validation optimizer selected residual correction with weight 1.0; weights on baseline, Fourier/STL, spectral and wavelet candidates were effectively zero. Therefore, the reported “ensemble” is a degenerate single-model selection rather than a useful multi-model blend. Do not claim ensemble diversity.

## Paired uncertainty intervals

The 300-replicate paired bootstrap resamples 3,000 store-product clusters and contiguous two-day date blocks. The evaluation has only seven dates, so temporal uncertainty is coarse and the intervals cannot replace later untouched temporal windows.

**Comparisons against baseline (percent relative improvement; positive means lower error):**

| Candidate | Metric | Mean improvement | 95% interval | Conclusion |
|---|---|---:|---:|---|
| Fourier + STL | MAE | +1.63% | +0.46% to +2.97% | Interval excludes zero |
| Fourier + STL | RMSE | +0.56% | -0.17% to +1.43% | Inconclusive |
| Spectral | MAE | +1.19% | +0.04% to +2.56% | Interval excludes zero |
| Spectral | RMSE | -1.15% | -2.38% to +0.37% | Inconclusive; point estimate worse |
| Wavelet augmented | MAE | +1.25% | -0.06% to +2.75% | Inconclusive |
| Wavelet augmented | RMSE | -0.95% | -2.32% to +0.52% | Inconclusive; point estimate worse |
| Residual corrected | MAE | +5.12% | +1.71% to +8.75% | Interval excludes zero |
| Residual corrected | RMSE | +9.42% | +4.71% to +13.57% | Interval excludes zero |

**Direct wavelet ablation against the spectral-only model:**

| Metric | Mean improvement | 95% interval | Conclusion |
|---|---:|---:|---|
| MAE | +0.06% | -0.31% to +0.47% | No evidence of improvement |
| RMSE | +0.20% | -0.30% to +0.82% | No evidence of improvement |
| WAPE | +0.06% | -0.31% to +0.47% | No evidence of improvement |

The direct ablation does not establish value from the wavelet features. Retain them as an experimental feature group, not as a promoted improvement.

## Decision

- Keep the original LightGBM baseline as the production reference until a later untouched period is scored.
- Retain Fourier/STL as a promising challenger: MAE and WAPE intervals exclude zero, but RMSE does not.
- Retain spectral features as exploratory: MAE/WAPE improve slightly but RMSE does not.
- Do not promote wavelet features based on this experiment; the direct paired ablation is inconclusive.
- Retain residual correction as the strongest challenger for the next untouched validation.
- Do not describe the validation-selected output as an ensemble blend: the selected weight is effectively 100% residual correction.
- Run the frozen candidates on a genuinely later untouched temporal holdout before any production promotion.

## Artifacts verified in R2

The workflow verified the following stable objects:
- `forecast-research-artifacts/forecast_research_summary.json`
- `forecast-research-artifacts/forecast_research_rolling_predictions.csv`
- `forecast-research-artifacts/forecast_research_final_predictions.csv`
- `forecast-research-artifacts/forecast_research_rolling_comparison.csv`
- `forecast-research-artifacts/forecast_research_final_comparison.csv`

The summary records source SHA-256 hashes and run/commit provenance. Stable output keys are overwritten by later successful runs; archive the summary JSON and the workflow run reference when retaining historical benchmark evidence.

## Limitations

- The sampled 3,000 series are not the full 50,000-series population.
- The seven-day evaluation period has already been used in previous project benchmarks and cannot serve as a new untouched test.
- The two-way bootstrap uses only seven target dates; time-block uncertainty is coarse.
- Annual Fourier basis functions cannot establish annual seasonal structure from only 90 days of training history.
- Target is observed sales, which can be censored by stockouts; it is not identified latent demand.
- Forecast improvements do not establish business impact or inventory savings.
