# Advanced time-series research benchmark

## Research question

Do classical decomposition and seasonal signals add useful information beyond the
existing LightGBM feature set, and can residual correction or validation-selected
ensembling improve held-out forecasts?

## Compared methods

1. Existing LightGBM feature baseline.
2. LightGBM plus weekly and annual Fourier harmonics.
3. LightGBM plus causal STL trend/seasonality/residual features and Fourier terms.
4. STL/Fourier plus FFT spectral features, without wavelets.
5. The same model with trailing db2 wavelet features.
6. Residual-corrected spectral/wavelet model.
7. Non-negative weighted ensemble selected on the validation window.

## Feature construction

For target date (t), all decomposition and spectrum features use only the preceding
28 observations. STL uses weekly period 7. The seasonal feature for target date (t)
is taken from the same weekday position one week earlier. The spectral block includes
weekly-frequency power share, low-frequency power share, spectral entropy, dominant
period and spectral centroid.

Fourier features use deterministic target-date calendar information (three weekly
harmonics and two annual harmonics). Wavelet summaries use a trailing 32-observation
window ending at (t-1). No contemporaneous target sales enter these feature families.

## Residual model and ensemble selection

The 14-day validation window is divided chronologically into two parts:

- First half: fit the residual learner on validation residuals from the frozen base
  spectral/wavelet model.
- Second half: estimate non-negative ensemble weights by minimizing MAE subject to
  weights summing to one.

For final evaluation, the residual learner is refit using the full validation period,
and the ensemble weights remain frozen. Evaluation labels are not used for feature,
residual or weight selection.

## Cohort and evaluation

The current script uses a deterministic random sample of 1,500 store-product series
(seed 20261128) to bound repeated STL fits. All candidate methods share the same
feature-eligible rows, chronological train/validation/evaluation dates and target.
This benchmark is not a full-population result.

The official FreshRetailNet evaluation partition still spans seven dates. The
evaluation metrics are useful for comparative research on the fixed holdout but do
not establish future temporal stability. The paired bootstrap resamples complete
store-product clusters and retains all dates for each sampled cluster; it is
conditional on those seven dates and does not capture shared date shocks or future
window uncertainty.

## Artifacts and API

Workflow: **Advanced Time-Series Research Benchmark**.

R2 objects under `timeseries-research-artifacts/`:

- `timeseries_research_summary.json`
- `timeseries_model_comparison.csv`
- `timeseries_wavelet_ablation.csv`
- `timeseries_evaluation_predictions.csv`
- `timeseries_feature_diagnostics.csv`
- `timeseries_ensemble_weights.json`
- `timeseries_manifest.json`

API:

- `GET /api/v1/research/time-series`
- `GET /api/v1/research/time-series/comparison`
- `GET /api/v1/research/time-series/ablation`
- `GET /api/v1/research/time-series/diagnostics`

## Promotion policy

A feature family or ensemble is not promoted solely because it wins on this seven-day
holdout. Require consistent fold-level improvements and a later untouched time window.
Wavelet inclusion should be judged by the same-row ablation; residual correction and
ensemble weights must remain validation-derived.
