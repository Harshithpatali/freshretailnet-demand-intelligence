# Advanced time-series research benchmark — results

**Status:** successful; all seven artifacts were uploaded and size-verified in Cloudflare R2.  
**Workflow:** [Advanced Time-Series Research Benchmark #46143898867](https://github.com/Harshithpatali/freshretailnet-demand-intelligence/actions/runs/46143898867)  
**Commit:** `ba4a1a5cb3e274918cb80a0e95f607b7d53654fd`  
**Evaluation:** 2024-06-26 to 2024-07-02 (seven dates)  
**Cohort:** deterministic sample of 1,500 out of 50,000 store-product series, seed 20261128  
**Evaluation population:** 10,500 store-product-day rows (1,500 series × 7 dates)

## Research setup

The benchmark compares:
- seasonal-naive lag-7;
- the existing LightGBM feature baseline;
- LightGBM with causal STL features and weekly/annual Fourier harmonics;
- STL/Fourier plus FFT spectral diagnostics, without wavelets;
- the same feature family with trailing db2 wavelet features;
- residual-corrected spectral/wavelet forecasts;
- a non-negative weighted ensemble selected by validation MAE.

The source Parquet files were downloaded from private R2. The run generated 103,500 STL/spectral feature rows and 97,500 wavelet feature rows for the deterministic cohort. Feature windows terminate strictly before their target dates. The 14-day validation window was divided into a residual-fitting block and a separate ensemble-selection block; no evaluation labels were used for residual fitting or ensemble weights.

## Held-out comparison

All metrics use normalized observed sales as the target. They are not metrics for identified latent customer demand.

| Model | MAE | RMSE | WAPE | Bias |
|---|---:|---:|---:|---:|
| Seasonal naive lag-7 | 0.513244 | 0.864998 | 40.8621% | +0.0476% |
| Baseline LightGBM | 0.414088 | 0.770711 | 32.9678% | -3.9477% |
| STL + Fourier LightGBM | 0.413374 | 0.764962 | 32.9109% | -3.0232% |
| Spectral, no wavelet | 0.414049 | 0.772055 | 32.9647% | -3.1848% |
| Spectral + wavelet | 0.415887 | 0.772911 | 33.1110% | -4.0311% |
| Residual-corrected spectral/wavelet | **0.401252** | **0.713439** | **31.9458%** | -0.8953% |
| Validation-selected ensemble | 0.402524 | 0.736995 | 32.0471% | -2.3783% |

Relative to baseline LightGBM, residual correction reduced MAE by about 3.10%, RMSE by about 7.43% and WAPE by about 3.10%. The validation-selected ensemble reduced MAE by about 2.80%, RMSE by about 4.37% and WAPE by about 2.80%.

STL/Fourier by itself improved MAE only marginally (about 0.17%). The spectral-only additions were essentially neutral.

## Wavelet ablation

The same-row comparison of spectral/STL/Fourier plus wavelets versus the same model without wavelets:

| Metric | No wavelet | With wavelet | Change |
|---|---:|---:|---:|
| MAE | 0.414049 | 0.415887 | +0.4437% |
| WAPE | 32.9647% | 33.1110% | +0.4437% |

A positive change means error increased. Wavelet features did not improve this experiment and should not be promoted on this evidence.

## Validation-selected ensemble weights

| Component | Weight |
|---|---:|
| STL + Fourier LightGBM | 61.35% |
| Residual-corrected spectral/wavelet | 30.90% |
| Spectral, no wavelet | 7.75% |
| Seasonal naive lag-7 | approximately 0% |
| Baseline LightGBM | approximately 0% |
| Spectral + wavelet | approximately 0% |

The weights are non-negative and sum to one. They were selected on the designated validation block, not on held-out evaluation outcomes.

## Paired store-product cluster bootstrap

The run used 500 paired bootstrap replicates over 1,500 store-product clusters, retaining the seven evaluation dates within each sampled cluster. Positive numbers indicate relative error improvement versus baseline LightGBM.

| Candidate | Metric | Improvement | 95% CI |
|---|---|---:|---:|
| STL + Fourier | MAE | +0.19% | -0.37% to +0.69% |
| STL + Fourier | RMSE | +0.73% | -0.48% to +1.74% |
| Residual-corrected | MAE | +3.11% | +1.30% to +5.23% |
| Residual-corrected | RMSE | +7.33% | +2.78% to +11.95% |
| Residual-corrected | WAPE | +3.11% | +1.30% to +5.23% |
| Validation-selected ensemble | MAE | +2.81% | +2.01% to +3.69% |
| Validation-selected ensemble | RMSE | +4.34% | +2.43% to +6.27% |
| Validation-selected ensemble | WAPE | +2.81% | +2.01% to +3.69% |
| Spectral + wavelet | MAE | -0.44% | -0.90% to +0.04% |

The residual-corrected and ensemble intervals exclude zero for the shown metrics. These are **cross-sectional cluster intervals conditional on this sample and the seven tested dates**. They do not establish robustness across future time windows or account fully for shared date shocks.

## Decision

1. Keep residual correction and the validation-selected ensemble as research challengers.
2. Do not promote wavelet features; the same-row ablation worsened MAE and WAPE.
3. Treat STL/Fourier as a small candidate improvement, not a proven segment-wide win.
4. Require additional chronological rolling windows and a later untouched evaluation before production promotion.
5. Keep the existing LightGBM production model as the reference until the new candidates pass those gates.

## Verified R2 artifacts

All were written to `timeseries-research-artifacts/` and also to the immutable run path
`experiment-runs/ba4a1a5cb3e2/46143898867-attempt-1/advanced-time-series-research-benchmark/`:

- `timeseries_research_summary.json` (14,337 bytes)
- `timeseries_model_comparison.csv` (943 bytes)
- `timeseries_wavelet_ablation.csv` (324 bytes)
- `timeseries_evaluation_predictions.csv` (1,472,276 bytes)
- `timeseries_feature_diagnostics.csv` (161,907 bytes)
- `timeseries_ensemble_weights.json` (312 bytes)
- `timeseries_manifest.json` (3,345 bytes)

The provenance manifest records source snapshot hashes, code commit, evaluation period and configuration. The target is observed sales and may be stockout-censored. The cohort is a 1,500-series research sample, not the full dataset population.
