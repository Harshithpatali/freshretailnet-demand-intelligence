# Forecast quality and interval monitoring

**Run:** [Forecast Quality and Drift Monitoring, run #1](https://github.com/Harshithpatali/freshretailnet-demand-intelligence/actions/runs/46109885915)  
**R2 output verification:** completed successfully.  
**Generated:** 2026-11-28  
**Evaluation window:** 2024-06-26 to 2024-07-02

## Held-out quality

| Metric | Result |
|---|---:|
| Evaluation rows | 350,000 |
| P50 MAE | 0.39358 |
| P50 RMSE | 0.69904 |
| P50 WAPE | 32.9894% |
| P50 bias | -8.0656% |
| Raw P10–P90 coverage | 73.0511% |
| Calibrated P10–P90 coverage | 77.4594% |
| Nominal interval coverage | 80.0000% |
| Calibrated mean interval width | 1.10751 |
| Raw quantile crossing rate | 0.1246% |

The calibrated interval remains below its 80% nominal target on this held-out period. The monitoring job emitted one warning: absolute P50 forecast bias of 8.07% exceeded the configured 5% threshold.

## Distribution-shift diagnostics

PSI compares the first chronological half of this seven-day evaluation window (2024-06-26 to 2024-06-28) with the second half (2024-06-29 to 2024-07-02). It is a short-window retrospective comparison, not a validated train-versus-live drift assessment. Thresholds of 0.10 and 0.25 are operational heuristics.

## Limitations

- This is batch monitoring of a held-out artifact, not a continuous production stream.
- Error and interval metrics require actual outcomes to mature.
- Observed-sales targets can be censored by stockouts.
- PSI and bias alert thresholds need calibration against longer historical performance and operational risk.

Artifacts are written to the private R2 paths under `monitoring/forecast-quality/`.
