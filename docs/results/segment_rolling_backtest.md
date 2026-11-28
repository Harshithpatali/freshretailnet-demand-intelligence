# Demand-segment specialist rolling backtest

## Design

The experiment compares the current LightGBM baseline with validation-gated demand-segment specialists. It uses three disjoint chronological test folds, expanding historical training windows, and a preceding validation window. The specialist-selection gate is based on validation history, not test outcomes.

The later calibration step estimates a segment-wise multiplicative adjustment on validation predictions, shrinks the factor toward 1, clips extreme factors and applies it to the untouched test predictions.

## Pooled results from the completed calibrated run

| Horizon | Model | MAE | RMSE | WAPE | Bias |
|---|---|---:|---:|---:|---:|
| 1 day | Current LightGBM | 0.39195 | 0.69816 | 34.11% | -2.69% |
| 1 day | Gated + calibrated specialists | 0.38716 | 0.67981 | 33.69% | -1.34% |
| 7 days | Current LightGBM | 0.43577 | 0.74910 | 36.75% | -4.42% |
| 7 days | Gated + calibrated specialists | 0.42901 | 0.73732 | 36.18% | -2.90% |
| 14 days | Current LightGBM | 0.47493 | 0.82361 | 39.16% | -7.25% |
| 14 days | Gated + calibrated specialists | 0.45893 | 0.78881 | 37.85% | -3.30% |

| Horizon | MAE improvement vs baseline |
|---|---:|
| 1 day | 1.22% |
| 7 days | 1.55% |
| 14 days | 3.37% |

Source workflow: [successful calibrated segment rolling backtest](https://github.com/Harshithpatali/freshretailnet-demand-intelligence/actions/runs/46097052636), commit `a9ded7e`.

## Interpretation

The calibrated challenger improved pooled MAE, RMSE and WAPE at all three horizons in this run, and reduced the magnitude of negative bias. The largest pooled MAE improvement was at 14 days.

## Sample-size and uncertainty qualification

This run used three chronological test folds. The evaluation contains many store-product rows, but only three distinct temporal test windows. Cross-sectional cluster uncertainty does not substitute for evidence across many future periods. Do not treat these results as proof of production superiority.

A subsequent workflow adds paired store-product cluster-bootstrap confidence intervals and a public comparison table. Its run must complete successfully before its uncertainty estimates are treated as published evidence: [comparison-report workflow](https://github.com/Harshithpatali/freshretailnet-demand-intelligence/actions/runs/46105979122).

The 1-, 7- and 14-day experiments are separate from the production one-day-ahead point model. The observed-sales target may be censored by stock availability.
