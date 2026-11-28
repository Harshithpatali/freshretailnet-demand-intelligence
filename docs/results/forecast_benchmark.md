# Point-forecast benchmark

<small>Public summary of the successful point-model workflow. Raw predictions and model artifacts remain in private Cloudflare R2.</small>

## Evaluation setup

| Field | Value |
|---|---|
| Dataset | FreshRetailNet-50K |
| Target | Observed daily `sale_amount`, not guaranteed uncensored demand |
| Forecast horizon | One day ahead |
| Training rows | 3,450,000 |
| Validation rows | 700,000 |
| Evaluation rows | 350,000 |
| Validation dates | 2024-06-12 to 2024-06-25 |
| Evaluation dates | 2024-06-26 to 2024-07-02 |
| Model | LightGBM |
| Benchmark | Seasonal naive, lag 7 |
| Workflow | [Train FreshRetailNet Demand Model](https://github.com/Harshithpatali/freshretailnet-demand-intelligence/actions/runs/45880756552) |
| Workflow commit | `b320274` |

## Held-out evaluation results

| Metric | LightGBM | Seasonal naive lag-7 | Relative change |
|---|---:|---:|---:|
| MAE | 0.38682 | 0.49907 | 22.49% lower |
| RMSE | 0.67889 | 0.83582 | 18.78% lower |
| WAPE | 32.42% | 41.83% | 9.41 percentage points lower |
| Bias | -5.40% | +0.28% | Model under-forecasts |

Relative change is calculated as (benchmark - LightGBM) / benchmark for error metrics. Bias is signed aggregate forecast error relative to observed sales; negative values indicate under-forecasting.

## Interpretation

On this seven-day held-out period, LightGBM reduced MAE, RMSE and WAPE relative to the lag-7 benchmark. The LightGBM model under-forecast aggregate observed sales by 5.40%, while the seasonal-naive benchmark was close to unbiased (+0.28%). The accuracy improvement therefore comes with a bias trade-off worth monitoring.

## Limitations

- The evaluation period contains seven dates; 350,000 rows do not represent 350,000 independent time observations.
- The target is observed sales. Stockouts may censor the target, so this is not a claim about latent customer demand.
- These results are one historical holdout, not a guarantee of future performance.
- The experiment-specific 1-, 7- and 14-day rolling results are separate from this production one-day-ahead model.

Re-run the point-model workflow after changing data or code and publish a new dated report rather than silently replacing historical results.
