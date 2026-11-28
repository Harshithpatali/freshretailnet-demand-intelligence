# Quantile forecast and interval evaluation

## Evaluation setup

| Field | Value |
|---|---|
| Target | Observed daily `sale_amount` |
| Quantiles | P10, P50, P90 |
| Nominal P10–P90 interval coverage | 80% |
| Training rows | 3,450,000 |
| Validation rows | 700,000 |
| Evaluation rows | 350,000 |
| Validation dates | 2024-06-12 to 2024-06-25 |
| Evaluation dates | 2024-06-26 to 2024-07-02 |
| Workflow | [Train FreshRetailNet Quantile Forecasts](https://github.com/Harshithpatali/freshretailnet-demand-intelligence/actions/runs/45936386460) |
| Workflow commit | `2a3dbe8` |

## Evaluation metrics

| Metric | Validation | Held-out evaluation |
|---|---:|---:|
| P10 pinball loss | 0.07331 | 0.08172 |
| P50 pinball loss | 0.17359 | 0.19679 |
| P90 pinball loss | 0.08565 | 0.09894 |
| P10–P90 coverage before conformal adjustment | 75.42% | 73.05% |
| Mean interval width before adjustment | 1.00346 | 1.04023 |
| Quantile crossing rate | 0.1134% | 0.1246% |
| Median MAE | 0.34717 | 0.39358 |
| Median WAPE | 30.80% | 32.99% |

## Split-conformal adjustment

The conformal adjustment was estimated on the prior validation period and then applied unchanged to the held-out evaluation period.

| Metric | Validation calibrated interval | Held-out calibrated interval |
|---|---:|---:|
| Coverage | 80.00% | 77.46% |
| Mean interval width | 1.07092 | 1.10751 |
| Nominal coverage | 80.00% | 80.00% |

The calibrated evaluation coverage is 2.54 percentage points below nominal. Calibration achieved nominal coverage on validation but did not fully maintain it on the later evaluation period. This is evidence of temporal distribution shift or calibration instability, not a result to hide.

## Held-out coverage by stockout-exposure stratum

| Exposure bucket | Rows | Calibrated coverage |
|---|---:|---:|
| 0 hours | 206,562 | 80.32% |
| 1–2 hours | 21,019 | 74.48% |
| 3–5 hours | 42,823 | 75.15% |
| 6–9 hours | 38,841 | 80.19% |
| 10–15 hours | 28,214 | 75.31% |
| 16+ hours | 12,541 | 39.64% |

Coverage varies materially by exposure bucket, with particularly low coverage in the 16+ hour group. These are descriptive strata; they do not establish that stockout exposure causes the coverage differences. Confirm status encoding and interpretation against the dataset documentation.

## Interpretation and limitations

- Do not describe the interval as achieving 80% future coverage: held-out coverage was 77.46%.
- The evaluation window spans seven dates. Row count is large, but temporal evidence remains limited.
- The target is observed sales, not guaranteed unconstrained demand.
- Monitor coverage by date and relevant segment; consider rolling recalibration and a later untouched evaluation period.
- Interval width is expressed in the target's observed-sales units.

Raw predictions and model files remain in private R2. This report publishes aggregate metrics only.
