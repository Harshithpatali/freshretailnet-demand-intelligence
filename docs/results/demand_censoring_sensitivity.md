# Demand-censoring sensitivity diagnostic

**Run:** [Demand Censoring Sensitivity, run #2](https://github.com/Harshithpatali/freshretailnet-demand-intelligence/actions/runs/46110128241)  
**R2 output verification:** completed successfully.  
**Generated:** 2026-11-28  
**Reference:** raw training split, in-stock hourly observations only  
**Evaluation:** 350,000 daily store-product rows from 2024-06-26 to 2024-07-02

## Observed exposure and imputation proxy

| Quantity | Result |
|---|---:|
| Observed evaluation sales | 417,568.523 units |
| Stockout-affected row rate | 40.9823% |
| Expected missed-sales proxy | 69,097.389 units |
| Stockout slots using global hourly fallback | 8,598 |

Hourly mean sales are estimated from training rows where the corresponding hourly stock status indicates availability, grouped by store, product and hour. When a store-product-hour reference rate is unavailable, the global in-stock mean for that hour is used. Only evaluation hours marked out of stock receive the imputed proxy.

The report additionally publishes lower, central and upper scenarios equal to observed sales plus 0.5x, 1.0x and 1.5x the estimated missing-sales proxy.

## Interpretation warning

The 69,097-unit value is **not an identified lost-sales total** and must not be interpreted as recovered latent demand. The central estimate assumes in-stock hourly sales rates represent demand during out-of-stock hours. That can fail when stockouts correlate with promotions, price, time-varying demand or customer substitution. Within-hour demand and substitutions are not observed.

Do not use these scenario values as a model target or causal business-impact estimate without independent validation.

Artifacts are written to private R2 paths under `diagnostics/demand-censoring/`.
