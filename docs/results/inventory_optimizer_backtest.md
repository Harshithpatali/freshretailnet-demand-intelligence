# Service-level constrained inventory optimizer: first backtest

**Workflow:** [Service-Level Inventory Decision Optimization #46142217478](https://github.com/Harshithpatali/freshretailnet-demand-intelligence/actions/runs/46142217478)  
**Status:** successful; six R2 artifacts published and verified in stable and immutable run-scoped paths  
**Commit:** `ac1b611d31fcb7975efb70bfb2bd4350e6954924`  
**Source population:** FreshRetailNet-LT, a separate external population (not FreshRetailNet-50K)

## Evaluation design

| Item | Result |
|---|---:|
| Complete continuous H1 store-product series | 2,613 |
| Stores | 842 |
| Continuous prediction horizon | 70 days |
| Calibration windows per series | 5 × 7 days |
| Later evaluation windows per series | 5 × 7 days |
| Calibration windows | 13,065 |
| Evaluation windows | 13,065 |

Each SKU's protection-period demand distribution is estimated from the first five non-overlapping seven-day windows. The model applies a mean forecast-error adjustment and empirical-Bayes shrinkage of residual standard deviation toward the cohort median. A moment-matched lognormal is used for the expected shortage/leftover calculations.

The MILP chooses one feasible order quantity per SKU. The store-level objective minimizes expected holding, shortage and fixed ordering costs, subject to:
- a store-level volume-weighted expected fill-rate target;
- shared storage capacity;
- shared replenishment capacity;
- SKU maximum order quantities;
- MOQ and pack-size multiples.

If a target is infeasible under the constraints, the fallback maximizes expected fulfilled demand and reports the feasibility gap.

## Scenario results

| Scenario | Target expected fill | Theoretical expected fill | Realized fill on later windows | Stores where expected target feasible | Capacity-limited stores | Simulated evaluation cost |
|---|---:|---:|---:|---:|---:|---:|
| balanced_90pct | 90% | 93.45% | **79.85%** | 767 / 842 | 75 | 133,913.67 |
| constrained_90pct | 90% | 71.02% | **60.28%** | 0 / 842 | 842 | 195,134.22 |
| tight_95pct | 95% | 49.77% | **42.69%** | 0 / 842 | 842 | 248,747.74 |

The balanced scenario used storage capacity of 14 demand-days and replenishment capacity of 7 demand-days. Its average storage utilization was 49.73%, while average replenishment utilization was 99.47%; replenishment capacity was close to binding. The constrained and tight scenarios reduced both capacity limits and became infeasible at every store.

For balanced_90pct, the empirical cycle service level was 32.75%, despite the theoretical expected fill rate of 93.45%. This confirms a substantial gap between the calibrated distribution and later realized observed sales. The expected fill constraint is satisfied for 767 stores in the model's own distribution, but that is not the same as achieving the target on later observations.

## Decision

**The optimizer execution and artifact publication passed; the empirical service target did not. Do not promote the resulting policy.**

The most important research finding is demand-distribution miscalibration and/or temporal drift. The first five calibration blocks did not adequately characterize the later evaluation blocks. Capacity is also binding: average replenishment utilization in the balanced case was 99.47%, and all stores become capacity-limited in the constrained and tight scenarios.

Next experiments should:
1. report SKU/store calibration of seven-day forecast residuals and demand variance before optimization;
2. compare residual-variance shrinkage strengths and robust uncertainty multipliers;
3. test empirical/conformal protection-period quantiles as alternatives to the lognormal approximation;
4. quantify service-level gaps across store-size and demand-pattern strata;
5. test the optimizer against retailer-supplied capacity, SKU volume, pack, MOQ and cost data.

## Interpretation limits

- The target is observed sales, which may be censored by stock availability; it is not identified latent demand.
- FreshRetailNet-LT is an external cohort and results must not be pooled with original FreshRetailNet-50K metrics.
- Only five calibration and five later evaluation windows per series are available.
- The lognormal model is a two-moment approximation whose tails were not independently validated.
- Cost, MOQ, pack-size, volume and capacity assumptions are illustrative normalized inputs.
- The simulator is a single protection-period replenishment decision; it is not a multi-period stochastic dynamic program with variable lead times, perishability or supplier uncertainty.

## Published R2 artifacts

- `inventory-optimizer-artifacts/inventory_optimizer_summary.json`
- `inventory-optimizer-artifacts/inventory_optimizer_store_results.csv`
- `inventory-optimizer-artifacts/inventory_optimizer_item_decisions.csv`
- `inventory-optimizer-artifacts/inventory_optimizer_rolling_evaluation.csv`
- `inventory-optimizer-artifacts/inventory_optimizer_demand_calibration.csv`
- `inventory-optimizer-artifacts/inventory_optimizer_manifest.json`

Each artifact also has an immutable copy under `experiment-runs/ac1b611d31fc/46142217478-attempt-1/`.
