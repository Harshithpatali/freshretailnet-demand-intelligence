# Stochastic inventory optimization benchmark

## Objective

The optimizer chooses one order-up-to quantity per store-product series. It minimizes
expected holding, shortage and ordering costs subject to a **store-level expected
fill-rate floor** and a shared store capacity. It also respects configurable minimum
order quantity (MOQ), pack-size rounding and maximum order quantity assumptions.

The decision is a replenishment recommendation for analysis. It does not submit
purchase orders or connect to a retailer ERP.

## Mathematical formulation

For SKU (i), the protection-period demand approximation is:

[
D_i sim mathcal{N}(\mu_i, \sigma_i^2)
]

where the protection period is lead time plus review period:

[
\mu_i = \sum_{t=1}^{L+R} P50_{i,t}, \qquad
\sigma_i^2 = \sum_{t=1}^{L+R}
\left(\frac{P90_{i,t}-P10_{i,t}}{2.5631}\right)^2
]

The variance aggregation assumes independent daily forecast errors. It is an explicit
planning approximation, not an empirically established dependence model.

For candidate order-up-to quantity (q_i), the Normal approximation gives expected
shortage, overage and fulfilled quantity. The optimizer chooses one candidate quantity
from a finite, pack-size-rounded set per SKU.

[
\min_{q_i} \sum_i
\left[
C_h\frac{R}{2}\mathbb{E}(q_i-D_i)^+
+ C_s\mathbb{E}(D_i-q_i)^+
+ C_o\mathbf{1}(q_i>0)
\right]
+ M\,s
]

subject to:

[
\sum_i q_i \leq C_{store}
]

[
\sum_i \mathbb{E}[\min(D_i,q_i)] + s
\geq \alpha\sum_i \mu_i
]

where (\alpha) is the target store fill rate and (s\geq0) is a heavily penalized
service-level shortfall slack. The slack allows the solver to return a transparent
least-violation plan when the capacity/service combination is infeasible, rather than
silently claiming the target was met.

Additional SKU constraints include MOQ, pack-size multiples, and maximum order quantity
expressed in mean forecast-demand days.

## Scenario grid

The GitHub Actions workflow evaluates:

- Lead/review combinations: 1/1, 3/2 and 5/2 days.
- Target expected fill rates: 85%, 90% and 95%.
- Shared store capacity: 2, 4 and 7 mean-demand days.
- Base cost assumptions: holding 0.05 per unit-day, shortage 2.0 per unit,
  fixed order cost 5.0, MOQ 0.2 and pack size 0.1 in normalized target units.

This creates 27 comparable configurations. The 3-day lead / 2-day review, 90% target,
4 demand-day capacity scenario is also published as a reference SKU plan.

## Artifacts and API

Workflow: **Stochastic Inventory Optimization Benchmark**.

R2 objects under `inventory-optimization-artifacts/`:

- `inventory_optimization_summary.json`
- `inventory_optimization_scenarios.csv`
- `inventory_optimization_store_summary.csv`
- `inventory_optimization_reference_plan.csv`
- `inventory_optimization_manifest.json`

API:

- `GET /api/v1/inventory/optimization`
- `GET /api/v1/inventory/optimization/scenarios`
- `GET /api/v1/inventory/optimization/plan?store_id=...&product_id=...&limit=...`

## Interpretation and limitations

- The target is normalized observed sales, not known unconstrained customer demand.
  Stockouts may censor the target.
- Daily demand errors are assumed independent when lead/review protection variance is
  aggregated. Positive temporal or cross-product correlations are not modelled.
- A Normal approximation can be poor for intermittent, zero-heavy or strongly
  right-skewed demand.
- Shared capacity is expressed in normalized observed-sales units, scaled by each
  store's summed mean daily forecast. The dataset does not provide physical shelf or
  warehouse capacity.
- MOQ, pack size, lead time, review cadence and costs are illustrative assumptions.
  They must be replaced by retailer and supplier data before operational use.
- A service target is marked met only when solver shortfall slack is effectively zero.
  Review both achieved expected fill rate and the proportion of stores meeting the
  target; a portfolio average can hide infeasible individual stores.
- The original labeled evaluation covers seven days. This experiment optimizes
  protection-period quantities from forecasts, but does not establish realized
  long-run service outcomes for supplier lead times longer than the available period.
