# Inventory policy backtest

## Scope and assumptions

The successful inventory simulation evaluated 50,000 store-product series using the held-out forecast rows.

| Setting | Value |
|---|---:|
| Store-product series | 50,000 |
| Lead time | 1 day |
| Review period | 1 day |
| Initial stock per series | 0 |
| Holding cost | 0.05 per unit per day |
| Shortage penalty | 2.00 per unfulfilled observed unit |
| Ordering cost | 5.00 per order |
| Demand proxy | Observed sales |
| Workflow | [Backtest FreshRetailNet Inventory Policies](https://github.com/Harshithpatali/freshretailnet-demand-intelligence/actions/runs/45941781156) |
| Workflow commit | `2a3dbe8` |

All cost values are simulation assumptions, not measured retailer costs. The simulator uses observed sales as a demand proxy; stockout-censored demand is not reconstructed.

## Full evaluation-horizon comparison

| Policy | Fill rate | Holding cost | Shortage cost | Ordering cost | Total simulated cost |
|---|---:|---:|---:|---:|---:|
| Seasonal naive lag-7 | 78.15% | 13,956.63 | 182,518.02 | 923,070.00 | 1,119,544.65 |
| P50 | 77.64% | 10,879.11 | 186,746.35 | 958,610.00 | 1,156,235.46 |
| P90 calibrated | 85.61% | 26,140.85 | 120,191.48 | 980,065.00 | 1,126,397.33 |

## Steady-state comparison

The simulator excludes the first lead-time day per series from these metrics to reduce cold-start effects.

| Policy | Steady-state fill rate | Steady-state holding cost | Steady-state shortage cost | Steady-state ordering cost | Steady-state total cost |
|---|---:|---:|---:|---:|---:|
| Seasonal naive lag-7 | 89.99% | 13,956.63 | 72,564.02 | 676,140.00 | 762,660.66 |
| P50 | 89.41% | 10,879.11 | 76,792.35 | 708,715.00 | 796,386.46 |
| P90 calibrated | 98.59% | 26,140.85 | 10,237.48 | 730,065.00 | 766,443.33 |

## What the trade-off says

- P90 calibrated produces the highest fill rate: 85.61% over the full horizon and 98.59% in steady state.
- Seasonal naive has the lowest simulated total cost under the stated assumptions.
- P90's higher service level comes with higher holding and ordering costs. Its full-horizon total simulated cost is about 0.61% higher than seasonal naive; steady-state cost is about 0.50% higher.
- P50 is not the preferred policy on either total cost or fill rate in this particular run.

There is no universal winner: the choice depends on the service-level target and the retailer's actual cost structure. This simulation does not establish real-world savings.

## Important caveats

- Fill rate is based on observed sales, not independently observed unconstrained demand.
- The cost assumptions are illustrative and should be replaced with retailer-specific inputs.
- Supplier capacity, minimum order quantities, stochastic lead times, shelf capacity and other operational constraints are not represented in this policy comparison.
- Orders that would arrive on or after the final evaluation day are suppressed and reported rather than charged.
- Full-horizon and steady-state totals answer different questions; do not mix them when comparing policies.

Raw series-level outputs remain in private R2. This report contains aggregate results from the successful workflow only.
