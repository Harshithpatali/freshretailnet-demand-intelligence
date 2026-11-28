# Inventory lead-time, review-cycle and replenishment-constraint scenarios

**Run:** [Inventory Lead-Time and Constraint Scenarios, run #1](https://github.com/Harshithpatali/freshretailnet-demand-intelligence/actions/runs/46110141403)  
**R2 output verification:** completed successfully for the comparison CSV and JSON summary.  
**Evaluation period:** 2024-06-26 to 2024-07-02  
**Population:** 350,000 daily rows across 50,000 store-product series  
**Demand proxy:** observed `sale_amount`, which may be censored by stock availability

## Structural scenario results

Base cost assumptions: holding cost 0.05 per unit-day, shortage penalty 2.0 per unit and order cost 5.0 per order. Initial stock is zero.

| Lead / review | Policy | Fill rate | Total simulated cost |
|---|---|---:|---:|
| 1 / 1 day | Seasonal naive | 78.15% | 1,119,545 |
| 1 / 1 day | P50 | 77.64% | 1,156,235 |
| 1 / 1 day | Calibrated P90 | 85.61% | 1,126,397 |
| 3 / 1 day | Seasonal naive | 56.87% | 785,647 |
| 3 / 1 day | P50 | 54.78% | 744,857 |
| 3 / 1 day | Calibrated P90 | 59.83% | 763,459 |
| 1 / 7 days | Seasonal naive | 83.62% | 456,350 |
| 1 / 7 days | P50 | 83.64% | 446,619 |
| 1 / 7 days | Calibrated P90 | 86.77% | 482,786 |

The seven-day review policy reduced order frequency, and the calibrated P90 policy achieved the highest fill rate in that scenario, but it also had the highest simulated cost of the three.

## Seven-day lead-time horizon limitation

The seven-day lead / one-day review scenarios and seven-day lead / seven-day review constrained scenario recorded **0% fill**. The test contains exactly seven evaluation days, and the simulator suppresses orders scheduled to arrive on or after the final evaluation day. As a result, replenishment does not arrive during the measured horizon. These rows are a horizon-boundary stress test only; they cannot be used to rank or claim the steady-state performance of a seven-day lead-time policy.

The constrained seven-day scenario used:
- storage capacity equal to 14 times each series' mean daily forecast;
- order cap equal to 7 times mean daily forecast;
- minimum order quantity 6;
- pack size 6.

The run tracked constraint-limited units and orders, but the short test horizon prevents service-level interpretation for this lead time. A later evaluation period of at least 14 days is required for a meaningful seven-day lead-time comparison.

## Shortage-penalty sensitivity

The workflow also evaluated 0.5x and 2.0x the base shortage penalty for the unconstrained 1-day lead / 1-day review case:

| Policy | 0.5x shortage penalty cost | 1.0x base cost | 2.0x shortage penalty cost |
|---|---:|---:|---:|
| Seasonal naive | 1,028,286 | 1,119,545 | 1,302,063 |
| P50 | 1,062,862 | 1,156,235 | 1,342,982 |
| Calibrated P90 | 1,066,302 | 1,126,397 | 1,246,589 |

At higher shortage penalties, the higher-fill P90 policy becomes more competitive on total simulated cost. This is a consequence of the explicit cost assumptions, not a measured business saving.

## What this does and does not establish

The scenario grid now varies lead time, review cadence, storage capacity, maximum order size, MOQ, pack size and shortage penalty. It also reports constrained order quantities.

It remains a controlled simulation, not a calibrated retailer replenishment system. In particular:
- demand is proxied by observed sales and may be censored during stockouts;
- all stores/products use uniform illustrative MOQ and pack size;
- lead time is deterministic;
- storage and maximum order capacity are scaled to the model forecast, not measured physical capacity;
- expiry, perishability, supplier calendars, stochastic delivery and minimum presentation stock are not modelled;
- the seven-day dataset evaluation horizon is too short for the seven-day lead cases.

Use retailer/supplier constraints and a longer untouched evaluation period before operational or financial conclusions.
