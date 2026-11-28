# Advanced forecasting research and service-level inventory optimization

This extension adds two reproducible research tracks. It does not replace the existing
LightGBM production reference or make a production-promotion claim.

## 1. Service-level constrained inventory decision engine

Implementation:
- `src/inventory_optimizer.py`
- `scripts/optimized_inventory_decision_engine.py`
- Workflow: **Service-Level Inventory Decision Optimization**

### Demand distribution

The engine uses the 70-day continuous H1 rolling predictions from the separate
FreshRetailNet-LT external population. It splits the horizon into five non-overlapping
seven-day calibration blocks followed by five non-overlapping seven-day evaluation
blocks. For each store-product series, calibration produces:

- a seven-day forecast mean adjusted by the mean historical forecast residual;
- a residual standard deviation shrunk toward the cohort median because only five
  calibration blocks are available;
- a moment-matched lognormal demand distribution with non-negative support.

The distribution is an approximation, not an identified latent-demand model. The target
remains observed sales and can be censored by stock availability.

### Optimization problem

For each store, each product selects one order quantity from a discrete feasible set:
zero or a minimum-order/pack-size multiple, bounded by its maximum order quantity.

The optimizer minimizes expected total cost:

```text
Expected holding cost
+ Expected shortage cost
+ Fixed ordering cost
```

Subject to:

- **Store service-level constraint:** expected volume-weighted fill rate must meet the
  scenario target.
- **Storage constraint:** the sum of SKU stock positions, weighted by volume, must
  remain within the store storage capacity.
- **Replenishment constraint:** total replenishment volume must remain within the
  store's replenishment capacity.
- **SKU constraints:** MOQ, pack multiples and per-SKU maximum order quantities.

This is a multiple-choice mixed-integer linear program solved with SciPy's
`milp` interface (HiGHS). If the service-level constraint is infeasible under the
capacity limits, the engine solves a second problem that maximizes expected fulfilled
demand under the same capacities and explicitly marks the SLA as infeasible. It does
not label an infeasible solution as successful.

### Scenarios

| Scenario | Expected fill-rate target | Storage capacity | Replenishment capacity |
|---|---:|---:|---:|
| balanced_90pct | 90% | 14 demand-days | 7 demand-days |
| constrained_90pct | 90% | 10 demand-days | 5 demand-days |
| tight_95pct | 95% | 7 demand-days | 3.5 demand-days |

MOQ 0.20, pack size 0.10 and normalized volume 1.0 are illustrative assumptions,
not supplier or physical case-pack data. Cost inputs are also illustrative.

### Evaluation

The selected order quantities are evaluated on the five later, disjoint seven-day blocks.
Reports include theoretical expected fill rate, empirical fill rate, cycle service level,
expected shortage, leftover stock, simulated costs, storage and replenishment utilization,
and the number of stores where the requested SLA was feasible.

R2 outputs:
- `inventory-optimizer-artifacts/inventory_optimizer_summary.json`
- `inventory-optimizer-artifacts/inventory_optimizer_store_results.csv`
- `inventory-optimizer-artifacts/inventory_optimizer_item_decisions.csv`
- `inventory-optimizer-artifacts/inventory_optimizer_rolling_evaluation.csv`
- `inventory-optimizer-artifacts/inventory_optimizer_demand_calibration.csv`
- `inventory-optimizer-artifacts/inventory_optimizer_manifest.json`

The API exposes the summary, store-level outcomes and SKU order recommendations under
`/api/v1/inventory/optimizer`. The Streamlit **Service-level optimizer** page compares
target versus realized fill rate and exposes capacity utilization and product decisions.

## 2. Advanced time-series forecasting research benchmark

Implementation:
- `experiments/forecast_research/forecast_research_benchmark.py`
- Existing causal wavelet feature builder: `experiments/wavelet/features.py`
- Workflow: **Forecast Research Benchmark (STL, Fourier, Spectral, Wavelet, Residual, Ensemble)**

### Data and validation design

The workflow takes a deterministic, seed-fixed sample of up to 3,000 complete
store-product series from FreshRetailNet-50K. It uses three adjacent seven-day rolling
test windows inside the 90-day training split, each preceded by a validation window and
an expanding training history. It also reports the original seven-day evaluation split
as a final research comparison.

The original evaluation split has already been used in earlier project benchmarks.
Therefore, the final comparison is **not** a new independent production-promotion test.
Rolling results are also sample-based, not full-population metrics.

### Model and feature comparisons

| Candidate | Added method |
|---|---|
| baseline | Existing leakage-aware lag and rolling LightGBM features |
| fourier_stl | Weekly Fourier harmonics, calendar harmonics and robust weekly STL summaries |
| spectral | Adds dominant period, weekly-band energy share, normalized spectral entropy and peak share |
| wavelet | Adds six causal db2 trailing-32-day wavelet features to the spectral feature set |
| residual_corrected | Adds a LightGBM residual model trained on validation prediction errors |
| ensemble | Non-negative weights selected on validation-tail squared error and frozen for test scoring |

STL and spectral summaries use observations strictly before each forecast block's origin.
Wavelet windows end at the day before their target. The benchmark excludes contemporaneous
discount, holiday/activity, weather, hourly sales and stock-status fields because their
forecast-time publication/availability is not established. Only historical lagged
availability summaries are retained.

### Statistics and selection

- MAE, RMSE, WAPE and signed bias are reported for each method and rolling fold.
- Wavelet value is assessed by paired ablation against the otherwise identical spectral
  feature model.
- Residual correction is trained on the first part of validation and checked on the
  later validation part.
- Ensemble weights are non-negative and sum to one; they are learned from validation
  predictions only.
- Paired uncertainty intervals use store-product cluster weights and contiguous
  two-day date-block weights. With only seven dates in the final evaluation, date-block
  inference is coarse and is reported with this limitation.

R2 outputs:
- `forecast-research-artifacts/forecast_research_summary.json`
- `forecast-research-artifacts/forecast_research_rolling_comparison.csv`
- `forecast-research-artifacts/forecast_research_final_comparison.csv`
- `forecast-research-artifacts/forecast_research_rolling_predictions.csv`
- `forecast-research-artifacts/forecast_research_final_predictions.csv`

The API exposes summary and comparison endpoints under `/api/v1/forecast-research`.
The Streamlit **Forecast research benchmark** page shows rolling results, final comparison,
validation-selected ensemble weights and bootstrap intervals.

## Reproduction and limitations

Both workflows are triggered by changes to their source files on the configured main or
research branch, and can also be run manually from GitHub Actions. They require the same
R2 secrets already used by the repository.

The optimizer and forecasting benchmark publish stable R2 aliases plus immutable
commit/run-scoped copies using the existing provenance helper. Each manifest records
the source artifact, commit, workflow run and configuration.

Do not claim a model improvement until the corresponding workflow succeeds and its
published report is inspected. These methods are experimental challengers. They do not
establish latent demand, causal lost sales, actual retailer service guarantees or measured
financial savings. A later untouched external temporal evaluation and retailer-specific
capacity, cost, volume, MOQ, pack-size and lead-time information remain necessary before
operational promotion.
