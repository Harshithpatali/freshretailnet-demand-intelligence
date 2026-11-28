# FreshRetailNet-LT external temporal validation

This is an external, later-dated evaluation population added to complement the original FreshRetailNet-50K benchmark. It is **not the same cohort** as the 50,000-series original project data, and its metrics must not be pooled with or directly compared as if the population were identical.

## Pinned source and verified ingestion

Source: [Dingdong-Inc/FreshRetailNet-LT](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-LT)  
Pinned revision: `01c3bb306415e09d3ccd2fe762f33eb3aeb2affd`  
License: CC BY 4.0  
R2 ingestion manifest: `external/freshretailnet-lt/v1.0/ingestion_manifest.json`

The ingestion workflow verified file checksums and R2 uploads:

| Split | Rows | Store-product series | Dates |
|---|---:|---:|---|
| Train | 7,869,549 | 22,939 | 2023-06-05 to 2025-07-13 |
| Official labeled evaluation | 70,000 | 10,000 | 2025-07-14 to 2025-07-20 |

The evaluation split is chronological and strictly later than the training split. The source's official labeled test spans seven days; that alone is too short to assess a seven-day replenishment cycle after arrival.

## Current external experiment

Workflow: [FreshRetailNet-LT External Temporal and Inventory Validation](https://github.com/Harshithpatali/freshretailnet-demand-intelligence/actions/workflows/freshretailnet-lt-validation.yml)

The workflow:
1. Downloads the pinned release, verifies SHA-256 and schema, then stages both splits in R2.
2. Runs the same leakage-aware rolling segment-specialist engine on the combined external time history with five chronological folds and 14 test days per fold (70 target dates in total).
3. Runs 500 paired store-product cluster-bootstrap replicates on the saved predictions.
4. Uses continuous H1 predictions over the 70-day block for inventory simulations with 1-, 3- and 7-day lead times, plus a constrained 7-day lead / 7-day review scenario.

The 70-day H1 inventory gate requires a contiguous global date range and at least 1,000 store-product series with a complete daily prediction record for the entire block. Incomplete series are excluded and reported. This avoids the original seven-day lead-time test-horizon boundary.

## Status and interpretation

The dataset ingestion step has succeeded and the files are checksum-verified in R2. The current rolling model workflow is still running; no external H7/H14 metric, confidence interval or inventory policy result should be claimed until the workflow completes and verifies its R2 outputs.

The official July 14–20, 2025 labeled evaluation is part of this external dataset's later period. The rolling evaluation must be interpreted as an expanding-window historical simulation on a new dataset population, not as an independent test of the same 50,000 series from the original benchmark. Keep the original and external results separate.

All inventory costs, initial stock, capacity, MOQ and pack-size inputs remain illustrative assumptions. Observed sales can be stockout-censored; the simulation does not establish latent demand or measured operating savings.
