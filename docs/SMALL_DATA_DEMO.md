# Reproducible small-data demo

The repository includes a deterministic synthetic demo that exercises the inventory
simulation without Cloudflare R2 credentials or the full FreshRetailNet dataset.

From the repository root:

```bash
pip install -r api/requirements.txt
python scripts/run_small_demo.py --output-dir demo_output
```

It writes:
- `synthetic_forecasts.csv`: 42 synthetic daily forecast rows across two series.
- `inventory_policy_results.csv`: P50/P90 policy simulation outputs.
- `demo_summary.json`: assumptions, aggregate comparisons and explicit caveats.

All values are generated for software demonstration only. They are not dataset
observations, model evaluation results, or evidence of business savings. The
full model-training and benchmark workflows still require the source data and
R2 artifacts documented in the root README.
