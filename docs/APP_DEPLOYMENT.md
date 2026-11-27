# Dashboard and API deployment

## Architecture

- Streamlit frontend: `dashboard/app.py`
- FastAPI read-only service: `api/main.py`
- Cloudflare R2 stores all trained models, prediction exports and diagnostics.
- R2 credentials belong only in the API service environment. Never add them to Streamlit secrets or the repository.

The API reads these existing objects:
- `model-artifacts/metrics.json`
- `quantile-model-artifacts/quantile_metrics.json`
- `quantile-model-artifacts/evaluation_quantile_predictions.csv`
- `diagnostics/stockout/stockout_diagnostics.json`
- `diagnostics/stockout/sales_by_stockout_bucket.csv`
- `diagnostics/stockout/hourly_stock_status_profile.csv`
- `inventory-simulation-artifacts/inventory_simulation_summary.json`
- `inventory-simulation-artifacts/inventory_simulation_by_series.csv`
- `inventory-simulation-artifacts/inventory_cost_sensitivity.csv`

## Deploy FastAPI on Render free tier

1. In Render, create a new Blueprint from this repository and select the root `render.yaml`.
2. Add the four required R2 values when prompted: `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, and `R2_BUCKET`.
3. Deploy. Check `https://YOUR-API.onrender.com/health` and then `/docs`.
4. The free service may sleep when idle; the first request after inactivity can be slow. Do not use it for latency-sensitive production traffic without a suitable hosting plan.

## Deploy Streamlit Community Cloud

1. Create an app from this repository.
2. Set the main file path to `dashboard/app.py`.
3. Set the app's Python dependencies file to `dashboard/requirements.txt` if prompted.
4. Add `API_BASE_URL` as a Streamlit secret or environment variable, with the deployed API base URL (no trailing slash).
5. Deploy and verify each dashboard tab.

## Local development

API:
```bash
cd api
python -m pip install -r requirements.txt
export R2_ACCOUNT_ID=...
export R2_ACCESS_KEY_ID=...
export R2_SECRET_ACCESS_KEY=...
export R2_BUCKET=...
uvicorn main:app --reload
```

Dashboard in a second terminal:
```bash
cd dashboard
python -m pip install -r requirements.txt
export API_BASE_URL=http://localhost:8000
streamlit run app.py
```

## API routes

- `GET /health`
- `GET /api/v1/overview`
- `GET /api/v1/model-metrics`
- `GET /api/v1/quantile-metrics`
- `GET /api/v1/forecasts` (filters: `store_id`, `product_id`, `date_from`, `date_to`, `limit`)
- `GET /api/v1/filters`
- `GET /api/v1/stockout`, `/stockout/buckets`, `/stockout/hourly`
- `GET /api/v1/inventory`, `/inventory/series`, `/inventory/sensitivity`

## Interpretation and limitations

Forecast targets are observed sales, not guaranteed uncensored demand. Stockout bucket comparisons are descriptive, not causal lost-sales estimates. The current evaluation interval coverage is below the nominal 80% target. Inventory costs and service levels are conditional on configurable simulation assumptions, a short held-out horizon and the observed-sales proxy.
