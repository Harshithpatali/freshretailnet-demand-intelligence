# API and dashboard artifact failure behaviour

## API contract

- `/health` is a process/configuration check. It reports whether the four R2
  environment variables are present; it does not guarantee that every object exists.
- Artifact-backed endpoints return HTTP 503 when an R2 object is missing, storage
  is unavailable, or the cached artifact cannot be read. The response identifies
  the artifact key and suggests checking R2 configuration/workflow outputs.
- Forecast and inventory-series endpoints validate required columns. A stale or
  malformed CSV returns HTTP 503 with the missing column names rather than an
  unhandled pandas `KeyError`.
- `/api/v1/overview` intentionally requires the four core overview artifacts.
  If one is missing, the dashboard startup view reports the API error rather
  than showing a misleading partial executive summary.

## Dashboard behaviour

- API health and the core overview are required for the main dashboard shell;
  startup failures are shown with the configured API URL and exception details.
- Forecast explorer, stockout bucket/hourly charts, inventory sensitivity, SQL
  EDA and experiment inputs are optional panels. Their failures are shown in
  the affected panel and do not prevent other tabs from rendering.
- Forecast query and experiment execution errors are surfaced in their own panel.
- A successful `/health` response only means the API process is alive; users
  should check artifact-backed endpoints when a particular panel is unavailable.

## Operational recovery

1. Check `/health` and confirm `r2_configured=true`.
2. Inspect the failed endpoint's HTTP 503 detail for the exact R2 key.
3. Re-run the workflow responsible for that artifact and verify its R2 upload.
4. Refresh the dashboard after the artifact has been published.

The offline tests monkeypatch storage access; they do not require R2 credentials
and do not claim that production objects are currently available.
