"""Read-only API for FreshRetailNet forecasting and inventory artifacts."""
from __future__ import annotations
import io, os
from functools import lru_cache
from typing import Any
import boto3
import pandas as pd
from botocore.config import Config
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="FreshRetailNet Demand Intelligence API", version="1.0.0",
              description="Read-only access to model, stockout and inventory backtest artifacts.")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
                   allow_credentials=False, allow_methods=["GET"], allow_headers=["*"])

def _client():
    required = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError("Missing environment variables: " + ", ".join(missing))
    return boto3.client("s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"], region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 4, "mode": "standard"}))

@lru_cache(maxsize=16)
def _read_json(key: str) -> dict[str, Any]:
    obj = _client().get_object(Bucket=os.environ["R2_BUCKET"], Key=key)
    import json
    return json.loads(obj["Body"].read())

@lru_cache(maxsize=8)
def _read_csv(key: str) -> pd.DataFrame:
    obj = _client().get_object(Bucket=os.environ["R2_BUCKET"], Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))

def _safe_json(key: str):
    try:
        return _read_json(key)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Artifact unavailable: {key}. Check R2 configuration and workflow outputs.") from exc

def _safe_csv(key: str):
    try:
        return _read_csv(key)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Artifact unavailable: {key}. Check R2 configuration and workflow outputs.") from exc

@app.get("/health")
def health():
    configured = all(os.getenv(k) for k in ("R2_ACCOUNT_ID","R2_ACCESS_KEY_ID","R2_SECRET_ACCESS_KEY","R2_BUCKET"))
    return {"status": "ok", "r2_configured": configured}

@app.get("/api/v1/overview")
def overview():
    return {
        "point_forecast": _safe_json("model-artifacts/metrics.json"),
        "quantile_forecast": _safe_json("quantile-model-artifacts/quantile_metrics.json"),
        "inventory": _safe_json("inventory-simulation-artifacts/inventory_simulation_summary.json"),
        "stockout": _safe_json("diagnostics/stockout/stockout_diagnostics.json"),
    }

@app.get("/api/v1/quantile-metrics")
def quantile_metrics():
    return _safe_json("quantile-model-artifacts/quantile_metrics.json")

@app.get("/api/v1/stockout")
def stockout():
    return _safe_json("diagnostics/stockout/stockout_diagnostics.json")

@app.get("/api/v1/stockout/buckets")
def stockout_buckets():
    return {"rows": _safe_csv("diagnostics/stockout/sales_by_stockout_bucket.csv").fillna("").to_dict(orient="records")}

@app.get("/api/v1/stockout/hourly")
def stockout_hourly():
    return {"rows": _safe_csv("diagnostics/stockout/hourly_stock_status_profile.csv").fillna("").to_dict(orient="records")}

@app.get("/api/v1/inventory")
def inventory():
    return _safe_json("inventory-simulation-artifacts/inventory_simulation_summary.json")

@app.get("/api/v1/inventory/series")
def inventory_series(policy: str | None = None, store_id: str | None = None,
                     product_id: str | None = None, limit: int = Query(500, ge=1, le=5000)):
    df = _safe_csv("inventory-simulation-artifacts/inventory_simulation_by_series.csv")
    for col, value in (("policy", policy), ("store_id", store_id), ("product_id", product_id)):
        if value is not None:
            df = df[df[col].astype(str) == value]
    return {"total": int(len(df)), "rows": df.head(limit).replace({float("nan"): None}).to_dict(orient="records")}

@app.get("/api/v1/inventory/sensitivity")
def inventory_sensitivity():
    return {"rows": _safe_csv("inventory-simulation-artifacts/inventory_cost_sensitivity.csv").fillna("").to_dict(orient="records")}

@app.get("/api/v1/forecasts")
def forecasts(store_id: str | None = None, product_id: str | None = None,
              date_from: str | None = None, date_to: str | None = None,
              limit: int = Query(1000, ge=1, le=10000)):
    df = _safe_csv("quantile-model-artifacts/evaluation_quantile_predictions.csv")
    if store_id is not None:
        df = df[df["store_id"].astype(str) == store_id]
    if product_id is not None:
        df = df[df["product_id"].astype(str) == product_id]
    if date_from:
        df = df[df["date"] >= date_from]
    if date_to:
        df = df[df["date"] <= date_to]
    total = len(df)
    return {"total": int(total), "rows": df.head(limit).fillna("").to_dict(orient="records")}

@app.get("/api/v1/filters")
def filters():
    df = _safe_csv("quantile-model-artifacts/evaluation_quantile_predictions.csv")
    return {"stores": sorted(df["store_id"].astype(str).unique().tolist()),
            "products": sorted(df["product_id"].astype(str).unique().tolist()),
            "dates": {"start": str(df["date"].min()), "end": str(df["date"].max())},
            "rows": int(len(df))}

@app.get("/api/v1/model-metrics")
def model_metrics():
    return _safe_json("model-artifacts/metrics.json")
