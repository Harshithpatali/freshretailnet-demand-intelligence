"""Read-only API for FreshRetailNet forecasting and inventory artifacts."""
from __future__ import annotations
import io, math, os, time
from functools import lru_cache
from typing import Any
import boto3
import pandas as pd
from botocore.config import Config
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="FreshRetailNet Demand Intelligence API",
    version="1.1.0",
    description="Analytics artifacts and evidence-grounded Groq stakeholder explanations.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class StakeholderBriefRequest(BaseModel):
    audience: str = Field(default="Executive leadership", max_length=80)
    focus: str = Field(
        default="Explain what the findings mean and what decisions should happen next.",
        max_length=600,
    )


def _stakeholder_evidence() -> tuple[dict[str, Any], list[str], list[str]]:
    """Read aggregate report JSON only; raw row-level predictions are not sent to Groq."""
    evidence: dict[str, Any] = {}
    available: list[str] = []
    missing: list[str] = []
    artifacts = {
        "inventory_optimizer": "inventory-optimizer-artifacts/inventory_optimizer_summary.json",
        "forecast_research": "forecast-research-artifacts/forecast_research_summary.json",
    }
    for name, key in artifacts.items():
        try:
            evidence[name] = _read_json(key)
            available.append(key)
        except Exception:
            evidence[name] = None
            missing.append(key)
    return evidence, available, missing


def _stakeholder_system_prompt() -> str:
    return (
        "You are a careful retail supply-chain analytics advisor. Use only facts in the "
        "provided JSON evidence. Separate measured backtest outcomes, model-expected "
        "outcomes, assumptions, and recommendations. Never describe simulated cost as "
        "realized savings, observed sales as uncensored latent demand, or a model-expected "
        "fill-rate constraint as achieved service. State uncertainty, sample limitations, "
        "and missing evidence. Do not invent dates, currency, causal explanations, capacity "
        "measurements, or business outcomes. Recommend reversible pilot decisions with "
        "owners and measurable acceptance tests. This is decision support only; it does "
        "not execute replenishment orders."
    )

def _client():
    required = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError("Missing environment variables: " + ", ".join(missing))
    return boto3.client("s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"], region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 4, "mode": "standard"}))

CACHE_TTL_SECONDS = max(1, int(os.getenv("ARTIFACT_CACHE_TTL_SECONDS", "300")))


def _json_safe(value):
    """Replace non-finite JSON values with null before FastAPI serialization."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@lru_cache(maxsize=16)
def _read_json_cached(key: str, cache_epoch: int) -> dict[str, Any]:
    obj = _client().get_object(Bucket=os.environ["R2_BUCKET"], Key=key)
    import json
    return _json_safe(json.loads(obj["Body"].read()))


def _read_json(key: str) -> dict[str, Any]:
    epoch = int(time.time() // CACHE_TTL_SECONDS)
    return _read_json_cached(key, epoch)


@lru_cache(maxsize=8)
def _read_csv_cached(key: str, cache_epoch: int) -> pd.DataFrame:
    obj = _client().get_object(Bucket=os.environ["R2_BUCKET"], Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def _read_csv(key: str) -> pd.DataFrame:
    epoch = int(time.time() // CACHE_TTL_SECONDS)
    return _read_csv_cached(key, epoch).copy()


def clear_artifact_caches() -> None:
    """Clear local API artifact caches; TTL remains the normal refresh mechanism."""
    _read_json_cached.cache_clear()
    _read_csv_cached.cache_clear()

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

def _require_columns(frame: pd.DataFrame, key: str, required: set[str]) -> pd.DataFrame:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"Artifact schema mismatch: {key} is missing columns {missing}. Republish the artifact.",
        )
    return frame

@app.get("/health")
def health():
    """Liveness endpoint; it does not depend on R2 availability."""
    configured = all(os.getenv(k) for k in (
        "R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"
    ))
    return {
        "status": "ok",
        "r2_configured": configured,
        "artifact_cache_ttl_seconds": CACHE_TTL_SECONDS,
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
        "groq_model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    }


@app.get("/ready")
def readiness():
    """Readiness endpoint that verifies the configured R2 bucket is reachable."""
    try:
        _client().head_bucket(Bucket=os.environ["R2_BUCKET"])
    except Exception as exc:
        raise HTTPException(status_code=503, detail="R2 bucket is not reachable from the API service.") from exc
    return {"status": "ready", "r2_reachable": True}


@app.get("/api/v1/overview")
def overview():
    """Return available overview artifacts independently instead of failing as a group."""
    artifact_keys = {
        "point_forecast": "model-artifacts/metrics.json",
        "quantile_forecast": "quantile-model-artifacts/quantile_metrics.json",
        "inventory": "inventory-simulation-artifacts/inventory_simulation_summary.json",
        "stockout": "diagnostics/stockout/stockout_diagnostics.json",
    }
    payload = {}
    status = {}
    for name, key in artifact_keys.items():
        try:
            payload[name] = _safe_json(key)
            status[name] = {"available": True, "key": key}
        except HTTPException:
            payload[name] = {}
            status[name] = {
                "available": False,
                "key": key,
                "message": "Artifact unavailable or unreadable. Other dashboard sections remain available.",
            }
    payload["artifact_status"] = status
    return payload

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
    df = _require_columns(df, "inventory-simulation-artifacts/inventory_simulation_by_series.csv", {"policy", "store_id", "product_id"})
    df = _require_columns(df, "inventory-simulation-artifacts/inventory_simulation_by_series.csv", {"policy", "store_id", "product_id"})
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
    df = _require_columns(df, "quantile-model-artifacts/evaluation_quantile_predictions.csv", {"store_id", "product_id", "date"})
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
    """Return compact metadata only; do not transfer all 50,000 series to the browser."""
    key = "quantile-model-artifacts/evaluation_quantile_predictions.csv"
    df = _safe_csv(key)
    df = _require_columns(df, key, {"store_id", "product_id", "date"})
    stores = df["store_id"].astype(str)
    products = df["product_id"].astype(str)
    return {
        "dates": {"start": str(df["date"].min()), "end": str(df["date"].max())},
        "rows": int(len(df)),
        "store_count": int(stores.nunique()),
        "product_count": int(products.nunique()),
        "series_count": int(df[["store_id", "product_id"]].drop_duplicates().shape[0]),
    }


@app.get("/api/v1/series/search")
def search_series(
    q: str | None = None,
    store_id: str | None = None,
    product_id: str | None = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
):
    """Search and paginate distinct store-product pairs on the API side."""
    key = "quantile-model-artifacts/evaluation_quantile_predictions.csv"
    df = _safe_csv(key)
    df = _require_columns(df, key, {"store_id", "product_id"})
    pairs = df.dropna(subset=["store_id", "product_id"])[["store_id", "product_id"]].astype(str).drop_duplicates()
    if store_id:
        pairs = pairs[pairs["store_id"].str.contains(store_id, case=False, regex=False, na=False)]
    if product_id:
        pairs = pairs[pairs["product_id"].str.contains(product_id, case=False, regex=False, na=False)]
    if q:
        mask = (
            pairs["store_id"].str.contains(q, case=False, regex=False, na=False)
            | pairs["product_id"].str.contains(q, case=False, regex=False, na=False)
        )
        pairs = pairs[mask]
    pairs = pairs.sort_values(["store_id", "product_id"])
    total = int(len(pairs))
    page = pairs.iloc[offset:offset + limit]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "rows": [
            {"store_id": row.store_id, "product_id": row.product_id}
            for row in page.itertuples(index=False)
        ],
    }

@app.get("/api/v1/series/options")
def series_options(store_id: str | None = None):
    """Return compact dropdown choices; product IDs are scoped to the selected store."""
    key = "quantile-model-artifacts/evaluation_quantile_predictions.csv"
    df = _safe_csv(key)
    df = _require_columns(df, key, {"store_id", "product_id"})
    pairs = df[["store_id", "product_id"]].astype(str).drop_duplicates()

    def identifier_key(value: str):
        return (0, int(value)) if value.isdigit() else (1, value.casefold())

    store_ids = sorted(pairs["store_id"].dropna().unique().tolist(), key=identifier_key)
    if store_id is None or store_id == "":
        product_ids = sorted(pairs["product_id"].dropna().unique().tolist(), key=identifier_key)
    else:
        selected = pairs[pairs["store_id"] == str(store_id)]
        product_ids = sorted(selected["product_id"].dropna().unique().tolist(), key=identifier_key)
    return {
        "store_ids": store_ids,
        "product_ids": product_ids,
        "selected_store_id": str(store_id) if store_id is not None else None,
    }


@app.get("/api/v1/dataset-eda")
def dataset_eda():
    """Return the full source-dataset EDA report, including all original fields."""
    return _safe_json("dataset-eda/full_dataset_eda.json")


@app.get("/api/v1/inventory/optimizer")
def inventory_optimizer_summary():
    """Service-level MILP results from the external FreshRetailNet-LT cohort."""
    return _safe_json("inventory-optimizer-artifacts/inventory_optimizer_summary.json")


@app.get("/api/v1/inventory/optimizer/stores")
def inventory_optimizer_stores(
    scenario: str | None = None,
    limit: int = Query(5000, ge=1, le=20000),
):
    key = "inventory-optimizer-artifacts/inventory_optimizer_store_results.csv"
    frame = _safe_csv(key)
    frame = _require_columns(
        frame, key,
        {"store_id", "scenario", "target_fill_rate", "achieved_fill_rate", "service_target_met"},
    )
    if scenario:
        frame = frame[frame["scenario"].astype(str) == scenario]
    return {
        "total": int(len(frame)),
        "rows": frame.head(limit).fillna("").to_dict(orient="records"),
    }


@app.get("/api/v1/inventory/optimizer/decisions")
def inventory_optimizer_decisions(
    scenario: str | None = None,
    store_id: str | None = None,
    limit: int = Query(2000, ge=1, le=10000),
):
    key = "inventory-optimizer-artifacts/inventory_optimizer_item_decisions.csv"
    frame = _safe_csv(key)
    frame = _require_columns(frame, key, {"store_id", "product_id", "scenario", "order_qty"})
    if scenario:
        frame = frame[frame["scenario"].astype(str) == scenario]
    if store_id:
        frame = frame[frame["store_id"].astype(str) == str(store_id)]
    return {
        "total": int(len(frame)),
        "rows": frame.head(limit).fillna("").to_dict(orient="records"),
    }


@app.get("/api/v1/inventory/optimizer/segments")
def inventory_optimizer_segments():
    key = "inventory-optimizer-artifacts/inventory_optimizer_segment_results.csv"
    frame = _safe_csv(key)
    frame = _require_columns(
        frame, key,
        {"scenario", "segment_type", "segment", "empirical_fill_rate", "shortage_units"},
    )
    return {"total": int(len(frame)), "rows": frame.fillna("").to_dict(orient="records")}


@app.get("/api/v1/inventory/optimizer/distributions")
def inventory_optimizer_distributions():
    key = "inventory-optimizer-artifacts/inventory_optimizer_distribution_diagnostics.csv"
    frame = _safe_csv(key)
    frame = _require_columns(
        frame, key,
        {"distribution", "segment_type", "segment", "median_mae",
         "central_80_coverage_pct", "central_80_mean_width"},
    )
    return {"total": int(len(frame)), "rows": frame.fillna("").to_dict(orient="records")}


@app.get("/api/v1/forecast-research/summary")
def forecast_research_summary():
    return _safe_json("forecast-research-artifacts/forecast_research_summary.json")


@app.get("/api/v1/forecast-research/rolling-comparison")
def forecast_research_rolling_comparison():
    key = "forecast-research-artifacts/forecast_research_rolling_comparison.csv"
    frame = _safe_csv(key)
    frame = _require_columns(frame, key, {"model", "mae", "rmse", "wape_pct", "bias_pct"})
    return {"total": int(len(frame)), "rows": frame.fillna("").to_dict(orient="records")}


@app.get("/api/v1/forecast-research/final-comparison")
def forecast_research_final_comparison():
    key = "forecast-research-artifacts/forecast_research_final_comparison.csv"
    frame = _safe_csv(key)
    frame = _require_columns(frame, key, {"model", "mae", "rmse", "wape_pct", "bias_pct"})
    return {"total": int(len(frame)), "rows": frame.fillna("").to_dict(orient="records")}


@app.get("/api/v1/model-metrics")
def model_metrics():
    return _safe_json("model-artifacts/metrics.json")

@app.get("/api/v1/sql-eda")
def sql_eda():
    return _safe_json("sql-eda/sql_eda_results.json")


@app.get("/api/v1/wavelet/metrics")
def wavelet_metrics():
    return _safe_json("wavelet-experiment-artifacts/wavelet_metrics.json")


@app.get("/api/v1/wavelet/feature-importance")
def wavelet_feature_importance(limit: int = Query(100, ge=1, le=2000)):
    df = _safe_csv("wavelet-experiment-artifacts/wavelet_feature_importance.csv")
    return {"total": int(len(df)), "rows": df.head(limit).fillna("").to_dict(orient="records")}


@app.get("/api/v1/wavelet/predictions")
def wavelet_predictions(
    store_id: str | None = None,
    product_id: str | None = None,
    limit: int = Query(5000, ge=1, le=10000),
):
    df = _safe_csv("wavelet-experiment-artifacts/wavelet_evaluation_predictions.csv")
    if store_id is not None and "store_id" in df.columns:
        df = df[df["store_id"].astype(str) == store_id]
    if product_id is not None and "product_id" in df.columns:
        df = df[df["product_id"].astype(str) == product_id]
    total = len(df)
    series = []
    if {"store_id", "product_id"}.issubset(df.columns):
        pairs = df[["store_id", "product_id"]].astype(str).drop_duplicates().sort_values(["store_id", "product_id"])
        series = [{"store_id": r.store_id, "product_id": r.product_id} for r in pairs.itertuples(index=False)]
    return {
        "total": total,
        "series": series,
        "rows": df.head(limit).replace({float("nan"): None}).to_dict(orient="records"),
    }


@app.get("/api/v1/monitoring/forecast-quality")
def forecast_quality_monitoring():
    return _safe_json("monitoring/forecast-quality/forecast_monitoring_summary.json")


@app.get("/api/v1/monitoring/forecast-quality/daily")
def forecast_quality_monitoring_daily(limit: int = Query(1000, ge=1, le=5000)):
    key = "monitoring/forecast-quality/forecast_monitoring_daily.csv"
    df = _safe_csv(key)
    df = _require_columns(df, key, {"date", "mae", "wape_pct", "bias_pct", "calibrated_coverage_pct"})
    return {"total": int(len(df)), "rows": df.head(limit).fillna("").to_dict(orient="records")}


@app.get("/api/v1/monitoring/forecast-quality/drift")
def forecast_quality_drift():
    key = "monitoring/forecast-quality/forecast_monitoring_drift.csv"
    df = _safe_csv(key)
    df = _require_columns(df, key, {"variable", "psi", "status"})
    return {"total": int(len(df)), "rows": df.fillna("").to_dict(orient="records")}


@app.get("/api/v1/demand-censoring/sensitivity")
def demand_censoring_sensitivity():
    return _safe_json("diagnostics/demand-censoring/demand_censoring_sensitivity.json")


@app.get("/api/v1/demand-censoring/sensitivity/buckets")
def demand_censoring_sensitivity_buckets():
    key = "diagnostics/demand-censoring/demand_censoring_sensitivity_by_bucket.csv"
    df = _safe_csv(key)
    df = _require_columns(df, key, {"stockout_bucket", "rows", "expected_lost_sales_proxy_units"})
    return {"total": int(len(df)), "rows": df.fillna("").to_dict(orient="records")}


@app.get("/api/v1/inventory/scenarios")
def inventory_scenario_summary():
    return _safe_json("inventory-scenario-artifacts/inventory_scenario_summary.json")


@app.get("/api/v1/inventory/scenarios/comparison")
def inventory_scenario_comparison():
    key = "inventory-scenario-artifacts/inventory_scenario_comparison.csv"
    df = _safe_csv(key)
    df = _require_columns(df, key, {"scenario", "policy", "fill_rate_pct", "total_simulated_cost"})
    return {"total": int(len(df)), "rows": df.fillna("").to_dict(orient="records")}



@app.get("/api/v1/monitoring/forecast-quality/exposure")
def forecast_quality_monitoring_exposure():
    key = "monitoring/forecast-quality/forecast_monitoring_exposure.csv"
    df = _safe_csv(key)
    df = _require_columns(df, key, {"stockout_exposure_bucket", "rows", "calibrated_coverage_pct", "calibrated_mean_interval_width"})
    return {"total": int(len(df)), "rows": df.fillna("").to_dict(orient="records")}


@app.get("/api/v1/monitoring/forecast-quality/time-exposure")
def forecast_quality_monitoring_time_exposure(limit: int = Query(5000, ge=1, le=20000)):
    key = "monitoring/forecast-quality/forecast_monitoring_time_exposure.csv"
    df = _safe_csv(key)
    df = _require_columns(df, key, {"date", "stockout_exposure_bucket", "rows", "calibrated_coverage_pct"})
    return {"total": int(len(df)), "rows": df.head(limit).fillna("").to_dict(orient="records")}


@app.post("/api/v1/stakeholder/brief")
def generate_stakeholder_brief(request: StakeholderBriefRequest):
    """Generate a factual stakeholder narrative from current R2 research summaries."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Groq is not configured. Add GROQ_API_KEY to the API service environment.",
        )
    evidence, sources, missing = _stakeholder_evidence()
    if not sources:
        raise HTTPException(
            status_code=503,
            detail="No inventory or forecasting research reports are available in R2.",
        )
    import json
    try:
        from groq import Groq

        model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        client = Groq(api_key=api_key, timeout=50.0, max_retries=1)
        prompt = {
            "audience": request.audience,
            "focus": request.focus,
            "evidence": evidence,
            "available_source_artifacts": sources,
            "missing_source_artifacts": missing,
            "required_sections": [
                "Executive takeaway",
                "Measured findings and their evaluation periods",
                "Operational implications and capacity bottlenecks",
                "Ranked decisions with owner and measurable acceptance test",
                "What should not be promoted or claimed yet",
                "Next validation plan",
            ],
        }
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _stakeholder_system_prompt()},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False, default=str)},
            ],
            temperature=0.2,
            max_completion_tokens=1800,
        )
        brief = completion.choices[0].message.content or ""
        if not brief.strip():
            raise RuntimeError("Groq returned an empty stakeholder brief.")
        usage = getattr(completion, "usage", None)
        return {
            "brief_markdown": brief,
            "model": model,
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_artifacts": sources,
            "missing_source_artifacts": missing,
            "usage": usage.model_dump() if usage is not None else None,
            "disclaimer": (
                "AI-generated explanation of published research. Verify source metrics "
                "and approve decisions with the responsible operational owner."
            ),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Groq stakeholder briefing failed ({type(exc).__name__}). Check model availability and API quota.",
        ) from exc
