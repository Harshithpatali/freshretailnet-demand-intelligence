"""Read-only API for FreshRetailNet forecasting and inventory artifacts."""
from __future__ import annotations
import io, math, os, time, logging
from collections.abc import Mapping
from functools import lru_cache
from typing import Any
import boto3
import pandas as pd
from botocore.config import Config
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="FreshRetailNet Demand Intelligence API",
    version="1.2.0",
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



def _pick_fields(payload: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    return {key: payload[key] for key in fields if key in payload}


def _compact_rows(rows: Any, fields: tuple[str, ...], limit: int = 8) -> list[dict[str, Any]]:
    """Retain only whitelisted scalar fields from a short list of aggregate rows."""
    if not isinstance(rows, list):
        return []
    compact = []
    for row in rows[:limit]:
        if isinstance(row, Mapping):
            compact.append(_pick_fields(row, fields))
    return compact


def _compact_capacity(items: Any) -> list[dict[str, Any]]:
    compact = []
    if not isinstance(items, list):
        return compact
    for item in items[:3]:
        if not isinstance(item, Mapping):
            continue
        replenishment = item.get("replenishment_utilization_pct", {})
        storage = item.get("storage_utilization_pct", {})
        sizes = item.get("by_store_size", [])
        compact.append({
            "scenario": item.get("scenario"),
            "store_count": item.get("store_count"),
            "replenishment_utilization_pct": _pick_fields(
                replenishment, ("p50", "p90", "p95", "max",
                                "share_at_least_95_pct", "share_at_least_99_pct")
            ),
            "storage_utilization_pct": _pick_fields(
                storage, ("p50", "p90", "p95", "max", "share_at_least_95_pct")
            ),
            "by_store_size": _compact_rows(
                sizes,
                ("store_size_segment", "stores",
                 "mean_replenishment_utilization_pct",
                 "share_replenishment_at_least_95_pct",
                 "service_target_met_stores", "mean_expected_fill_rate"),
                limit=3,
            ),
        })
    return compact


def _compact_bootstrap(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    comparison = payload.get("comparisons_vs_baseline", {})
    compact_comparison = {}
    if isinstance(comparison, Mapping):
        for model, metrics in list(comparison.items())[:6]:
            if not isinstance(metrics, Mapping):
                continue
            compact_comparison[str(model)] = {}
            for metric_name in ("mae", "rmse", "wape"):
                metric = metrics.get(metric_name)
                if isinstance(metric, Mapping):
                    compact_comparison[str(model)][metric_name] = _pick_fields(
                        metric,
                        ("mean_improvement_pct", "ci_95_low_pct",
                         "ci_95_high_pct", "probability_improvement",
                         "ci_excludes_zero"),
                    )
    return {
        "method": payload.get("method"),
        "replicates": payload.get("replicates"),
        "evaluation_dates": payload.get("evaluation_dates"),
        "comparisons_vs_baseline": compact_comparison,
        "caveat": payload.get("caveat"),
    }


def _compact_stakeholder_evidence(raw: dict[str, Any]) -> dict[str, Any]:
    """Create short, high-signal evidence packs for the two briefing requests."""
    inventory = raw.get("inventory_optimizer") or {}
    if not isinstance(inventory, Mapping):
        inventory = {}

    scenario_source = inventory.get("scenario_results", [])
    balanced = next(
        (
            row for row in scenario_source
            if isinstance(row, Mapping) and row.get("scenario") == "balanced_90pct"
        ),
        scenario_source[0] if isinstance(scenario_source, list) and scenario_source else {},
    )
    scenario = _compact_rows(
        [balanced] if isinstance(balanced, Mapping) else [],
        (
            "scenario", "target_fill_rate", "theoretical_expected_fill_rate",
            "empirical_fill_rate", "empirical_cycle_service_level",
            "capacity_limited_store_count", "service_target_met_store_count",
            "store_count", "store_product_series", "evaluation_windows",
            "unfulfilled_protection_demand", "leftover_stock", "holding_cost",
            "shortage_cost", "ordering_cost", "total_simulated_cost",
            "mean_storage_utilization_pct", "mean_replenishment_utilization_pct",
            "service_target_met_empirically",
        ),
        limit=1,
    )

    capacity_rows = inventory.get("capacity_diagnostics", [])
    capacity_row = next(
        (
            row for row in capacity_rows
            if isinstance(row, Mapping) and row.get("scenario") == "balanced_90pct"
        ),
        {},
    ) if isinstance(capacity_rows, list) else {}
    replenishment = capacity_row.get("replenishment_utilization_pct", {})
    storage = capacity_row.get("storage_utilization_pct", {})
    size_rows = capacity_row.get("by_store_size", [])
    capacity = {
        "replenishment_utilization_pct": _pick_fields(
            replenishment,
            ("p50", "p90", "p95", "max", "share_at_least_90_pct",
             "share_at_least_95_pct", "share_at_least_99_pct"),
        ),
        "storage_utilization_pct": _pick_fields(
            storage, ("p50", "p90", "p95", "max", "share_at_least_95_pct")
        ),
        "by_store_size": _compact_rows(
            size_rows,
            ("store_size_segment", "stores", "mean_replenishment_utilization_pct",
             "share_replenishment_at_least_95_pct", "service_target_met_stores"),
            limit=3,
        ),
    }

    segment_source = inventory.get("segment_results", [])
    segment_source = [
        row for row in segment_source
        if isinstance(row, Mapping) and row.get("scenario") == "balanced_90pct"
    ] if isinstance(segment_source, list) else []
    segments = _compact_rows(
        segment_source,
        ("segment_type", "segment", "stores", "empirical_fill_rate", "shortage_units"),
        limit=6,
    )

    distribution = inventory.get("distribution_diagnostics", {})
    distribution_rows = distribution.get("results", []) if isinstance(distribution, Mapping) else []
    distribution_rows = [
        row for row in distribution_rows
        if isinstance(row, Mapping) and row.get("segment_type") == "overall"
    ] if isinstance(distribution_rows, list) else []
    distribution_scores = _compact_rows(
        distribution_rows,
        ("distribution", "median_mae", "pinball_q10", "pinball_q50",
         "pinball_q90", "central_80_coverage_pct", "central_80_mean_width"),
        limit=3,
    )
    population = inventory.get("evaluation_population", {})
    calibration = inventory.get("calibration_design", {})
    evaluation_design = inventory.get("evaluation_design", {})
    compact_inventory = {
        "population": _pick_fields(
            population,
            ("target_date_start", "target_date_end", "target_dates",
             "complete_continuous_series"),
        ),
        "calibration_windows_per_series": (
            calibration.get("windows_per_series") if isinstance(calibration, Mapping) else None
        ),
        "evaluation_windows_per_series": (
            evaluation_design.get("windows_per_series")
            if isinstance(evaluation_design, Mapping) else None
        ),
        "scenario": scenario[0] if scenario else {},
        "capacity": capacity,
        "segments": segments,
        "reference_distribution": (
            distribution.get("reference_optimizer_distribution")
            if isinstance(distribution, Mapping) else None
        ),
        "distribution_challengers": (
            distribution.get("challengers") if isinstance(distribution, Mapping) else None
        ),
        "distribution_scores": distribution_scores,
    }

    forecast = raw.get("forecast_research") or {}
    if not isinstance(forecast, Mapping):
        forecast = {}
    rolling = forecast.get("rolling_design", {})
    final = forecast.get("final_research_comparison", {})
    if not isinstance(rolling, Mapping):
        rolling = {}
    if not isinstance(final, Mapping):
        final = {}

    def metric_rows(source: Any) -> list[dict[str, Any]]:
        if not isinstance(source, list):
            return []
        priority = (
            "baseline", "fourier_stl", "spectral", "wavelet",
            "residual_corrected", "ensemble",
        )
        by_name = {
            str(row.get("model")): row
            for row in source if isinstance(row, Mapping)
        }
        selected = []
        baseline = by_name.get("baseline", {})
        for name in priority:
            row = by_name.get(name)
            if not isinstance(row, Mapping):
                continue
            compact = _pick_fields(row, ("model", "mae", "rmse", "wape_pct", "bias_pct"))
            if name != "baseline":
                for metric in ("mae", "rmse", "wape_pct"):
                    base_value = baseline.get(metric) if isinstance(baseline, Mapping) else None
                    model_value = row.get(metric)
                    if (
                        isinstance(base_value, (int, float))
                        and isinstance(model_value, (int, float))
                        and float(base_value) > 0
                    ):
                        compact[f"{metric}_improvement_pct"] = round(
                            (float(base_value) - float(model_value))
                            / float(base_value) * 100.0,
                            2,
                        )
            selected.append(compact)
        return selected

    def selected_bootstrap(payload: Any, models: tuple[str, ...]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            return {}
        comparisons = payload.get("comparisons_vs_baseline", {})
        if not isinstance(comparisons, Mapping):
            return {}
        compact_comparisons = {}
        for model_name in models:
            metrics = comparisons.get(model_name)
            if not isinstance(metrics, Mapping):
                continue
            metric_pack = {}
            for metric_name in ("mae", "rmse", "wape"):
                metric = metrics.get(metric_name)
                if isinstance(metric, Mapping):
                    metric_pack[metric_name] = _pick_fields(
                        metric,
                        ("mean_improvement_pct", "ci_95_low_pct",
                         "ci_95_high_pct", "ci_excludes_zero"),
                    )
            if metric_pack:
                compact_comparisons[model_name] = metric_pack
        return {
            "replicates": payload.get("replicates"),
            "comparisons": compact_comparisons,
        }

    bootstrap = final.get("bootstrap_vs_baseline", {})
    wavelet_ablation = final.get("wavelet_ablation_vs_spectral", {})
    weights = final.get("ensemble_weights", {})
    if not isinstance(weights, Mapping):
        weights = {}
    compact_weights = {
        str(name): round(float(value), 2)
        for name, value in list(weights.items())[:6]
        if isinstance(value, (int, float))
    }
    diversity = final.get("ensemble_diversity", {})
    compact_forecast = {
        "sample_series": forecast.get("series_sample_size_complete"),
        "rolling_folds": (
            len(rolling.get("folds", [])) if isinstance(rolling.get("folds"), list) else None
        ),
        "rolling_metrics": metric_rows(rolling.get("pooled_test_metrics", [])),
        "evaluation_period": final.get("evaluation_period"),
        "final_metrics": metric_rows(final.get("metrics", [])),
        "ensemble_weights": compact_weights,
        "ensemble_effective_models": (
            diversity.get("effective_model_count") if isinstance(diversity, Mapping) else None
        ),
        "bootstrap_vs_baseline": selected_bootstrap(
            bootstrap, ("fourier_stl", "spectral", "wavelet", "residual_corrected")
        ),
        "wavelet_ablation_vs_spectral": selected_bootstrap(
            wavelet_ablation, ("wavelet",)
        ),
        "experiment_methods": {
            "fourier_stl": "Calendar harmonics and weekly STL trend, seasonal and residual features.",
            "spectral": "Dominant historical periods, weekly energy share and spectral entropy.",
            "wavelet": "Causal db2 trailing-32-day coefficient features added to spectral features.",
            "residual_corrected": "LightGBM trained to correct baseline forecast residuals.",
            "ensemble": "Non-negative model weights selected on validation data and frozen for evaluation.",
        },
        "holdout_status": str(final.get("status", ""))[:150],
    }
    return {"inventory": compact_inventory, "forecast": compact_forecast}


def _stakeholder_system_prompt() -> str:
    return (
        "You are a retail analytics advisor. Use only supplied evidence. Define metrics in plain "
        "language; distinguish expected from achieved service, simulated costs from savings, and "
        "observed sales from latent demand. Lower MAE/RMSE/WAPE is better; bias near zero is better. "
        "Positive bootstrap improvement means lower error; a 95% interval crossing zero is inconclusive. "
        "Do not invent values or claim production readiness. Inventory uses FreshRetailNet-LT; "
        "forecasting uses a separate FreshRetailNet-50K sample. Write readable Markdown with headings, "
        "short paragraphs and compact tables, followed by ranked actions with owners and acceptance tests."
    )


def _stakeholder_prompt_parts(
    request: StakeholderBriefRequest,
    evidence: dict[str, Any],
    sources: list[str],
    missing: list[str],
) -> list[dict[str, str]]:
    import json

    inventory_payload = {
        "part": "1/3 Inventory optimisation",
        "dataset": "FreshRetailNet-LT",
        "audience": request.audience,
        "focus": request.focus,
        "source_available": any("inventory_optimizer_summary" in item for item in sources),
        "source_missing": any("inventory_optimizer_summary" in item for item in missing),
        "evidence": evidence.get("inventory", {}),
        "instructions": (
            "Write about 450-500 words. Explain the 70-day population, 5 calibration and 5 later "
            "evaluation windows. Define target fill rate, theoretical expected fill, empirical fill, "
            "cycle service level, percentage-point gap, replenishment/storage utilization percentiles, "
            "capacity-limited stores, segment shortages, median MAE, pinball loss, central-80% coverage "
            "and interval width. Interpret each provided result and the lognormal/bootstrap/conformal "
            "comparison. Explain simulated holding/shortage/ordering cost as normalized assumptions, "
            "not real savings. Finish with limitations and 2 practical next steps with owners/tests."
        ),
    }

    forecast = evidence.get("forecast", {})
    forecast_metrics = {
        "dataset": "FreshRetailNet-50K",
        "sample_series": forecast.get("sample_series"),
        "rolling_folds": forecast.get("rolling_folds"),
        "rolling_metrics": forecast.get("rolling_metrics", []),
        "evaluation_period": forecast.get("evaluation_period"),
        "final_metrics": forecast.get("final_metrics", []),
        "experiment_methods": forecast.get("experiment_methods", {}),
    }
    forecast_metrics_payload = {
        "part": "2/3 Forecast accuracy and model comparisons",
        "audience": request.audience,
        "focus": request.focus,
        "evidence": forecast_metrics,
        "instructions": (
            "Write about 450-500 words. Define MAE (average absolute prediction error), RMSE "
            "(square-root average squared error, penalizing large misses), WAPE (total absolute "
            "error divided by total actual volume) and signed bias (over/under-forecast tendency). "
            "Compare every supplied candidate across pooled rolling and final-period results. Explain "
            "relative improvements where supplied; lower MAE/RMSE/WAPE is better and bias near zero "
            "is preferable. Explain Fourier/STL, spectral, wavelet, residual correction and ensemble "
            "methods. Do not infer business savings from forecast accuracy."
        ),
    }

    forecast_experiments = {
        "dataset": "FreshRetailNet-50K",
        "evaluation_period": forecast.get("evaluation_period"),
        "ensemble_weights": forecast.get("ensemble_weights", {}),
        "ensemble_effective_models": forecast.get("ensemble_effective_models"),
        "bootstrap_vs_baseline": forecast.get("bootstrap_vs_baseline", {}),
        "wavelet_ablation_vs_spectral": forecast.get("wavelet_ablation_vs_spectral", {}),
        "holdout_status": forecast.get("holdout_status", ""),
    }
    forecast_experiments_payload = {
        "part": "3/3 Statistical evidence and experiment decisions",
        "audience": request.audience,
        "focus": request.focus,
        "evidence": forecast_experiments,
        "instructions": (
            "Write about 400-450 words. Explain paired bootstrap mean improvements and 95% intervals; "
            "say an interval crossing zero is inconclusive. Interpret the direct wavelet-vs-spectral "
            "ablation. Explain whether ensemble weights show a genuine blend or model concentration. "
            "State that the original seven-day evaluation was previously used, not untouched, and the "
            "inventory and forecasting datasets are separate. Rank 2-3 next experiments with owners "
            "and numeric acceptance tests. Do not promote a challenger before a later untouched holdout."
        ),
    }
    return [
        {
            "title": "Inventory optimisation and service level",
            "user_message": json.dumps(inventory_payload, ensure_ascii=False, default=str, separators=(",", ":")),
        },
        {
            "title": "Forecast accuracy and model comparisons",
            "user_message": json.dumps(forecast_metrics_payload, ensure_ascii=False, default=str, separators=(",", ":")),
        },
        {
            "title": "Statistical evidence and experiment decisions",
            "user_message": json.dumps(forecast_experiments_payload, ensure_ascii=False, default=str, separators=(",", ":")),
        },
    ]


def _groq_http_exception(exc: Exception, model: str) -> HTTPException:
    """Translate Groq SDK failures into safe, actionable API responses."""
    upstream_status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    message = getattr(exc, "message", None) or str(exc)
    if isinstance(body, Mapping):
        error_payload = body.get("error", body)
        if isinstance(error_payload, Mapping):
            message = str(error_payload.get("message") or message)
    message = str(message).replace(os.getenv("GROQ_API_KEY", "\\0"), "[redacted]")
    message = message[:700]

    if upstream_status == 429:
        action = "Groq rate limit reached. Wait for the retry window or check the Groq project limits."
        status_code = 429
    elif upstream_status == 401:
        action = "Groq rejected the API key. Replace GROQ_API_KEY with an active key from the intended Groq project."
        status_code = 503
    elif upstream_status == 403:
        action = "The Groq key or project lacks access to this model. Check model permissions in the Groq console."
        status_code = 503
    elif upstream_status == 404:
        action = f"Groq could not find model '{model}'. Set GROQ_MODEL to an active model ID."
        status_code = 503
    elif upstream_status is not None and int(upstream_status) >= 500:
        action = "Groq returned an upstream server error. Retry later and check Groq service status."
        status_code = 503
    elif upstream_status is not None:
        action = "Groq rejected the request. Review the upstream message and model parameter support."
        status_code = 502
    elif type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}:
        action = "The API service could not reach Groq or the request timed out. Check outbound connectivity and retry."
        status_code = 503
    else:
        action = "Unexpected stakeholder briefing failure. Check the API service logs using the exception type."
        status_code = 500

    detail = {
        "code": "groq_upstream_error",
        "upstream_status": upstream_status,
        "exception_type": type(exc).__name__,
        "model": model,
        "message": message,
        "action": action,
    }
    return HTTPException(status_code=status_code, detail=detail)


def _stakeholder_system_prompt() -> str:
    return (
        "You are a retail supply-chain analytics advisor. Use only supplied evidence. "
        "Separate empirical results, model expectations and assumptions. Never call "
        "simulated cost savings, expected fill achieved service, or observed sales latent "
        "demand. Do not invent facts. State key limitations. Recommend reversible next "
        "decisions with an owner and measurable acceptance test. This is decision support; "
        "do not execute orders."
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
    """Generate a detailed three-part stakeholder explanation using bounded Groq calls."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Groq is not configured. Add GROQ_API_KEY to the API service environment.",
        )
    raw_evidence, sources, missing = _stakeholder_evidence()
    if not sources:
        raise HTTPException(
            status_code=503,
            detail="No inventory or forecasting research reports are available in R2.",
        )
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    try:
        from groq import Groq

        evidence = _compact_stakeholder_evidence(raw_evidence)
        parts = _stakeholder_prompt_parts(request, evidence, sources, missing)
        system_message = _stakeholder_system_prompt()
        max_prompt_characters = 5_000
        max_completion_tokens = 850
        prompt_sizes = [
            len(system_message) + len(part["user_message"])
            for part in parts
        ]
        oversized = [
            size for size in prompt_sizes if size > max_prompt_characters
        ]
        estimated_reserved_tokens = sum(
            int(math.ceil(size / 2.5)) + max_completion_tokens
            for size in prompt_sizes
        )
        max_estimated_reserved_tokens = 6_800
        if oversized or estimated_reserved_tokens > max_estimated_reserved_tokens:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "stakeholder_prompt_too_large",
                    "message": (
                        "The three-part stakeholder briefing exceeds the safe limit of "
                        f"{max_prompt_characters:,} characters per section or the estimated "
                        f"Groq token budget. Part sizes: {prompt_sizes}; estimated reserved "
                        f"tokens: {estimated_reserved_tokens:,}."
                    ),
                    "prompt_characters": max(prompt_sizes, default=0),
                    "maximum_prompt_characters": max_prompt_characters,
                    "prompt_characters_by_part": prompt_sizes,
                    "maximum_prompt_characters_per_part": max_prompt_characters,
                    "estimated_reserved_tokens": estimated_reserved_tokens,
                    "maximum_estimated_reserved_tokens": max_estimated_reserved_tokens,
                    "action": (
                        "The report is split into three bounded calls. Reduce the requested focus or "
                        "inspect the compact evidence fields before retrying; no Groq call was made."
                    ),
                },
            )

        client = Groq(api_key=api_key, timeout=35.0, max_retries=0)
        generated_sections = []
        usage_by_part = []
        for index, part in enumerate(parts, start=1):
            request_options = {}
            if model.startswith("openai/gpt-oss-"):
                request_options["reasoning_effort"] = "low"
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": part["user_message"]},
                ],
                temperature=0.2,
                max_completion_tokens=max_completion_tokens,
                **request_options,
            )
            content = completion.choices[0].message.content or ""
            if not content.strip():
                raise RuntimeError(f"Groq returned an empty section for part {index}.")
            generated_sections.append(content.strip())
            usage = getattr(completion, "usage", None)
            usage_by_part.append(
                usage.model_dump() if usage is not None else None
            )

        brief = (
            "# FreshRetailNet Demand Intelligence — Stakeholder Report\n\n"
            + "\n\n---\n\n".join(
                f"## Part {index}. {part['title']}\n\n{section}"
                for index, (part, section) in enumerate(zip(parts, generated_sections), start=1)
            )
        )
        numeric_usage = [
            item for item in usage_by_part if isinstance(item, Mapping)
        ]
        usage_totals = {}
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            values = [
                item.get(field) for item in numeric_usage
                if isinstance(item.get(field), (int, float))
            ]
            if values:
                usage_totals[field] = sum(values)
        usage_summary = {
            "parts": usage_by_part,
            "totals": usage_totals,
            "estimated_reserved_tokens": estimated_reserved_tokens,
        }
        return {
            "brief_markdown": brief,
            "model": model,
            "generation_strategy": "three_part_groq",
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_artifacts": sources,
            "missing_source_artifacts": missing,
            "usage": usage_summary,
            "disclaimer": (
                "AI-generated explanation of published research. Verify source metrics "
                "and approve decisions with the responsible operational owner."
            ),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "Three-part stakeholder brief generation failed: exception_type=%s model=%s",
            type(exc).__name__,
            model,
        )
        raise _groq_http_exception(exc, model) from exc
