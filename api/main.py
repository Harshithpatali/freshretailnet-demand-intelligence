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
    """Build separate inventory and forecasting evidence packs for two Groq calls."""
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
    scenarios = _compact_rows(
        [balanced] if isinstance(balanced, Mapping) else [],
        (
            "scenario", "target_fill_rate", "empirical_fill_rate",
            "theoretical_expected_fill_rate", "empirical_cycle_service_level",
            "actual_protection_demand", "fulfilled_protection_demand",
            "unfulfilled_protection_demand", "leftover_stock",
            "holding_cost", "shortage_cost", "ordering_cost",
            "total_simulated_cost", "theoretical_expected_cost_per_protection_period",
            "capacity_limited_store_count", "service_target_met_store_count",
            "service_target_met_empirically", "store_count", "store_product_series",
            "evaluation_windows_per_series", "evaluation_windows",
            "mean_storage_utilization_pct", "mean_replenishment_utilization_pct",
        ),
        limit=1,
    )

    capacity_source = inventory.get("capacity_diagnostics", [])
    capacity_source = [
        row for row in capacity_source
        if isinstance(row, Mapping) and row.get("scenario") == "balanced_90pct"
    ] if isinstance(capacity_source, list) else []
    capacity = _compact_capacity(capacity_source[:1])

    segments_source = inventory.get("segment_results", [])
    segments_source = [
        row for row in segments_source
        if isinstance(row, Mapping) and row.get("scenario") == "balanced_90pct"
    ] if isinstance(segments_source, list) else []
    segments = _compact_rows(
        segments_source,
        (
            "segment_type", "segment", "stores", "series", "evaluation_windows",
            "actual_demand", "fulfilled_demand", "empirical_fill_rate",
            "shortage_units", "leftover_units", "simulated_cost",
        ),
        limit=6,
    )

    dist = inventory.get("distribution_diagnostics", {})
    dist_rows = dist.get("results", []) if isinstance(dist, Mapping) else []
    dist_rows = [
        row for row in dist_rows
        if isinstance(row, Mapping) and row.get("segment_type") == "overall"
    ] if isinstance(dist_rows, list) else []
    distributions = _compact_rows(
        dist_rows,
        (
            "distribution", "evaluation_windows", "median_mae",
            "pinball_q10", "pinball_q50", "pinball_q90",
            "central_80_coverage_pct", "central_80_mean_width", "median_bias_pct",
        ),
        limit=3,
    )

    population = inventory.get("evaluation_population", {})
    evaluation_design = inventory.get("evaluation_design", {})
    compact_inventory = {
        "source_population": "FreshRetailNet-LT (external population; separate from FreshRetailNet-50K)",
        "population": _pick_fields(
            population,
            (
                "target_date_start", "target_date_end", "target_dates",
                "complete_continuous_series", "total_series_with_any_prediction",
            ),
        ),
        "calibration_windows_per_series": (
            inventory.get("calibration_design", {}).get("windows_per_series")
            if isinstance(inventory.get("calibration_design"), Mapping) else None
        ),
        "evaluation_windows_per_series": (
            evaluation_design.get("windows_per_series")
            if isinstance(evaluation_design, Mapping) else None
        ),
        "target_definition": "Store-wide volume-weighted expected fill rate over a seven-day protection period",
        "scenario": scenarios[0] if scenarios else {},
        "capacity": capacity,
        "segments": segments,
        "distribution_reference": (
            dist.get("reference_optimizer_distribution")
            if isinstance(dist, Mapping) else None
        ),
        "distribution_challengers": (
            dist.get("challengers") if isinstance(dist, Mapping) else None
        ),
        "distribution_scored_on": (
            dist.get("scored_on") if isinstance(dist, Mapping) else None
        ),
        "distribution_scores": distributions,
        "distribution_limitations": (
            str(dist.get("limitations", ""))[:260] if isinstance(dist, Mapping) else ""
        ),
        "optimizer_objective": (
            "Minimize expected holding cost + expected shortage cost + fixed ordering cost, "
            "subject to service, storage, replenishment, MOQ, pack-size and SKU order-cap constraints."
        ),
        "cost_and_pack_assumptions": (
            "Illustrative normalized assumptions, not retailer-measured economics or supplier pack data."
        ),
        "limitations": [
            str(item)[:220] for item in inventory.get("limitations", [])[:6]
        ] if isinstance(inventory.get("limitations"), list) else [],
    }

    forecast = raw.get("forecast_research") or {}
    if not isinstance(forecast, Mapping):
        forecast = {}
    rolling = forecast.get("rolling_design", {})
    final = forecast.get("final_research_comparison", {})
    methods = forecast.get("methods", {})
    if not isinstance(rolling, Mapping):
        rolling = {}
    if not isinstance(final, Mapping):
        final = {}
    if not isinstance(methods, Mapping):
        methods = {}

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
        selected = [
            _pick_fields(
                by_name[name],
                ("model", "mae", "rmse", "wape_pct", "bias_pct"),
            )
            for name in priority if name in by_name
        ]
        baseline = next((row for row in selected if row.get("model") == "baseline"), None)
        if isinstance(baseline, Mapping):
            for row in selected:
                if row.get("model") == "baseline":
                    continue
                improvements = {}
                for metric in ("mae", "rmse", "wape_pct"):
                    base_value = baseline.get(metric)
                    model_value = row.get(metric)
                    if (
                        isinstance(base_value, (int, float))
                        and isinstance(model_value, (int, float))
                        and float(base_value) > 0
                    ):
                        improvements[metric] = round(
                            (float(base_value) - float(model_value))
                            / float(base_value) * 100.0,
                            3,
                        )
                if improvements:
                    row["improvement_vs_baseline_pct"] = improvements
        return selected

    weights = final.get("ensemble_weights", {})
    if not isinstance(weights, Mapping):
        weights = {}
    weights = {
        str(name): round(float(value), 3)
        for name, value in list(weights.items())[:6]
        if isinstance(value, (int, float))
    }

    diversity = final.get("ensemble_diversity", {})
    compact_forecast = {
        "source_population": "FreshRetailNet-50K",
        "sample_series": forecast.get("series_sample_size_complete"),
        "sample_policy": str(forecast.get("sample_policy", ""))[:220],
        "seed": forecast.get("seed"),
        "experiment": str(forecast.get("experiment", ""))[:240],
        "rolling_design": {
            "note": str(rolling.get("note", ""))[:300],
            "warmup_days": rolling.get("warmup_days"),
            "fold_count": len(rolling.get("folds", [])) if isinstance(rolling.get("folds"), list) else None,
            "pooled_test_metrics": metric_rows(rolling.get("pooled_test_metrics", [])),
        },
        "validation_period": final.get("validation_period"),
        "evaluation_period": final.get("evaluation_period"),
        "final_metrics": metric_rows(final.get("metrics", [])),
        "ensemble_weights": weights,
        "ensemble_weight_cap": final.get("ensemble_weight_cap"),
        "ensemble_diversity": _pick_fields(
            diversity,
            ("effective_model_count", "contributing_models", "max_weight"),
        ),
        "bootstrap_vs_baseline": _compact_bootstrap(
            final.get("bootstrap_vs_baseline", {})
        ),
        "wavelet_ablation_vs_spectral": _compact_bootstrap(
            final.get("wavelet_ablation_vs_spectral", {})
        ),
        "methods": _pick_fields(
            methods,
            ("fourier", "stl", "spectral", "wavelet", "residual_model", "ensemble"),
        ),
        "limitations": [
            str(item)[:240] for item in forecast.get("limitations", [])[:7]
        ] if isinstance(forecast.get("limitations"), list) else [],
        "promotion_status": str(final.get("status", ""))[:180],
    }
    return {"inventory": compact_inventory, "forecast": compact_forecast}


def _stakeholder_system_prompt() -> str:
    return (
        "You are a senior retail supply-chain and forecasting analytics advisor. "
        "Use only supplied evidence; do not invent missing values. Explain every reported "
        "metric in plain language and state how it is calculated or interpreted. Distinguish "
        "targets, theoretical expectations, empirical outcomes, estimates and assumptions. "
        "Lower forecast error is better; signed bias near zero is preferable. Positive bootstrap "
        "improvement means lower error than the named baseline, and an interval crossing zero "
        "is inconclusive. Never claim simulated costs are realized savings, expected fill is "
        "achieved service, observed sales are latent demand, or a research challenger is "
        "production-ready. Keep FreshRetailNet-LT inventory findings separate from "
        "FreshRetailNet-50K forecasting findings. Write a detailed, readable Markdown report "
        "for the requested audience, with explanatory headings, short paragraphs and compact "
        "tables. End with actionable next steps and measurable acceptance tests."
    )


def _stakeholder_prompt_parts(
    request: StakeholderBriefRequest,
    evidence: dict[str, Any],
    sources: list[str],
    missing: list[str],
) -> list[dict[str, str]]:
    import json

    inventory_payload = {
        "part": "1 of 2: Inventory optimisation and service-level evidence",
        "audience": request.audience,
        "decision_focus": request.focus,
        "source_artifacts": [
            item for item in sources if "inventory_optimizer_summary" in item
        ],
        "missing_source_artifacts": [
            item for item in missing if "inventory_optimizer_summary" in item
        ],
        "evidence": evidence.get("inventory", {}),
        "required_coverage": [
            "Open with a concise inventory-specific finding, not a generic executive paragraph.",
            "Explain the evaluation population, horizon, calibration and later evaluation windows.",
            "Define target fill rate, theoretical expected fill, empirical fill, empirical cycle service level, "
            "the percentage-point service gap, replenishment and storage utilisation percentiles, "
            "capacity-limited stores, segment shortage units, and every distribution score present.",
            "Compare theoretical expected fill with empirical later-window fill. Explain why the gap matters.",
            "Explain the balanced scenario and the capacity bottleneck, then interpret each store-size and "
            "SKU-demand segment present in the evidence.",
            "Explain the reference demand distribution and challenger distribution experiments, including "
            "median MAE, pinball loss, central-80% coverage and interval width where supplied.",
            "Explain simulated cost components only as illustrative normalized assumptions, never as realized savings.",
            "End with limitations and 2-3 reversible operational/research next steps with owners and numeric acceptance tests.",
        ],
        "length_and_style": (
            "Aim for 500-650 words. Use a short metric glossary table with columns Metric, Meaning, "
            "How to read this result. Do not make a dense one-line summary or omit definitions."
        ),
    }
    forecast_payload = {
        "part": "2 of 2: Forecast research, model comparison and combined decisions",
        "audience": request.audience,
        "decision_focus": request.focus,
        "source_artifacts": [
            item for item in sources if "forecast_research_summary" in item
        ],
        "missing_source_artifacts": [
            item for item in missing if "forecast_research_summary" in item
        ],
        "evidence": evidence.get("forecast", {}),
        "required_coverage": [
            "Explain the deterministic sample, chronological validation approach, three rolling seven-day folds, "
            "original evaluation dates, and why the previously used seven-day period is not an untouched holdout.",
            "Define MAE, RMSE, WAPE and signed bias, with clear directionality and formulas in plain language.",
            "Compare every available model in both rolling results and final-period results; explicitly quantify "
            "relative error reductions from improvement_vs_baseline_pct without implying business savings.",
            "Explain Fourier/STL, spectral, wavelet, residual correction and validation-selected ensemble experiments.",
            "Discuss paired bootstrap confidence intervals against baseline and the direct wavelet-versus-spectral "
            "ablation. State which intervals exclude zero and which experiments are inconclusive.",
            "Interpret ensemble weights and effective model count. Do not call it a diverse blend if only one model "
            "has material weight or if effective model count indicates concentration.",
            "Explain what is genuinely better, what is only promising, what failed to show added value, and why "
            "the production baseline must remain until a later untouched temporal evaluation.",
            "Close with a short integrated decision list, owners, measurable acceptance tests and limitations. "
            "Explicitly state that forecast and inventory populations are separate and must not be pooled.",
        ],
        "length_and_style": (
            "Aim for 550-700 words. Include a compact model-comparison table, a separate experiment/uncertainty "
            "interpretation, and ranked recommendations. Avoid repeating the entire inventory glossary."
        ),
    }
    return [
        {
            "title": "Inventory optimisation and service-level explanation",
            "user_message": json.dumps(
                inventory_payload, ensure_ascii=False, default=str, separators=(",", ":")
            ),
        },
        {
            "title": "Forecast experiments, model comparisons and decisions",
            "user_message": json.dumps(
                forecast_payload, ensure_ascii=False, default=str, separators=(",", ":")
            ),
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
    """Generate a detailed two-part stakeholder explanation using two bounded Groq calls."""
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
        max_completion_tokens = 1_400
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
        max_estimated_reserved_tokens = 7_000
        if oversized or estimated_reserved_tokens > max_estimated_reserved_tokens:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "stakeholder_prompt_too_large",
                    "message": (
                        "The two-part stakeholder briefing exceeds the safe limit of "
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
                        "The report is split into two bounded calls. Reduce the requested focus or "
                        "inspect the compact evidence fields before retrying; no Groq call was made."
                    ),
                },
            )

        client = Groq(api_key=api_key, timeout=60.0, max_retries=0)
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
            "## Part 1. Inventory optimisation and service-level evidence\n\n"
            + generated_sections[0]
            + "\n\n---\n\n"
            + "## Part 2. Forecast research, model comparisons and decisions\n\n"
            + generated_sections[1]
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
            "generation_strategy": "two_part_groq",
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
            "Two-part stakeholder brief generation failed: exception_type=%s model=%s",
            type(exc).__name__,
            model,
        )
        raise _groq_http_exception(exc, model) from exc
