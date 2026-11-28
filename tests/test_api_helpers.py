"""Offline tests for API artifact handling and forecast simulation."""
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import main as api


def test_safe_json_returns_artifact(monkeypatch):
    monkeypatch.setattr(api, "_read_json", lambda key: {"artifact": key})
    assert api._safe_json("metrics.json") == {"artifact": "metrics.json"}


def test_safe_json_maps_missing_artifact_to_503(monkeypatch):
    monkeypatch.setattr(api, "_read_json", lambda _key: (_ for _ in ()).throw(FileNotFoundError()))
    with pytest.raises(HTTPException) as exc:
        api._safe_json("missing.json")
    assert exc.value.status_code == 503
    assert "missing.json" in exc.value.detail


def test_safe_csv_maps_storage_failure_to_503(monkeypatch):
    monkeypatch.setattr(api, "_read_csv", lambda _key: (_ for _ in ()).throw(RuntimeError("R2 down")))
    with pytest.raises(HTTPException) as exc:
        api._safe_csv("forecasts.csv")
    assert exc.value.status_code == 503
    assert "forecasts.csv" in exc.value.detail


def test_schema_mismatch_is_reported_as_503():
    with pytest.raises(HTTPException) as exc:
        api._require_columns(pd.DataFrame({"store_id": ["s1"]}), "forecasts.csv",
                             {"store_id", "product_id", "date"})
    assert exc.value.status_code == 503
    assert "product_id" in exc.value.detail


def test_health_reports_unconfigured_r2(monkeypatch):
    for name in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
        monkeypatch.delenv(name, raising=False)
    health = api.health()
    assert health["status"] == "ok"
    assert health["r2_configured"] is False
    assert health["artifact_cache_ttl_seconds"] >= 1
    assert health["groq_configured"] is False


def test_forecast_simulation_handles_multiple_series():
    from src.simulate_inventory import simulate
    dates = pd.date_range("2025-01-01", periods=4)
    frame = pd.DataFrame({
        "date": list(dates) * 2, "store_id": ["s1"] * 4 + ["s2"] * 4,
        "product_id": ["p1"] * 4 + ["p2"] * 4,
        "actual_sale_amount": [2, 3, 1, 4, 4, 2, 5, 3],
        "forecast": [2, 2, 2, 3, 3, 3, 4, 4],
    })
    result = simulate(frame, "forecast", "test", 1, 1, .05, 2, 1, 2)
    assert len(result) == 2
    assert set(result.store_id) == {"s1", "s2"}
    assert result["fill_rate_pct"].between(0, 100).all()
    assert result["total_simulated_cost"].map(pd.notna).all()


def test_forecast_simulation_rejects_missing_prediction_column():
    from src.simulate_inventory import simulate
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-01"]), "store_id": ["s1"],
        "product_id": ["p1"], "actual_sale_amount": [1.0],
    })
    with pytest.raises(KeyError):
        simulate(frame, "missing_forecast", "bad", 1, 1, .05, 2, 1, 0)

def test_small_demo_writes_reproducible_synthetic_artifacts(tmp_path):
    from scripts.run_small_demo import run
    result = run(tmp_path)
    assert result["demo_type"] == "synthetic_only_not_benchmark_evidence"
    assert result["input_rows"] == 42
    assert result["store_product_series"] == 2
    assert (tmp_path / "synthetic_forecasts.csv").exists()
    assert (tmp_path / "inventory_policy_results.csv").exists()
    assert (tmp_path / "demo_summary.json").exists()



def test_overview_degrades_gracefully_when_one_artifact_is_missing(monkeypatch):
    missing_key = "quantile-model-artifacts/quantile_metrics.json"

    def read_artifact(key):
        if key == missing_key:
            raise HTTPException(status_code=503, detail="missing")
        return {"source": key}

    monkeypatch.setattr(api, "_safe_json", read_artifact)
    result = api.overview()
    assert result["point_forecast"]["source"] == "model-artifacts/metrics.json"
    assert result["quantile_forecast"] == {}
    assert result["artifact_status"]["point_forecast"]["available"] is True
    assert result["artifact_status"]["quantile_forecast"]["available"] is False


def test_filters_returns_compact_metadata_without_series_payload(monkeypatch):
    frame = pd.DataFrame({
        "store_id": ["s1", "s1", "s2"],
        "product_id": ["p1", "p2", "p1"],
        "date": ["2024-01-01", "2024-01-02", "2024-01-01"],
    })
    monkeypatch.setattr(api, "_safe_csv", lambda _key: frame)
    result = api.filters()
    assert result["series_count"] == 3
    assert result["store_count"] == 2
    assert result["product_count"] == 2
    assert "series" not in result


def test_series_search_filters_and_paginates_on_server(monkeypatch):
    frame = pd.DataFrame({
        "store_id": ["s1", "s1", "s2", "s3", "s3"],
        "product_id": ["p1", "p2", "p1", "p1", "p3"],
    })
    monkeypatch.setattr(api, "_safe_csv", lambda _key: frame)
    result = api.search_series(q="p1", offset=1, limit=1)
    assert result["total"] == 3
    assert result["rows"] == [{"store_id": "s2", "product_id": "p1"}]
    assert result["offset"] == 1
    assert result["limit"] == 1


def test_json_safe_replaces_nan_and_infinity_with_null():
    result = api._json_safe({
        "nan": float("nan"),
        "positive_inf": float("inf"),
        "nested": [{"negative_inf": float("-inf"), "value": 3.0}],
    })
    assert result == {
        "nan": None,
        "positive_inf": None,
        "nested": [{"negative_inf": None, "value": 3.0}],
    }


def test_series_options_returns_store_dropdown_and_store_scoped_products(monkeypatch):
    frame = pd.DataFrame({
        "store_id": ["10", "2", "1", "2", "10", "1"],
        "product_id": ["p3", "p2", "p1", "p4", "p5", "p2"],
    })
    monkeypatch.setattr(api, "_safe_csv", lambda _key: frame)

    result = api.series_options(store_id="2")

    assert result["store_ids"] == ["1", "2", "10"]
    assert result["product_ids"] == ["p2", "p4"]
    assert result["selected_store_id"] == "2"


def test_dataset_eda_endpoint_returns_published_report(monkeypatch):
    expected = {"dataset": "FreshRetailNet-50K", "feature_count": 19}
    monkeypatch.setattr(api, "_safe_json", lambda key: expected if key == "dataset-eda/full_dataset_eda.json" else {})
    assert api.dataset_eda() == expected


def test_inventory_optimizer_summary_api_reads_published_report(monkeypatch):
    expected = {"scenario_results": [{"scenario": "balanced_90pct"}]}
    monkeypatch.setattr(api, "_safe_json", lambda key: expected if key == "inventory-optimizer-artifacts/inventory_optimizer_summary.json" else {})
    assert api.inventory_optimizer_summary() == expected


def test_inventory_optimizer_store_endpoint_filters_scenario(monkeypatch):
    frame = pd.DataFrame({
        "store_id": ["s1", "s2", "s1"],
        "scenario": ["balanced_90pct", "balanced_90pct", "tight_95pct"],
        "target_fill_rate": [0.9, 0.9, 0.95],
        "achieved_fill_rate": [0.91, 0.88, 0.94],
        "service_target_met": [True, False, False],
    })
    monkeypatch.setattr(api, "_safe_csv", lambda _key: frame)
    result = api.inventory_optimizer_stores(scenario="balanced_90pct", limit=10)
    assert result["total"] == 2
    assert {row["store_id"] for row in result["rows"]} == {"s1", "s2"}


def test_forecast_research_summary_api_reads_published_report(monkeypatch):
    expected = {"final_research_comparison": {"ensemble_weights": {"baseline": 1.0}}}
    monkeypatch.setattr(api, "_safe_json", lambda key: expected if key == "forecast-research-artifacts/forecast_research_summary.json" else {})
    assert api.forecast_research_summary() == expected



def test_stakeholder_brief_requires_server_side_groq_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    request = api.StakeholderBriefRequest(
        audience="Executive leadership",
        focus="Explain the service-level gap and next decisions.",
    )
    with pytest.raises(HTTPException) as exc:
        api.generate_stakeholder_brief(request)
    assert exc.value.status_code == 503
    assert "GROQ_API_KEY" in exc.value.detail


def test_stakeholder_evidence_uses_only_aggregate_json_artifacts(monkeypatch):
    artifacts = {
        "inventory-optimizer-artifacts/inventory_optimizer_summary.json": {
            "scenario_results": [{"scenario": "balanced_90pct", "empirical_fill_rate": 0.8}]
        },
        "forecast-research-artifacts/forecast_research_summary.json": {
            "final_research_comparison": {"ensemble_weight_cap": 0.65}
        },
    }
    monkeypatch.setattr(api, "_read_json", lambda key: artifacts[key])
    evidence, sources, missing = api._stakeholder_evidence()
    assert evidence["inventory_optimizer"]["scenario_results"][0]["empirical_fill_rate"] == 0.8
    assert len(sources) == 2
    assert missing == []



def test_nested_dashboard_payloads_are_flattened_without_json_dumps():
    dashboard_path = ROOT / "dashboard"
    if str(dashboard_path) not in sys.path:
        sys.path.insert(0, str(dashboard_path))
    from presentation import flatten_structured_data

    payload = {
        "nominal_coverage_pct": 80.0,
        "validation": {"coverage_pct": 79.5, "interval_width": 1.25},
        "group_adjustments": [{"bucket": "low exposure", "adjustment": 0.2}],
        "limitations": ["short validation window", "observed-sales proxy"],
    }
    frame = flatten_structured_data(payload)
    values = dict(zip(frame["Field"], frame["Value"]))

    assert values["nominal_coverage_pct"] == "80"
    assert values["validation / coverage_pct"] == "79.5"
    assert values["group_adjustments 1 / bucket"] == "low exposure"
    assert values["limitations"] == "short validation window · observed-sales proxy"
    assert not any(value.startswith("{") or value.startswith("[") for value in frame["Value"])


def test_dashboard_source_does_not_render_raw_json_widgets():
    dashboard_files = [
        ROOT / "dashboard" / "app.py",
        ROOT / "dashboard" / "wavelet_tab.py",
    ]
    for path in dashboard_files:
        source = path.read_text(encoding="utf-8")
        assert "st.json(" not in source
        assert "st.write({" not in source
