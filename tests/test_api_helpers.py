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



def test_stakeholder_evidence_is_compacted_before_sending_to_groq(monkeypatch):
    raw_inventory = {
        "generated_at_utc": "2026-11-28T00:00:00Z",
        "evaluation_population": {
            "target_date_start": "2024-01-01",
            "target_date_end": "2024-03-10",
            "target_dates": 70,
            "complete_continuous_series": 2600,
            "total_series_with_any_prediction": 3000,
        },
        "scenario_results": [{
            "scenario": "balanced_90pct",
            "target_fill_rate": 0.9,
            "empirical_fill_rate": 0.8,
            "feature_gain_dump": ["must not be forwarded"],
        }],
    }
    raw_forecast = {
        "series_sample_size_complete": 3000,
        "rolling_design": {
            "pooled_test_metrics": [{"model": "baseline", "mae": 0.4}],
            "folds": [{
                "fold": "fold-1",
                "test_dates": ["2024-01-01", "2024-01-07"],
                "test_metrics": [{"model": "residual", "mae": 0.38}],
                "feature_gain": [{"feature": f"f{i}", "gain": i} for i in range(500)],
            }],
        },
        "final_research_comparison": {
            "ensemble_weights": {"residual": 1.0},
            "bootstrap_vs_baseline": {"comparisons_vs_baseline": {"mae": {"mean": 5.0}}},
        },
    }
    monkeypatch.setattr(
        api, "_stakeholder_evidence",
        lambda: (
            {"inventory_optimizer": raw_inventory, "forecast_research": raw_forecast},
            ["inventory.json", "forecast.json"],
            [],
        ),
    )
    raw, sources, missing = api._stakeholder_evidence()
    compact = api._compact_stakeholder_evidence(raw)

    assert compact["inventory"]["scenario"]["empirical_fill_rate"] == 0.8
    assert "feature_gain_dump" not in compact["inventory"]["scenario"]
    assert compact["forecast"]["rolling_folds"] == 1
    assert "feature_gain" not in str(compact)
    assert compact["forecast"]["rolling_metrics"][0]["model"] == "baseline"
    assert compact["forecast"]["sample_series"] == 3000
    assert sources == ["inventory.json", "forecast.json"]
    assert missing == []


def test_groq_rate_limit_is_returned_as_actionable_429():
    from types import SimpleNamespace

    error = SimpleNamespace(
        status_code=429,
        body={"error": {"message": "rate limit exceeded"}},
        message="upstream rate limit",
    )
    response = api._groq_http_exception(error, "openai/gpt-oss-120b")
    assert response.status_code == 429
    assert response.detail["upstream_status"] == 429
    assert "rate limit" in response.detail["action"].lower()


def test_groq_authentication_error_is_reported_as_configuration_issue():
    from types import SimpleNamespace

    error = SimpleNamespace(
        status_code=401,
        body={"error": {"message": "invalid api key"}},
        message="authentication failed",
    )
    response = api._groq_http_exception(error, "openai/gpt-oss-120b")
    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.detail["action"]



def test_compacted_stakeholder_evidence_omits_fold_level_payloads_and_limits_segments():
    raw = {
        "inventory_optimizer": {
            "evaluation_population": {"target_dates": 70, "complete_continuous_series": 2600},
            "scenario_results": [
                {"scenario": f"scenario-{i}", "empirical_fill_rate": 0.8, "items": ["x"] * 1000}
                for i in range(3)
            ],
            "capacity_diagnostics": [{
                "scenario": "balanced_90pct",
                "replenishment_utilization_pct": {
                    "p50": 90, "p90": 99, "p95": 100, "max": 100,
                    "share_at_least_95_pct": 60, "share_at_least_99_pct": 30,
                    "raw_store_rows": ["x"] * 2000,
                },
                "storage_utilization_pct": {"p50": 40, "p90": 90, "p95": 99, "max": 100},
                "by_store_size": [
                    {"store_size_segment": s, "stores": 10, "debug": ["x"] * 1000}
                    for s in ("small", "medium", "large")
                ],
            }],
            "segment_results": [
                {"scenario": "balanced_90pct", "segment_type": typ, "segment": seg,
                 "empirical_fill_rate": 0.8, "debug": ["x"] * 1000}
                for typ, seg in [
                    ("store_size", "small"), ("store_size", "medium"), ("store_size", "large"),
                    ("sku_demand", "low"), ("sku_demand", "medium"), ("sku_demand", "high"),
                ]
            ],
            "distribution_diagnostics": {
                "reference_optimizer_distribution": "moment_lognormal",
                "challengers": ["empirical_bootstrap", "conformal_residual"],
                "results": [
                    {"segment_type": "overall", "distribution": name, "median_mae": .2, "debug": ["x"] * 500}
                    for name in ("moment_lognormal", "empirical_bootstrap", "conformal_residual")
                ] + [{"segment_type": "demand_segment", "debug": ["x"] * 500} for _ in range(20)],
            },
            "limitations": ["one", "two"],
        },
        "forecast_research": {
            "series_sample_size_complete": 3000,
            "rolling_design": {
                "folds": [{"feature_gain": ["x"] * 5000}],
                "pooled_test_metrics": [{"model": "baseline", "mae": .4, "debug": ["x"] * 5000}],
            },
            "final_research_comparison": {
                "evaluation_period": ["2024-06-26", "2024-07-02"],
                "evaluation_rows": 21000,
                "metrics": [{"model": "baseline", "mae": .4, "debug": ["x"] * 5000}],
                "ensemble_weights": {"residual_corrected": .65, "spectral": .35},
                "bootstrap_vs_baseline": {
                    "comparisons_vs_baseline": {
                        "residual_corrected": {
                            "mae": {"mean_improvement_pct": 5.0, "debug": ["x"] * 5000}
                        }
                    }
                },
                "wavelet_ablation_vs_spectral": {"comparisons_vs_baseline": {"wavelet": {}}},
            },
            "limitations": ["not untouched", "observed sales"],
        },
    }
    compact = api._compact_stakeholder_evidence(raw)
    encoded = __import__("json").dumps(compact, separators=(",", ":"))

    assert len(compact["inventory"]["segments"]) == 6
    assert len(compact["inventory"]["distribution_scores"]) == 3
    assert "debug" not in encoded
    assert "feature_gain" not in encoded
    assert "raw_store_rows" not in encoded
    assert compact["forecast"]["rolling_folds"] == 1
    assert len(encoded) < 3500
    parts = api._stakeholder_prompt_parts(
        api.StakeholderBriefRequest(audience="Executive leadership", focus="Explain the evidence."),
        compact, ["inventory_optimizer_summary.json", "forecast_research_summary.json"], [],
    )
    assert len(parts) == 4
    assert all(len(api._stakeholder_system_prompt()) + len(part["user_message"]) < 5000 for part in parts)
    estimated_reserved_tokens = sum(
        int(__import__("math").ceil((len(api._stakeholder_system_prompt()) + len(part["user_message"])) / 2.5)) + 400
        for part in parts
    )
    assert estimated_reserved_tokens < 7800
    assert all(len(part["user_message"]) < 5000 for part in parts)
    assert len(parts[0]["user_message"]) < 3000
    assert len(parts[1]["user_message"]) < 3000
    assert len(parts[2]["user_message"]) < 4000
    assert len(parts[3]["user_message"]) < 3500




def test_stakeholder_brief_waits_for_both_latest_summary_artifacts(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(
        api,
        "_stakeholder_evidence",
        lambda: (
            {"inventory_optimizer": {"scenario_results": []}, "forecast_research": None},
            ["inventory-optimizer-artifacts/inventory_optimizer_summary.json"],
            ["forecast-research-artifacts/forecast_research_summary.json"],
        ),
    )
    with pytest.raises(HTTPException) as exc:
        api.generate_stakeholder_brief(
            api.StakeholderBriefRequest(audience="Executive leadership", focus="Explain the findings.")
        )
    assert exc.value.status_code == 503
    assert exc.value.detail["code"] == "stakeholder_sources_not_ready"
    assert "forecast-research-artifacts/forecast_research_summary.json" in exc.value.detail["missing_artifacts"]



def test_stakeholder_prompt_size_guard_raises_actionable_413(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(
        api, "_stakeholder_evidence",
        lambda: ({"inventory_optimizer": {}, "forecast_research": {}}, ["inventory.json"], []),
    )
    monkeypatch.setattr(api, "_compact_stakeholder_evidence", lambda _raw: {"inventory": {"population": "x" * 15000}, "forecast": {}})
    request = api.StakeholderBriefRequest(audience="Executive leadership", focus="Summarize results.")
    with pytest.raises(HTTPException) as exc:
        api.generate_stakeholder_brief(request)
    assert exc.value.status_code == 413
    assert exc.value.detail["code"] == "stakeholder_prompt_too_large"
    assert exc.value.detail["prompt_characters"] > exc.value.detail["maximum_prompt_characters"]
    assert exc.value.detail["maximum_prompt_characters"] == 5000
    assert "safe limit" in exc.value.detail["message"]





def test_stakeholder_brief_generates_four_sections_and_combines_them(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "")
    monkeypatch.setattr(
        api,
        "_stakeholder_evidence",
        lambda: (
            {"inventory_optimizer": {"scenario_results": []}, "forecast_research": {}},
            ["inventory_optimizer_summary.json", "forecast_research_summary.json"],
            [],
        ),
    )
    monkeypatch.setattr(
        api,
        "_compact_stakeholder_evidence",
        lambda _raw: {
            "inventory": {"population": "FreshRetailNet-LT", "empirical_fill_rate": 0.8},
            "forecast": {"sample_series": 3000},
        },
    )

    calls = []
    contents = [
        "Inventory service outcomes explained.",
        "Inventory constraints and distributions explained.",
        "Forecast model comparisons explained.",
        "Bootstrap, ablation and experiment decisions explained.",
    ]

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            content = contents[len(calls) - 1]
            usage = SimpleNamespace(
                model_dump=lambda: {
                    "prompt_tokens": 100,
                    "completion_tokens": 200,
                    "total_tokens": 300,
                }
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=usage,
            )

    class FakeGroq:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "groq", SimpleNamespace(Groq=FakeGroq))
    result = api.generate_stakeholder_brief(
        api.StakeholderBriefRequest(audience="Executive leadership", focus="Explain all metrics.")
    )

    assert len(calls) == 4
    assert all(call["model"] == "openai/gpt-oss-120b" for call in calls)
    assert "FreshRetailNet-LT" in calls[0]["messages"][1]["content"]
    assert "FreshRetailNet-LT" in calls[1]["messages"][1]["content"]
    assert "FreshRetailNet-50K" in calls[2]["messages"][1]["content"]
    assert "FreshRetailNet-50K" in calls[3]["messages"][1]["content"]
    assert all(call["max_completion_tokens"] == 400 for call in calls)
    for content in contents:
        assert content in result["brief_markdown"]
    assert result["generation_strategy"] == "four_part_groq"
    assert result["usage"]["totals"]["total_tokens"] == 1200



def test_stakeholder_compact_snapshot_stays_well_below_free_tier_prompt_budget():
    raw = {
        "inventory_optimizer": {
            "evaluation_population": {
                "target_date_start": "2024-01-01", "target_date_end": "2024-03-10",
                "target_dates": 70, "complete_continuous_series": 2613,
                "notes": "x" * 5000,
            },
            "calibration_design": {"windows_per_series": 5},
            "scenario_results": [{
                "scenario": "balanced_90pct", "target_fill_rate": .9,
                "empirical_fill_rate": .798, "theoretical_expected_fill_rate": .934,
                "total_simulated_cost": 133913, "capacity_limited_store_count": 75,
                "store_count": 842, "mean_replenishment_utilization_pct": 99.47,
                "notes": "x" * 2000,
            }],
            "capacity_diagnostics": [{
                "scenario": "balanced_90pct",
                "replenishment_utilization_pct": {
                    "p50": 99, "p95": 100, "share_at_least_95_pct": 85,
                },
            }],
            "segment_results": [
                {"scenario": "balanced_90pct", "segment_type": "store_size",
                 "segment": name, "empirical_fill_rate": .8, "shortage_units": 100,
                 "notes": "x" * 1000}
                for name in ("small", "medium", "large")
            ],
            "distribution_diagnostics": {
                "reference_optimizer_distribution": "moment_lognormal",
                "results": [
                    {"segment_type": "overall", "distribution": name,
                     "median_mae": .3, "central_80_coverage_pct": 80}
                    for name in ("moment_lognormal", "empirical_bootstrap", "conformal_residual")
                ],
            },
        },
        "forecast_research": {
            "series_sample_size_complete": 3000,
            "rolling_design": {
                "pooled_test_metrics": [
                    {"model": "baseline", "mae": .38, "rmse": .67, "wape_pct": 34.3},
                    {"model": "residual_corrected", "mae": .37, "rmse": .66, "wape_pct": 34.1},
                ],
                "folds": [{"feature_gain": "x" * 5000}],
            },
            "final_research_comparison": {
                "evaluation_period": ["2024-06-26", "2024-07-02"],
                "status": "previously used holdout; not untouched",
                "metrics": [
                    {"model": name, "mae": .4, "rmse": .7, "wape_pct": 33.6}
                    for name in ("baseline", "residual_corrected", "wavelet", "ensemble")
                ],
                "ensemble_weights": {"residual_corrected": .65, "spectral": .35},
                "ensemble_diversity": {"effective_model_count": 1.9},
                "bootstrap_vs_baseline": {
                    "comparisons_vs_baseline": {
                        "residual_corrected": {"mae": {
                            "mean_improvement_pct": 5.1, "ci_95_low_pct": 1.7,
                            "ci_95_high_pct": 8.7, "ci_excludes_zero": True,
                        }}
                    }
                },
            },
        },
    }
    raw["inventory_optimizer"]["scenario_results"][0].update({
        "empirical_cycle_service_level": .3275,
        "unfulfilled_protection_demand": 1000,
        "leftover_stock": 250,
        "holding_cost": 5000,
        "shortage_cost": 2000,
        "ordering_cost": 100,
        "mean_storage_utilization_pct": 49.73,
        "mean_replenishment_utilization_pct": 99.47,
        "service_target_met_store_count": 767,
        "evaluation_windows": 13065,
    })
    raw["inventory_optimizer"]["capacity_diagnostics"][0]["replenishment_utilization_pct"].update({
        "p90": 100, "share_at_least_90_pct": 90, "share_at_least_99_pct": 65,
    })
    raw["inventory_optimizer"]["capacity_diagnostics"][0]["storage_utilization_pct"] = {
        "p50": 50, "p90": 80, "p95": 95, "max": 100,
    }
    raw["inventory_optimizer"]["distribution_diagnostics"]["results"] = [
        {"segment_type": "overall", "distribution": name, "median_mae": .3,
         "pinball_q10": .1, "pinball_q50": .2, "pinball_q90": .3,
         "central_80_coverage_pct": 78, "central_80_mean_width": .7}
        for name in ("moment_lognormal", "empirical_bootstrap", "conformal_residual")
    ]
    raw["forecast_research"]["rolling_design"]["pooled_test_metrics"] = [
        {"model": name, "mae": .38, "rmse": .67, "wape_pct": 34.3, "bias_pct": -2.5}
        for name in ("baseline", "fourier_stl", "spectral", "wavelet", "residual_corrected", "ensemble")
    ]
    raw["forecast_research"]["final_research_comparison"]["metrics"] = [
        {"model": name, "mae": .4, "rmse": .7, "wape_pct": 33.6, "bias_pct": -.6}
        for name in ("baseline", "fourier_stl", "spectral", "wavelet", "residual_corrected", "ensemble")
    ]
    raw["forecast_research"]["final_research_comparison"]["bootstrap_vs_baseline"] = {
        "replicates": 300,
        "comparisons_vs_baseline": {
            name: {
                metric: {"mean_improvement_pct": 5.1, "ci_95_low_pct": 1.7,
                         "ci_95_high_pct": 8.7, "ci_excludes_zero": True}
                for metric in ("mae", "rmse", "wape")
            }
            for name in ("fourier_stl", "spectral", "wavelet", "residual_corrected")
        },
    }
    raw["forecast_research"]["final_research_comparison"]["wavelet_ablation_vs_spectral"] = {
        "replicates": 300,
        "comparisons_vs_baseline": {
            "wavelet": {
                metric: {"mean_improvement_pct": .1, "ci_95_low_pct": -.3,
                         "ci_95_high_pct": .5, "ci_excludes_zero": False}
                for metric in ("mae", "rmse", "wape")
            }
        },
    }
    raw["forecast_research"]["methods"] = {
        "fourier": "Calendar harmonics.",
        "stl": "Weekly decomposition.",
        "spectral": "Periodogram features.",
        "wavelet": "Causal db2 decomposition.",
        "residual_model": "Residual correction.",
        "ensemble": "Validation-selected weights.",
    }
    evidence = api._compact_stakeholder_evidence(raw)
    encoded = __import__("json").dumps(evidence, separators=(",", ":"))
    assert len(encoded) < 6800  # Combined evidence is intentionally split into three Groq calls.
    parts = api._stakeholder_prompt_parts(
        api.StakeholderBriefRequest(audience="Executive leadership", focus="Explain the findings."),
        evidence, ["inventory_optimizer_summary.json", "forecast_research_summary.json"], [],
    )
    assert len(parts) == 4
    assert all(len(api._stakeholder_system_prompt()) + len(part["user_message"]) < 5000 for part in parts)
    estimated_reserved_tokens = sum(
        int(__import__("math").ceil((len(api._stakeholder_system_prompt()) + len(part["user_message"])) / 2.5)) + 400
        for part in parts
    )
    assert estimated_reserved_tokens < 7800
