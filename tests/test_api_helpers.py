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
    assert api.health() == {"status": "ok", "r2_configured": False}


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
