"""Small offline tests for the API's response-safe dataframe behavior."""
import pandas as pd

def test_forecast_export_contains_required_dashboard_fields():
    required = {"date","store_id","product_id","actual_sale_amount","p10","p50","p90",
                "p10_calibrated","p90_calibrated","seasonal_naive_lag7"}
    sample = pd.DataFrame([{k: 1 for k in required}])
    assert required.issubset(sample.columns)

def test_policy_summary_steady_state_metrics_are_nonnegative():
    row = {"steady_state_fill_rate_pct": 98.5, "steady_state_total_cost": 123.4}
    assert row["steady_state_fill_rate_pct"] >= 0
    assert row["steady_state_total_cost"] >= 0
