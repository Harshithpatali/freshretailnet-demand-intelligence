"""Regression tests for the inventory simulator's core accounting behavior."""
import pandas as pd
import pytest

from src.simulate_inventory import simulate


def sample_forecasts():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "store_id": ["s1"] * 3,
            "product_id": ["p1"] * 3,
            "actual_sale_amount": [2.0, 3.0, 1.0],
            "forecast": [3.0, 3.0, 2.0],
        }
    )


def test_simulation_returns_nonnegative_inventory_and_bounded_fill_rate():
    result = simulate(
        sample_forecasts(),
        forecast_col="forecast",
        policy_label="test",
        lead_time=1,
        review_period=1,
        holding_cost=0.1,
        shortage_cost=2.0,
        order_cost=1.0,
        initial_stock=2.0,
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["ending_inventory"] >= 0
    assert 0 <= row["fill_rate_pct"] <= 100
    assert row["units_fulfilled"] <= row["observed_sales"]
    assert row["total_simulated_cost"] == pytest.approx(
        row["holding_cost"] + row["shortage_cost"] + row["ordering_cost"]
    )


def test_terminal_orders_are_suppressed_when_lead_time_exceeds_remaining_horizon():
    result = simulate(
        sample_forecasts().iloc[:1],
        forecast_col="forecast",
        policy_label="long_lead",
        lead_time=2,
        review_period=1,
        holding_cost=0.0,
        shortage_cost=1.0,
        order_cost=1.0,
        initial_stock=0.0,
    )

    row = result.iloc[0]
    assert row["orders_placed"] == 0
    assert row["orders_suppressed_at_horizon"] == 1
    assert row["units_suppressed_at_horizon"] > 0
