"""Reproducible synthetic demo for the inventory simulator.

Run from the repository root:
    python scripts/run_small_demo.py --output-dir demo_output

The generated data are synthetic and must not be reported as benchmark results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.simulate_inventory import aggregate, simulate


def build_demo_frame() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=21, freq="D")
    rows = []
    for store, product, scale in (("demo_store_1", "demo_product_A", 1.0),
                                  ("demo_store_2", "demo_product_B", 1.6)):
        for i, date in enumerate(dates):
            actual = scale * (5 + (i % 7) + (2 if i in (8, 9, 17) else 0))
            rows.append({
                "date": date, "store_id": store, "product_id": product,
                "actual_sale_amount": actual,
                "forecast_p50": scale * (5.5 + (i % 7)),
                "forecast_p90": scale * (7.0 + (i % 7)),
            })
    return pd.DataFrame(rows)


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = build_demo_frame()
    params = dict(lead_time=2, review_period=1, holding_cost=0.05,
                  shortage_cost=2.0, order_cost=1.0, initial_stock=8.0)
    detail = pd.concat([
        simulate(frame, "forecast_p50", "synthetic_p50", **params),
        simulate(frame, "forecast_p90", "synthetic_p90", **params),
    ], ignore_index=True)
    summary = {
        "demo_type": "synthetic_only_not_benchmark_evidence",
        "seed": 20261128,
        "input_rows": int(len(frame)),
        "store_product_series": int(frame[["store_id", "product_id"]].drop_duplicates().shape[0]),
        "assumptions": params,
        "comparison": [
            aggregate(detail[detail.policy == policy], "synthetic_demo", params)
            for policy in ("synthetic_p50", "synthetic_p90")
        ],
        "limitations": [
            "Synthetic demand and forecasts are illustrative, not FreshRetailNet observations.",
            "This demo validates local execution and inventory accounting only.",
        ],
    }
    frame.to_csv(output_dir / "synthetic_forecasts.csv", index=False)
    detail.to_csv(output_dir / "inventory_policy_results.csv", index=False)
    (output_dir / "demo_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("demo_output"))
    args = parser.parse_args()
    result = run(args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
