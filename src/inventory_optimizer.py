"""Service-level constrained stochastic inventory optimization.

Each store is solved as a multiple-choice mixed-integer program. Every SKU
chooses one admissible order quantity (zero or a MOQ/pack-size multiple).
The objective minimizes expected holding, shortage and fixed ordering costs,
subject to store-level fill-rate, storage-capacity and replenishment-capacity
constraints. Demand is represented by a moment-matched lognormal distribution.

All cost, volume, capacity and MOQ inputs are explicit planning assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log, sqrt
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.stats import norm


REQUIRED_COLUMNS = {
    "store_id", "product_id", "demand_mean", "demand_std",
    "inventory_position", "unit_holding_cost", "unit_shortage_cost",
    "fixed_order_cost", "pack_size", "min_order_qty", "max_order_qty",
    "volume_per_unit",
}


@dataclass(frozen=True)
class DemandMoments:
    mean: float
    std: float


def lognormal_expected_shortage(mean: float, std: float, stock: float) -> float:
    """E[(D-stock)+] for non-negative lognormal demand with supplied moments."""
    if mean < 0 or std < 0 or stock < 0:
        raise ValueError("Demand moments and stock must be non-negative")
    if mean == 0:
        return 0.0
    if std <= 1e-12:
        return max(mean - stock, 0.0)
    if stock == 0:
        return mean
    sigma2 = log(1.0 + (std / mean) ** 2)
    sigma = sqrt(sigma2)
    mu = log(mean) - 0.5 * sigma2
    z = (log(stock) - mu) / sigma
    truncated_mean = mean * norm.cdf((log(stock) - mu - sigma2) / sigma)
    expected_short = mean - truncated_mean - stock * norm.sf(z)
    return float(max(0.0, expected_short))


def lognormal_expected_leftover(mean: float, std: float, stock: float) -> float:
    """E[(stock-D)+] for non-negative lognormal demand."""
    if mean < 0 or std < 0 or stock < 0:
        raise ValueError("Demand moments and stock must be non-negative")
    if stock == 0:
        return 0.0
    if mean == 0:
        return float(stock)
    if std <= 1e-12:
        return max(stock - mean, 0.0)
    sigma2 = log(1.0 + (std / mean) ** 2)
    sigma = sqrt(sigma2)
    mu = log(mean) - 0.5 * sigma2
    z = (log(stock) - mu) / sigma
    truncated_mean = mean * norm.cdf((log(stock) - mu - sigma2) / sigma)
    return float(max(0.0, stock * norm.cdf(z) - truncated_mean))


def lognormal_cycle_service(mean: float, std: float, stock: float) -> float:
    """P(D <= stock) for moment-matched lognormal demand."""
    if mean < 0 or std < 0 or stock < 0:
        raise ValueError("Demand moments and stock must be non-negative")
    if mean == 0:
        return 1.0
    if std <= 1e-12:
        return 1.0 if stock >= mean else 0.0
    if stock <= 0:
        return 0.0
    sigma2 = log(1.0 + (std / mean) ** 2)
    sigma = sqrt(sigma2)
    mu = log(mean) - 0.5 * sigma2
    return float(norm.cdf((log(stock) - mu) / sigma))


def _candidate_orders(row: pd.Series, max_options: int = 60) -> np.ndarray:
    pack = float(row.pack_size)
    minimum = float(row.min_order_qty)
    maximum = max(0.0, float(row.max_order_qty))
    if pack <= 0 or minimum < 0 or maximum < 0:
        raise ValueError("pack_size must be positive; MOQ and max_order_qty non-negative")
    first_pack = max(1, int(ceil((minimum - 1e-12) / pack)))
    last_pack = int(np.floor((maximum + 1e-12) / pack))
    if last_pack < first_pack:
        return np.array([0.0])
    counts = np.arange(first_pack, last_pack + 1, dtype=int)
    if len(counts) > max_options:
        selected = np.unique(np.rint(np.linspace(0, len(counts) - 1, max_options)).astype(int))
        counts = counts[selected]
    return np.concatenate(([0.0], counts.astype(float) * pack))


def _candidate_table(row: pd.Series, max_options: int) -> pd.DataFrame:
    orders = _candidate_orders(row, max_options=max_options)
    mean = float(row.demand_mean)
    std = float(row.demand_std)
    position = float(row.inventory_position)
    volume = float(row.volume_per_unit)
    hold_cost = float(row.unit_holding_cost)
    shortage_cost = float(row.unit_shortage_cost)
    fixed_order = float(row.fixed_order_cost)
    if min(mean, std, position, volume, hold_cost, shortage_cost, fixed_order) < 0 or volume <= 0:
        raise ValueError(f"Invalid non-negative input for store/product {row.store_id}/{row.product_id}")
    stock = position + orders
    short = np.array([lognormal_expected_shortage(mean, std, float(s)) for s in stock])
    leftover = np.array([lognormal_expected_leftover(mean, std, float(s)) for s in stock])
    cycle = np.array([lognormal_cycle_service(mean, std, float(s)) for s in stock])
    fulfilled = np.maximum(0.0, mean - short)
    costs = hold_cost * leftover + shortage_cost * short + (orders > 0) * fixed_order
    return pd.DataFrame({
        "order_qty": orders,
        "stock_position": stock,
        "expected_fulfilled": fulfilled,
        "expected_shortage": short,
        "expected_leftover": leftover,
        "cycle_service_level": cycle,
        "expected_cost": costs,
        "storage_volume": stock * volume,
        "replenishment_volume": orders * volume,
    })


def optimize_store_orders(
    items: pd.DataFrame,
    *,
    target_fill_rate: float = 0.90,
    storage_capacity: float,
    replenishment_capacity: float,
    max_options_per_sku: int = 60,
    time_limit_seconds: float = 30.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Optimize one store's assortment subject to service and capacity constraints.

    The fill-rate constraint is volume-weighted expected fulfilled demand divided
    by total expected demand. If the requested SLA is infeasible under the given
    capacities, a second optimization maximizes expected fulfilled demand and
    reports the resulting service gap instead of claiming the SLA was achieved.
    """
    if not 0 < target_fill_rate <= 1:
        raise ValueError("target_fill_rate must be in (0, 1]")
    if storage_capacity < 0 or replenishment_capacity < 0:
        raise ValueError("Capacity constraints must be non-negative")
    missing = REQUIRED_COLUMNS - set(items.columns)
    if missing:
        raise ValueError(f"Missing optimizer columns: {sorted(missing)}")
    if items.empty:
        raise ValueError("Cannot optimize an empty store assortment")
    if items.product_id.astype(str).duplicated().any():
        raise ValueError("Each store assortment must contain at most one row per product")

    items = items.reset_index(drop=True).copy()
    candidate_frames = [
        _candidate_table(row, max_options_per_sku)
        for _, row in items.iterrows()
    ]
    starts = np.cumsum([0] + [len(frame) for frame in candidate_frames])
    n_variables = int(starts[-1])
    n_items = len(items)
    total_mean = float(items.demand_mean.sum())
    if total_mean <= 0:
        chosen = [0] * n_items
        selected = [frame.iloc[index] for frame, index in zip(candidate_frames, chosen)]
        decisions = items.copy()
        for column in selected[0].index:
            decisions[column] = [row[column] for row in selected]
        return decisions, {
            "status": "NO_POSITIVE_FORECAST_DEMAND",
            "solver": "scipy.optimize.milp",
            "target_fill_rate": target_fill_rate,
            "achieved_fill_rate": 1.0,
            "service_target_met": True,
            "expected_demand": total_mean,
            "storage_capacity": storage_capacity,
            "replenishment_capacity": replenishment_capacity,
        }

    # Rows: one selected option per SKU, storage capacity, replenishment capacity,
    # and the store-wide expected fill-rate service constraint.
    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []
    lower = np.full(n_items + 3, -np.inf, dtype=float)
    upper = np.full(n_items + 3, np.inf, dtype=float)
    lower[:n_items] = 1.0
    upper[:n_items] = 1.0
    upper[n_items] = float(storage_capacity)
    upper[n_items + 1] = float(replenishment_capacity)
    lower[n_items + 2] = float(target_fill_rate * total_mean)

    costs = np.empty(n_variables, dtype=float)
    fulfilled = np.empty(n_variables, dtype=float)
    for item_index, candidates in enumerate(candidate_frames):
        start, end = int(starts[item_index]), int(starts[item_index + 1])
        for local_index, (_, option) in enumerate(candidates.iterrows()):
            variable = start + local_index
            row_indices.extend([item_index, n_items, n_items + 1, n_items + 2])
            col_indices.extend([variable] * 4)
            values.extend([
                1.0,
                float(option.storage_volume),
                float(option.replenishment_volume),
                float(option.expected_fulfilled),
            ])
            costs[variable] = float(option.expected_cost)
            fulfilled[variable] = float(option.expected_fulfilled)

    matrix = coo_matrix(
        (values, (row_indices, col_indices)),
        shape=(n_items + 3, n_variables),
    ).tocsr()
    constraints = LinearConstraint(matrix, lower, upper)
    bounds = Bounds(np.zeros(n_variables), np.ones(n_variables))
    integrality = np.ones(n_variables, dtype=int)

    options = {"time_limit": float(time_limit_seconds), "mip_rel_gap": 0.01}
    result = milp(
        c=costs,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options=options,
    )
    used_fallback = False
    if result.x is None:
        # Infeasible service-level target: maximize expected fulfilled demand
        # under the same warehouse and replenishment constraints.
        fallback_cost = -fulfilled + 1e-7 * costs
        fallback = milp(
            c=fallback_cost,
            integrality=integrality,
            bounds=bounds,
            constraints=LinearConstraint(
                matrix[:n_items + 2],
                lower[:n_items + 2],
                upper[:n_items + 2],
            ),
            options=options,
        )
        if fallback.x is None:
            raise RuntimeError(
                f"Inventory MILP failed for store {items.store_id.iloc[0]}: "
                f"primary={result.message}; fallback={fallback.message}"
            )
        result = fallback
        used_fallback = True

    chosen = np.zeros(n_items, dtype=int)
    for item_index, candidates in enumerate(candidate_frames):
        start, end = int(starts[item_index]), int(starts[item_index + 1])
        chosen[item_index] = int(np.argmax(result.x[start:end]))
    selected_options = [
        candidate_frames[i].iloc[chosen[i]]
        for i in range(n_items)
    ]
    decisions = items.copy()
    for column in selected_options[0].index:
        decisions[column] = [option[column] for option in selected_options]

    expected_demand = float(items.demand_mean.sum())
    expected_fulfilled = float(decisions.expected_fulfilled.sum())
    expected_shortage = float(decisions.expected_shortage.sum())
    total_order = float(decisions.order_qty.sum())
    used_storage = float(decisions.storage_volume.sum())
    used_replenishment = float(decisions.replenishment_volume.sum())
    achieved_fill = expected_fulfilled / max(expected_demand, 1e-12)
    weighted_cycle = float(
        np.average(decisions.cycle_service_level, weights=np.maximum(items.demand_mean, 1e-12))
    )
    total_cost = float(decisions.expected_cost.sum())
    service_target_met = achieved_fill + 1e-8 >= target_fill_rate
    status = "OPTIMAL_TARGET_MET" if service_target_met and not used_fallback else (
        "CAPACITY_LIMITED_SLA_INFEASIBLE" if not service_target_met else "FEASIBLE_INCUMBENT"
    )
    if getattr(result, "status", None) == 1 and service_target_met:
        status = "FEASIBLE_TIME_LIMIT_INCUMBENT"

    return decisions, {
        "status": status,
        "solver": "scipy.optimize.milp (HiGHS)",
        "solver_message": str(result.message),
        "solver_objective": float(result.fun) if result.fun is not None else None,
        "mip_gap": float(getattr(result, "mip_gap", np.nan)),
        "target_fill_rate": float(target_fill_rate),
        "achieved_fill_rate": float(achieved_fill),
        "service_target_met": bool(service_target_met),
        "weighted_cycle_service_level": weighted_cycle,
        "expected_demand": expected_demand,
        "expected_fulfilled": expected_fulfilled,
        "expected_shortage": expected_shortage,
        "expected_leftover": float(decisions.expected_leftover.sum()),
        "expected_cost": total_cost,
        "total_order_qty": total_order,
        "storage_capacity": float(storage_capacity),
        "storage_used": used_storage,
        "storage_utilization_pct": 100 * used_storage / max(storage_capacity, 1e-12),
        "replenishment_capacity": float(replenishment_capacity),
        "replenishment_used": used_replenishment,
        "replenishment_utilization_pct": 100 * used_replenishment / max(replenishment_capacity, 1e-12),
        "sku_count": int(len(items)),
        "ordered_sku_count": int((decisions.order_qty > 0).sum()),
        "infeasible_service_constraint_fallback": bool(used_fallback),
    }


def optimize_network(
    items: pd.DataFrame,
    *,
    target_fill_rate: float = 0.90,
    protection_days: int = 7,
    storage_capacity_days: float = 14.0,
    replenishment_capacity_days: float = 7.0,
    max_order_days_per_sku: float = 14.0,
    pack_size: float = 0.10,
    min_order_qty: float = 0.20,
    unit_holding_cost: float = 0.05,
    unit_shortage_cost: float = 2.0,
    fixed_order_cost: float = 5.0,
    max_options_per_sku: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Optimize each store's replenishment allocation under shared capacity.

    Demand mean/std describe the protection-period demand distribution.
    Capacity settings are expressed as days of each store's average daily demand.
    """
    if protection_days < 1:
        raise ValueError("protection_days must be >= 1")
    if min(storage_capacity_days, replenishment_capacity_days, max_order_days_per_sku) < 0:
        raise ValueError("Capacity-day settings must be non-negative")
    required = {"store_id", "product_id", "demand_mean", "demand_std"}
    missing = required - set(items.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")
    data = items.copy()
    data["inventory_position"] = pd.to_numeric(
        data.get("inventory_position", 0.0), errors="coerce"
    ).fillna(0.0) if isinstance(data.get("inventory_position", 0.0), pd.Series) else float(
        data.get("inventory_position", 0.0)
    )
    defaults = {
        "unit_holding_cost": unit_holding_cost,
        "unit_shortage_cost": unit_shortage_cost,
        "fixed_order_cost": fixed_order_cost,
        "pack_size": pack_size,
        "min_order_qty": min_order_qty,
        "volume_per_unit": 1.0,
    }
    for column, default in defaults.items():
        if column not in data:
            data[column] = default
        data[column] = pd.to_numeric(data[column], errors="raise")
    # Optimizer holding input is per unit per day; each plan covers the protection period.
    data["unit_holding_cost"] = data["unit_holding_cost"] * protection_days
    data["max_order_qty"] = (
        data["demand_mean"] / protection_days * max_order_days_per_sku
    )
    all_decisions = []
    summaries = []
    for store_id, group in data.groupby("store_id", sort=False):
        mean_protection_demand = float(group.demand_mean.sum())
        daily_store_mean = mean_protection_demand / protection_days
        storage_capacity = storage_capacity_days * daily_store_mean
        replenishment_capacity = replenishment_capacity_days * daily_store_mean
        decisions, summary = optimize_store_orders(
            group,
            target_fill_rate=target_fill_rate,
            storage_capacity=storage_capacity,
            replenishment_capacity=replenishment_capacity,
            max_options_per_sku=max_options_per_sku,
        )
        decisions["store_id"] = str(store_id)
        decisions["target_fill_rate"] = target_fill_rate
        decisions["protection_days"] = protection_days
        all_decisions.append(decisions)
        summaries.append({
            "store_id": str(store_id),
            "storage_capacity_days": storage_capacity_days,
            "replenishment_capacity_days": replenishment_capacity_days,
            "max_order_days_per_sku": max_order_days_per_sku,
            "protection_days": protection_days,
            **summary,
        })
    return (
        pd.concat(all_decisions, ignore_index=True) if all_decisions else pd.DataFrame(),
        pd.DataFrame(summaries),
    )
