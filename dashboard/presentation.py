"""Small display helpers for long/nested API values in the Streamlit dashboard."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


def _display_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, Mapping):
        return f"{len(value):,} nested fields"
    if isinstance(value, (list, tuple, set)):
        if not value:
            return "—"
        return " · ".join(_display_value(item) for item in value)
    if isinstance(value, float):
        if pd.isna(value):
            return "—"
        return f"{value:,.5g}"
    return str(value)


def flatten_structured_data(payload: Any) -> pd.DataFrame:
    """Flatten nested dict/list payloads into complete, readable key/value rows."""
    rows: list[dict[str, str]] = []

    def visit(label: str, value: Any) -> None:
        if isinstance(value, Mapping):
            if not value:
                rows.append({"Field": label, "Value": "—"})
                return
            for key, item in value.items():
                next_label = f"{label} / {key}" if label else str(key)
                visit(next_label, item)
            return
        if isinstance(value, (list, tuple)) and value and all(
            isinstance(item, Mapping) for item in value
        ):
            for index, item in enumerate(value, start=1):
                for key, child in item.items():
                    visit(f"{label} {index} / {key}", child)
            return
        rows.append({"Field": label or "Value", "Value": _display_value(value)})

    visit("", payload)
    return pd.DataFrame(rows, columns=["Field", "Value"])
