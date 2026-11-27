"""Leakage-safe, batched trailing-window wavelet features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pywt

WAVELET = "db2"
LEVEL = 2
WINDOW = 32


def build_wavelet_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create features for date t using only the 32 observed sales before t.

    The first 32 observations per store-product series have no wavelet features.
    A fixed window and periodization mode make coefficient shapes consistent.
    """
    required = {"store_id", "product_id", "date", "sale_amount"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing wavelet input columns: {sorted(missing)}")

    ordered = frame.sort_values(["store_id", "product_id", "date"])
    chunks: list[pd.DataFrame] = []
    names = [
        "wavelet_approx_energy",
        "wavelet_detail_energy_l1",
        "wavelet_detail_energy_l2",
        "wavelet_coefficient_entropy",
        "wavelet_high_low_energy_ratio",
        "wavelet_smoothed_mean_32",
    ]

    for (store_id, product_id), group in ordered.groupby(
        ["store_id", "product_id"], sort=False, observed=True
    ):
        sales = group["sale_amount"].to_numpy(dtype=np.float64, copy=False)
        dates = group["date"].to_numpy(dtype="datetime64[D]")
        if len(sales) <= WINDOW:
            continue

        # Each row in windows is the trailing 32-day history immediately
        # preceding one target date; no current/future target enters a window.
        windows = np.lib.stride_tricks.sliding_window_view(
            sales, WINDOW
        )[:-1]
        # windows[k] covers sales[k:k+32], and predicts date index k+32.
        coeffs = pywt.wavedec(
            windows, WAVELET, mode="periodization", level=LEVEL, axis=1
        )
        approx, detail2, detail1 = coeffs
        energies = [
            np.mean(np.square(approx), axis=1),
            np.mean(np.square(detail1), axis=1),
            np.mean(np.square(detail2), axis=1),
        ]
        energy_matrix = np.column_stack(energies)
        total_energy = np.maximum(energy_matrix.sum(axis=1), 1e-12)
        probabilities = energy_matrix / total_energy[:, None]
        entropy = -np.sum(
            probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1
        )
        ratio = (energy_matrix[:, 1:].sum(axis=1) + 1e-12) / (
            energy_matrix[:, 0] + 1e-12
        )
        values = np.column_stack(
            [*energies, entropy, ratio, np.mean(windows, axis=1)]
        )
        chunks.append(
            pd.DataFrame(
                values,
                columns=names,
            ).assign(
                store_id=store_id,
                product_id=product_id,
                date=pd.to_datetime(dates[WINDOW:]).date,
            )
        )

    if not chunks:
        raise ValueError("No series had enough history for wavelet features")
    result = pd.concat(chunks, ignore_index=True)
    if result[names].isna().any().any() or not np.isfinite(result[names]).all().all():
        raise ValueError("Wavelet feature generation produced non-finite values")
    return result
