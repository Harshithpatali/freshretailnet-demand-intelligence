"""Rolling benchmark for Fourier/STL, spectral, wavelet, residual and ensemble models.

Uses a deterministic panel sample and three expanding-window rolling folds from
FreshRetailNet-50K's 90-day training split. The original seven-day evaluation
split is scored as a final research comparison, but it is not an untouched
independent holdout because the repository's earlier benchmark already used it.

All target-day predictors are forecast-time safe: the fixed-origin STL/spectral
features use data strictly before each split's origin; wavelet and lag features
use trailing history only; contemporaneous price, weather, holiday/activity and
stock-status fields are excluded.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import boto3
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from botocore.config import Config
from scipy.optimize import minimize
from scipy.signal import periodogram
from statsmodels.tsa.seasonal import STL

from experiments.wavelet.features import build_wavelet_features
from src.train_model import RAW, load_features_from_frame, metric_pack


GROUP = ["store_id", "product_id"]
OUT = "forecast-research-artifacts"
SEED = 20261128
SERIES_SAMPLE = int(os.getenv("FORECAST_RESEARCH_SERIES_SAMPLE", "3000"))
BOOTSTRAP_REPS = int(os.getenv("FORECAST_RESEARCH_BOOTSTRAP_REPS", "300"))
WARMUP_DAYS = 56
WAVELET_FAMILY = "db2"

BASE_FEATURES = [
    "city_id", "store_id", "management_group_id", "first_category_id",
    "second_category_id", "third_category_id", "product_id",
    "day_of_week", "day_of_month", "month", "day_of_year", "week_of_year",
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "stock_status_lag_1", "stock_status_lag_7", "stock_status_lag_14",
    "stock_status_lag_28", "sales_roll_mean_7", "sales_roll_mean_28",
    "sales_roll_std_7",
]
FOURIER_FEATURES = [
    *(f"fourier_week_sin_{k}" for k in (1, 2, 3)),
    *(f"fourier_week_cos_{k}" for k in (1, 2, 3)),
    "fourier_month_sin", "fourier_month_cos",
    "fourier_year_sin", "fourier_year_cos",
]
STL_FEATURES = [
    "stl_trend_level", "stl_trend_slope_14", "stl_residual_mean_14",
    "stl_residual_std_28", "stl_residual_last",
    "stl_seasonal_for_weekday", "stl_days_since_origin",
    "stl_fit_available",
]
SPECTRAL_FEATURES = [
    "spectral_dominant_period_days", "spectral_weekly_energy_share",
    "spectral_entropy", "spectral_peak_share",
]
WAVELET_FEATURES = [
    "wavelet_approx_energy", "wavelet_detail_energy_l1",
    "wavelet_detail_energy_l2", "wavelet_coefficient_entropy",
    "wavelet_high_low_energy_ratio", "wavelet_smoothed_mean_32",
]
MODEL_FEATURES = {
    "baseline": BASE_FEATURES,
    "fourier_stl": BASE_FEATURES + FOURIER_FEATURES + STL_FEATURES,
    "spectral": BASE_FEATURES + FOURIER_FEATURES + STL_FEATURES + SPECTRAL_FEATURES,
    "wavelet": BASE_FEATURES + FOURIER_FEATURES + STL_FEATURES + SPECTRAL_FEATURES + WAVELET_FEATURES,
}
CATEGORICAL = [
    "city_id", "store_id", "management_group_id", "first_category_id",
    "second_category_id", "third_category_id", "product_id",
    "day_of_week", "month",
]


def env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def r2_client():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{env('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
        aws_access_key_id=env("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=env("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_deterministic_series_sample(
    pairs: pd.DataFrame,
    sample_size: int = SERIES_SAMPLE,
) -> pd.DataFrame:
    """Sample the same IDs regardless of the upstream distinct-row ordering."""
    stable_pairs = (
        pairs[GROUP]
        .drop_duplicates()
        .sort_values(GROUP, kind="mergesort")
        .reset_index(drop=True)
    )
    if len(stable_pairs) < sample_size:
        raise RuntimeError(f"Need {sample_size} distinct series; found {len(stable_pairs)}")
    return (
        stable_pairs.sample(n=sample_size, random_state=SEED)
        .sort_values(GROUP, kind="mergesort")
        .reset_index(drop=True)
    )


def load_deterministic_panel(s3, bucket: str, root: Path):
    source_paths = {}
    for split, key in RAW.items():
        path = root / f"{split}.parquet"
        print(f"Downloading {key}", flush=True)
        s3.download_file(bucket, key, str(path))
        source_paths[split] = path

    train_lazy = pl.scan_parquet(source_paths["train"])
    pairs = train_lazy.select(GROUP).unique().collect(engine="streaming").to_pandas()
    sampled = select_deterministic_series_sample(pairs, SERIES_SAMPLE)
    sampled_keys = pl.from_pandas(sampled)

    frames = {}
    for split, path in source_paths.items():
        lf = (
            pl.scan_parquet(path)
            .join(sampled_keys.lazy(), on=GROUP, how="inner")
            .with_columns(pl.col("dt").cast(pl.String).str.to_date(strict=False).alias("date"))
        )
        frames[split] = lf.collect(engine="streaming")

    train_dates = sorted(frames["train"].get_column("date").unique().to_list())
    eval_dates = sorted(frames["eval"].get_column("date").unique().to_list())
    if len(train_dates) < 90 or len(eval_dates) < 7:
        raise RuntimeError(
            f"Expected >=90 train and >=7 evaluation dates; got {len(train_dates)} and {len(eval_dates)}"
        )
    train_counts = frames["train"].group_by(GROUP).agg(pl.col("date").n_unique().alias("days"))
    eval_counts = frames["eval"].group_by(GROUP).agg(pl.col("date").n_unique().alias("days"))
    complete_train = train_counts.filter(pl.col("days") >= 90).select(GROUP)
    complete_eval = eval_counts.filter(pl.col("days") >= 7).select(GROUP)
    complete = complete_train.join(complete_eval, on=GROUP, how="inner")
    frames = {
        split: frame.join(complete, on=GROUP, how="inner")
        for split, frame in frames.items()
    }
    n_series = frames["train"].select(GROUP).unique().height
    if n_series < min(1000, SERIES_SAMPLE):
        raise RuntimeError(f"Only {n_series} complete sample series remain")
    source_hashes = {split: sha256_file(path) for split, path in source_paths.items()}
    return frames, train_dates, eval_dates, n_series, source_hashes


def _fourier(date_value: pd.Timestamp) -> dict[str, float]:
    day_of_week = int(date_value.day_of_week)
    row = {}
    for harmonic in (1, 2, 3):
        angle = 2.0 * np.pi * harmonic * day_of_week / 7.0
        row[f"fourier_week_sin_{harmonic}"] = float(np.sin(angle))
        row[f"fourier_week_cos_{harmonic}"] = float(np.cos(angle))
    month_angle = 2.0 * np.pi * (date_value.day_of_year - 1) / 30.4375
    year_angle = 2.0 * np.pi * (date_value.day_of_year - 1) / 365.25
    row["fourier_month_sin"] = float(np.sin(month_angle))
    row["fourier_month_cos"] = float(np.cos(month_angle))
    row["fourier_year_sin"] = float(np.sin(year_angle))
    row["fourier_year_cos"] = float(np.cos(year_angle))
    return row


def _spectral_summary(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 14:
        return {
            "spectral_dominant_period_days": 0.0,
            "spectral_weekly_energy_share": 0.0,
            "spectral_entropy": 0.0,
            "spectral_peak_share": 0.0,
        }
    x = x - float(np.mean(x))
    frequencies, power = periodogram(x, fs=1.0, detrend="linear", scaling="spectrum")
    valid = (frequencies > 0) & np.isfinite(power)
    frequencies, power = frequencies[valid], np.maximum(power[valid], 0.0)
    if not len(power) or float(power.sum()) <= 1e-15:
        return {
            "spectral_dominant_period_days": 0.0,
            "spectral_weekly_energy_share": 0.0,
            "spectral_entropy": 0.0,
            "spectral_peak_share": 0.0,
        }
    periods = 1.0 / frequencies
    candidate = (periods >= 2.0) & (periods <= 28.0)
    if candidate.any():
        idx = int(np.argmax(power[candidate]))
        dominant = float(periods[candidate][idx])
    else:
        dominant = 0.0
    weekly = (periods >= 6.0) & (periods <= 8.0)
    probabilities = power / power.sum()
    positive = probabilities[probabilities > 0]
    entropy = (
        -float(np.sum(positive * np.log(positive))) / float(np.log(len(positive)))
        if len(positive) > 1 else 0.0
    )
    return {
        "spectral_dominant_period_days": dominant,
        "spectral_weekly_energy_share": float(power[weekly].sum() / power.sum()),
        "spectral_entropy": float(entropy),
        "spectral_peak_share": float(power.max() / power.sum()),
    }


def origin_features(
    raw_panel: pd.DataFrame,
    target_rows: pd.DataFrame,
    origin_date: pd.Timestamp,
) -> pd.DataFrame:
    """Build STL/spectral context using only records strictly before origin_date."""
    targets = target_rows[GROUP + ["date"]].copy()
    records = []
    histories = raw_panel[raw_panel["date"] < origin_date]
    history_by_series = {
        key: group.sort_values("date").tail(56)
        for key, group in histories.groupby(GROUP, sort=False, observed=True)
    }
    for (store_id, product_id), target_group in targets.groupby(GROUP, sort=False, observed=True):
        hist = history_by_series.get((store_id, product_id))
        if hist is None:
            hist = histories.iloc[0:0]
        values = hist["sale_amount"].to_numpy(dtype=float)
        hist_dates = pd.to_datetime(hist["date"]).to_numpy()
        stl_ok = 0.0
        seasonal_by_weekday = np.zeros(7, dtype=float)
        if len(values) >= 28 and np.isfinite(values).all():
            try:
                decomposition = STL(values, period=7, robust=True).fit()
                trend = np.asarray(decomposition.trend, dtype=float)
                seasonal = np.asarray(decomposition.seasonal, dtype=float)
                residual = np.asarray(decomposition.resid, dtype=float)
                hist_weekdays = pd.to_datetime(hist_dates).day_of_week.to_numpy()
                for weekday in range(7):
                    select = hist_weekdays[-28:] == weekday
                    seasonal_by_weekday[weekday] = (
                        float(np.mean(seasonal[-28:][select])) if select.any() else 0.0
                    )
                stl_level = float(trend[-1])
                slope_n = min(14, len(trend) - 1)
                stl_slope = float((trend[-1] - trend[-1 - slope_n]) / max(slope_n, 1))
                residual_mean = float(np.mean(residual[-14:]))
                residual_std = float(np.std(residual[-28:]))
                residual_last = float(residual[-1])
                stl_ok = 1.0
            except Exception:
                stl_level = float(np.mean(values)) if len(values) else 0.0
                stl_slope = 0.0
                residual_mean = 0.0
                residual_std = float(np.std(values)) if len(values) else 0.0
                residual_last = 0.0
                seasonal_by_weekday[:] = 0.0
        else:
            stl_level = float(np.mean(values)) if len(values) else 0.0
            stl_slope = 0.0
            residual_mean = 0.0
            residual_std = float(np.std(values)) if len(values) else 0.0
            residual_last = 0.0
        spectral = _spectral_summary(values)
        for target_date in pd.to_datetime(target_group["date"]):
            row = {
                "store_id": store_id,
                "product_id": product_id,
                "date": pd.Timestamp(target_date),
                "stl_trend_level": stl_level,
                "stl_trend_slope_14": stl_slope,
                "stl_residual_mean_14": residual_mean,
                "stl_residual_std_28": residual_std,
                "stl_residual_last": residual_last,
                "stl_seasonal_for_weekday": float(seasonal_by_weekday[target_date.day_of_week]),
                "stl_days_since_origin": float((target_date - origin_date).days),
                "stl_fit_available": stl_ok,
                **spectral,
                **_fourier(pd.Timestamp(target_date)),
            }
            records.append(row)
    return pd.DataFrame.from_records(records)


def attach_origin_features(
    base_panel: pd.DataFrame,
    raw_panel: pd.DataFrame,
    wavelet_panel: pd.DataFrame,
    target_dates: list[pd.Timestamp],
    origin_date: pd.Timestamp,
) -> pd.DataFrame:
    target = base_panel[base_panel["date"].isin(target_dates)].copy()
    context = origin_features(raw_panel, target, origin_date)
    merged = target.merge(context, on=GROUP + ["date"], how="left", validate="one_to_one")
    merged = merged.merge(
        wavelet_panel, on=GROUP + ["date"], how="left", validate="one_to_one"
    )
    feature_fill = FOURIER_FEATURES + STL_FEATURES + SPECTRAL_FEATURES + WAVELET_FEATURES
    for column in feature_fill:
        if column in merged:
            merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    return merged


def align_categories(frames: list[pd.DataFrame], categorical: list[str]):
    copies = [frame.copy() for frame in frames]
    for column in categorical:
        levels = pd.Index(
            pd.concat([frame[column].astype(str) for frame in copies], ignore_index=True)
            .dropna().unique()
        )
        dtype = pd.CategoricalDtype(categories=levels)
        for frame in copies:
            frame[column] = frame[column].astype(str).astype(dtype)
    return copies


def fit_model(
    train: pd.DataFrame,
    val_head: pd.DataFrame,
    val_tail: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    label: str,
):
    cats = [column for column in CATEGORICAL if column in features]
    tr, vh, vt, te = align_categories([train, val_head, val_tail, test], cats)
    params = {
        "objective": "regression",
        "metric": "l1",
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_data_in_leaf": 80,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 3.0,
        "verbosity": -1,
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
        "data_random_seed": SEED,
        "num_threads": 4,
        "deterministic": True,
        "force_col_wise": True,
    }
    dtrain = lgb.Dataset(
        tr[features], label=tr["sale_amount"], categorical_feature=cats, free_raw_data=False
    )
    dhead = lgb.Dataset(
        vh[features], label=vh["sale_amount"], categorical_feature=cats,
        reference=dtrain, free_raw_data=False,
    )
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=600,
        valid_sets=[dhead],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    best_iter = int(model.best_iteration or 600)
    pred_head = np.maximum(model.predict(vh[features], num_iteration=best_iter), 0.0)
    pred_tail = np.maximum(model.predict(vt[features], num_iteration=best_iter), 0.0)
    pred_test = np.maximum(model.predict(te[features], num_iteration=best_iter), 0.0)
    gains = dict(zip(features, model.feature_importance(importance_type="gain").astype(float).tolist()))
    print(
        f"{label}: iter={best_iter}, head_mae="
        f"{np.mean(np.abs(vh.sale_amount.to_numpy() - pred_head)):.5f}",
        flush=True,
    )
    return {
        "model": model,
        "best_iteration": best_iter,
        "pred_head": pred_head,
        "pred_tail": pred_tail,
        "pred_test": pred_test,
        "feature_gain": gains,
    }


def residual_correction(
    train: pd.DataFrame,
    val_head: pd.DataFrame,
    val_tail: pd.DataFrame,
    test: pd.DataFrame,
    base_head: np.ndarray,
    base_tail: np.ndarray,
    base_test: np.ndarray,
    features: list[str],
):
    """Fit residual correction on validation head, score tail, refit on all validation."""
    def augmented(frame: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
        result = frame[features].copy()
        result["base_prediction"] = prediction
        result["base_minus_lag7"] = prediction - frame["sales_lag_7"].to_numpy(float)
        return result

    cats = [column for column in CATEGORICAL if column in features]
    head_x = augmented(val_head, base_head)
    tail_x = augmented(val_tail, base_tail)
    test_x = augmented(test, base_test)
    all_valid_x = pd.concat([head_x, tail_x], ignore_index=True)
    all_valid_y = np.concatenate([
        val_head["sale_amount"].to_numpy(float) - base_head,
        val_tail["sale_amount"].to_numpy(float) - base_tail,
    ])
    tr_head, va_tail, te = align_categories([head_x, tail_x, test_x], cats)
    params = {
        "objective": "regression",
        "metric": "l1",
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_data_in_leaf": 80,
        "lambda_l2": 5.0,
        "verbosity": -1,
        "seed": SEED,
        "num_threads": 4,
        "deterministic": True,
        "force_col_wise": True,
    }
    dtrain = lgb.Dataset(
        tr_head, label=val_head["sale_amount"].to_numpy(float) - base_head,
        categorical_feature=cats, free_raw_data=False,
    )
    dvalid = lgb.Dataset(
        va_tail, label=val_tail["sale_amount"].to_numpy(float) - base_tail,
        categorical_feature=cats, reference=dtrain, free_raw_data=False,
    )
    model = lgb.train(
        params, dtrain, num_boost_round=300, valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    best_iter = int(model.best_iteration or 300)
    tail_corrected = np.maximum(
        0.0, base_tail + model.predict(va_tail, num_iteration=best_iter)
    )

    full_x, test_x_aligned = align_categories([all_valid_x, augmented(test, base_test)], cats)
    full_model = lgb.train(
        params,
        lgb.Dataset(full_x, label=all_valid_y, categorical_feature=cats, free_raw_data=False),
        num_boost_round=best_iter,
    )
    test_corrected = np.maximum(
        0.0, base_test + full_model.predict(test_x_aligned, num_iteration=best_iter)
    )
    return tail_corrected, test_corrected, best_iter


def select_ensemble_weights(
    actual: np.ndarray, candidates: dict[str, np.ndarray]
) -> dict[str, float]:
    names = list(candidates)
    matrix = np.column_stack([candidates[name] for name in names])
    n_models = len(names)
    start = np.full(n_models, 1.0 / n_models)

    def objective(weights):
        residual = np.asarray(actual, dtype=float) - matrix @ weights
        return float(np.mean(residual ** 2) + 1e-7 * np.sum(weights ** 2))

    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints=[{"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}],
        options={"maxiter": 500, "ftol": 1e-10},
    )
    weights = result.x if result.success else start
    weights = np.maximum(weights, 0.0)
    weights /= max(float(weights.sum()), 1e-12)
    return {name: float(weight) for name, weight in zip(names, weights)}


def two_way_bootstrap(
    predictions: pd.DataFrame,
    baseline: str,
    candidates: list[str],
) -> dict[str, Any]:
    """Paired store-product cluster and contiguous two-day date-block bootstrap."""
    frame = predictions.reset_index(drop=True)
    y = frame["actual"].to_numpy(float)
    base = frame[baseline].to_numpy(float)
    series = (frame.store_id.astype(str) + "::" + frame.product_id.astype(str)).to_numpy()
    unique_series, series_index = np.unique(series, return_inverse=True)
    dates = pd.to_datetime(frame.date).dt.normalize().to_numpy()
    unique_dates = np.array(sorted(pd.unique(dates)))
    date_index = np.searchsorted(unique_dates, dates)
    blocks = [np.arange(i, min(i + 2, len(unique_dates))) for i in range(0, len(unique_dates), 2)]
    rng = np.random.default_rng(SEED)
    draws = {candidate: {"mae": [], "rmse": [], "wape": []} for candidate in candidates}
    for _ in range(BOOTSTRAP_REPS):
        series_weights = np.bincount(
            rng.integers(0, len(unique_series), len(unique_series)),
            minlength=len(unique_series),
        )
        date_weights = np.zeros(len(unique_dates), dtype=float)
        for block_index in rng.integers(0, len(blocks), len(blocks)):
            date_weights[blocks[block_index]] += 1.0
        weights = series_weights[series_index] * date_weights[date_index]
        if weights.sum() <= 0:
            continue
        denominator = max(float(np.sum(weights * np.abs(y))), 1e-12)
        base_abs = np.abs(y - base)
        base_metrics = {
            "mae": float(np.sum(weights * base_abs) / weights.sum()),
            "rmse": float(np.sqrt(np.sum(weights * np.square(y - base)) / weights.sum())),
            "wape": float(np.sum(weights * base_abs) / denominator),
        }
        for candidate in candidates:
            pred = frame[candidate].to_numpy(float)
            errors = np.abs(y - pred)
            metric_values = {
                "mae": float(np.sum(weights * errors) / weights.sum()),
                "rmse": float(np.sqrt(np.sum(weights * np.square(y - pred)) / weights.sum())),
                "wape": float(np.sum(weights * errors) / denominator),
            }
            for metric in draws[candidate]:
                improvement = (
                    (base_metrics[metric] - metric_values[metric])
                    / max(base_metrics[metric], 1e-12) * 100
                )
                draws[candidate][metric].append(improvement)
    output = {}
    for candidate, metrics in draws.items():
        output[candidate] = {}
        for metric, values in metrics.items():
            values = np.asarray(values, dtype=float)
            low, high = np.percentile(values, [2.5, 97.5])
            output[candidate][metric] = {
                "mean_improvement_pct": float(values.mean()),
                "ci_95_low_pct": float(low),
                "ci_95_high_pct": float(high),
                "probability_improvement": float(np.mean(values > 0)),
                "ci_excludes_zero": bool(low > 0 or high < 0),
            }
    return {
        "method": "paired two-way bootstrap: store-product cluster and contiguous two-day date blocks",
        "replicates": BOOTSTRAP_REPS,
        "seed": SEED,
        "series_clusters": int(len(unique_series)),
        "evaluation_dates": int(len(unique_dates)),
        "comparisons_vs_baseline": output,
        "caveat": (
            "The original evaluation split contains only seven dates. Date-block uncertainty "
            "is therefore coarse and does not replace a genuinely later untouched temporal holdout."
        ),
    }


def score_prediction_frame(frame: pd.DataFrame, candidate_columns: list[str]) -> list[dict[str, Any]]:
    actual = frame["actual"].to_numpy(float)
    return [
        {"model": name, **metric_pack(actual, frame[name].to_numpy(float))}
        for name in candidate_columns
    ]


def run_fold(
    base_panel: pd.DataFrame,
    raw_panel: pd.DataFrame,
    wavelet_panel: pd.DataFrame,
    train_dates: list[pd.Timestamp],
    valid_dates: list[pd.Timestamp],
    test_dates: list[pd.Timestamp],
    fold_label: str,
):
    train_origin = train_dates[0]
    valid_origin = valid_dates[0]
    test_origin = test_dates[0]
    train = attach_origin_features(base_panel, raw_panel, wavelet_panel, train_dates, train_origin)
    valid = attach_origin_features(base_panel, raw_panel, wavelet_panel, valid_dates, valid_origin)
    test = attach_origin_features(base_panel, raw_panel, wavelet_panel, test_dates, test_origin)

    split = max(1, len(valid_dates) // 2)
    head_dates = valid_dates[:split]
    tail_dates = valid_dates[split:]
    if not head_dates or not tail_dates:
        raise RuntimeError(f"{fold_label}: validation must have at least two target dates")
    val_head = valid[valid["date"].isin(head_dates)].copy()
    val_tail = valid[valid["date"].isin(tail_dates)].copy()
    if train.empty or val_head.empty or val_tail.empty or test.empty:
        raise RuntimeError(f"{fold_label}: empty chronological split")

    candidates = {}
    candidate_heads = {}
    test_predictions = {}
    feature_gains = {}
    iteration_counts = {}
    for model_name, features in MODEL_FEATURES.items():
        features = [column for column in features if column in train.columns]
        fitted = fit_model(
            train, val_head, val_tail, test, features, model_name
        )
        candidates[model_name] = fitted["pred_tail"]
        candidate_heads[model_name] = fitted["pred_head"]
        test_predictions[model_name] = fitted["pred_test"]
        feature_gains[model_name] = fitted["feature_gain"]
        iteration_counts[model_name] = fitted["best_iteration"]

    spectral_features = [column for column in MODEL_FEATURES["spectral"] if column in train.columns]
    residual_tail, residual_test, residual_iterations = residual_correction(
        train,
        val_head,
        val_tail,
        test,
        candidate_heads["spectral"],
        candidates["spectral"],
        test_predictions["spectral"],
        spectral_features,
    )
    candidates["residual_corrected"] = residual_tail
    test_predictions["residual_corrected"] = residual_test

    selection_actual = val_tail["sale_amount"].to_numpy(float)
    weights = select_ensemble_weights(selection_actual, candidates)
    ensemble_prediction = sum(
        weights[name] * test_predictions[name] for name in weights
    )
    test_output = test[GROUP + ["date", "sale_amount"]].copy()
    test_output = test_output.rename(columns={"sale_amount": "actual"})
    for name, prediction in test_predictions.items():
        test_output[name] = prediction
    test_output["ensemble"] = ensemble_prediction
    test_output["fold"] = fold_label

    validation_metrics = score_prediction_frame(
        val_tail.assign(actual=val_tail["sale_amount"].to_numpy(float), **{
            name: prediction for name, prediction in candidates.items()
        }),
        list(candidates),
    )
    test_metrics = score_prediction_frame(
        test_output.rename(columns={"actual": "actual"}),
        list(test_predictions) + ["ensemble"],
    )
    summary = {
        "fold": fold_label,
        "training_target_dates": [str(min(train_dates).date()), str(max(train_dates).date())],
        "validation_dates": [str(min(valid_dates).date()), str(max(valid_dates).date())],
        "validation_selection_dates": [str(min(tail_dates).date()), str(max(tail_dates).date())],
        "test_dates": [str(min(test_dates).date()), str(max(test_dates).date())],
        "train_rows": int(len(train)),
        "validation_head_rows": int(len(val_head)),
        "validation_selection_rows": int(len(val_tail)),
        "test_rows": int(len(test)),
        "best_iterations": iteration_counts,
        "residual_correction_best_iteration": residual_iterations,
        "ensemble_weights_selected_on_validation_tail": weights,
        "validation_selection_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "wavelet_ablation": {
            "reference_model": "spectral",
            "wavelet_augmented_model": "wavelet",
            "interpretation": "wavelet adds db2 trailing-32-day coefficient features to the spectral feature set",
        },
        "feature_gain": {
            name: sorted(
                [{"feature": key, "gain": value} for key, value in gains.items()],
                key=lambda item: item["gain"], reverse=True,
            )[:30]
            for name, gains in feature_gains.items()
        },
    }
    return test_output, summary


def main():
    bucket = env("R2_BUCKET")
    s3 = r2_client()
    with tempfile.TemporaryDirectory(prefix="frn-forecast-research-") as temporary:
        root = Path(temporary)
        frames, train_dates_raw, eval_dates_raw, n_series, source_hashes = load_deterministic_panel(
            s3, bucket, root
        )
        train_dates = [pd.Timestamp(value) for value in train_dates_raw[:90]]
        eval_dates = [pd.Timestamp(value) for value in eval_dates_raw[:7]]
        raw_polars = pl.concat([frames["train"], frames["eval"]], how="vertical_relaxed").sort(GROUP + ["date"])
        base_features = load_features_from_frame(
            raw_polars.drop("date")
        )
        base_panel = base_features.to_pandas()
        base_panel["date"] = pd.to_datetime(base_panel["date"])
        base_panel = base_panel.sort_values(GROUP + ["date"]).reset_index(drop=True)

        raw_panel = (
            raw_polars.select(GROUP + ["date", "sale_amount"])
            .to_pandas()
        )
        raw_panel["date"] = pd.to_datetime(raw_panel["date"])
        raw_panel = raw_panel.sort_values(GROUP + ["date"]).reset_index(drop=True)

        base_columns = [column for column in BASE_FEATURES if column in base_panel.columns]
        missing_base = sorted(set(BASE_FEATURES) - set(base_columns))
        if missing_base:
            raise RuntimeError(f"Base feature contract missing columns: {missing_base}")
        # Store/product identifiers are also part of BASE_FEATURES; preserve
        # each source column only once to keep Pandas groupby/joins well-defined.
        selected_columns = list(dict.fromkeys(GROUP + ["date", "sale_amount"] + base_columns))
        base_panel = base_panel[selected_columns].copy()

        wavelet_panel = build_wavelet_features(
            raw_panel[GROUP + ["date", "sale_amount"]], WAVELET_FAMILY
        )
        wavelet_panel["date"] = pd.to_datetime(wavelet_panel["date"])
        wavelet_panel = wavelet_panel[GROUP + ["date"] + WAVELET_FEATURES]
        if wavelet_panel.duplicated(GROUP + ["date"]).any():
            raise RuntimeError("Duplicate wavelet feature keys")
        print(
            f"Sampled {n_series:,} complete series; built {len(wavelet_panel):,} causal wavelet rows",
            flush=True,
        )

        fold_definitions = [
            (62, 69, 76, "rolling_fold_1"),
            (69, 76, 83, "rolling_fold_2"),
            (76, 83, 90, "rolling_fold_3"),
        ]
        all_predictions = []
        fold_reports = []
        for train_end, valid_end, test_end, label in fold_definitions:
            train_target_dates = train_dates[WARMUP_DAYS:train_end]
            valid_target_dates = train_dates[train_end:valid_end]
            test_target_dates = train_dates[valid_end:test_end]
            prediction, report = run_fold(
                base_panel, raw_panel, wavelet_panel,
                train_target_dates, valid_target_dates, test_target_dates, label,
            )
            all_predictions.append(prediction)
            fold_reports.append(report)

        # Final comparison reproduces the source benchmark's split: 76 target
        # dates for fitting (after warmup), 14 validation dates, seven evaluation dates.
        final_train_dates = train_dates[WARMUP_DAYS:76]
        final_valid_dates = train_dates[76:90]
        final_eval_dates = eval_dates
        final_predictions, final_report = run_fold(
            base_panel, raw_panel, wavelet_panel,
            final_train_dates, final_valid_dates, final_eval_dates,
            "original_seven_day_evaluation",
        )
        rolling_predictions = pd.concat(all_predictions, ignore_index=True)
        rolling_candidates = [
            "baseline", "fourier_stl", "spectral", "wavelet",
            "residual_corrected", "ensemble",
        ]
        rolling_metrics = score_prediction_frame(rolling_predictions, rolling_candidates)
        final_metrics = score_prediction_frame(final_predictions, rolling_candidates)
        bootstrap = two_way_bootstrap(
            final_predictions,
            "baseline",
            [name for name in rolling_candidates if name != "baseline"],
        )
        wavelet_ablation_bootstrap = two_way_bootstrap(
            final_predictions,
            "spectral",
            ["wavelet"],
        )
        final_weights = final_report["ensemble_weights_selected_on_validation_tail"]
        summary = {
            "project": "FreshRetailNet Demand Intelligence",
            "experiment": "STL + Fourier + spectral + wavelet ablation + residual correction + validation-selected ensemble",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository_commit": os.getenv("GITHUB_SHA"),
            "workflow_run_id": os.getenv("GITHUB_RUN_ID"),
            "seed": SEED,
            "series_sample_size_requested": SERIES_SAMPLE,
            "series_sample_size_complete": n_series,
            "sample_policy": "deterministic random sample of complete train/evaluation series, sorted before panel construction",
            "source_sha256": source_hashes,
            "source_rows": {
                "train": int(frames["train"].height),
                "evaluation": int(frames["eval"].height),
            },
            "rolling_design": {
                "warmup_days": WARMUP_DAYS,
                "folds": fold_reports,
                "pooled_test_metrics": rolling_metrics,
                "note": "Three adjacent seven-day rolling test blocks inside the 90-day training partition; each has a preceding validation window and expanding training history.",
            },
            "final_research_comparison": {
                "validation_period": [str(final_valid_dates[0].date()), str(final_valid_dates[-1].date())],
                "evaluation_period": [str(final_eval_dates[0].date()), str(final_eval_dates[-1].date())],
                "evaluation_rows": int(len(final_predictions)),
                "metrics": final_metrics,
                "ensemble_weights": final_weights,
                "bootstrap_vs_baseline": bootstrap,
                "wavelet_ablation_vs_spectral": wavelet_ablation_bootstrap,
                "status": "research comparison on the repository's previously used seven-day holdout; not an untouched production promotion test",
            },
            "methods": {
                "stl": "robust STL, period=7, fit on the 56 observations before each forecast origin; trend, weekday seasonal, and residual summaries",
                "fourier": "three weekly harmonics plus 30.4375-day and 365.25-day calendar harmonics",
                "spectral": "periodogram diagnostics on pre-origin history: dominant 2-28-day period, 6-8-day energy share, normalized entropy and peak share",
                "wavelet": "db2 discrete wavelet decomposition on trailing 32 observed sales strictly before each target date; six energy/entropy/ratio/smoothed-mean features",
                "residual_model": "LightGBM residual correction trained on the first half of validation residuals, checked on the second half, then refit on all validation residuals for test scoring",
                "ensemble": "non-negative simplex weights minimizing validation-tail squared error; weights frozen for test dates",
            },
            "feature_availability": {
                "excluded_target_day_fields": [
                    "discount", "holiday_flag", "activity_flag", "precpt",
                    "avg_temperature", "avg_humidity", "avg_wind_level",
                    "hours_sale", "hours_stock_status", "stock_hour6_22_cnt",
                ],
                "included_history": "lagged observed sales and lagged stock-status aggregates only",
            },
            "limitations": [
                "The original seven-day evaluation split was already used by prior repository benchmark reports; treat this as a research comparison, not a fresh independent holdout.",
                "STL/spectral origin features are fixed within each training/validation/test block using only history before that block's origin.",
                "The deterministic 3,000-series sample reduces computation and results should not be represented as full-population metrics.",
                "Observed sales may be stockout-censored; target is not identified latent demand.",
                "Fourier monthly/yearly harmonics are calendar basis functions; the 90-day source history is too short to establish annual seasonality.",
                "Two-way bootstrap time uncertainty is coarse because the final evaluation has seven dates.",
            ],
        }

        outputs = {
            "forecast_research_summary.json": summary,
            "forecast_research_rolling_predictions.csv": rolling_predictions,
            "forecast_research_final_predictions.csv": final_predictions,
            "forecast_research_rolling_comparison.csv": pd.DataFrame(rolling_metrics),
            "forecast_research_final_comparison.csv": pd.DataFrame(final_metrics),
        }
        for filename, payload in outputs.items():
            path = root / filename
            if filename.endswith(".csv"):
                payload.to_csv(path, index=False)
                content_type = "text/csv"
            else:
                path.write_text(json.dumps(payload, indent=2, default=str, allow_nan=False))
                content_type = "application/json"
            key = f"{OUT}/{filename}"
            s3.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": content_type})
            verified = s3.head_object(Bucket=bucket, Key=key)
            if int(verified["ContentLength"]) != path.stat().st_size:
                raise RuntimeError(f"R2 upload verification failed for {key}")
            print(f"Verified s3://{bucket}/{key} ({verified['ContentLength']:,} bytes)", flush=True)

        print(json.dumps({
            "sample_series": n_series,
            "rolling_metrics": rolling_metrics,
            "final_metrics": final_metrics,
            "ensemble_weights": final_weights,
            "bootstrap": bootstrap,
            "wavelet_ablation_vs_spectral": wavelet_ablation_bootstrap,
            "report_key": f"{OUT}/forecast_research_summary.json",
        }, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
