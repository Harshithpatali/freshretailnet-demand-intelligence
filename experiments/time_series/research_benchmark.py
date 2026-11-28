"""Reproducible time-series research benchmark for FreshRetailNet.

Benchmarks causal STL, Fourier, spectral and wavelet features; performs a
same-row wavelet ablation; learns a residual correction from a disjoint part
of the validation window; and selects non-negative ensemble weights using
only the remaining validation dates. The official evaluation partition is
scored once after all model/ensemble decisions are frozen.

A deterministic store-product sample keeps the STL experiment bounded. Results
must not be represented as full-population evidence.
"""
from __future__ import annotations

import json
import os
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
import sys

import boto3
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from botocore.config import Config
from scipy.optimize import minimize
from statsmodels.tsa.seasonal import STL

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.train_model import RAW, env, load_features_from_frame, metric_pack
from experiments.wavelet.features import build_wavelet_features
from scripts.artifact_provenance import build_provenance, upload_versioned_artifact

OUT = "timeseries-research-artifacts"
SEED = 20261128
SAMPLE_SERIES = 1500
STL_WINDOW = 28
WAVELET_WINDOW = 32
BOOT_REPS = 500
CAT_COLUMNS = (
    "city_id", "store_id", "management_group_id", "first_category_id",
    "second_category_id", "third_category_id", "product_id", "day_of_week", "month",
)


def r2_client():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{env('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
        aws_access_key_id=env("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=env("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
    )


def add_fourier_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic weekly and annual calendar harmonics."""
    out = frame.copy()
    dates = pd.to_datetime(out["date"])
    periods = {
        "week": (dates.dt.dayofweek.to_numpy(dtype=float), 7.0, 3),
        "year": (dates.dt.dayofyear.to_numpy(dtype=float), 365.25, 2),
    }
    for name, (position, period, max_harmonic) in periods.items():
        for harmonic in range(1, max_harmonic + 1):
            angle = 2.0 * np.pi * harmonic * position / period
            out[f"fourier_{name}_sin_{harmonic}"] = np.sin(angle)
            out[f"fourier_{name}_cos_{harmonic}"] = np.cos(angle)
    return out


def build_stl_spectral_features(
    raw: pd.DataFrame, window: int = STL_WINDOW, period: int = 7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build STL and FFT features for date t from the immediately prior window only."""
    required = {"store_id", "product_id", "date", "sale_amount"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing STL/spectral columns: {sorted(missing)}")
    if window < 2 * period:
        raise ValueError("STL window must contain at least two seasonal cycles")
    chunks = []
    diagnostic_rows = []
    for (store, product), group in raw.sort_values(
        ["store_id", "product_id", "date"]
    ).groupby(["store_id", "product_id"], sort=False):
        values = group["sale_amount"].to_numpy(dtype=float)
        dates = pd.to_datetime(group["date"]).to_numpy()
        if len(values) <= window:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(values, window)[:-1]
        demeaned = windows - windows.mean(axis=1, keepdims=True)
        spectrum = np.fft.rfft(demeaned, axis=1)
        power = np.square(np.abs(spectrum))
        non_dc = power[:, 1:]
        total_power = np.maximum(non_dc.sum(axis=1), 1e-12)
        probabilities = non_dc / total_power[:, None]
        entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1)
        frequencies = np.fft.rfftfreq(window, d=1.0)[1:]
        dominant = np.argmax(non_dc, axis=1) + 1
        dominant_period = window / dominant.astype(float)
        weekly_bin = int(round(window / period)) - 1
        weekly_share = non_dc[:, weekly_bin] / total_power if weekly_bin < non_dc.shape[1] else np.zeros(len(windows))
        low_frequency_share = non_dc[:, : min(2, non_dc.shape[1])].sum(axis=1) / total_power
        spectral_centroid = (probabilities * frequencies[None, :]).sum(axis=1)

        stl_values = []
        for history in windows:
            # The STL window is strictly prior to its corresponding target date.
            fit = STL(history, period=period, robust=False).fit()
            seasonal = np.asarray(fit.seasonal, dtype=float)
            trend = np.asarray(fit.trend, dtype=float)
            resid = np.asarray(fit.resid, dtype=float)
            denom = float(np.var(seasonal + resid))
            seasonal_strength = (
                max(0.0, 1.0 - float(np.var(resid)) / denom) if denom > 1e-12 else 0.0
            )
            stl_values.append([
                float(trend[-1]),
                float(trend[-1] - trend[-8]),
                float(seasonal[-period]),
                float(np.std(resid)),
                seasonal_strength,
            ])
        stl_array = np.asarray(stl_values, dtype=float)
        features = pd.DataFrame({
            "store_id": str(store),
            "product_id": str(product),
            "date": pd.to_datetime(dates[window:]),
            "stl_trend_last": stl_array[:, 0],
            "stl_trend_slope_7": stl_array[:, 1],
            "stl_next_day_seasonal": stl_array[:, 2],
            "stl_residual_std": stl_array[:, 3],
            "stl_seasonal_strength": stl_array[:, 4],
            "spectral_weekly_power_share": weekly_share,
            "spectral_low_frequency_power_share": low_frequency_share,
            "spectral_entropy": entropy,
            "spectral_dominant_period": dominant_period,
            "spectral_centroid": spectral_centroid,
        })
        chunks.append(features)
        diagnostic_rows.append({
            "store_id": str(store),
            "product_id": str(product),
            "mean_weekly_power_share": float(np.mean(weekly_share)),
            "mean_spectral_entropy": float(np.mean(entropy)),
            "mean_dominant_period": float(np.mean(dominant_period)),
            "mean_stl_seasonal_strength": float(np.mean(stl_array[:, 4])),
            "mean_stl_residual_std": float(np.mean(stl_array[:, 3])),
            "history_window": int(window),
            "seasonal_period": int(period),
        })
    if not chunks:
        raise ValueError("No eligible series for STL/spectral features")
    features = pd.concat(chunks, ignore_index=True)
    numeric = features.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("STL/spectral features contain non-finite values")
    return features, pd.DataFrame(diagnostic_rows)


def _align_categories(frames: list[pd.DataFrame], columns: list[str]) -> list[pd.DataFrame]:
    aligned = [frame.copy() for frame in frames]
    for column in columns:
        levels = pd.Index(pd.concat([frame[column].astype(str) for frame in aligned]).unique())
        dtype = pd.CategoricalDtype(categories=levels)
        for frame in aligned:
            frame[column] = frame[column].astype(str).astype(dtype)
    return aligned


def fit_lightgbm(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    evaluation: pd.DataFrame,
    features: list[str],
    categories: list[str],
    label: str,
) -> tuple[lgb.Booster, np.ndarray, np.ndarray]:
    tr, va, ev = _align_categories([train, valid, evaluation], categories)
    model = lgb.train(
        {
            "objective": "regression", "metric": "l1", "learning_rate": 0.04,
            "num_leaves": 31, "max_depth": -1, "min_data_in_leaf": 80,
            "feature_fraction": 0.90, "bagging_fraction": 0.85, "bagging_freq": 1,
            "lambda_l2": 5.0, "verbosity": -1, "seed": SEED, "num_threads": 4,
        },
        lgb.Dataset(tr[features], label=tr["sale_amount"], categorical_feature=categories, free_raw_data=False),
        num_boost_round=900,
        valid_sets=[lgb.Dataset(va[features], label=va["sale_amount"], categorical_feature=categories)],
        callbacks=[lgb.early_stopping(70, verbose=False)],
    )
    pv = np.maximum(model.predict(va[features], num_iteration=model.best_iteration), 0.0)
    pe = np.maximum(model.predict(ev[features], num_iteration=model.best_iteration), 0.0)
    print(f"{label}: best_iteration={model.best_iteration}")
    return model, pv, pe


def fit_residual_model(
    fit_frame: pd.DataFrame,
    base_predictions: np.ndarray,
    score_frame: pd.DataFrame,
    score_base_predictions: np.ndarray,
    features: list[str],
    categories: list[str],
    label: str,
) -> np.ndarray:
    """Fit a small residual learner to a prior validation block and score a later block."""
    train = fit_frame.copy()
    score = score_frame.copy()
    train["__base_prediction"] = np.maximum(base_predictions, 0.0)
    score["__base_prediction"] = np.maximum(score_base_predictions, 0.0)
    target_residual = train["sale_amount"].to_numpy(dtype=float) - train["__base_prediction"].to_numpy(dtype=float)
    model_features = features + ["__base_prediction"]
    tr, sc = _align_categories([train, score], categories)
    residual_model = lgb.train(
        {
            "objective": "regression", "metric": "l1", "learning_rate": 0.025,
            "num_leaves": 15, "min_data_in_leaf": 100, "feature_fraction": 0.9,
            "lambda_l2": 10.0, "verbosity": -1, "seed": SEED + 1, "num_threads": 4,
        },
        lgb.Dataset(tr[model_features], label=target_residual, categorical_feature=categories, free_raw_data=False),
        num_boost_round=350,
    )
    correction = residual_model.predict(sc[model_features])
    return np.maximum(score_base_predictions + correction, 0.0)


def select_ensemble_weights(
    actual: np.ndarray, predictions: dict[str, np.ndarray]
) -> dict[str, float]:
    """Choose non-negative weights summing to one by validation MAE only."""
    names = list(predictions)
    matrix = np.column_stack([np.asarray(predictions[name], dtype=float) for name in names])
    y = np.asarray(actual, dtype=float)

    def objective(weights):
        combined = np.maximum(matrix @ weights, 0.0)
        return float(np.mean(np.abs(y - combined)) + 1e-8 * np.square(weights).sum())

    n = len(names)
    result = minimize(
        objective,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)}],
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Ensemble weight optimization failed: {result.message}")
    return {name: float(weight) for name, weight in zip(names, result.x)}


def paired_cluster_bootstrap(
    frame: pd.DataFrame, baseline_col: str, candidate_cols: list[str], reps: int = BOOT_REPS
) -> dict:
    """Paired cluster bootstrap, resampling whole sampled store-product series."""
    ids = (frame["store_id"].astype(str) + "::" + frame["product_id"].astype(str)).to_numpy()
    unique_ids, inverse = np.unique(ids, return_inverse=True)
    members = [np.flatnonzero(inverse == index) for index in range(len(unique_ids))]
    y = frame["actual_sale_amount"].to_numpy(dtype=float)
    baseline = frame[baseline_col].to_numpy(dtype=float)
    rng = np.random.default_rng(SEED)
    draws = {name: {"mae": [], "rmse": [], "wape": []} for name in candidate_cols}
    for _ in range(reps):
        chosen = rng.integers(0, len(members), size=len(members))
        idx = np.concatenate([members[index] for index in chosen])
        denominator = max(float(np.abs(y[idx]).sum()), 1e-12)
        base_abs = np.abs(y[idx] - baseline[idx])
        base = {
            "mae": float(base_abs.mean()),
            "rmse": float(np.sqrt(np.square(y[idx] - baseline[idx]).mean())),
            "wape": float(base_abs.sum() / denominator),
        }
        for name in candidate_cols:
            pred = frame[name].to_numpy(dtype=float)[idx]
            error = np.abs(y[idx] - pred)
            candidate = {
                "mae": float(error.mean()),
                "rmse": float(np.sqrt(np.square(y[idx] - pred).mean())),
                "wape": float(error.sum() / denominator),
            }
            for metric in draws[name]:
                draws[name][metric].append(
                    (base[metric] - candidate[metric]) / max(base[metric], 1e-12) * 100
                )
    output = {}
    for name, metrics in draws.items():
        output[name] = {}
        for metric, values in metrics.items():
            values = np.asarray(values, dtype=float)
            low, high = np.percentile(values, [2.5, 97.5])
            output[name][metric] = {
                "improvement_pct": float(np.mean(values)),
                "ci_95_low_pct": float(low),
                "ci_95_high_pct": float(high),
                "probability_improvement": float(np.mean(values > 0)),
                "interval_excludes_zero": bool(low > 0 or high < 0),
            }
    return {
        "method": "paired store-product cluster bootstrap; all evaluation dates retained within each sampled cluster",
        "replicates": reps,
        "seed": SEED,
        "clusters": int(len(unique_ids)),
        "caveat": "This interval quantifies cross-sectional series uncertainty conditional on the seven evaluation dates; it does not quantify uncertainty over future time windows or shared date shocks.",
        "comparisons_vs_baseline": output,
    }


def main() -> None:
    bucket = env("R2_BUCKET")
    s3 = r2_client()
    with tempfile.TemporaryDirectory(prefix="frn-timeseries-research-") as td:
        root = Path(td)
        raw_splits = {}
        for split, key in RAW.items():
            path = root / f"{split}.parquet"
            print(f"Downloading {key}")
            s3.download_file(bucket, key, str(path))
            raw_splits[split] = pl.read_parquet(path)

        series = raw_splits["train"].select(["store_id", "product_id"]).unique().to_pandas()
        if len(series) < SAMPLE_SERIES:
            sample_keys = series
        else:
            sample_keys = series.sample(n=SAMPLE_SERIES, random_state=SEED).sort_values(
                ["store_id", "product_id"]
            ).reset_index(drop=True)
        sample_pl = pl.from_pandas(sample_keys)
        combined_raw = pl.concat([raw_splits["train"], raw_splits["eval"]], how="vertical_relaxed")
        sampled_raw = combined_raw.join(sample_pl, on=["store_id", "product_id"], how="semi")
        print(f"Deterministic cohort: {len(sample_keys):,} of {len(series):,} available store-product series")

        evaluation_dates = set(raw_splits["eval"].get_column("dt").unique().to_list())
        base = load_features_from_frame(sampled_raw).to_pandas()
        base["date"] = pd.to_datetime(base["date"])
        base = add_fourier_features(base)

        history = sampled_raw.select(["store_id", "product_id", "dt", "sale_amount"]).with_columns(
            pl.col("dt").str.to_date(strict=False).alias("date")
        ).drop("dt").to_pandas()
        stl_spectral, diagnostics = build_stl_spectral_features(history, STL_WINDOW, 7)
        stl_wave_input = history.rename(columns={"date": "date"})
        wavelet = build_wavelet_features(stl_wave_input, wavelet_name="db2")
        print(f"Joining leakage-safe feature families: STL/spectral={len(stl_spectral):,}, wavelet={len(wavelet):,}")

        frame = base.merge(stl_spectral, on=["store_id", "product_id", "date"], how="inner", validate="one_to_one")
        frame = frame.merge(wavelet, on=["store_id", "product_id", "date"], how="inner", validate="one_to_one")
        frame = frame.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)
        feature_families = {
            "base": [c for c in base.columns if c not in {"date", "sale_amount"}],
            "fourier": [c for c in base.columns if c.startswith("fourier_")],
            "stl": [c for c in frame.columns if c.startswith("stl_")],
            "spectral": [c for c in frame.columns if c.startswith("spectral_")],
            "wavelet": [c for c in frame.columns if c.startswith("wavelet_")],
        }
        key_cols = ["date", "store_id", "product_id", "sale_amount"]
        base_features = feature_families["base"]
        stl_fourier_features = base_features + feature_families["fourier"] + feature_families["stl"]
        spectral_no_wavelet_features = stl_fourier_features + feature_families["spectral"]
        full_features = spectral_no_wavelet_features + feature_families["wavelet"]
        if not feature_families["wavelet"]:
            raise RuntimeError("Wavelet feature generator returned no features")

        train_dates = sorted(pd.to_datetime(frame.loc[~frame["date"].dt.strftime("%Y-%m-%d").isin(evaluation_dates), "date"].unique()))
        validation_dates = train_dates[-14:]
        training_dates = train_dates[:-14]
        validation_dates_sorted = sorted(validation_dates)
        if len(validation_dates_sorted) < 4 or len(training_dates) < 10:
            raise RuntimeError("Insufficient chronological rows after STL/wavelet history eligibility")
        validation_mid = len(validation_dates_sorted) // 2
        residual_fit_dates = set(validation_dates_sorted[:validation_mid])
        ensemble_validation_dates = set(validation_dates_sorted[validation_mid:])
        train = frame[frame["date"].isin(training_dates)].copy()
        valid = frame[frame["date"].isin(validation_dates_sorted)].copy()
        evaluation = frame[frame["date"].dt.strftime("%Y-%m-%d").isin(evaluation_dates)].copy()
        if train.empty or valid.empty or evaluation.empty:
            raise RuntimeError("One or more chronological modelling splits are empty")

        categories = [c for c in CAT_COLUMNS if c in full_features]
        predictions: dict[str, dict[str, np.ndarray]] = {}
        models = {}
        model_specs = [
            ("baseline_lightgbm", base_features),
            ("stl_fourier_lightgbm", stl_fourier_features),
            ("spectral_no_wavelet_lightgbm", spectral_no_wavelet_features),
            ("spectral_wavelet_lightgbm", full_features),
        ]
        y_validation = valid["sale_amount"].to_numpy(dtype=float)
        y_evaluation = evaluation["sale_amount"].to_numpy(dtype=float)
        for name, feature_cols in model_specs:
            model, valid_pred, evaluation_pred = fit_lightgbm(
                train, valid, evaluation, feature_cols, categories, name
            )
            models[name] = model
            predictions[name] = {"validation": valid_pred, "evaluation": evaluation_pred}

        valid_fit_mask = valid["date"].isin(residual_fit_dates).to_numpy()
        valid_select_mask = valid["date"].isin(ensemble_validation_dates).to_numpy()
        if not valid_fit_mask.any() or not valid_select_mask.any():
            raise RuntimeError("Validation window could not be split for residual/ensemble selection")

        full_valid_pred = predictions["spectral_wavelet_lightgbm"]["validation"]
        full_eval_pred = predictions["spectral_wavelet_lightgbm"]["evaluation"]
        residual_validation_select = fit_residual_model(
            valid.loc[valid_fit_mask], full_valid_pred[valid_fit_mask],
            valid.loc[valid_select_mask], full_valid_pred[valid_select_mask],
            full_features, categories, "residual model (validation first-half -> second-half)",
        )
        residual_evaluation = fit_residual_model(
            valid, full_valid_pred, evaluation, full_eval_pred,
            full_features, categories, "residual model (full validation -> evaluation)",
        )
        # Build the validation-only model pool; no evaluation labels enter weight selection.
        validation_selection_predictions = {
            "seasonal_naive_lag7": valid.loc[valid_select_mask, "sales_lag_7"].to_numpy(dtype=float),
            "baseline_lightgbm": predictions["baseline_lightgbm"]["validation"][valid_select_mask],
            "stl_fourier_lightgbm": predictions["stl_fourier_lightgbm"]["validation"][valid_select_mask],
            "spectral_no_wavelet_lightgbm": predictions["spectral_no_wavelet_lightgbm"]["validation"][valid_select_mask],
            "spectral_wavelet_lightgbm": full_valid_pred[valid_select_mask],
            "residual_corrected_spectral_wavelet": residual_validation_select,
        }
        weights = select_ensemble_weights(
            valid.loc[valid_select_mask, "sale_amount"].to_numpy(dtype=float),
            validation_selection_predictions,
        )
        evaluation_predictions = {
            "seasonal_naive_lag7": evaluation["sales_lag_7"].to_numpy(dtype=float),
            "baseline_lightgbm": predictions["baseline_lightgbm"]["evaluation"],
            "stl_fourier_lightgbm": predictions["stl_fourier_lightgbm"]["evaluation"],
            "spectral_no_wavelet_lightgbm": predictions["spectral_no_wavelet_lightgbm"]["evaluation"],
            "spectral_wavelet_lightgbm": full_eval_pred,
            "residual_corrected_spectral_wavelet": residual_evaluation,
        }
        ensemble_eval = np.zeros(len(evaluation), dtype=float)
        for name, weight in weights.items():
            ensemble_eval += weight * evaluation_predictions[name]
        evaluation_predictions["validation_selected_ensemble"] = np.maximum(ensemble_eval, 0.0)

        comparison_rows = []
        for name, prediction in evaluation_predictions.items():
            comparison_rows.append({
                "model": name,
                **{f"evaluation_{key}": value for key, value in metric_pack(y_evaluation, prediction).items()},
                "validation_selection_weight": weights.get(name, 0.0),
            })
        comparison = pd.DataFrame(comparison_rows)
        ablation_rows = []
        no_wavelet = evaluation_predictions["spectral_no_wavelet_lightgbm"]
        with_wavelet = evaluation_predictions["spectral_wavelet_lightgbm"]
        no_metrics = metric_pack(y_evaluation, no_wavelet)
        wave_metrics = metric_pack(y_evaluation, with_wavelet)
        ablation_rows.append({
            "comparison": "spectral_stl_fourier_plus_wavelet_vs_no_wavelet",
            "no_wavelet_mae": no_metrics["mae"],
            "with_wavelet_mae": wave_metrics["mae"],
            "mae_change_pct": (wave_metrics["mae"] / max(no_metrics["mae"], 1e-12) - 1) * 100,
            "no_wavelet_wape_pct": no_metrics["wape_pct"],
            "with_wavelet_wape_pct": wave_metrics["wape_pct"],
            "wape_change_pct": (wave_metrics["wape_pct"] / max(no_metrics["wape_pct"], 1e-12) - 1) * 100,
            "negative_change_means_wavelet_improvement": True,
        })

        prediction_frame = evaluation[["date", "store_id", "product_id", "sale_amount"]].copy()
        prediction_frame = prediction_frame.rename(columns={"sale_amount": "actual_sale_amount"})
        for name, prediction in evaluation_predictions.items():
            prediction_frame[name] = prediction
        bootstrap = paired_cluster_bootstrap(
            prediction_frame,
            "baseline_lightgbm",
            ["stl_fourier_lightgbm", "spectral_no_wavelet_lightgbm",
             "spectral_wavelet_lightgbm", "residual_corrected_spectral_wavelet",
             "validation_selected_ensemble"],
        )
        diagnostics_summary = {
            "sampled_series": int(len(sample_keys)),
            "source_population_series": int(len(series)),
            "history_window_days": STL_WINDOW,
            "wavelet_window_days": WAVELET_WINDOW,
            "seasonal_period_days": 7,
            "feature_rows": int(len(frame)),
            "stl_spectral_feature_count": int(len(feature_families["stl"]) + len(feature_families["spectral"])),
            "wavelet_feature_count": int(len(feature_families["wavelet"])),
            "mean_stl_seasonal_strength": float(diagnostics["mean_stl_seasonal_strength"].mean()),
            "mean_spectral_weekly_power_share": float(diagnostics["mean_weekly_power_share"].mean()),
            "dominant_period_distribution": diagnostics["mean_dominant_period"].describe().to_dict(),
        }
        configuration = {
            "cohort": f"deterministic random sample of {SAMPLE_SERIES} series (seed {SEED})",
            "sampled_series": int(len(sample_keys)),
            "source_population_series": int(len(series)),
            "sample_seed": SEED,
            "stl": {"window_days": STL_WINDOW, "period_days": 7, "robust": False, "causal_policy": "history ends at t-1"},
            "fourier": {"weekly_harmonics": 3, "annual_harmonics": 2},
            "spectral": {"window_days": STL_WINDOW, "features": feature_families["spectral"]},
            "wavelet": {"family": "db2", "window_days": WAVELET_WINDOW, "causal_policy": "window ends at t-1"},
            "residual_model": "trained on first half of validation; evaluation residual correction is refit on all validation rows",
            "ensemble_selection": "non-negative simplex weights minimize MAE on second half of validation only",
            "validation_period": {"start": str(min(validation_dates_sorted).date()), "end": str(max(validation_dates_sorted).date())},
            "residual_fit_period": {"start": str(min(residual_fit_dates)), "end": str(max(residual_fit_dates))},
            "ensemble_selection_period": {"start": str(min(ensemble_validation_dates)), "end": str(max(ensemble_validation_dates))},
            "evaluation_period": {"start": str(evaluation["date"].min().date()), "end": str(evaluation["date"].max().date())},
            "training_rows": int(len(train)), "validation_rows": int(len(valid)), "evaluation_rows": int(len(evaluation)),
            "target": "sale_amount",
            "target_semantics": "Observed sales; may be censored by stock availability.",
            "feature_timing": "STL, spectral and wavelet windows use only observations strictly before target date.",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        summary = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": configuration,
            "feature_diagnostics": diagnostics_summary,
            "validation_ensemble_weights": weights,
            "comparison": comparison.to_dict(orient="records"),
            "wavelet_ablation": ablation_rows,
            "paired_cluster_bootstrap": bootstrap,
            "interpretation": "Model selection and residual fitting use chronological validation only. Evaluation metrics are reported once on the existing seven-day holdout and must not be treated as independent future-window evidence.",
            "limitations": [
                "The benchmark uses a deterministic 1,500-series sample, not the full 50,000-series population.",
                "The existing evaluation partition spans seven dates; uncertainty over future time windows is not estimated.",
                "STL is fit independently on a trailing 28-day history for each eligible target row; this is computationally bounded by the sampled cohort.",
                "Observed sales may be censored by stock availability.",
                "Wavelet/STL/spectral features cannot identify demand during stockouts.",
            ],
        }

        input_path = root / "evaluation_quantile_predictions.csv"
        # Data provenance records the public model prediction artifact used for comparison,
        # while source Parquet hashes are recorded by its stable R2 object key.
        raw_train = root / "train.parquet"
        raw_eval = root / "eval.parquet"
        # Re-download files are available through the initial R2 objects and hashed by the
        # artifact-provenance helper to provide immutable input references.
        s3.download_file(bucket, RAW["train"], str(raw_train))
        s3.download_file(bucket, RAW["eval"], str(raw_eval))
        output_keys = [
            f"{OUT}/timeseries_research_summary.json",
            f"{OUT}/timeseries_model_comparison.csv",
            f"{OUT}/timeseries_wavelet_ablation.csv",
            f"{OUT}/timeseries_evaluation_predictions.csv",
            f"{OUT}/timeseries_feature_diagnostics.csv",
            f"{OUT}/timeseries_ensemble_weights.json",
            f"{OUT}/timeseries_manifest.json",
        ]
        provenance = build_provenance(
            "Advanced Time-Series Research Benchmark",
            {RAW["train"]: raw_train, RAW["eval"]: raw_eval},
            str(evaluation["date"].min().date()),
            str(evaluation["date"].max().date()),
            configuration,
            outputs=output_keys,
        )
        summary["provenance"] = provenance
        artifacts = [
            (root / "timeseries_research_summary.json", json.dumps(summary, indent=2, default=str), "application/json"),
            (root / "timeseries_model_comparison.csv", comparison.to_csv(index=False), "text/csv"),
            (root / "timeseries_wavelet_ablation.csv", pd.DataFrame(ablation_rows).to_csv(index=False), "text/csv"),
            (root / "timeseries_evaluation_predictions.csv", prediction_frame.to_csv(index=False), "text/csv"),
            (root / "timeseries_feature_diagnostics.csv", diagnostics.to_csv(index=False), "text/csv"),
            (root / "timeseries_ensemble_weights.json", json.dumps(weights, indent=2), "application/json"),
            (root / "timeseries_manifest.json", json.dumps(provenance, indent=2), "application/json"),
        ]
        for path, payload, content_type in artifacts:
            path.write_text(payload, encoding="utf-8")
            base_key = f"{OUT}/{path.name}"
            destinations = upload_versioned_artifact(s3, bucket, path, base_key, content_type, provenance)
            head = s3.head_object(Bucket=bucket, Key=destinations[0])
            if int(head["ContentLength"]) != path.stat().st_size:
                raise RuntimeError(f"R2 size verification failed: {destinations[0]}")
            print(f"Verified {destinations[0]} ({head['ContentLength']:,} bytes)")

        print("ADVANCED_TIME_SERIES_RESEARCH_BENCHMARK")
        print(comparison.to_string(index=False))
        print("WAVELET_ABLATION")
        print(json.dumps(ablation_rows, indent=2))
        print("VALIDATION_SELECTED_ENSEMBLE_WEIGHTS")
        print(json.dumps(weights, indent=2))
        print("PAIRED_CLUSTER_BOOTSTRAP")
        print(json.dumps(bootstrap, indent=2))


if __name__ == "__main__":
    main()
