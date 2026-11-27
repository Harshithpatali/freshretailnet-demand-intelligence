"""Compare engineered LightGBM features with causal multi-signal wavelet features.

All features for target date t use observations strictly before t. Models share
one 64-observation-eligible row universe and identical chronological splits.
"""
from __future__ import annotations

import json
import tempfile
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import pywt
from botocore.config import Config

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.train_model import RAW, env, load_features_from_frame, metric_pack

OUT = "wavelet-engineered-artifacts"
GROUP = ["store_id", "product_id"]
WINDOWS = (32, 64)
FAMILIES = ("db2", "db4")
SIGNALS = ("sales", "sales_residual_7", "sales_momentum_7_28")
BOOT_REPS = 2000
BOOT_SEED = 20261128


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{env('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
        aws_access_key_id=env("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=env("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
    )


def add_engineered_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Add causal lag, rolling, intermittency, momentum and calendar features."""
    group = GROUP
    frame = frame.sort(GROUP + ["date"])
    expressions = []
    for lag in (2, 3, 14, 28, 56):
        expressions.append(pl.col("sale_amount").shift(lag).over(group).alias(f"sales_lag_{lag}"))
    shifted = pl.col("sale_amount").shift(1)
    for window in (14, 28, 56):
        expressions.append(
            shifted.rolling_mean(window_size=window, min_samples=1).over(group).alias(f"sales_roll_mean_{window}")
        )
        expressions.append(
            shifted.rolling_std(window_size=window, min_samples=2).over(group).fill_null(0).alias(f"sales_roll_std_{window}")
        )
    expressions.extend([
        shifted.rolling_median(window_size=7, min_samples=1).over(group).alias("sales_roll_median_7"),
        shifted.rolling_median(window_size=28, min_samples=1).over(group).alias("sales_roll_median_28"),
        (pl.col("sale_amount").shift(1).rolling_sum(window_size=28, min_samples=1).over(group)).alias("sales_roll_sum_28"),
        (pl.col("sale_amount").shift(1).rolling_map(lambda s: float((s == 0).mean()), window_size=28, min_samples=1).over(group)).alias("sales_zero_rate_28"),
    ])
    frame = frame.with_columns(expressions)
    frame = frame.with_columns([
        (pl.col("sales_roll_mean_7") / (pl.col("sales_roll_mean_28") + 1e-6)).alias("sales_momentum_7_28"),
        (pl.col("sales_roll_std_28") / (pl.col("sales_roll_mean_28") + 1e-6)).alias("sales_cv_28"),
        (pl.col("sales_lag_1") / (pl.col("sales_lag_7") + 1e-6)).alias("sales_lag_ratio_1_7"),
        (2 * np.pi * pl.col("date").dt.weekday() / 7).sin().alias("dow_sin"),
        (2 * np.pi * pl.col("date").dt.weekday() / 7).cos().alias("dow_cos"),
        (2 * np.pi * pl.col("date").dt.ordinal_day() / 365.25).sin().alias("doy_sin"),
        (2 * np.pi * pl.col("date").dt.ordinal_day() / 365.25).cos().alias("doy_cos"),
    ])
    return frame


def wavelet_features(raw: pl.DataFrame, window: int, family: str) -> pd.DataFrame:
    """Extract causal DWT summaries from sales, residual and momentum histories."""
    chunks = []
    names = []
    for signal in SIGNALS:
        names.extend([
            f"wv_{signal}_approx_energy", f"wv_{signal}_detail_energy",
            f"wv_{signal}_entropy", f"wv_{signal}_detail_to_approx",
            f"wv_{signal}_smooth_mean",
        ])
    ordered = raw.sort(GROUP + ["date"])
    for (store, product), group in ordered.group_by(GROUP, maintain_order=True):
        sales = group.get_column("sale_amount").to_numpy().astype(float)
        dates = group.get_column("date").to_numpy()
        s = pd.Series(sales)
        residual = s - s.shift(1).rolling(7, min_periods=1).mean()
        mean7 = s.shift(1).rolling(7, min_periods=1).mean()
        mean28 = s.shift(1).rolling(28, min_periods=1).mean()
        momentum = mean7 / (mean28 + 1e-6)
        signal_arrays = {
            "sales": sales,
            "sales_residual_7": residual.fillna(0).to_numpy(dtype=float),
            "sales_momentum_7_28": momentum.fillna(0).to_numpy(dtype=float),
        }
        if len(sales) <= window:
            continue
        windows = {name: np.lib.stride_tricks.sliding_window_view(values, window)[:-1]
                   for name, values in signal_arrays.items()}
        features = []
        for signal in SIGNALS:
            coeffs = pywt.wavedec(windows[signal], family, mode="periodization", level=2, axis=1)
            approx, detail2, detail1 = coeffs
            ae = np.mean(np.square(approx), axis=1)
            de1 = np.mean(np.square(detail1), axis=1)
            de2 = np.mean(np.square(detail2), axis=1)
            de = de1 + de2
            total = np.maximum(ae + de, 1e-12)
            p = np.column_stack([ae / total, de / total])
            entropy = -np.sum(p * np.log(np.maximum(p, 1e-12)), axis=1)
            ratio = de / (ae + 1e-12)
            smooth = np.mean(windows[signal], axis=1)
            features.extend([ae, de, entropy, ratio, smooth])
        values = np.column_stack(features)
        chunks.append(pd.DataFrame(values, columns=names).assign(
            store_id=store, product_id=product, date=pd.to_datetime(dates[window:]).date
        ))
    if not chunks:
        raise ValueError(f"No eligible series for {family}/{window}")
    result = pd.concat(chunks, ignore_index=True)
    if not np.isfinite(result[names].to_numpy()).all():
        raise ValueError(f"Non-finite wavelet features for {family}/{window}")
    return result


def fit_predict(train, valid, evaluation, columns, cats, label):
    frames = [train.copy(), valid.copy(), evaluation.copy()]
    for frame in frames:
        for col in cats:
            frame[col] = frame[col].astype("category")
    train, valid, evaluation = frames
    model = lgb.train(
        {"objective": "regression", "metric": "l1", "learning_rate": 0.05,
         "num_leaves": 63, "max_depth": -1, "min_data_in_leaf": 100,
         "feature_fraction": 0.85, "bagging_fraction": 0.85, "bagging_freq": 1,
         "lambda_l2": 2.0, "verbosity": -1, "seed": 42, "num_threads": 4},
        lgb.Dataset(train[columns], label=train.sale_amount, categorical_feature=cats, free_raw_data=False),
        num_boost_round=1200,
        valid_sets=[lgb.Dataset(valid[columns], label=valid.sale_amount, categorical_feature=cats)],
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(200)],
    )
    pv = np.maximum(model.predict(valid[columns], num_iteration=model.best_iteration), 0)
    pe = np.maximum(model.predict(evaluation[columns], num_iteration=model.best_iteration), 0)
    print(f"{label}: best_iteration={model.best_iteration}")
    return model, pv, pe


def cluster_bootstrap(y, baseline, candidates, series_ids):
    """Paired bootstrap resampling whole store-product series, retaining all dates."""
    y = np.asarray(y, dtype=float)
    baseline = np.maximum(np.asarray(baseline, dtype=float), 0)
    series_ids = np.asarray(series_ids).astype(str)
    unique_ids, inverse = np.unique(series_ids, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(unique_ids))]
    rng = np.random.default_rng(BOOT_SEED)
    draws = {name: {"mae": [], "rmse": [], "wape": []} for name in candidates}
    for _ in range(BOOT_REPS):
        selected = rng.integers(0, len(members), size=len(members))
        idx = np.concatenate([members[i] for i in selected])
        denom = max(float(np.abs(y[idx]).sum()), 1e-12)
        b_abs = np.abs(y[idx] - baseline[idx])
        b_rmse = np.sqrt(np.square(y[idx] - baseline[idx]).mean())
        b = {"mae": b_abs.mean(), "rmse": b_rmse, "wape": b_abs.sum() / denom}
        for name, prediction in candidates.items():
            p = np.maximum(np.asarray(prediction)[idx], 0)
            err = np.abs(y[idx] - p)
            c = {"mae": err.mean(), "rmse": np.sqrt(np.square(y[idx] - p).mean()), "wape": err.sum() / denom}
            for metric in draws[name]:
                draws[name][metric].append((b[metric] - c[metric]) / max(b[metric], 1e-12) * 100)
    result = {}
    for name, metrics in draws.items():
        result[name] = {}
        for metric, values in metrics.items():
            values = np.asarray(values)
            lo, hi = np.percentile(values, [2.5, 97.5])
            result[name][metric] = {
                "mean_improvement_pct": float(values.mean()),
                "ci_95_low_pct": float(lo), "ci_95_high_pct": float(hi),
                "probability_improvement": float((values > 0).mean()),
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
            }
    return {
        "method": "paired nonparametric store-product series cluster bootstrap",
        "replicates": BOOT_REPS, "seed": BOOT_SEED,
        "evaluation_rows": len(y), "series_clusters": len(unique_ids),
        "comparisons": result,
        "caveat": "Clusters preserve within-series time dependence. Shared date shocks across different series are not fully addressed; consider two-way/date-block bootstrap for final inference.",
    }


def main():
    s3, bucket = s3_client(), env("R2_BUCKET")
    with tempfile.TemporaryDirectory(prefix="frn-engineered-wavelet-") as td:
        root = Path(td)
        raw = {}
        for split, key in RAW.items():
            path = root / f"{split}.parquet"
            print(f"Downloading {key}")
            s3.download_file(bucket, key, str(path))
            raw[split] = pl.read_parquet(path)
        eval_dates = set(raw["eval"].get_column("dt").unique().to_list())
        combined = pl.concat([raw["train"], raw["eval"]], how="vertical_relaxed")
        base = load_features_from_frame(combined)
        engineered = add_engineered_features(base)
        # The 64-day eligibility mask gives every model identical rows.
        eligibility = combined.select(GROUP + ["dt"]).with_columns(
            pl.col("dt").str.to_date(strict=False).alias("date")
        ).sort(GROUP + ["date"]).with_columns(
            pl.col("date").cum_count().over(GROUP).alias("_series_n")
        ).filter(pl.col("_series_n") > 64).select(GROUP + ["date"])
        frame = engineered.join(eligibility, on=GROUP + ["date"], how="inner").sort(GROUP + ["date"])
        dates = frame.filter(~pl.col("date").cast(pl.String).is_in(list(eval_dates))).select(
            pl.col("date").unique().sort()
        ).to_series().to_list()
        valid_dates = set(dates[-14:])
        eval_pd = frame.filter(pl.col("date").cast(pl.String).is_in(list(eval_dates))).to_pandas()
        valid_pd = frame.filter(pl.col("date").is_in(list(valid_dates))).to_pandas()
        train_pd = frame.filter(~pl.col("date").cast(pl.String).is_in(list(eval_dates))
                                & ~pl.col("date").is_in(list(valid_dates))).to_pandas()
        base_cols = [c for c in frame.columns if c not in ("sale_amount", "date")]
        engineered_only = [c for c in base_cols if c not in (
            "sales_lag_2", "sales_lag_3", "sales_lag_14", "sales_lag_28", "sales_lag_56",
            "sales_roll_mean_14", "sales_roll_mean_56", "sales_roll_std_14", "sales_roll_std_28",
            "sales_roll_std_56", "sales_roll_median_7", "sales_roll_median_28", "sales_roll_sum_28",
            "sales_zero_rate_28", "sales_momentum_7_28", "sales_cv_28", "sales_lag_ratio_1_7",
            "dow_sin", "dow_cos", "doy_sin", "doy_cos"
        )]
        new_cols = [c for c in base_cols if c not in engineered_only]
        cats = [c for c in ("city_id", "store_id", "management_group_id", "first_category_id",
                             "second_category_id", "third_category_id", "product_id", "day_of_week", "month")
                if c in base_cols]
        yv, ye = valid_pd.sale_amount.to_numpy(), eval_pd.sale_amount.to_numpy()
        models = {}
        predictions = {}
        summaries = []
        _, pv, pe = fit_predict(train_pd, valid_pd, eval_pd, engineered_only, cats, "existing baseline")
        models["existing_baseline"] = pe
        summaries.append({"model": "existing_baseline", **{f"validation_{k}":v for k,v in metric_pack(yv,pv).items()},
                          **{f"evaluation_{k}":v for k,v in metric_pack(ye,pe).items()}})
        _, pv, pe = fit_predict(train_pd, valid_pd, eval_pd, base_cols, cats, "engineered LightGBM")
        models["engineered_lightgbm"] = pe
        summaries.append({"model": "engineered_lightgbm", **{f"validation_{k}":v for k,v in metric_pack(yv,pv).items()},
                          **{f"evaluation_{k}":v for k,v in metric_pack(ye,pe).items()}})
        wave_raw = combined.select(GROUP + ["dt", "sale_amount"]).with_columns(
            pl.col("dt").str.to_date(strict=False).alias("date")
        ).drop("dt")
        wave_predictions = {}
        for window in WINDOWS:
            for family in FAMILIES:
                print(f"Generating engineered-signal wavelets: {family}, window={window}")
                wf = wavelet_features(wave_raw, window, family)
                wf_pl = pl.from_pandas(wf).with_columns(pl.col("date").cast(pl.Date))
                joined = frame.join(wf_pl, on=GROUP + ["date"], how="inner").sort(GROUP + ["date"])
                wcols = [c for c in wf.columns if c not in GROUP + ["date"]]
                tr = joined.filter(~pl.col("date").is_in(list(valid_dates))
                                   & ~pl.col("date").cast(pl.String).is_in(list(eval_dates))).to_pandas()
                va = joined.filter(pl.col("date").is_in(list(valid_dates))).to_pandas()
                ev = joined.filter(pl.col("date").cast(pl.String).is_in(list(eval_dates))).to_pandas()
                # Assert same row keys and order as the common evaluation universe.
                keys = ["store_id", "product_id", "date"]
                if not ev[keys].reset_index(drop=True).equals(eval_pd[keys].reset_index(drop=True)):
                    raise RuntimeError(f"Evaluation row alignment mismatch for {family}/{window}")
                _, pv, pe = fit_predict(tr, va, ev, base_cols + wcols, cats, f"{family}/{window}")
                name = f"{family}_window_{window}"
                wave_predictions[name] = pe
                summaries.append({"model": name, **{f"validation_{k}":v for k,v in metric_pack(yv,pv).items()},
                                  **{f"evaluation_{k}":v for k,v in metric_pack(ye,pe).items()}})
        baseline = models["engineered_lightgbm"]
        candidates = {name: pred for name, pred in wave_predictions.items()}
        series_ids = eval_pd["store_id"].astype(str) + "::" + eval_pd["product_id"].astype(str)
        bootstrap = cluster_bootstrap(ye, baseline, candidates, series_ids)
        summary = {
            "models": summaries, "cluster_bootstrap_vs_engineered_baseline": bootstrap,
            "feature_groups": {
                "existing_baseline_feature_count": len(engineered_only),
                "engineered_additions": new_cols,
                "wavelet_signals": list(SIGNALS), "wavelet_families": list(FAMILIES),
                "wavelet_windows": list(WINDOWS),
            },
            "rows": {"train": len(train_pd), "validation": len(valid_pd), "evaluation": len(eval_pd)},
            "validation_period": {"start": str(min(valid_dates)), "end": str(max(valid_dates))},
            "evaluation_period": {"start": str(eval_pd.date.min()), "end": str(eval_pd.date.max())},
            "target": "sale_amount",
            "target_semantics": "Observed sales; not necessarily uncensored demand",
            "leakage_control": "All engineered lags/rolling windows are shifted; wavelet windows end at t-1; common 64-observation eligibility across all models.",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        predictions = pd.DataFrame({
            "date": eval_pd.date.astype(str), "store_id": eval_pd.store_id.astype(str),
            "product_id": eval_pd.product_id.astype(str), "actual_sale_amount": ye,
            "existing_baseline": models["existing_baseline"],
            "engineered_lightgbm": baseline,
            **{f"wavelet_{name}": pred for name, pred in wave_predictions.items()},
        })
        artifacts = {
            "engineered_wavelet_model_comparison.csv": pd.DataFrame(summaries),
            "engineered_wavelet_cluster_bootstrap.json": bootstrap,
            "engineered_wavelet_config.json": summary,
            "engineered_wavelet_predictions.csv": predictions,
        }
        for filename, payload in artifacts.items():
            path = root / filename
            if filename.endswith(".csv"):
                payload.to_csv(path, index=False)
                content_type = "text/csv"
            else:
                path.write_text(json.dumps(payload, indent=2, default=str))
                content_type = "application/json"
            key = f"{OUT}/{filename}"
            s3.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": content_type})
            head = s3.head_object(Bucket=bucket, Key=key)
            if head["ContentLength"] != path.stat().st_size:
                raise RuntimeError(f"R2 verification failed: {key}")
            print(f"Verified s3://{bucket}/{key} ({path.stat().st_size:,} bytes)")
        print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
