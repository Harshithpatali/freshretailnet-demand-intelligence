"""Fair same-row comparison of seasonal naive, LightGBM, and wavelet LightGBM."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import sys

import boto3
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from botocore.config import Config
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.train_model import RAW, env, load_features_from_frame, metric_pack
from experiments.wavelet.features import build_wavelet_features

OUT = "wavelet-experiment-artifacts"
WAVELET_COLUMNS = [
    "wavelet_approx_energy",
    "wavelet_detail_energy_l1",
    "wavelet_detail_energy_l2",
    "wavelet_detail_energy_l3",
    "wavelet_coefficient_entropy",
    "wavelet_high_low_energy_ratio",
    "wavelet_smoothed_mean_32",
]


def r2_client():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{env('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
        aws_access_key_id=env("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=env("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
    )


def train_and_score(train_pd, valid_pd, eval_pd, feature_cols, cat_cols):
    frames = [train_pd.copy(), valid_pd.copy(), eval_pd.copy()]
    for frame in frames:
        for col in cat_cols:
            frame[col] = frame[col].astype("category")
    tr, va, ev = frames
    model = lgb.train(
        {
            "objective": "regression", "metric": "l1", "learning_rate": 0.05,
            "num_leaves": 63, "max_depth": -1, "min_data_in_leaf": 100,
            "feature_fraction": 0.85, "bagging_fraction": 0.85,
            "bagging_freq": 1, "lambda_l2": 2.0, "verbosity": -1,
            "seed": 42, "num_threads": 4,
        },
        lgb.Dataset(tr[feature_cols], label=tr["sale_amount"],
                    categorical_feature=cat_cols, free_raw_data=False),
        num_boost_round=1200,
        valid_sets=[lgb.Dataset(va[feature_cols], label=va["sale_amount"],
                               categorical_feature=cat_cols)],
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(100)],
    )
    pv = np.maximum(model.predict(va[feature_cols], num_iteration=model.best_iteration), 0)
    pe = np.maximum(model.predict(ev[feature_cols], num_iteration=model.best_iteration), 0)
    return model, pv, pe


def main():
    bucket = env("R2_BUCKET")
    s3 = r2_client()
    with tempfile.TemporaryDirectory(prefix="frn-wavelet-") as td:
        root = Path(td)
        raw = {}
        for split, key in RAW.items():
            local = root / f"{split}.parquet"
            print(f"Downloading {key}")
            s3.download_file(bucket, key, str(local))
            raw[split] = pl.read_parquet(local)

        eval_dates = set(raw["eval"].get_column("dt").unique().to_list())
        combined = pl.concat([raw["train"], raw["eval"]], how="vertical_relaxed")
        print("Building the existing leakage-aware feature frame...")
        base = load_features_from_frame(combined)
        print("Building 32-day trailing db4 wavelet features by store-product series...")
        wave_input = combined.select(["store_id", "product_id", "dt", "sale_amount"]).with_columns(
            pl.col("dt").str.to_date(strict=False).alias("date")
        ).drop("dt").to_pandas()
        wave = build_wavelet_features(wave_input)
        wave_pl = pl.from_pandas(wave)
        merged = base.join(
            wave_pl, on=["store_id", "product_id", "date"], how="inner"
        )
        eval_df = merged.filter(pl.col("date").cast(pl.String).is_in(list(eval_dates)))
        train = merged.filter(~pl.col("date").cast(pl.String).is_in(list(eval_dates)))
        dates = train.select(pl.col("date").unique().sort()).to_series().to_list()
        valid_dates = set(dates[-14:])
        train_pd = train.filter(~pl.col("date").is_in(list(valid_dates))).drop("date").to_pandas()
        valid_pd = train.filter(pl.col("date").is_in(list(valid_dates))).drop("date").to_pandas()
        eval_pd = eval_df.drop("date").to_pandas()

        # Inner join intentionally restricts both models and naive benchmark to
        # identical rows with 32 days of history, ensuring a fair comparison.
        target = "sale_amount"
        all_features = [c for c in train_pd.columns if c != target]
        wave_features = [c for c in all_features if c not in WAVELET_COLUMNS]
        cat_cols = [c for c in (
            "city_id", "store_id", "management_group_id", "first_category_id",
            "second_category_id", "third_category_id", "product_id",
            "day_of_week", "month"
        ) if c in all_features]
        print(
            f"Same-row sample sizes: train={len(train_pd):,}, "
            f"validation={len(valid_pd):,}, evaluation={len(eval_pd):,}; "
            f"base_features={len(wave_features)}, wavelet_features={len(WAVELET_COLUMNS)}"
        )

        print("Training same-row LightGBM control...")
        base_model, base_v, base_e = train_and_score(
            train_pd, valid_pd, eval_pd, wave_features, cat_cols
        )
        print("Training wavelet-enhanced LightGBM...")
        wave_model, wave_v, wave_e = train_and_score(
            train_pd, valid_pd, eval_pd, all_features, cat_cols
        )

        yv = valid_pd[target].to_numpy()
        ye = eval_pd[target].to_numpy()
        valid_naive = valid_pd["sales_lag_7"].to_numpy()
        eval_naive = eval_pd["sales_lag_7"].to_numpy()
        results = {
            "validation": {
                "seasonal_naive_lag7": metric_pack(yv, valid_naive),
                "lightgbm_same_rows": metric_pack(yv, base_v),
                "wavelet_lightgbm": metric_pack(yv, wave_v),
            },
            "evaluation": {
                "seasonal_naive_lag7": metric_pack(ye, eval_naive),
                "lightgbm_same_rows": metric_pack(ye, base_e),
                "wavelet_lightgbm": metric_pack(ye, wave_e),
            },
            "comparison": {
                "evaluation_mae_change_pct_vs_same_row_lightgbm": float(
                    (metric_pack(ye, wave_e)["mae"] / metric_pack(ye, base_e)["mae"] - 1) * 100
                ),
                "evaluation_wape_change_pct_vs_same_row_lightgbm": float(
                    (metric_pack(ye, wave_e)["wape_pct"] / metric_pack(ye, base_e)["wape_pct"] - 1) * 100
                ),
                "negative_values_mean_improvement": True,
            },
            "experiment": {
                "wavelet": "db4", "decomposition_level": 3, "history_window_days": 32,
                "features": WAVELET_COLUMNS,
                "validation_period": {"start": str(min(valid_dates)), "end": str(max(valid_dates))},
                "evaluation_period": {"start": str(eval_df["date"].min()), "end": str(eval_df["date"].max())},
                "rows": {"train": len(train_pd), "validation": len(valid_pd), "evaluation": len(eval_pd)},
                "base_feature_count": len(wave_features), "wavelet_feature_count": len(WAVELET_COLUMNS),
                "best_iteration_same_row_lightgbm": int(base_model.best_iteration),
                "best_iteration_wavelet_lightgbm": int(wave_model.best_iteration),
                "target": "sale_amount",
                "target_semantics": "Observed daily sales, not guaranteed uncensored demand.",
                "fair_comparison": "All three methods scored on identical rows with at least 32 prior observations.",
                "leakage_control": "Each wavelet window ends at t-1; no current-day or future sales enter features.",
                "limitations": "Wavelet features cannot recover unobserved demand during stockouts.",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        }
        metrics_path = root / "wavelet_metrics.json"
        metrics_path.write_text(json.dumps(results, indent=2))
        pred_path = root / "wavelet_evaluation_predictions.csv"
        pd.DataFrame({
            "actual_sale_amount": ye,
            "seasonal_naive_lag7": eval_naive,
            "lightgbm_same_rows": base_e,
            "wavelet_lightgbm": wave_e,
        }).to_csv(pred_path, index=False)
        importance_path = root / "wavelet_feature_importance.csv"
        pd.DataFrame({
            "feature": all_features,
            "gain": wave_model.feature_importance(importance_type="gain"),
            "split": wave_model.feature_importance(importance_type="split"),
        }).sort_values("gain", ascending=False).to_csv(importance_path, index=False)

        for path in (metrics_path, pred_path, importance_path):
            key = f"{OUT}/{path.name}"
            s3.upload_file(str(path), bucket, key, ExtraArgs={
                "ContentType": "application/json" if path.suffix == ".json" else "text/csv"
            })
            head = s3.head_object(Bucket=bucket, Key=key)
            if head["ContentLength"] != path.stat().st_size:
                raise RuntimeError(f"R2 upload verification failed: {key}")
            print(f"Verified s3://{bucket}/{key} ({path.stat().st_size:,} bytes)")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
