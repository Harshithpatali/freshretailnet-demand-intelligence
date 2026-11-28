"""Fast, post-hoc paired cluster bootstrap for saved rolling-backtest predictions.

This job does not train models. It aggregates one row per store-product cluster,
then uses batched multinomial bootstrap weights to avoid Python loops over
clusters and to cap memory use.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from botocore.config import Config

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.artifact_provenance import build_provenance, upload_versioned_artifact

PREFIX = "segment-rolling-backtest-artifacts"
PREDICTIONS_KEY = f"{PREFIX}/segment_rolling_backtest_predictions.csv"
REPORT_KEY = f"{PREFIX}/segment_rolling_backtest_report.json"
COMPARISON_CSV_KEY = f"{PREFIX}/segment_rolling_backtest_comparison.csv"
COMPARISON_MD_KEY = f"{PREFIX}/segment_rolling_backtest_comparison.md"
GROUP = ["store_id", "product_id"]
REQUIRED = GROUP + [
    "horizon_days", "fold", "target_date", "actual",
    "baseline_prediction", "calibrated_prediction",
]


def r2_client():
    required = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"Missing R2 environment variables: {missing}")
    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
    )


def _cluster_sufficient_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Vectorized aggregation: one sufficient-statistics row per entity cluster."""
    data = frame[GROUP + ["actual", "baseline_prediction", "calibrated_prediction"]].copy()
    actual = data["actual"].to_numpy(dtype=np.float64)
    baseline = np.maximum(data["baseline_prediction"].to_numpy(dtype=np.float64), 0.0)
    challenger = np.maximum(data["calibrated_prediction"].to_numpy(dtype=np.float64), 0.0)
    data["_n"] = 1
    data["_actual_abs"] = np.abs(actual)
    data["_actual_sum"] = actual
    data["_baseline_abs_error"] = np.abs(actual - baseline)
    data["_challenger_abs_error"] = np.abs(actual - challenger)
    data["_baseline_squared_error"] = np.square(actual - baseline)
    data["_challenger_squared_error"] = np.square(actual - challenger)
    data["_baseline_signed_error"] = baseline - actual
    data["_challenger_signed_error"] = challenger - actual
    return data.groupby(GROUP, sort=False, observed=True).agg(
        n=("_n", "sum"),
        actual_abs=("_actual_abs", "sum"),
        actual_sum=("_actual_sum", "sum"),
        baseline_abs=("_baseline_abs_error", "sum"),
        challenger_abs=("_challenger_abs_error", "sum"),
        baseline_sq=("_baseline_squared_error", "sum"),
        challenger_sq=("_challenger_squared_error", "sum"),
        baseline_signed=("_baseline_signed_error", "sum"),
        challenger_signed=("_challenger_signed_error", "sum"),
    ).reset_index(drop=True)


def paired_cluster_bootstrap(frame: pd.DataFrame, reps: int = 500, seed: int = 20261128,
                             batch_size: int = 32) -> list[dict]:
    """Paired cluster bootstrap, retaining all dates/folds for each sampled series."""
    if reps < 100:
        raise ValueError("Use at least 100 bootstrap replicates for reportable intervals")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rows = []
    rng = np.random.default_rng(seed)

    for horizon, group in frame.groupby("horizon_days", sort=True):
        sufficient = _cluster_sufficient_statistics(group)
        stats = sufficient.drop(columns=[]).to_numpy(dtype=np.float64)
        n_clusters = stats.shape[0]
        if n_clusters < 2:
            raise RuntimeError(f"Horizon {horizon} has fewer than two store-product clusters")
        # columns: n, actual_abs, actual_sum, baseline_abs, challenger_abs,
        # baseline_sq, challenger_sq, baseline_signed, challenger_signed
        point = stats.sum(axis=0)
        replicate_values = np.empty((reps, 4), dtype=np.float64)
        cursor = 0
        p = np.full(n_clusters, 1.0 / n_clusters, dtype=np.float64)
        while cursor < reps:
            size = min(batch_size, reps - cursor)
            # A multinomial row is equivalent to drawing n_clusters clusters
            # with replacement, but enables a fast matrix multiplication.
            weights = rng.multinomial(n_clusters, p, size=size)
            totals = weights @ stats
            count = np.maximum(totals[:, 0], 1.0)
            denom = np.maximum(totals[:, 1], 1e-9)
            actual_sum = np.maximum(totals[:, 2], 1e-9)
            base_mae = totals[:, 3] / count
            challenger_mae = totals[:, 4] / count
            base_rmse = np.sqrt(totals[:, 5] / count)
            challenger_rmse = np.sqrt(totals[:, 6] / count)
            base_wape = 100.0 * totals[:, 3] / denom
            challenger_wape = 100.0 * totals[:, 4] / denom
            base_bias = 100.0 * totals[:, 7] / actual_sum
            challenger_bias = 100.0 * totals[:, 8] / actual_sum
            replicate_values[cursor:cursor + size, 0] = (
                100.0 * (base_mae - challenger_mae) / np.maximum(base_mae, 1e-9)
            )
            replicate_values[cursor:cursor + size, 1] = (
                100.0 * (base_rmse - challenger_rmse) / np.maximum(base_rmse, 1e-9)
            )
            replicate_values[cursor:cursor + size, 2] = base_wape - challenger_wape
            replicate_values[cursor:cursor + size, 3] = challenger_bias - base_bias
            cursor += size

        count, actual_abs, actual_sum, base_abs, challenger_abs, base_sq, challenger_sq, base_signed, challenger_signed = point
        base_mae = base_abs / max(count, 1.0)
        challenger_mae = challenger_abs / max(count, 1.0)
        base_rmse = np.sqrt(base_sq / max(count, 1.0))
        challenger_rmse = np.sqrt(challenger_sq / max(count, 1.0))
        base_wape = 100.0 * base_abs / max(actual_abs, 1e-9)
        challenger_wape = 100.0 * challenger_abs / max(actual_abs, 1e-9)
        base_bias = 100.0 * base_signed / max(actual_sum, 1e-9)
        challenger_bias = 100.0 * challenger_signed / max(actual_sum, 1e-9)
        point_values = [
            100.0 * (base_mae - challenger_mae) / max(base_mae, 1e-9),
            100.0 * (base_rmse - challenger_rmse) / max(base_rmse, 1e-9),
            base_wape - challenger_wape,
            challenger_bias - base_bias,
        ]
        estimands = [
            "mae_improvement_pct", "rmse_improvement_pct",
            "wape_improvement_pp", "bias_change_pp",
        ]
        eval_rows = int(group.shape[0])
        folds = int(group["fold"].nunique())
        dates = int(group["target_date"].nunique())
        print(f"Bootstrap complete for horizon={horizon}: clusters={n_clusters:,}, rows={eval_rows:,}")
        for idx, estimand in enumerate(estimands):
            rows.append({
                "horizon_days": int(horizon),
                "estimand": estimand,
                "point_estimate": float(point_values[idx]),
                "ci_95_lower": float(np.quantile(replicate_values[:, idx], 0.025)),
                "ci_95_upper": float(np.quantile(replicate_values[:, idx], 0.975)),
                "bootstrap_replicates": int(reps),
                "evaluation_rows": eval_rows,
                "store_product_clusters": int(n_clusters),
                "test_folds": folds,
                "target_dates": dates,
                "uncertainty_scope": "cross-sectional paired cluster bootstrap; temporal-window uncertainty not covered",
            })
    return rows


def main():
    bucket = os.getenv("R2_BUCKET")
    if not bucket:
        raise RuntimeError("Missing R2_BUCKET")
    reps = int(os.getenv("BOOTSTRAP_REPS", "500"))
    seed = int(os.getenv("BOOTSTRAP_SEED", "20261128"))
    s3 = r2_client()
    with tempfile.TemporaryDirectory(prefix="frn-cluster-bootstrap-") as td:
        root = Path(td)
        prediction_path = root / "predictions.csv"
        report_path = root / "report.json"
        print(f"Downloading saved predictions: s3://{bucket}/{PREDICTIONS_KEY}")
        s3.download_file(bucket, PREDICTIONS_KEY, str(prediction_path))
        frame = pd.read_csv(prediction_path, usecols=REQUIRED, dtype={"store_id": str, "product_id": str})
        missing = set(REQUIRED) - set(frame.columns)
        if missing:
            raise RuntimeError(f"Prediction artifact missing columns: {sorted(missing)}")
        if frame[REQUIRED].isna().any().any():
            raise RuntimeError("Prediction artifact contains null values in bootstrap fields")
        values = frame[["actual", "baseline_prediction", "calibrated_prediction"]].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise RuntimeError("Non-finite prediction values; refusing to publish intervals")
        uncertainty = paired_cluster_bootstrap(frame, reps=reps, seed=seed)
        eval_start = str(pd.to_datetime(frame["target_date"]).min().date())
        eval_end = str(pd.to_datetime(frame["target_date"]).max().date())
        configuration = {
            "bootstrap_method": "paired store-product cluster bootstrap",
            "replicates": reps,
            "seed": seed,
            "batch_size": 32,
            "estimands": ["relative MAE improvement", "relative RMSE improvement", "WAPE improvement pp", "bias change pp"],
            "uncertainty_scope": "cross-sectional clusters conditional on the temporal test windows",
        }
        output_keys = [COMPARISON_CSV_KEY, COMPARISON_MD_KEY, REPORT_KEY]
        provenance = build_provenance(
            "Paired Cluster Bootstrap for Segment Rolling Backtest",
            {PREDICTIONS_KEY: prediction_path},
            eval_start, eval_end, configuration, outputs=output_keys,
        )
        comparison = []
        for horizon, group in frame.groupby("horizon_days", sort=True):
            n = len(group)
            actual = group["actual"].to_numpy(dtype=float)
            baseline = np.maximum(group["baseline_prediction"].to_numpy(dtype=float), 0.0)
            challenger = np.maximum(group["calibrated_prediction"].to_numpy(dtype=float), 0.0)
            denom = max(float(np.abs(actual).sum()), 1e-9)
            base = {
                "mae": float(np.abs(actual - baseline).mean()),
                "rmse": float(np.sqrt(np.square(actual - baseline).mean())),
                "wape_pct": float(100 * np.abs(actual - baseline).sum() / denom),
                "bias_pct": float(100 * (baseline - actual).sum() / max(float(actual.sum()), 1e-9)),
            }
            challenger_score = {
                "mae": float(np.abs(actual - challenger).mean()),
                "rmse": float(np.sqrt(np.square(actual - challenger).mean())),
                "wape_pct": float(100 * np.abs(actual - challenger).sum() / denom),
                "bias_pct": float(100 * (challenger - actual).sum() / max(float(actual.sum()), 1e-9)),
            }
            metric_map = {
                "mae_improvement_pct": ("mae", 100 * (base["mae"] - challenger_score["mae"]) / max(base["mae"], 1e-9)),
                "rmse_improvement_pct": ("rmse", 100 * (base["rmse"] - challenger_score["rmse"]) / max(base["rmse"], 1e-9)),
                "wape_improvement_pp": ("wape_pct", base["wape_pct"] - challenger_score["wape_pct"]),
                "bias_change_pp": ("bias_pct", challenger_score["bias_pct"] - base["bias_pct"]),
            }
            for metric, (value_key, estimate) in metric_map.items():
                ci = next(x for x in uncertainty if x["horizon_days"] == int(horizon) and x["estimand"] == metric)
                comparison.append({
                    "horizon_days": int(horizon), "comparison": "calibrated_specialists_vs_current_lightgbm",
                    "metric": metric, "baseline_value": base[value_key],
                    "challenger_value": challenger_score[value_key],
                    "improvement_point_estimate": float(estimate),
                    "ci_95_lower": ci["ci_95_lower"], "ci_95_upper": ci["ci_95_upper"],
                    "evaluation_rows": n, "store_product_clusters": ci["store_product_clusters"],
                    "test_folds": ci["test_folds"], "bootstrap_replicates": reps,
                    "uncertainty_scope": ci["uncertainty_scope"],
                })

        comparison_frame = pd.DataFrame(comparison)
        csv_path = root / "comparison.csv"
        comparison_frame.to_csv(csv_path, index=False)
        md_lines = [
            "# Rolling-backtest experiment comparison", "",
            "Calibrated validation-gated segment specialists versus the current LightGBM baseline.",
            f"Paired store-product cluster bootstrap with {reps:,} replicates; 95% percentile intervals.",
            "Intervals quantify cross-sectional cluster uncertainty, not uncertainty across future time windows.", "",
            "| Horizon | Metric | Baseline | Challenger | Improvement | 95% CI | Rows | Clusters | Folds | Dates |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in comparison:
            md_lines.append(
                f"| {row['horizon_days']}d | {row['metric']} | {row['baseline_value']:.4f} | "
                f"{row['challenger_value']:.4f} | {row['improvement_point_estimate']:.4f} | "
                f"[{row['ci_95_lower']:.4f}, {row['ci_95_upper']:.4f}] | "
                f"{row['evaluation_rows']:,} | {row['store_product_clusters']:,} | "
                f"{row['test_folds']} | {next(x['target_dates'] for x in uncertainty if x['horizon_days'] == row['horizon_days'])} |"
            )
        md_lines.extend([
            "", "**Interpretation:** positive MAE/RMSE/WAPE improvements favour the challenger; bias change is a signed percentage-point difference.",
            "These cluster intervals do not capture variation between future date windows. Review fold-level metrics and collect additional live periods before production promotion.",
        ])
        md_path = root / "comparison.md"
        md_path.write_text("\n".join(md_lines) + "\n")
        report = {}
        try:
            s3.download_file(bucket, REPORT_KEY, str(report_path))
            report = json.loads(report_path.read_text())
        except Exception as exc:
            print(f"Existing JSON report unavailable; publishing a standalone uncertainty report: {exc}")
        report["provenance"] = provenance
        report["paired_cluster_bootstrap_uncertainty"] = uncertainty
        report["comparison_sample_sizes"] = [
            {key: row[key] for key in ("horizon_days", "evaluation_rows", "store_product_clusters", "test_folds", "target_dates")}
            for row in uncertainty[::4]
        ]
        report["comparison_report"] = "segment_rolling_backtest_comparison.csv"
        report["uncertainty_methodology"] = (
            f"Paired store-product cluster bootstrap with {reps} replicates, resampling entities "
            "and retaining all held-out dates/folds for each sampled entity. This captures "
            "cross-sectional cluster uncertainty, not temporal-window uncertainty."
        )
        report["uncertainty_generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        report_path.write_text(json.dumps(report, indent=2, default=str))
        uploads = [
            (csv_path, COMPARISON_CSV_KEY, "text/csv"),
            (md_path, COMPARISON_MD_KEY, "text/markdown"),
            (report_path, REPORT_KEY, "application/json"),
        ]
        for path, key, content_type in uploads:
            s3.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": content_type})
            size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
            if size != path.stat().st_size:
                raise RuntimeError(f"R2 verification failed for {key}")
            print(f"Verified s3://{bucket}/{key} ({size:,} bytes)")
        print(json.dumps(uncertainty, indent=2))


if __name__ == "__main__":
    main()
