"""Batch forecast-quality monitoring for held-out or appended prediction records.

Produces daily error/coverage metrics plus temporal distribution-shift diagnostics.
The initial run uses the published evaluation prediction artifact; future production
monitoring requires appending matured forecast/actual records to that artifact.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from botocore.config import Config

REQUIRED = [
    "date", "actual_sale_amount", "p10", "p50", "p90",
    "p10_calibrated", "p90_calibrated",
]


def population_stability_index(reference, current, bins: int = 10) -> float:
    """PSI using reference quantile bins; returns 0 for identical constant samples."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0 if np.allclose(ref, cur) else float("inf")
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_counts = np.histogram(ref, bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur, bins=edges)[0].astype(float)
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, 1)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, 1)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _overall_metrics(frame: pd.DataFrame) -> dict:
    y = frame["actual_sale_amount"].to_numpy(dtype=float)
    p50 = np.maximum(frame["p50"].to_numpy(dtype=float), 0.0)
    lower = np.maximum(frame["p10_calibrated"].to_numpy(dtype=float), 0.0)
    upper = np.maximum(frame["p90_calibrated"].to_numpy(dtype=float), lower)
    raw_lower = np.maximum(frame["p10"].to_numpy(dtype=float), 0.0)
    raw_upper = np.maximum(frame["p90"].to_numpy(dtype=float), raw_lower)
    denominator = max(float(np.abs(y).sum()), 1e-9)
    return {
        "rows": int(len(frame)),
        "mae": float(np.abs(y - p50).mean()),
        "rmse": float(np.sqrt(np.square(y - p50).mean())),
        "wape_pct": float(100 * np.abs(y - p50).sum() / denominator),
        "bias_pct": float(100 * (p50 - y).sum() / max(float(y.sum()), 1e-9)),
        "calibrated_p10_p90_coverage_pct": float(((y >= lower) & (y <= upper)).mean() * 100),
        "calibrated_mean_interval_width": float((upper - lower).mean()),
        "raw_p10_p90_coverage_pct": float(((y >= raw_lower) & (y <= raw_upper)).mean() * 100),
        "raw_mean_interval_width": float((raw_upper - raw_lower).mean()),
        "raw_quantile_crossing_rate_pct": float(((frame["p10"] > frame["p50"]) |
                                                   (frame["p50"] > frame["p90"])).mean() * 100),
    }


def build_monitoring_report(frame: pd.DataFrame, nominal_coverage_pct: float = 80.0,
                            coverage_alert_floor_pct: float = 75.0,
                            bias_alert_pct: float = 5.0,
                            psi_alert: float = 0.25) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Return aggregate monitoring state, daily metrics and temporal PSI diagnostics."""
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"prediction frame missing required columns: {sorted(missing)}")
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise").dt.normalize()
    metric_cols = ["actual_sale_amount", "p10", "p50", "p90", "p10_calibrated", "p90_calibrated"]
    if not np.isfinite(data[metric_cols].to_numpy(dtype=float)).all():
        raise ValueError("forecast monitoring inputs contain non-finite values")
    data = data.sort_values("date")
    y = data["actual_sale_amount"].to_numpy(dtype=float)
    p50 = np.maximum(data["p50"].to_numpy(dtype=float), 0.0)
    lower = np.maximum(data["p10_calibrated"].to_numpy(dtype=float), 0.0)
    upper = np.maximum(data["p90_calibrated"].to_numpy(dtype=float), lower)
    raw_lower = np.maximum(data["p10"].to_numpy(dtype=float), 0.0)
    raw_upper = np.maximum(data["p90"].to_numpy(dtype=float), raw_lower)
    data["_abs_error"] = np.abs(y - p50)
    data["_signed_error"] = p50 - y
    data["_covered"] = (y >= lower) & (y <= upper)
    data["_interval_width"] = upper - lower
    data["_quantile_crossing"] = (data["p10"] > data["p50"]) | (data["p50"] > data["p90"])
    daily = data.groupby("date", sort=True).agg(
        rows=("_abs_error", "size"),
        mae=("_abs_error", "mean"),
        absolute_error_sum=("_abs_error", "sum"),
        actual_abs_sum=("actual_sale_amount", lambda x: float(np.abs(x).sum())),
        actual_sum=("actual_sale_amount", "sum"),
        signed_error_sum=("_signed_error", "sum"),
        calibrated_coverage_pct=("_covered", lambda x: float(x.mean() * 100)),
        calibrated_mean_interval_width=("_interval_width", "mean"),
        quantile_crossing_rate_pct=("_quantile_crossing", lambda x: float(x.mean() * 100)),
    ).reset_index()
    daily["wape_pct"] = 100 * daily["absolute_error_sum"] / daily["actual_abs_sum"].clip(lower=1e-9)
    daily["bias_pct"] = 100 * daily["signed_error_sum"] / daily["actual_sum"].clip(lower=1e-9)

    unique_dates = sorted(data["date"].unique())
    split_at = max(1, len(unique_dates) // 2)
    reference = data[data["date"].isin(unique_dates[:split_at])]
    current = data[data["date"].isin(unique_dates[split_at:])]
    if current.empty:
        current = reference
        reference_label = "first half of evaluation dates (fallback: only one date available)"
    else:
        reference_label = "first chronological half of evaluation dates"
    drift_variables = {
        "observed_sales": "actual_sale_amount",
        "median_forecast": "p50",
        "calibrated_interval_width": None,
    }
    drift_rows = []
    for label, column in drift_variables.items():
        ref_values = (reference["p90_calibrated"] - reference["p10_calibrated"]
                      if column is None else reference[column])
        cur_values = (current["p90_calibrated"] - current["p10_calibrated"]
                      if column is None else current[column])
        psi = population_stability_index(ref_values, cur_values)
        drift_rows.append({
            "variable": label, "psi": psi,
            "status": "high" if psi >= psi_alert else "moderate" if psi >= 0.10 else "low",
            "reference_rows": int(len(ref_values)), "current_rows": int(len(cur_values)),
        })
    if "stock_hour6_22_cnt" in data.columns:
        psi = population_stability_index(reference["stock_hour6_22_cnt"], current["stock_hour6_22_cnt"])
        drift_rows.append({
            "variable": "stockout_hours_06_22", "psi": psi,
            "status": "high" if psi >= psi_alert else "moderate" if psi >= 0.10 else "low",
            "reference_rows": int(len(reference)), "current_rows": int(len(current)),
        })
    drift = pd.DataFrame(drift_rows)
    metrics = _overall_metrics(data)
    alerts = []
    if metrics["calibrated_p10_p90_coverage_pct"] < coverage_alert_floor_pct:
        alerts.append({
            "severity": "warning", "code": "INTERVAL_COVERAGE_LOW",
            "message": f"Calibrated interval coverage {metrics['calibrated_p10_p90_coverage_pct']:.2f}% is below alert floor {coverage_alert_floor_pct:.2f}%.",
        })
    if abs(metrics["bias_pct"]) > bias_alert_pct:
        alerts.append({
            "severity": "warning", "code": "FORECAST_BIAS_HIGH",
            "message": f"Absolute forecast bias {abs(metrics['bias_pct']):.2f}% exceeds {bias_alert_pct:.2f}%.",
        })
    if metrics["raw_quantile_crossing_rate_pct"] > 0.5:
        alerts.append({
            "severity": "warning", "code": "QUANTILE_CROSSING",
            "message": f"Raw quantile crossing rate {metrics['raw_quantile_crossing_rate_pct']:.3f}% exceeds 0.5%.",
        })
    for row in drift_rows:
        if np.isfinite(row["psi"]) and row["psi"] >= psi_alert:
            alerts.append({
                "severity": "warning", "code": "DISTRIBUTION_SHIFT",
                "message": f"{row['variable']} PSI {row['psi']:.3f} reached the high-shift threshold {psi_alert:.2f}.",
            })
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "monitoring_mode": "batch retrospective evaluation-artifact monitor",
        "evaluation_period": {"start": str(data["date"].min().date()), "end": str(data["date"].max().date())},
        "nominal_interval_coverage_pct": nominal_coverage_pct,
        "overall": metrics,
        "alerts": alerts,
        "alert_count": len(alerts),
        "drift_reference": reference_label,
        "drift_reference_period": {"start": str(reference["date"].min().date()), "end": str(reference["date"].max().date())},
        "drift_current_period": {"start": str(current["date"].min().date()), "end": str(current["date"].max().date())},
        "drift_thresholds": {"low_below": 0.10, "moderate_below": psi_alert, "high_at_or_above": psi_alert},
        "operational_thresholds": {
            "coverage_alert_floor_pct": coverage_alert_floor_pct,
            "absolute_bias_alert_pct": bias_alert_pct,
            "quantile_crossing_alert_pct": 0.5,
        },
        "limitations": [
            "The initial artifact is a held-out evaluation batch, not a live production stream.",
            "Temporal PSI compares the early and later halves of the available evaluation window; it is not a validated training-versus-production drift test.",
            "Coverage and error metrics require matured actual outcomes and are therefore lagging indicators.",
            "PSI thresholds are operational heuristics and should be calibrated against business risk and historical variation.",
            "Actual targets are observed sales and may be censored by stockouts.",
        ],
    }
    return report, daily, drift


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


def main():
    bucket = os.getenv("R2_BUCKET")
    if not bucket:
        raise RuntimeError("Missing R2_BUCKET")
    s3 = r2_client()
    with tempfile.TemporaryDirectory(prefix="frn-forecast-monitor-") as td:
        root = Path(td)
        source = root / "evaluation_quantile_predictions.csv"
        s3.download_file(bucket, "quantile-model-artifacts/evaluation_quantile_predictions.csv", str(source))
        frame = pd.read_csv(source)
        report, daily, drift = build_monitoring_report(frame)
        report_path = root / "forecast_monitoring_summary.json"
        daily_path = root / "forecast_monitoring_daily.csv"
        drift_path = root / "forecast_monitoring_drift.csv"
        report_path.write_text(json.dumps(report, indent=2, default=str))
        daily.to_csv(daily_path, index=False)
        drift.to_csv(drift_path, index=False)
        uploads = [
            (report_path, "monitoring/forecast-quality/forecast_monitoring_summary.json", "application/json"),
            (daily_path, "monitoring/forecast-quality/forecast_monitoring_daily.csv", "text/csv"),
            (drift_path, "monitoring/forecast-quality/forecast_monitoring_drift.csv", "text/csv"),
        ]
        for path, key, content_type in uploads:
            s3.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": content_type})
            size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
            if size != path.stat().st_size:
                raise RuntimeError(f"R2 verification failed for {key}")
            print(f"Verified s3://{bucket}/{key} ({size:,} bytes)")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
