"""Forecast-time availability policy for target-date features."""
from __future__ import annotations

DETERMINISTIC_TARGET_DATE_FEATURES = (
    "target_day_of_week",
    "target_day_of_month",
    "target_month",
    "target_dow_sin",
    "target_dow_cos",
)

DATASET_TARGET_DATE_FLAGS = ("holiday_flag", "activity_flag")


def target_calendar_feature_audit(dataset_columns) -> dict:
    """Permit deterministic date transforms; exclude flags without a publication-time contract."""
    present = sorted(set(DATASET_TARGET_DATE_FLAGS).intersection(set(dataset_columns)))
    return {
        "deterministic_target_date_features_used": list(DETERMINISTIC_TARGET_DATE_FEATURES),
        "dataset_target_date_flags_excluded": present,
        "excluded_reason": (
            "The dataset card defines these as holiday/activity indicators but does not "
            "document their publication time relative to the forecast issue time. "
            "They are therefore not admitted as future covariates without an operational feed/SLA."
        ),
        "forecast_time_availability": {
            "deterministic_date_fields": "known once the target date is selected",
            "holiday_flag": "not certified as available at forecast issue time; excluded",
            "activity_flag": "not certified as available at forecast issue time; excluded",
        },
        "historical_origin_date_flags": (
            "Origin-date values may remain in historical feature rows; this audit concerns "
            "values joined from the future target date."
        ),
        "promotion_gate": "Re-enable a target-date flag only with a timestamped source and evidence it is published before the forecast cutoff.",
    }
