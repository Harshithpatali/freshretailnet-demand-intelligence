"""Regression checks for the original FreshRetailNet source-field contract."""

from scripts.full_dataset_eda import FEATURE_CATALOG, CORRELATION_FIELDS


def test_full_dataset_eda_profiles_all_nineteen_original_fields():
    names = [feature["name"] for feature in FEATURE_CATALOG]

    assert len(names) == 19
    assert len(set(names)) == 19
    assert names == [
        "city_id",
        "store_id",
        "management_group_id",
        "first_category_id",
        "second_category_id",
        "third_category_id",
        "product_id",
        "dt",
        "sale_amount",
        "hours_sale",
        "stock_hour6_22_cnt",
        "hours_stock_status",
        "discount",
        "holiday_flag",
        "activity_flag",
        "precpt",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level",
    ]


def test_hourly_source_fields_are_explicit_vector_features():
    by_name = {feature["name"]: feature for feature in FEATURE_CATALOG}

    assert by_name["hours_sale"]["kind"] == "vector"
    assert by_name["hours_stock_status"]["kind"] == "vector"
    assert "sale_amount" in CORRELATION_FIELDS
    assert "stock_hour6_22_cnt" in CORRELATION_FIELDS


def test_feature_catalog_has_plain_language_explanations():
    assert all(feature["label"] and feature["description"] for feature in FEATURE_CATALOG)
