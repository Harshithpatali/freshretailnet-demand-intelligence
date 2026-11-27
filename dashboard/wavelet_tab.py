"""Streamlit view for the published wavelet benchmark artifacts."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_wavelet_tab(api_base, get_json, metric, compact_table):
    st.subheader("Wavelet research & benchmark")
    st.caption(
        "A same-row comparison of seasonal-naive lag-7, baseline LightGBM, "
        "and wavelet-enhanced LightGBM. Metrics are computed on the published held-out evaluation rows."
    )
    try:
        metrics = get_json("/api/v1/wavelet/metrics")
        importance_payload = get_json("/api/v1/wavelet/feature-importance", {"limit": 2000})
        prediction_payload = get_json("/api/v1/wavelet/predictions", {"limit": 1})
    except Exception as exc:
        st.warning(
            "Wavelet artifacts are not available through the API yet. Confirm that the "
            "Wavelet Feature Experiment workflow uploaded its three files to R2, then redeploy "
            "the API and refresh this page."
        )
        st.caption(f"Artifact/API detail: {exc}")
        return

    experiment = metrics.get("experiment", {})
    validation = metrics.get("validation", {})
    evaluation = metrics.get("evaluation", {})
    comparisons = metrics.get("comparison", {})
    model_keys = [
        ("seasonal_naive_lag7", "Seasonal naive lag-7"),
        ("lightgbm_same_rows", "LightGBM (same rows)"),
        ("wavelet_lightgbm", "Wavelet LightGBM"),
    ]

    st.markdown("#### Evaluation model comparison")
    rows = []
    for key, label in model_keys:
        values = evaluation.get(key, {})
        rows.append({
            "Model": label,
            "MAE": values.get("mae"),
            "RMSE": values.get("rmse"),
            "WAPE (%)": values.get("wape_pct"),
        })
    baseline_metrics = evaluation.get("lightgbm_same_rows", {})
    naive_metrics = evaluation.get("seasonal_naive_lag7", {})
    wave_metrics = evaluation.get("wavelet_lightgbm", {})
    for row in rows:
        key = next(k for k, label in model_keys if label == row["Model"])
        values = evaluation.get(key, {})
        for metric_name, metric_key in (("MAE", "mae"), ("RMSE", "rmse"), ("WAPE", "wape_pct")):
            base_value = baseline_metrics.get(metric_key)
            naive_value = naive_metrics.get(metric_key)
            current_value = values.get(metric_key)
            row[f"{metric_name} change vs LightGBM (%)"] = (
                (current_value / base_value - 1) * 100
                if current_value is not None and base_value else None
            )
            row[f"{metric_name} change vs naive (%)"] = (
                (current_value / naive_value - 1) * 100
                if current_value is not None and naive_value else None
            )
    comparison_df = pd.DataFrame(rows)
    compact_table(comparison_df, 240)

    delta_cols = st.columns(3)
    for column, metric_name, metric_key in zip(
        delta_cols,
        ("MAE", "RMSE", "WAPE"),
        ("mae", "rmse", "wape_pct"),
    ):
        base_value = baseline_metrics.get(metric_key)
        current_value = wave_metrics.get(metric_key)
        if base_value and current_value is not None:
            change = (current_value / base_value - 1) * 100
            with column:
                metric(f"Wavelet {metric_name} change vs LightGBM", f"{change:+.2f}%",
                       "Negative means improvement; positive means degradation.")

    st.caption(
        "Percentage change = (candidate metric / comparator metric − 1) × 100. "
        "Negative values indicate improvement; positive values indicate degradation. "
        "The table reports changes against both same-row LightGBM and seasonal naive."
    )
    metric_rows = []
    for split_name, split_data in (("Validation", validation), ("Evaluation", evaluation)):
        for key, label in model_keys:
            item = split_data.get(key, {})
            metric_rows.append({
                "Split": split_name, "Model": label,
                "MAE": item.get("mae"), "RMSE": item.get("rmse"),
                "WAPE (%)": item.get("wape_pct"),
            })
    st.markdown("#### Validation vs evaluation")
    split_df = pd.DataFrame(metric_rows)
    compact_table(split_df, 260)
    if not split_df.empty:
        selected_metric = st.selectbox("Metric chart", ["MAE", "RMSE", "WAPE (%)"], key="wavelet_metric_chart")
        chart = px.bar(
            split_df, x="Model", y=selected_metric, color="Split", barmode="group",
            title=f"{selected_metric} by model and data split",
        )
        chart.update_layout(height=390, margin={"l": 15, "r": 15, "t": 55, "b": 20})
        st.plotly_chart(chart, use_container_width=True)

    st.markdown("#### Actual vs predicted — evaluation")
    pred = pd.DataFrame(prediction_payload.get("rows", []))
    if not pred.empty and {"store_id", "product_id"}.issubset(pred.columns):
        pairs = pd.DataFrame(prediction_payload.get("series", []))
        if not pairs.empty:
            labels = [f"Store {r.store_id} · Product {r.product_id}" for r in pairs.itertuples(index=False)]
            selected = st.selectbox("Store-product series", range(len(labels)),
                                    format_func=lambda i: labels[i], key="wavelet_series")
            pair = pairs.iloc[selected]
            prediction_payload = get_json("/api/v1/wavelet/predictions", {
                "store_id": str(pair["store_id"]), "product_id": str(pair["product_id"]), "limit": 10000
            })
            pred = pd.DataFrame(prediction_payload.get("rows", []))
    elif not pred.empty and "date" not in pred.columns:
        prediction_payload = get_json("/api/v1/wavelet/predictions", {"limit": 10000})
        pred = pd.DataFrame(prediction_payload.get("rows", []))
    if pred.empty:
        st.info("No evaluation prediction rows were returned.")
    else:
        if "date" in pred.columns:
            pred["date"] = pd.to_datetime(pred["date"], errors="coerce")
            pred = pred.sort_values("date")
            x_col = "date"
            x_title = "Date"
        else:
            pred = pred.reset_index(drop=True)
            pred["evaluation_row"] = pred.index + 1
            x_col = "evaluation_row"
            x_title = "Evaluation row (legacy artifact; rerun workflow to publish dates)"
            st.warning("This prediction artifact predates date and series identifiers. The chart uses row order until the experiment workflow is rerun.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pred[x_col], y=pred["actual_sale_amount"],
                                 mode="lines", name="Actual observed sales"))
        for col, label in (
            ("seasonal_naive_lag7", "Seasonal naive lag-7"),
            ("lightgbm_same_rows", "LightGBM (same rows)"),
            ("wavelet_lightgbm", "Wavelet LightGBM"),
        ):
            if col in pred:
                fig.add_trace(go.Scatter(x=pred[x_col], y=pred[col], mode="lines", name=label))
        fig.update_layout(
            title="Actual observed sales vs model predictions",
            xaxis_title=x_title, yaxis_title="Observed sales units",
            height=440, margin={"l": 15, "r": 15, "t": 55, "b": 20},
            legend={"orientation": "h", "y": -0.25},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Showing {len(pred):,} evaluation rows from {prediction_payload.get('total', len(pred)):,} returned rows.")
        st.download_button(
            "Download evaluation predictions (CSV)", pred.to_csv(index=False),
            "wavelet_evaluation_predictions.csv", "text/csv",
        )

    st.markdown("#### Wavelet feature importance")
    importance = pd.DataFrame(importance_payload.get("rows", []))
    if importance.empty:
        st.info("No feature-importance rows were returned.")
    else:
        if "gain" in importance:
            importance["gain"] = pd.to_numeric(importance["gain"], errors="coerce")
            top = importance.sort_values("gain", ascending=False).head(25)
            fig = px.bar(top.sort_values("gain"), x="gain", y="feature", orientation="h",
                         title="Top 25 features by LightGBM gain")
            fig.update_layout(height=650, xaxis_title="Split gain", yaxis_title="Feature",
                              margin={"l": 15, "r": 15, "t": 55, "b": 20})
            st.plotly_chart(fig, use_container_width=True)
        wave_cols = experiment.get("features", [])
        st.caption("Wavelet feature rows")
        wave_rows = importance[importance["feature"].isin(wave_cols)].sort_values("gain", ascending=False)
        if not wave_rows.empty:
            compact_table(wave_rows, 240)
        else:
            st.info("The artifact does not contain rows matching the wavelet feature names in the experiment metadata.")
        with st.expander("Full feature-importance table"):
            compact_table(importance, 420)

    st.markdown("#### Experiment configuration & reproducibility")
    config = {
        "wavelet": experiment.get("wavelet"),
        "decomposition_level": experiment.get("decomposition_level"),
        "history_window_days": experiment.get("history_window_days"),
        "features": experiment.get("features", []),
        "validation_period": experiment.get("validation_period"),
        "evaluation_period": experiment.get("evaluation_period"),
        "rows": experiment.get("rows"),
        "base_feature_count": experiment.get("base_feature_count"),
        "wavelet_feature_count": experiment.get("wavelet_feature_count"),
        "best_iteration_same_row_lightgbm": experiment.get("best_iteration_same_row_lightgbm"),
        "best_iteration_wavelet_lightgbm": experiment.get("best_iteration_wavelet_lightgbm"),
        "target": experiment.get("target"),
        "target_semantics": experiment.get("target_semantics"),
        "fair_comparison": experiment.get("fair_comparison"),
        "leakage_control": experiment.get("leakage_control"),
        "limitations": experiment.get("limitations"),
        "generated_at_utc": experiment.get("generated_at_utc"),
    }
    st.json(config)
    st.warning(
        "The target is observed sales, not necessarily uncensored demand. Wavelet features use only "
        "the preceding 32 observations; they cannot recover demand hidden by stockouts. "
        "A lower evaluation error is evidence of predictive improvement, not proof of inventory savings."
    )
