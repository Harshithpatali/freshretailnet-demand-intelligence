"""Streamlit dashboard for FreshRetailNet Demand Intelligence."""
import os
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="FreshRetailNet | Demand Intelligence", page_icon="📦", layout="wide")
API = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

st.markdown("""
<style>
.block-container {padding: 1.25rem 1.4rem 2.5rem; max-width: 1480px;}
[data-testid="stDataFrame"], [data-testid="stTable"] {width: 100%; overflow-x: auto; border-radius: 12px;}
[data-testid="stMetric"] {background: var(--secondary-background-color); border: 1px solid rgba(128,128,128,.18); padding: 13px 15px; border-radius: 14px;}
[data-testid="stMetricValue"] {font-size: clamp(1.25rem, 3vw, 2rem);}
[data-testid="stTabs"] [role="tablist"] {gap: .35rem; border-bottom: 1px solid rgba(128,128,128,.20);}
[data-testid="stTabs"] button[role="tab"] {border-radius: 10px 10px 0 0; padding: .55rem .9rem;}
.stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button {border-radius: 11px; min-height: 2.7rem;}
[data-baseweb="select"] > div {border-radius: 10px;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 14px;}
@media (max-width: 760px) {
  .block-container {padding: .8rem .75rem 2rem;}
  h1 {font-size: 1.6rem !important;}
  h2 {font-size: 1.22rem !important;}
  h3 {font-size: 1.05rem !important;}
  [data-testid="stMetric"] {padding: 10px;}
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300, show_spinner=False)
def get_json(path, params=None):
    response = requests.get(f"{API}{path}", params=params, timeout=90)
    response.raise_for_status()
    return response.json()

def metric(label, value, help_text=None):
    st.metric(label, value, help=help_text)

def compact_table(frame, height=300):
    st.dataframe(frame, use_container_width=True, hide_index=True, height=height)

def select_store_product(prefix: str):
    """Render searchable, dependent Store and Product dropdowns."""
    try:
        available = get_json("/api/v1/series/options")
        store_ids = available.get("store_ids", [])
        if not store_ids:
            st.info("No store-product choices are available in the published forecast artifact.")
            return None
        with st.container(border=True):
            st.markdown("**Choose a store and product**")
            store_col, product_col = st.columns(2)
            with store_col:
                selected_store = st.selectbox(
                    "Store",
                    store_ids,
                    key=f"{prefix}_store_dropdown",
                    format_func=lambda value: f"Store {value}",
                    help="Start typing a store ID to find it quickly.",
                )
            product_options = get_json(
                "/api/v1/series/options", {"store_id": str(selected_store)}
            ).get("product_ids", [])
            with product_col:
                if not product_options:
                    st.warning("No evaluation products are available for this store.")
                    return None
                selected_product = st.selectbox(
                    "Product",
                    product_options,
                    key=f"{prefix}_product_dropdown",
                    format_func=lambda value: f"Product {value}",
                    help="The product list updates when you change the store.",
                )
            st.caption(
                f"Selected series: Store {selected_store} · Product {selected_product}. "
                "Type inside a dropdown to search its choices."
            )
        return {"store_id": str(selected_store), "product_id": str(selected_product)}
    except Exception as exc:
        st.warning(f"Could not load Store/Product dropdown choices: {exc}")
        return None


st.title("FreshRetailNet Demand Intelligence")
st.caption("Observed-sales forecasting · stockout diagnostics · probabilistic intervals · inventory policy simulation")
overview = {}
health = {}
try:
    health = get_json("/health")
except Exception as exc:
    st.error(f"Could not reach the FastAPI service at {API}. Check API_BASE_URL and the API deployment. Details: {exc}")
    st.stop()

if not health.get("r2_configured"):
    st.warning("The API is alive, but R2 credentials are not configured. Artifact-backed panels will show their own availability messages.")
else:
    try:
        readiness = get_json("/ready")
        st.caption(f"API readiness: {readiness.get('status', 'unknown')} · R2 reachable")
    except Exception as exc:
        st.warning(f"The API is alive but its R2 bucket is not ready: {exc}. Available dashboard sections may be limited.")
    try:
        overview = get_json("/api/v1/overview")
    except Exception as exc:
        st.warning(f"The API is reachable, but the overview could not be loaded. Individual panels may still work: {exc}")
        overview = {}

artifact_status = overview.get("artifact_status", {})
unavailable_overview = [
    name for name, state in artifact_status.items() if not state.get("available", False)
]
if unavailable_overview:
    st.warning(
        "Some overview artifacts are unavailable: "
        + ", ".join(unavailable_overview)
        + ". Other dashboard sections remain available."
    )

point = overview.get("point_forecast") or {}
quant = overview.get("quantile_forecast") or {}
stock = overview.get("stockout") or {}
inventory = overview.get("inventory") or {}
freshness = {
    "Point forecast": point.get("generated_at_utc"),
    "Quantile forecast": quant.get("generated_at_utc"),
    "Stockout diagnostics": stock.get("generated_at_utc"),
}
workspace_pages = {
    "Overview": ["Executive overview"],
    "Explore data": ["Dataset explorer"],
    "Forecasting": ["Forecast explorer", "Uncertainty & calibration"],
    "Availability & risk": ["Stockout diagnostics", "Forecast monitoring", "Demand censoring"],
    "Inventory decisions": ["Inventory optimisation", "Inventory scenarios", "Service-level optimizer"],
    "Experiments": ["Experiment lab", "Wavelet benchmark", "Forecast research benchmark"],
}
st.sidebar.title("Navigation")
st.sidebar.caption("Choose a task area, then open the page you need.")
workspace = st.sidebar.selectbox("Workspace", list(workspace_pages), key="workspace_choice")
active_page = st.sidebar.radio(
    "Page",
    workspace_pages[workspace],
    key=f"workspace_page_{workspace}",
)
st.sidebar.divider()
st.sidebar.caption(
    "Start with the overview, explore the original data, then review forecasts "
    "and operational decisions. Advanced experiments are grouped separately."
)

if active_page == "Executive overview":
    ev = point.get("evaluation", {})
    model = ev.get("model", {})
    baseline = ev.get("seasonal_naive_lag7", {})
    cal = quant.get("conformal_calibration", {}).get("evaluation_calibrated_interval", {})
    inv = inventory.get("base_policy_comparison", [])
    mae_gain = (1 - model["mae"] / baseline["mae"]) * 100 if model.get("mae") and baseline.get("mae") else None
    wape_gain = (1 - model["wape_pct"] / baseline["wape_pct"]) * 100 if model.get("wape_pct") and baseline.get("wape_pct") else None
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric("Evaluation MAE", f'{model.get("mae", float("nan")):.3f}')
    with c2: metric("Evaluation WAPE", f'{model.get("wape_pct", float("nan")):.1f}%')
    with c3: metric("Calibrated interval coverage", f'{cal.get("coverage_pct", float("nan")):.1f}%', "Nominal target: 80%; evaluation coverage can shift.")
    with c4: metric("Training rows", f'{point.get("rows", {}).get("train", 0):,}')
    st.subheader("Improvement over seasonal-naive benchmark")
    g1,g2 = st.columns(2)
    with g1: metric("Relative MAE reduction", f"{mae_gain:.1f}%" if mae_gain is not None else "N/A")
    with g2: metric("Relative WAPE reduction", f"{wape_gain:.1f}%" if wape_gain is not None else "N/A")
    st.caption("Relative reductions compare evaluation metrics against the seasonal-naive lag-7 baseline; they are not business-impact estimates.")
    st.subheader("Point forecast vs seasonal-naive benchmark")
    compare = pd.DataFrame([{"Model":"LightGBM","MAE":model.get("mae"),"WAPE (%)":model.get("wape_pct")},
                            {"Model":"Seasonal naive lag-7","MAE":baseline.get("mae"),"WAPE (%)":baseline.get("wape_pct")}])
    compact_table(compare, 150)
    st.subheader("Inventory policy snapshot")
    if inv:
        inv_df = pd.DataFrame(inv)
        cols = [c for c in ["policy","fill_rate_pct","steady_state_fill_rate_pct","steady_state_total_cost"] if c in inv_df]
        compact_table(inv_df[cols], 180)
    st.subheader("Artifact freshness (UTC)")
    freshness_rows = [{"Artifact": k, "Generated at": v or "Timestamp unavailable"} for k,v in freshness.items()]
    compact_table(pd.DataFrame(freshness_rows), 160)
    st.info("All forecast targets are observed daily sales. They are not guaranteed to represent uncensored customer demand.")

if active_page == "Forecast explorer":
    st.subheader("Evaluation forecast explorer")
    try:
        filters = get_json("/api/v1/filters")
    except Exception as exc:
        filters = None
        st.warning(f"Forecast filters are unavailable: {exc}")
    chosen = select_store_product("forecast") if filters else None
    if not chosen:
        st.info("Select a store and product above to view one series at a time.")
    else:
        start = st.text_input("Start date (YYYY-MM-DD)", filters["dates"]["start"] if filters else "")
        end = st.text_input("End date (YYYY-MM-DD)", filters["dates"]["end"] if filters else "")
        params = {
            "limit": 10000, "date_from": start, "date_to": end,
            "store_id": chosen["store_id"], "product_id": chosen["product_id"],
        }
        try:
            if not filters:
                raise RuntimeError("Forecast filter artifact is unavailable; republish quantile predictions.")
            data = get_json("/api/v1/forecasts", params)
            f = pd.DataFrame(data["rows"])
            if f.empty:
                st.warning("No observations match this store, product and date range.")
            else:
                f["date"] = pd.to_datetime(f["date"])
                f = f.sort_values("date")
                st.caption(f'Showing {len(f):,} of {data["total"]:,} observations for Store {chosen["store_id"]} · Product {chosen["product_id"]}.')
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=f["date"], y=f["p90_calibrated"], mode="lines",
                    line={"width": 0}, name="Calibrated upper", showlegend=False,
                ))
                fig.add_trace(go.Scatter(
                    x=f["date"], y=f["p10_calibrated"], mode="lines",
                    line={"width": 0}, fill="tonexty",
                    fillcolor="rgba(31,119,180,0.18)", name="Calibrated P10–P90 band",
                ))
                fig.add_trace(go.Scatter(
                    x=f["date"], y=f["actual_sale_amount"], mode="lines+markers",
                    name="Observed sales",
                ))
                fig.add_trace(go.Scatter(
                    x=f["date"], y=f["p50"], mode="lines+markers", name="P50 forecast",
                ))
                fig.update_layout(
                    title="Observed sales, median forecast and calibrated interval",
                    xaxis_title="Date", yaxis_title="Normalized daily observed sales",
                    height=420, margin={"l": 15, "r": 15, "t": 55, "b": 15},
                    legend={"orientation": "h", "y": -0.25},
                )
                st.plotly_chart(fig, use_container_width=True)
                display_cols = [
                    c for c in [
                        "date", "actual_sale_amount", "p10", "p50", "p90",
                        "p10_calibrated", "p90_calibrated", "stock_hour6_22_cnt",
                    ] if c in f
                ]
                compact_table(f[display_cols], 300)
        except Exception as exc:
            st.error(f"Forecast query failed: {exc}")


if active_page == "Uncertainty & calibration":
    st.subheader("Probabilistic forecast quality")
    v = quant.get("validation", {})
    e = quant.get("evaluation", {})
    cc = quant.get("conformal_calibration", {})
    rows = []
    for split, obj in (("Validation",v),("Evaluation",e)):
        rows.append({"Split":split,"Raw P10–P90 coverage (%)":obj.get("p10_p90_coverage_pct"),
                     "Raw interval width":obj.get("p10_p90_mean_width"),
                     "P50 MAE":obj.get("median_mae"),"P50 WAPE (%)":obj.get("median_wape_pct")})
    compact_table(pd.DataFrame(rows), 180)
    st.write("Global split-conformal calibration")
    st.json({"nominal_coverage_pct":cc.get("nominal_coverage_pct"),
             "validation":cc.get("validation_calibrated_interval"),
             "evaluation":cc.get("evaluation_calibrated_interval")})

    conditional = quant.get("conditional_conformal_calibration", {})
    if conditional:
        st.markdown("#### Candidate: prior-exposure conditional conformal")
        st.caption(
            "The interval adjustment is chosen using the maximum stockout exposure in the previous seven days. "
            "It does not use the current evaluation-day stock status to choose the interval."
        )
        base_strata = quant.get("evaluation_calibrated_stockout_exposure_strata", {})
        cond_strata = quant.get("evaluation_conditional_calibrated_stockout_exposure_strata", {})
        compare_rows = []
        for bucket, candidate in cond_strata.items():
            baseline = base_strata.get(bucket, {})
            compare_rows.append({
                "Prior exposure context": bucket,
                "Rows": candidate.get("rows", 0),
                "Global coverage (%)": baseline.get("coverage_pct"),
                "Conditional coverage (%)": candidate.get("coverage_pct"),
                "Coverage change (pp)": (
                    candidate.get("coverage_pct", 0) - baseline.get("coverage_pct", 0)
                    if baseline else None
                ),
                "Global width": baseline.get("mean_interval_width"),
                "Conditional width": candidate.get("mean_interval_width"),
            })
        if compare_rows:
            comparison = pd.DataFrame(compare_rows)
            compact_table(comparison, 260)
            high_group = comparison[comparison["Prior exposure context"].astype(str) == "16_hours"]
            if not high_group.empty:
                high = high_group.iloc[0]
                k1, k2, k3 = st.columns(3)
                with k1: metric("16+ hour rows", f"{int(high['Rows']):,}")
                with k2: metric("Global coverage", f"{high['Global coverage (%)']:.1f}%")
                with k3: metric("Conditional coverage", f"{high['Conditional coverage (%)']:.1f}%")
                width_change = (
                    high["Conditional width"] / high["Global width"] - 1
                    if pd.notna(high["Global width"]) and high["Global width"] > 0
                    else float("nan")
                )
                metric("16+ hour interval-width change", f"{width_change * 100:+.1f}%" if pd.notna(width_change) else "N/A")
        st.json({
            "method": conditional.get("method"),
            "context_feature": conditional.get("context_feature"),
            "validation_interval": conditional.get("validation_interval"),
            "evaluation_interval": conditional.get("evaluation_interval"),
            "evaluation_metrics": conditional.get("evaluation_metrics"),
            "validation_group_adjustments": conditional.get("validation_group_adjustments"),
        })
        st.warning(
            "This is a candidate calibration. Higher coverage is not automatically better if interval width grows excessively. "
            "Compare coverage and width together, including the 16+ hour group, before promoting."
        )
    else:
        st.info(
            "No conditional-calibration experiment results are published yet. "
            "The global split-conformal interval remains the active reported method."
        )

    risk_cal = quant.get("risk_conditioned_conformal_calibration", {})
    if risk_cal:
        st.markdown("#### Candidate: predicted high-stockout risk decile calibration")
        st.caption(
            "A LightGBM classifier estimates the probability of 16+ stockout hours using only "
            "prior sales/availability history and calendar features. Conformal adjustments are "
            "fit by validation risk decile and reused on evaluation rows."
        )
        risk_metrics = risk_cal.get("risk_model_metrics", {})
        val_risk = risk_metrics.get("validation", {})
        eval_risk = risk_metrics.get("evaluation", {})
        r1, r2, r3, r4 = st.columns(4)
        with r1: metric("Risk-model evaluation ROC-AUC", f"{eval_risk.get('roc_auc', float('nan')):.3f}")
        with r2: metric("Risk-model evaluation PR-AUC", f"{eval_risk.get('average_precision', float('nan')):.3f}")
        with r3: metric("Top-decile precision", f"{risk_metrics.get('evaluation_top_decile_precision_pct', float('nan')):.1f}%")
        with r4: metric("Top-decile recall", f"{risk_metrics.get('evaluation_top_decile_recall_pct', float('nan')):.1f}%")
        global_risk_strata = quant.get("evaluation_calibrated_stockout_exposure_strata", {})
        candidate_risk_strata = quant.get("evaluation_risk_conditioned_stockout_exposure_strata", {})
        risk_rows = []
        for bucket, candidate in candidate_risk_strata.items():
            baseline = global_risk_strata.get(bucket, {})
            risk_rows.append({
                "Actual exposure": bucket,
                "Rows": candidate.get("rows", 0),
                "Global coverage (%)": baseline.get("coverage_pct"),
                "Risk-conditioned coverage (%)": candidate.get("coverage_pct"),
                "Coverage change (pp)": (
                    candidate.get("coverage_pct", 0) - baseline.get("coverage_pct", 0)
                    if baseline else None
                ),
                "Global width": baseline.get("mean_interval_width"),
                "Risk-conditioned width": candidate.get("mean_interval_width"),
            })
        if risk_rows:
            risk_table = pd.DataFrame(risk_rows)
            compact_table(risk_table, 260)
            high = risk_table[risk_table["Actual exposure"].astype(str) == "16_hours"]
            if not high.empty:
                hr = high.iloc[0]
                c1, c2, c3 = st.columns(3)
                with c1: metric("16+ global coverage", f"{hr['Global coverage (%)']:.1f}%")
                with c2: metric("16+ risk-conditioned coverage", f"{hr['Risk-conditioned coverage (%)']:.1f}%")
                with c3:
                    change = (
                        hr["Risk-conditioned width"] / hr["Global width"] - 1
                        if pd.notna(hr["Global width"]) and hr["Global width"] > 0
                        else float("nan")
                    )
                    metric("16+ width change", f"{change * 100:+.1f}%" if pd.notna(change) else "N/A")
        st.markdown("#### Predicted-risk decile diagnostics")
        compact_table(pd.DataFrame(risk_cal.get("evaluation_risk_deciles", [])), 300)
        st.warning(
            "Risk-conditioned intervals are experimental. Keep the global interval active unless the 16+ hour "
            "coverage improves materially and the interval-width increase is acceptable."
        )
    else:
        st.info("The forecast-time stockout-risk model and risk-decile calibration are not yet published.")
    st.warning("Global held-out coverage remains below the 80% nominal target in the existing benchmark. Read all exposure buckets with their sample sizes; intervals are not yet uniformly calibrated.")

if active_page == "Stockout diagnostics":
    st.subheader("Stockout exposure and observed sales")
    c1,c2,c3 = st.columns(3)
    with c1: metric("Training rows", f'{stock.get("rows",0):,}')
    with c2: metric("Rows with stockout exposure", f'{stock.get("stockout_affected_row_rate",0)*100:.1f}%')
    with c3: metric("Mean stockout hours (06–21)", f'{stock.get("mean_stockout_hours_06_22",0):.2f}')
    try:
        buckets = pd.DataFrame(get_json("/api/v1/stockout/buckets")["rows"])
    except Exception as exc:
        buckets = pd.DataFrame()
        st.warning(f"Stockout bucket artifact unavailable: {exc}")
    if not buckets.empty:
        buckets["zero_sales_rate_pct"] = pd.to_numeric(buckets["zero_sales_rate"], errors="coerce")*100
        fig = px.bar(buckets, x="stockout_bucket", y="mean_observed_sales", title="Mean observed sales by stockout-exposure bucket")
        fig.update_layout(height=380, margin={"l":15,"r":15,"t":55,"b":20})
        st.plotly_chart(fig, use_container_width=True)
        compact_table(buckets, 300)
    try:
        hourly = pd.DataFrame(get_json("/api/v1/stockout/hourly")["rows"])
    except Exception as exc:
        hourly = pd.DataFrame()
        st.warning(f"Hourly stockout artifact unavailable: {exc}")
    if not hourly.empty:
        fig = px.line(hourly, x="hour_of_day", y="out_of_stock_rate", title="Out-of-stock rate by vector index")
        fig.update_layout(height=350, margin={"l":15,"r":15,"t":55,"b":20})
        st.plotly_chart(fig, use_container_width=True)
    st.caption(stock.get("interpretation","Descriptive association only."))
    st.caption(stock.get("hour_index_note","Hourly vector positions are indices; do not infer local-time labels."))

if active_page == "Inventory optimisation":
    st.subheader("Inventory policy backtest")
    base = pd.DataFrame(inventory.get("base_policy_comparison", []))
    if not base.empty:
        compact_cols = [c for c in ["policy","fill_rate_pct","steady_state_fill_rate_pct","steady_state_total_cost","orders_placed"] if c in base]
        compact_table(base[compact_cols], 220)
        fig = px.bar(base, x="policy", y="steady_state_fill_rate_pct", title="Steady-state fill rate by policy", text_auto=".1f")
        fig.update_layout(height=360, yaxis_title="Fill rate (%)", margin={"l":15,"r":15,"t":55,"b":20})
        st.plotly_chart(fig, use_container_width=True)
        cost = base.copy()
        cost["steady_state_cost_thousands"] = pd.to_numeric(cost["steady_state_total_cost"]) / 1000
        fig = px.bar(cost, x="policy", y="steady_state_cost_thousands", title="Steady-state simulated cost by policy", text=cost["steady_state_cost_thousands"].map(lambda x: f"{x:,.1f}k"))
        fig.update_layout(height=360, yaxis_title="Simulated cost (thousands of cost units)",
                          yaxis={"tickformat":",.0f"}, margin={"l":15,"r":15,"t":55,"b":20})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Costs are dimensionless simulation cost units derived from configured penalties and order costs; no currency is specified by the dataset or simulator.")
    try:
        sens = pd.DataFrame(get_json("/api/v1/inventory/sensitivity")["rows"])
    except Exception as exc:
        sens = pd.DataFrame()
        st.warning(f"Inventory sensitivity artifact unavailable: {exc}")
    if not sens.empty:
        st.subheader("Shortage penalty sensitivity")
        cols = [c for c in ["scenario","policy","steady_state_fill_rate_pct","steady_state_total_cost","steady_state_shortage_cost"] if c in sens]
        compact_table(sens[cols], 320)
    st.warning("Costs, lead time, review cadence and starting inventory are simulation assumptions. The short evaluation horizon and observed-sales proxy limit operational interpretation.")


if active_page == "Dataset explorer":
    st.subheader("Explore the original FreshRetailNet dataset")
    st.caption(
        "A visual guide to the original train and evaluation data: all 19 source fields, "
        "data quality, distributions, relationships, daily patterns and hourly signals."
    )
    try:
        eda = get_json("/api/v1/dataset-eda")
        split = eda.get("split_overview", {})
        combined = split.get("combined", {})
        train_summary = split.get("train", {})
        eval_summary = split.get("evaluation", {})
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric("Original observations", f"{int(combined.get('row_count', 0)):,}")
        with c2:
            metric("Source fields", f"{int(eda.get('feature_count', 0))}")
        with c3:
            metric("Stores", f"{int(combined.get('store_count', 0)):,}")
        with c4:
            metric("Store-product series", f"{int(combined.get('store_product_series', 0)):,}")

        if eda.get("generated_at_utc"):
            st.caption(f"Report generated (UTC): {eda['generated_at_utc']} · Includes both source splits")
        st.info(eda.get("source_semantics", "Observed-sales EDA is descriptive; sales can be censored by stock availability."))

        overview_tab, feature_tab, relationship_tab, hourly_tab = st.tabs(
            ["Dataset overview", "Explore every feature", "Relationships & patterns", "Hourly signals"]
        )

        with overview_tab:
            st.markdown("#### Train and evaluation coverage")
            split_rows = []
            for name, summary in (("Train", train_summary), ("Evaluation", eval_summary)):
                split_rows.append({
                    "Split": name,
                    "Rows": summary.get("row_count"),
                    "Cities": summary.get("city_count"),
                    "Stores": summary.get("store_count"),
                    "Products": summary.get("product_count"),
                    "Store-product series": summary.get("store_product_series"),
                    "Dates": summary.get("date_count"),
                    "Start": summary.get("date_start"),
                    "End": summary.get("date_end"),
                    "Duplicate key excess": summary.get("duplicate_key_excess"),
                })
            compact_table(pd.DataFrame(split_rows), 135)

            trend = pd.DataFrame(eda.get("daily_trend", []))
            if not trend.empty:
                trend["date"] = pd.to_datetime(trend["date"])
                st.markdown("#### Daily sales pattern")
                trend_fig = px.line(
                    trend, x="date", y="total_observed_sales", color="split",
                    markers=True, title="Total observed sales by day and source split",
                    hover_data=["store_product_rows", "mean_observed_sales", "zero_sales_rate_pct"],
                )
                trend_fig.update_layout(height=370, xaxis_title="Date", yaxis_title="Normalized observed sales")
                st.plotly_chart(trend_fig, use_container_width=True)
                trend_metric = px.line(
                    trend, x="date", y="zero_sales_rate_pct", color="split",
                    title="Share of store-product rows with zero observed sales",
                )
                trend_metric.update_layout(height=320, yaxis_title="Zero-sales rows (%)")
                st.plotly_chart(trend_metric, use_container_width=True)

            st.markdown("#### Data quality across all original fields")
            quality = pd.DataFrame(eda.get("data_quality", {}).get("features", []))
            if not quality.empty:
                quality["null_pct"] = pd.to_numeric(quality["null_pct"], errors="coerce")
                quality = quality.rename(columns={
                    "feature": "Source field", "label": "Field name",
                    "kind": "Field type", "null_count": "Missing values",
                    "null_pct": "Missing (%)", "distinct_count": "Distinct values",
                })
                compact_table(quality, 380)
                if quality["Missing (%)"].fillna(0).max() == 0:
                    st.success("No missing values were found in the profiled source fields.")
                else:
                    st.warning("Some source fields contain missing values. Review the field-level profile before modelling.")
            st.caption(
                f"Duplicate store-product-date key excess in the combined source: "
                f"{int(eda.get('data_quality', {}).get('duplicate_key_excess', 0)):,}. "
                "The report profiles the original fields before modelling feature engineering."
            )

            st.markdown("#### Original data dictionary")
            catalog = pd.DataFrame(eda.get("feature_catalog", []))
            if not catalog.empty:
                compact_table(catalog.rename(columns={
                    "name": "Source column", "label": "Friendly name",
                    "group": "Feature group", "kind": "Data type", "description": "Meaning",
                }), 430)

        with feature_tab:
            catalog = eda.get("feature_catalog", [])
            if catalog:
                by_label = {item["label"]: item["name"] for item in catalog}
                selected_label = st.selectbox(
                    "Select a source feature",
                    list(by_label.keys()),
                    key="full_eda_feature_choice",
                    help="Search by typing a field name, or browse the feature groups below.",
                )
                feature_name = by_label[selected_label]
                feature_info = next(item for item in catalog if item["name"] == feature_name)
                profile = eda.get("profiles", {}).get(feature_name, {})
                st.markdown(f"#### {selected_label}")
                st.caption(f"{feature_info['group']} · {feature_info['description']}")

                a, b, c = st.columns(3)
                with a:
                    metric("Missing values", f"{int(profile.get('null_count', 0)):,}",
                           f"{profile.get('null_pct', 0):.3f}% of combined rows")
                with b:
                    metric("Distinct values", f"{int(profile.get('distinct_count', 0) or 0):,}"
                           if profile.get("distinct_count") is not None else "Not applicable")
                with c:
                    metric("Field type", feature_info["kind"].replace("_", " ").title())

                if feature_info["kind"] in ("numeric", "target"):
                    stat_cols = st.columns(5)
                    stat_values = [
                        ("Minimum", "min"), ("25th percentile", "p25"),
                        ("Median", "median"), ("75th percentile", "p75"), ("Maximum", "max"),
                    ]
                    for column, (label, field) in zip(stat_cols, stat_values):
                        value = profile.get(field)
                        with column:
                            metric(label, f"{value:,.3f}" if isinstance(value, (int, float)) and value is not None else "N/A")
                    histogram = pd.DataFrame(profile.get("histogram", []))
                    if not histogram.empty:
                        histogram["range"] = histogram.apply(
                            lambda row: f"{row['bin_start']:.3g} – {row['bin_end']:.3g}", axis=1
                        )
                        histogram_fig = px.bar(
                            histogram, x="range", y="count",
                            title=f"Distribution of {selected_label}",
                            labels={"range": "Value range", "count": "Observations"},
                        )
                        histogram_fig.update_layout(height=370, xaxis_tickangle=-35)
                        st.plotly_chart(histogram_fig, use_container_width=True)
                    st.caption(
                        f"Mean: {profile.get('mean', float('nan')):.4g} · "
                        f"Standard deviation: {profile.get('std', float('nan')):.4g} · "
                        f"1st–99th percentile: {profile.get('p01', float('nan')):.4g} to {profile.get('p99', float('nan')):.4g}"
                    )
                elif feature_info["kind"] in ("identifier", "binary"):
                    top_values = pd.DataFrame(profile.get("top_values", []))
                    if not top_values.empty:
                        top_values["value"] = top_values["value"].astype(str)
                        fig = px.bar(
                            top_values.sort_values("count"), x="count", y="value",
                            orientation="h", title=f"Most frequent values in {selected_label}",
                            labels={"count": "Rows", "value": "Encoded value"},
                            text="count",
                        )
                        fig.update_layout(height=390, yaxis_title="")
                        st.plotly_chart(fig, use_container_width=True)
                        compact_table(top_values, 280)
                    st.caption(
                        "IDs are encoded categories, not measured quantities. Their numeric order "
                        "does not imply a higher or lower business value."
                    )
                elif feature_info["kind"] == "date":
                    st.write({
                        "First date": profile.get("min"),
                        "Last date": profile.get("max"),
                        "Distinct dates": profile.get("distinct_count"),
                    })
                elif feature_info["kind"] == "vector":
                    st.markdown("**24-hour sequence diagnostics**")
                    v1, v2, v3 = st.columns(3)
                    with v1:
                        metric("Shortest vector", str(profile.get("min_length", "N/A")))
                    with v2:
                        metric("Average vector length", f"{profile.get('mean_length', float('nan')):.2f}")
                    with v3:
                        metric("Longest vector", str(profile.get("max_length", "N/A")))
                    st.info(
                        "This field contains one value per hour. Use the Hourly signals tab "
                        "to compare the full 24-position profile across train and evaluation."
                    )
            else:
                st.info("The feature catalog is not included in the published EDA report.")

        with relationship_tab:
            st.markdown("#### Numeric relationships")
            correlations = eda.get("correlations", {})
            corr_fields = correlations.get("fields", [])
            corr_matrix = correlations.get("matrix", [])
            if corr_fields and corr_matrix:
                corr_df = pd.DataFrame(corr_matrix, index=corr_fields, columns=corr_fields)
                corr_fig = px.imshow(
                    corr_df, text_auto=".2f", zmin=-1, zmax=1,
                    color_continuous_scale="RdBu_r", origin="lower",
                    title="Correlation between numeric source features",
                    labels={"color": "Correlation"},
                )
                corr_fig.update_layout(height=520)
                st.plotly_chart(corr_fig, use_container_width=True)
                st.caption("Correlation describes linear association; it does not establish causality.")

            left, right = st.columns(2)
            weekday = pd.DataFrame(eda.get("weekday_profile", []))
            with left:
                if not weekday.empty:
                    fig = px.bar(
                        weekday, x="weekday", y="mean_observed_sales",
                        title="Mean observed sales by weekday",
                        hover_data=["observations", "median_observed_sales", "zero_sales_rate_pct"],
                    )
                    fig.update_layout(height=350, xaxis_title="")
                    st.plotly_chart(fig, use_container_width=True)
            calendar = pd.DataFrame(eda.get("calendar_profile", []))
            with right:
                if not calendar.empty:
                    calendar["calendar_group"] = calendar.apply(
                        lambda row: f"Holiday {int(row['holiday_flag'])} · Activity {int(row['activity_flag'])}",
                        axis=1,
                    )
                    fig = px.bar(
                        calendar, x="calendar_group", y="mean_observed_sales",
                        title="Observed sales by calendar flags",
                        hover_data=["observations", "median_observed_sales", "zero_sales_rate_pct"],
                    )
                    fig.update_layout(height=350, xaxis_title="", xaxis_tickangle=-25)
                    st.plotly_chart(fig, use_container_width=True)

            discount = pd.DataFrame(eda.get("discount_profile", []))
            stockout = pd.DataFrame(eda.get("stockout_profile", []))
            left, right = st.columns(2)
            with left:
                if not discount.empty:
                    fig = px.bar(
                        discount, x="discount_band", y="mean_observed_sales",
                        title="Observed sales by discount band",
                        hover_data=["observations", "mean_discount_factor"],
                    )
                    fig.update_layout(height=350, xaxis_title="")
                    st.plotly_chart(fig, use_container_width=True)
            with right:
                if not stockout.empty:
                    fig = px.bar(
                        stockout, x="exposure_band", y="mean_observed_sales",
                        title="Observed sales by stockout exposure",
                        hover_data=["observations", "mean_stockout_hours", "zero_sales_rate_pct"],
                    )
                    fig.update_layout(height=350, xaxis_title="")
                    st.plotly_chart(fig, use_container_width=True)

            store_df = pd.DataFrame(eda.get("top_stores", []))
            product_df = pd.DataFrame(eda.get("top_products", []))
            left, right = st.columns(2)
            with left:
                if not store_df.empty:
                    fig = px.bar(
                        store_df.sort_values("total_observed_sales"),
                        x="total_observed_sales", y="store_id", orientation="h",
                        title="Top stores by observed sales",
                    )
                    fig.update_layout(height=430, yaxis_title="")
                    st.plotly_chart(fig, use_container_width=True)
            with right:
                if not product_df.empty:
                    fig = px.bar(
                        product_df.sort_values("total_observed_sales"),
                        x="total_observed_sales", y="product_id", orientation="h",
                        title="Top products by observed sales",
                    )
                    fig.update_layout(height=430, yaxis_title="")
                    st.plotly_chart(fig, use_container_width=True)

            categories = pd.DataFrame(eda.get("category_profile", []))
            if not categories.empty:
                st.markdown("#### Product hierarchy")
                compact_table(categories, 300)

            st.warning(
                "These are descriptive comparisons. Sales can be censored by product availability, "
                "and differences across discounts, weather or calendar groups are not causal effects."
            )

        with hourly_tab:
            st.markdown("#### Hourly observed-sales profile")
            hourly_sales = pd.DataFrame(eda.get("hourly_sales", []))
            if not hourly_sales.empty:
                sales_fig = px.line(
                    hourly_sales, x="hour_index", y="mean_hourly_sales",
                    color="split", markers=True,
                    title="Average normalized observed sales at each hourly position",
                    hover_data=["observations", "median_hourly_sales", "zero_hour_rate_pct"],
                )
                sales_fig.update_layout(height=380, xaxis_title="Hour index (0–23)", yaxis_title="Mean normalized sales")
                st.plotly_chart(sales_fig, use_container_width=True)
                zero_fig = px.line(
                    hourly_sales, x="hour_index", y="zero_hour_rate_pct",
                    color="split", markers=True,
                    title="Zero-sales share by hourly position",
                )
                zero_fig.update_layout(height=330, xaxis_title="Hour index (0–23)", yaxis_title="Zero-sales hours (%)")
                st.plotly_chart(zero_fig, use_container_width=True)

            st.markdown("#### Hourly stock-status profile")
            hourly_status = pd.DataFrame(eda.get("hourly_status", []))
            if not hourly_status.empty:
                status_fig = px.line(
                    hourly_status, x="hour_index", y="mean_status_value",
                    color="split", markers=True,
                    title="Average stock-status value by hourly position",
                    hover_data=["observations", "value_one_pct"],
                )
                status_fig.update_layout(height=360, xaxis_title="Hour index (0–23)", yaxis_title="Mean encoded status")
                st.plotly_chart(status_fig, use_container_width=True)
                st.caption(
                    "The source stores 24 hourly positions. The chart retains the source's zero-based "
                    "hour indices; confirm time-zone alignment before translating an index into local clock time."
                )
        with st.expander("Advanced: inspect the precomputed SQL queries"):
            try:
                sql_eda = get_json("/api/v1/sql-eda")
                query_items = sql_eda.get("queries", [])
                if query_items:
                    titles = [item["title"] for item in query_items]
                    selected_query = st.selectbox(
                        "SQL query", range(len(titles)),
                        format_func=lambda index: titles[index],
                        key="advanced_sql_query_choice",
                    )
                    item = query_items[selected_query]
                    st.code(item["sql"], language="sql")
                    result = pd.DataFrame(item.get("rows", []), columns=item.get("columns", []))
                    compact_table(result, 260)
                else:
                    st.info("No precomputed SQL results are published yet.")
            except Exception as exc:
                st.caption(f"Advanced SQL output is unavailable: {exc}")
    except requests.HTTPError as exc:
        st.warning(
            "The full dataset EDA report is not published yet. Run the Full Dataset EDA workflow "
            "from GitHub Actions, then refresh this page. Details: " + str(exc)
        )
    except Exception as exc:
        st.error(f"Could not load the full dataset EDA report: {exc}")


if active_page == "Experiment lab":
    st.subheader("Experiment lab")
    st.caption("Run a configurable, reproducible what-if inventory experiment on one held-out store-product series. Results are a small-sample simulation, not causal estimates or operational savings.")
    try:
        exp_filters = get_json("/api/v1/filters")
    except Exception as exc:
        exp_filters = {}
        st.warning(f"Experiment inputs are unavailable: {exc}")
    exp_chosen = select_store_product("experiment") if exp_filters else None
    if exp_chosen:
        selected_series_key = f"{exp_chosen['store_id']}::{exp_chosen['product_id']}"
        if st.session_state.get("freshretail_experiment_series_key") != selected_series_key:
            for state_key in (
                "freshretail_last_experiment",
                "freshretail_last_experiment_curves",
                "freshretail_last_experiment_config",
            ):
                st.session_state.pop(state_key, None)
            st.session_state["freshretail_experiment_series_key"] = selected_series_key
    if not exp_chosen:
        st.info("Choose a store and product above to configure an inventory experiment.")
    else:
        p1, p2, p3 = st.columns(3)
        with p1:
            lead_days = st.number_input("Lead time (days)", min_value=1, max_value=7, value=1, step=1)
            review_days = st.number_input("Review period (days)", min_value=1, max_value=7, value=1, step=1)
        with p2:
            starting_stock = st.number_input("Starting inventory (units)", min_value=0.0, value=0.0, step=1.0)
            holding_rate = st.number_input("Holding cost / unit / day", min_value=0.0, value=0.05, step=0.05, format="%.3f")
        with p3:
            shortage_rate = st.number_input("Shortage penalty / unit", min_value=0.0, value=2.0, step=0.5)
            order_rate = st.number_input("Ordering cost / order", min_value=0.0, value=5.0, step=1.0)
        exp_start = exp_filters.get("dates", {}).get("start", "")
        exp_end = exp_filters.get("dates", {}).get("end", "")
        if st.button("Run experiment", type="primary", use_container_width=True, disabled=exp_chosen is None):
            try:
                with st.spinner("Loading held-out series and simulating policies..."):
                    exp_data = get_json("/api/v1/forecasts", {
                        "store_id": exp_chosen["store_id"], "product_id": exp_chosen["product_id"],
                        "date_from": exp_start, "date_to": exp_end, "limit": 100
                    })
                experiment_df = pd.DataFrame(exp_data.get("rows", []))
                if experiment_df.empty:
                    st.error("No evaluation observations were returned for this series.")
                else:
                    experiment_df["date"] = pd.to_datetime(experiment_df["date"])
                    experiment_df = experiment_df.sort_values("date").reset_index(drop=True)
                    policy_cols = {
                        "Seasonal naive lag-7": "seasonal_naive_lag7",
                        "Median forecast P50": "p50",
                        "Calibrated upper P90": "p90_calibrated",
                    }
                    outputs = []
                    curves = []
                    for policy_name, forecast_col in policy_cols.items():
                        inventory_level = float(starting_stock)
                        pipeline = {}
                        fulfilled = 0.0
                        shortage_units = 0.0
                        holding_total = 0.0
                        ordering_total = 0.0
                        orders = 0
                        stock_path = []
                        for day_idx, row in experiment_df.iterrows():
                            inventory_level += pipeline.pop(day_idx, 0.0)
                            if day_idx % int(review_days) == 0:
                                horizon = min(len(experiment_df), day_idx + int(lead_days) + int(review_days))
                                target = max(0.0, float(experiment_df.loc[day_idx:horizon-1, forecast_col].clip(lower=0).sum()))
                                position = inventory_level + sum(pipeline.values())
                                quantity = max(0.0, float(__import__("math").ceil(target - position)))
                                arrival = day_idx + int(lead_days)
                                if quantity > 0 and arrival < len(experiment_df):
                                    pipeline[arrival] = pipeline.get(arrival, 0.0) + quantity
                                    orders += 1
                                    ordering_total += float(order_rate)
                            actual = max(0.0, float(row["actual_sale_amount"]))
                            sold = min(inventory_level, actual)
                            fulfilled += sold
                            shortage_units += actual - sold
                            inventory_level = max(0.0, inventory_level - sold)
                            holding_total += inventory_level * float(holding_rate)
                            stock_path.append(inventory_level)
                            curves.append({"date": row["date"], "policy": policy_name, "ending_inventory": inventory_level})
                        observed = float(experiment_df["actual_sale_amount"].clip(lower=0).sum())
                        total_cost = holding_total + shortage_units * float(shortage_rate) + ordering_total
                        outputs.append({
                            "policy": policy_name, "observed_sales_proxy_units": observed,
                            "fulfilled_units": fulfilled,
                            "fill_rate_pct": fulfilled / observed * 100 if observed else 0.0,
                            "unfulfilled_observed_sales_units": shortage_units,
                            "orders_placed": orders, "holding_cost": holding_total,
                            "shortage_cost": shortage_units * float(shortage_rate),
                            "ordering_cost": ordering_total, "total_simulated_cost": total_cost,
                        })
                    result_df = pd.DataFrame(outputs)
                    st.session_state["freshretail_last_experiment"] = result_df
                    st.session_state["freshretail_last_experiment_curves"] = pd.DataFrame(curves)
                    st.session_state["freshretail_last_experiment_config"] = {
                        "store_id": exp_chosen["store_id"], "product_id": exp_chosen["product_id"],
                        "lead_time_days": int(lead_days), "review_period_days": int(review_days),
                        "initial_stock_units": float(starting_stock), "holding_cost_per_unit_day": float(holding_rate),
                        "shortage_penalty_per_unit": float(shortage_rate), "ordering_cost_per_order": float(order_rate),
                        "evaluation_start": str(experiment_df["date"].min().date()),
                        "evaluation_end": str(experiment_df["date"].max().date()),
                    }
                    st.success("Experiment completed. Results are stored in this browser session.")
            except Exception as exc:
                st.error(f"Experiment failed: {exc}")
        if "freshretail_last_experiment" in st.session_state:
            result_df = st.session_state["freshretail_last_experiment"]
            curves_df = st.session_state["freshretail_last_experiment_curves"]
            config = st.session_state["freshretail_last_experiment_config"]
            st.markdown("#### Experiment results")
            k1, k2, k3 = st.columns(3)
            best_fill = result_df.loc[result_df["fill_rate_pct"].idxmax()]
            best_cost = result_df.loc[result_df["total_simulated_cost"].idxmin()]
            with k1: metric("Highest fill rate", f"{best_fill['fill_rate_pct']:.1f}%", best_fill["policy"])
            with k2: metric("Lowest simulated cost", f"{best_cost['total_simulated_cost']:,.2f}", best_cost["policy"])
            with k3: metric("Series", f"{config['store_id']} / {config['product_id']}")
            compact_table(result_df, 260)
            cost_fig = px.bar(result_df, x="policy", y="total_simulated_cost", title="Simulated cost by policy", text_auto=",.2f")
            cost_fig.update_layout(height=360, yaxis_title="Simulation cost units")
            st.plotly_chart(cost_fig, use_container_width=True)
            inv_fig = px.line(curves_df, x="date", y="ending_inventory", color="policy", title="Ending inventory trajectory")
            inv_fig.update_layout(height=360, yaxis_title="Ending inventory (units)")
            st.plotly_chart(inv_fig, use_container_width=True)
            st.download_button("Download experiment results (CSV)", result_df.to_csv(index=False), "freshretailnet_experiment_results.csv", "text/csv")
            st.download_button("Download experiment configuration (JSON)", __import__("json").dumps(config, indent=2), "freshretailnet_experiment_config.json", "application/json")
            st.warning("Interpretation limits: this experiment uses observed sales as a demand proxy, has a short evaluation horizon, and assumes deterministic lead time. It does not estimate causal effects or real-world savings. Orders arriving after the evaluation horizon are suppressed.")


if active_page == "Wavelet benchmark":
    from wavelet_tab import render_wavelet_tab
    render_wavelet_tab(API, get_json, metric, compact_table)



if active_page == "Forecast monitoring":
    st.subheader("Forecast quality monitoring")
    st.caption("Batch monitoring of the published evaluation predictions. Error and coverage are lagging metrics and require actual outcomes.")
    try:
        monitor = get_json("/api/v1/monitoring/forecast-quality")
        overall = monitor.get("overall", {})
        m1, m2, m3, m4 = st.columns(4)
        with m1: metric("MAE", f"{overall.get('mae', float('nan')):.3f}")
        with m2: metric("WAPE", f"{overall.get('wape_pct', float('nan')):.1f}%")
        with m3: metric("Calibrated coverage", f"{overall.get('calibrated_p10_p90_coverage_pct', float('nan')):.1f}%", f"Nominal {monitor.get('nominal_interval_coverage_pct', 80):.0f}%")
        with m4: metric("Forecast bias", f"{overall.get('bias_pct', float('nan')):.1f}%")
        st.caption(f"Monitoring period: {monitor.get('evaluation_period', {}).get('start')} to {monitor.get('evaluation_period', {}).get('end')}")
        alerts = monitor.get("alerts", [])
        if alerts:
            for alert in alerts:
                st.warning(f"{alert.get('code')}: {alert.get('message')}")
        else:
            st.success("No configured alert thresholds were breached in this batch.")
        try:
            drift_data = get_json("/api/v1/monitoring/forecast-quality/drift")
            drift_df = pd.DataFrame(drift_data.get("rows", []))
            st.markdown("#### Distribution shift (PSI)")
            if not drift_df.empty:
                compact_table(drift_df, 220)
        except Exception as exc:
            st.info(f"Drift report is not available yet: {exc}")
        try:
            daily_data = get_json("/api/v1/monitoring/forecast-quality/daily")
            daily_df = pd.DataFrame(daily_data.get("rows", []))
            if not daily_df.empty:
                daily_df["date"] = pd.to_datetime(daily_df["date"])
                st.markdown("#### Daily forecast error and interval coverage")
                left_metric, right_metric = st.columns(2)
                with left_metric:
                    mae_fig = px.line(daily_df, x="date", y="mae", markers=True,
                                      title="Daily P50 MAE")
                    mae_fig.update_layout(height=330, yaxis_title="Observed-sales units")
                    st.plotly_chart(mae_fig, use_container_width=True)
                with right_metric:
                    coverage_fig = px.line(daily_df, x="date", y="calibrated_coverage_pct",
                                            markers=True, title="Daily calibrated P10–P90 coverage")
                    coverage_fig.add_hline(y=80, line_dash="dash", annotation_text="80% nominal")
                    coverage_fig.update_layout(height=330, yaxis_title="Coverage (%)")
                    st.plotly_chart(coverage_fig, use_container_width=True)
                compact_table(daily_df, 260)
        except Exception as exc:
            st.info(f"Daily metrics are not available yet: {exc}")
        try:
            exposure_data = get_json("/api/v1/monitoring/forecast-quality/exposure")
            exposure_df = pd.DataFrame(exposure_data.get("rows", []))
            st.markdown("#### Calibrated interval coverage by stockout exposure")
            if not exposure_df.empty:
                exposure_fig = px.bar(
                    exposure_df, x="stockout_exposure_bucket", y="calibrated_coverage_pct",
                    text="rows", title="P10–P90 coverage by stockout hours (06:00–22:00)"
                )
                exposure_fig.add_hline(y=80, line_dash="dash", annotation_text="80% nominal")
                exposure_fig.update_layout(height=340, yaxis_title="Coverage (%)", xaxis_title="Observed stockout hours")
                st.plotly_chart(exposure_fig, use_container_width=True)
                compact_table(exposure_df, 220)
            time_exposure_data = get_json("/api/v1/monitoring/forecast-quality/time-exposure")
            time_exposure_df = pd.DataFrame(time_exposure_data.get("rows", []))
            if not time_exposure_df.empty:
                time_exposure_df["date"] = pd.to_datetime(time_exposure_df["date"])
                time_fig = px.line(
                    time_exposure_df, x="date", y="calibrated_coverage_pct",
                    color="stockout_exposure_bucket", markers=True,
                    title="Coverage over time by stockout-exposure segment"
                )
                time_fig.add_hline(y=80, line_dash="dash", annotation_text="80% nominal")
                time_fig.update_layout(height=400, yaxis_title="Coverage (%)")
                st.plotly_chart(time_fig, use_container_width=True)
        except Exception as exc:
            st.info(f"Stockout-segment coverage artifacts are not published yet: {exc}")
        st.caption("PSI compares the first and second chronological halves of the available evaluation window. It is a retrospective drift signal, not a live training-versus-production test. Exposure-stratified coverage is descriptive and should always be read alongside sample sizes.")

    except Exception as exc:
        st.warning(f"Monitoring artifacts are not published yet. Run the Forecast Quality and Drift Monitoring workflow. Details: {exc}")


if active_page == "Demand censoring":
    st.subheader("Demand censoring sensitivity")
    st.caption("Observed sales during out-of-stock hours can understate demand. These are explicit imputation scenarios, not causal lost-sales estimates.")
    try:
        censoring = get_json("/api/v1/demand-censoring/sensitivity")
        c1, c2, c3 = st.columns(3)
        with c1: metric("Evaluation rows", f"{censoring.get('evaluation_rows', 0):,}")
        with c2: metric("Stockout-affected rows", f"{censoring.get('stockout_affected_row_rate_pct', float('nan')):.1f}%")
        with c3: metric("Central imputation proxy", f"{censoring.get('expected_lost_sales_proxy_units', 0):,.1f}")
        st.caption(f"Evaluation period: {censoring.get('evaluation_start')} to {censoring.get('evaluation_end')}")
        validation = censoring.get("reconstruction_validation", {})
        if validation:
            st.markdown("#### Out-of-time pseudo-censoring validation")
            st.caption("Known in-stock evaluation hours are predicted with training-only rates and compared with observed sales. This validates available-hour predictions, not latent demand during real stockouts.")
            v1, v2, v3, v4 = st.columns(4)
            with v1: metric("Validation hours", f"{validation.get('rows', 0):,}")
            with v2: metric("Hourly MAE", f"{validation.get('hourly_mae', float('nan')):.3f}")
            with v3: metric("Hourly WAPE", f"{validation.get('hourly_wape_pct', float('nan')):.1f}%")
            with v4: metric("Fallback rate", f"{validation.get('global_fallback_rate_pct', float('nan')):.1f}%")
        st.info(censoring.get("interpretation", "Sensitivity assumptions only."))
        try:
            bucket_data = get_json("/api/v1/demand-censoring/sensitivity/buckets")
            bucket_df = pd.DataFrame(bucket_data.get("rows", []))
            if not bucket_df.empty:
                st.markdown("#### Sales and imputation sensitivity by stockout exposure")
                compact_table(bucket_df, 300)
                chart_cols = [x for x in ["counterfactual_sales_lower_units", "counterfactual_sales_central_units", "counterfactual_sales_upper_units"] if x in bucket_df]
                if chart_cols:
                    chart_df = bucket_df.melt(id_vars=["stockout_bucket"], value_vars=chart_cols,
                                              var_name="scenario", value_name="units")
                    fig = px.bar(chart_df, x="stockout_bucket", y="units", color="scenario",
                                 barmode="group", title="Counterfactual sales sensitivity by exposure")
                    fig.update_layout(height=380, yaxis_title="Scenario units")
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.info(f"Exposure breakdown is not available yet: {exc}")
        st.warning("The central proxy imputes each out-of-stock hour using historical in-stock sales for the same store-product-hour, with a global hourly fallback. It assumes those rates represent demand during stockouts; validate before using as a label.")
    except Exception as exc:
        st.warning(f"Demand-censoring artifacts are not published yet. Run the Demand Censoring Sensitivity workflow. Details: {exc}")


if active_page == "Inventory scenarios":
    st.subheader("Inventory policy scenario grid")
    st.caption("Compare replenishment policies across lead times, review cycles, storage capacity, minimum order quantities and pack-size assumptions.")
    try:
        scenario_summary = get_json("/api/v1/inventory/scenarios")
        scenario_rows = get_json("/api/v1/inventory/scenarios/comparison")
        scenario_df = pd.DataFrame(scenario_rows.get("rows", []))
        st.caption(
            f"{scenario_summary.get('evaluation_rows', 0):,} evaluation rows · "
            f"{scenario_summary.get('store_product_series', 0):,} store-product series · "
            f"{scenario_summary.get('evaluation_start')} to {scenario_summary.get('evaluation_end')}"
        )
        if not scenario_df.empty:
            scenario_names = sorted(scenario_df["scenario"].dropna().astype(str).unique())
            chosen_scenario = st.selectbox("Scenario", scenario_names, key="inventory_scenario_choice")
            selected_scenario = scenario_df[scenario_df["scenario"].astype(str) == chosen_scenario]
            display_cols = [c for c in [
                "policy", "lead_time_days", "review_period_days", "fill_rate_pct",
                "steady_state_fill_rate_pct", "total_simulated_cost",
                "steady_state_total_cost", "constraint_limited_units",
                "constraint_limited_orders", "shortage_penalty_per_unit",
                "storage_capacity_basis", "max_order_basis", "pack_size",
            ] if c in selected_scenario]
            compact_table(selected_scenario[display_cols], 300)
            fig = px.scatter(
                selected_scenario, x="total_simulated_cost", y="fill_rate_pct",
                color="policy", hover_data=[c for c in ["lead_time_days", "review_period_days", "pack_size"] if c in selected_scenario],
                title="Fill rate versus simulated cost"
            )
            fig.update_layout(height=380, xaxis_title="Total simulated cost", yaxis_title="Fill rate (%)")
            st.plotly_chart(fig, use_container_width=True)
        st.warning("All capacity, MOQ, pack-size, cost and lead-time values are simulation assumptions. Observed sales may be censored; these results are not measured operational savings.")
    except Exception as exc:
        st.warning(f"Inventory scenario artifacts are not published yet. Run the Inventory Lead-Time and Constraint Scenarios workflow. Details: {exc}")


if active_page == "Service-level optimizer":
    st.subheader("Service-level inventory decision engine")
    st.caption(
        "Store-level mixed-integer optimization of product replenishment quantities. "
        "The optimizer balances expected shortage, holding and ordering costs while "
        "respecting the selected fill-rate target and shared capacity limits."
    )
    try:
        optimizer = get_json("/api/v1/inventory/optimizer")
        population = optimizer.get("evaluation_population", {})
        st.info(
            "This experiment uses the separate FreshRetailNet-LT external population, "
            "not the original FreshRetailNet-50K cohort. Costs, MOQ, pack size and capacity "
            "are normalized illustrative assumptions."
        )
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            metric("Continuous evaluation days", f"{population.get('target_dates', 0):,}")
        with k2:
            metric("Complete store-product series", f"{population.get('complete_continuous_series', 0):,}")
        with k3:
            metric("Calibration windows / SKU", "5")
        with k4:
            metric("Evaluation windows / SKU", "5")

        scenario_results = pd.DataFrame(optimizer.get("scenario_results", []))
        store_payload = get_json("/api/v1/inventory/optimizer/stores")
        store_rows = pd.DataFrame(store_payload.get("rows", []))
        if not scenario_results.empty:
            st.markdown("#### Service-level and cost comparison")
            display = scenario_results.copy()
            display["theoretical_expected_fill_rate_pct"] = (
                display["theoretical_expected_fill_rate"] * 100
            )
            display["empirical_fill_rate_pct"] = display["empirical_fill_rate"] * 100
            summary_cols = [
                "scenario", "target_fill_rate", "theoretical_expected_fill_rate_pct",
                "empirical_fill_rate_pct", "empirical_cycle_service_level",
                "total_simulated_cost", "capacity_limited_store_count",
                "service_target_met_store_count",
            ]
            summary_cols = [column for column in summary_cols if column in display]
            compact_table(display[summary_cols], 240)

            fig = px.bar(
                display, x="scenario",
                y=["theoretical_expected_fill_rate_pct", "empirical_fill_rate_pct"],
                barmode="group", title="Expected service target versus later realized fill rate",
                labels={"value": "Fill rate (%)", "variable": "Measurement"},
            )
            fig.update_layout(height=370, xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

            chosen_scenario = st.selectbox(
                "Inspect a capacity scenario",
                display["scenario"].astype(str).tolist(),
                key="inventory_optimizer_scenario",
            )
            selected_summary = display[display["scenario"].astype(str) == chosen_scenario].iloc[0]
            s1, s2, s3 = st.columns(3)
            with s1:
                metric("Target fill rate", f"{selected_summary['target_fill_rate'] * 100:.1f}%")
            with s2:
                metric("Empirical fill rate", f"{selected_summary['empirical_fill_rate'] * 100:.1f}%")
            with s3:
                metric(
                    "Stores where target was feasible",
                    f"{int(selected_summary['service_target_met_store_count']):,} / {int(selected_summary['store_count']):,}",
                )

            if not store_rows.empty:
                chosen_stores = store_rows[store_rows["scenario"].astype(str) == chosen_scenario].copy()
                if not chosen_stores.empty:
                    chosen_stores["achieved_fill_pct"] = pd.to_numeric(
                        chosen_stores["achieved_fill_rate"], errors="coerce"
                    ) * 100
                    chosen_stores["target_fill_pct"] = pd.to_numeric(
                        chosen_stores["target_fill_rate"], errors="coerce"
                    ) * 100
                    scatter = px.scatter(
                        chosen_stores,
                        x="storage_utilization_pct",
                        y="achieved_fill_pct",
                        color="service_target_met",
                        hover_data=["store_id", "replenishment_utilization_pct", "expected_cost"],
                        title="Store capacity utilization versus optimized expected fill rate",
                        labels={
                            "storage_utilization_pct": "Storage utilization (%)",
                            "achieved_fill_pct": "Expected fill rate (%)",
                            "service_target_met": "Target feasible",
                        },
                    )
                    scatter.add_hline(
                        y=float(selected_summary["target_fill_rate"]) * 100,
                        line_dash="dash", annotation_text="Target service level",
                    )
                    scatter.update_layout(height=390)
                    st.plotly_chart(scatter, use_container_width=True)
                    compact_table(
                        chosen_stores.sort_values("achieved_fill_pct").head(100)[
                            [column for column in [
                                "store_id", "status", "target_fill_pct", "achieved_fill_pct",
                                "storage_utilization_pct", "replenishment_utilization_pct",
                                "expected_shortage", "expected_cost",
                            ] if column in chosen_stores]
                        ],
                        300,
                    )
            store_options = sorted(
                store_rows["store_id"].astype(str).unique().tolist()
            ) if not store_rows.empty else []
            if store_options:
                selected_store = st.selectbox(
                    "Inspect product-level order recommendations",
                    store_options,
                    key="inventory_optimizer_store",
                )
                decisions = get_json(
                    "/api/v1/inventory/optimizer/decisions",
                    {"scenario": chosen_scenario, "store_id": selected_store, "limit": 2000},
                )
                decision_df = pd.DataFrame(decisions.get("rows", []))
                if not decision_df.empty:
                    cols = [
                        column for column in [
                            "product_id", "demand_mean", "demand_std", "inventory_position",
                            "order_qty", "stock_position", "expected_fulfilled",
                            "expected_shortage", "expected_leftover", "cycle_service_level",
                            "expected_cost",
                        ] if column in decision_df
                    ]
                    compact_table(decision_df[cols], 320)
            st.warning(
                "The theoretical fill-rate constraint is enforced in the optimizer, but actual "
                "service is measured on five later non-overlapping seven-day periods. A gap is "
                "useful evidence of calibration/model risk, not proof that the SLA is guaranteed."
            )
        else:
            st.info("Inventory optimizer summary contains no scenario results yet.")
    except Exception as exc:
        st.warning(
            "Inventory optimizer artifacts have not been published or the API cannot read them. "
            "Run the Service-Level Inventory Decision Optimization workflow. Details: " + str(exc)
        )


if active_page == "Forecast research benchmark":
    st.subheader("Advanced time-series research benchmark")
    st.caption(
        "Chronological comparisons of Fourier/STL features, spectral diagnostics, "
        "wavelet ablation, residual correction and validation-selected model ensembles."
    )
    try:
        research = get_json("/api/v1/forecast-research/summary")
        rolling = pd.DataFrame(
            get_json("/api/v1/forecast-research/rolling-comparison").get("rows", [])
        )
        final = pd.DataFrame(
            get_json("/api/v1/forecast-research/final-comparison").get("rows", [])
        )
        st.info(
            f"Deterministic sample: {research.get('series_sample_size_complete', 0):,} complete "
            "store-product series. These are research-sample metrics, not full-population estimates."
        )
        if not rolling.empty:
            st.markdown("#### Expanding-window rolling benchmark")
            roll_fig = px.bar(
                rolling, x="model", y="mae",
                title="Pooled MAE across three chronological test windows",
                hover_data=["rmse", "wape_pct", "bias_pct"],
            )
            roll_fig.update_layout(height=370, xaxis_title="", yaxis_title="MAE")
            st.plotly_chart(roll_fig, use_container_width=True)
            compact_table(rolling, 260)

        if not final.empty:
            st.markdown("#### Original seven-day evaluation comparison")
            final_fig = px.bar(
                final, x="model", y="mae",
                title="Research model comparison on the original evaluation split",
                hover_data=["rmse", "wape_pct", "bias_pct"],
            )
            final_fig.update_layout(height=370, xaxis_title="", yaxis_title="MAE")
            st.plotly_chart(final_fig, use_container_width=True)
            compact_table(final, 260)

        final_design = research.get("final_research_comparison", {})
        weights = final_design.get("ensemble_weights", {})
        if weights:
            st.markdown("#### Ensemble weights selected on validation data")
            weight_df = pd.DataFrame(
                [{"model": key, "weight": value} for key, value in weights.items()]
            )
            weight_fig = px.bar(
                weight_df, x="model", y="weight",
                title="Non-negative ensemble weights (sum to 1)",
            )
            weight_fig.update_layout(height=330, yaxis_title="Weight", xaxis_title="")
            st.plotly_chart(weight_fig, use_container_width=True)
            compact_table(weight_df, 200)

        bootstrap = final_design.get("bootstrap_vs_baseline", {})
        if bootstrap.get("comparisons_vs_baseline"):
            st.markdown("#### Paired bootstrap uncertainty")
            rows = []
            for model_name, metrics in bootstrap["comparisons_vs_baseline"].items():
                for metric_name, values in metrics.items():
                    rows.append({
                        "Model": model_name,
                        "Metric": metric_name,
                        "Mean improvement (%)": values.get("mean_improvement_pct"),
                        "95% CI low (%)": values.get("ci_95_low_pct"),
                        "95% CI high (%)": values.get("ci_95_high_pct"),
                        "CI excludes zero": values.get("ci_excludes_zero"),
                    })
            compact_table(pd.DataFrame(rows), 320)

        st.caption(
            "Interpretation: Fourier and STL features are fixed-origin and history-only; "
            "wavelet inputs are trailing and causal; ensemble weights are frozen from validation. "
            "The seven-day source evaluation was already used in earlier project benchmarks, "
            "so these figures are not a new independent production-promotion test."
        )
    except Exception as exc:
        st.warning(
            "Forecast research results are not published yet. Run the Forecast Research Benchmark "
            "workflow. Details: " + str(exc)
        )
