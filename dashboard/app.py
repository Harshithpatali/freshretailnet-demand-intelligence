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
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px;}
[data-testid="stDataFrame"], [data-testid="stTable"] {width: 100%; overflow-x: auto;}
[data-testid="stMetricValue"] {font-size: clamp(1.25rem, 4vw, 2rem);}
@media (max-width: 640px) {
  .block-container {padding-left: .8rem; padding-right: .8rem;}
  h1 {font-size: 1.65rem !important;}
  h2 {font-size: 1.25rem !important;}
  h3 {font-size: 1.05rem !important;}
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

st.title("FreshRetailNet Demand Intelligence")
st.caption("Observed-sales forecasting · stockout diagnostics · probabilistic intervals · inventory policy simulation")
try:
    health = get_json("/health")
    if not health.get("r2_configured"):
        st.warning("The API is reachable, but R2 credentials are not configured on the API service.")
        st.stop()
    overview = get_json("/api/v1/overview")
except Exception as exc:
    st.error(f"Could not load the API at {API}. Verify API_BASE_URL and API deployment. Details: {exc}")
    st.stop()

point = overview["point_forecast"]
quant = overview["quantile_forecast"]
stock = overview["stockout"]
inventory = overview["inventory"]
freshness = {
    "Point forecast": point.get("generated_at_utc"),
    "Quantile forecast": quant.get("generated_at_utc"),
    "Stockout diagnostics": stock.get("generated_at_utc"),
}
tabs = st.tabs(["Executive overview", "Forecast explorer", "Uncertainty & calibration", "Stockout diagnostics", "Inventory optimisation", "SQL EDA"])

with tabs[0]:
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

with tabs[1]:
    st.subheader("Evaluation forecast explorer")
    filters = get_json("/api/v1/filters")
    series_options = filters.get("series", [])
    if series_options:
        labels = [f"Store {x['store_id']} · Product {x['product_id']}" for x in series_options]
        selected = st.selectbox("Select one store-product series", range(len(labels)), format_func=lambda i: labels[i])
        chosen = series_options[selected]
        st.caption(f"Selected series: store {chosen['store_id']} / product {chosen['product_id']}")
    else:
        st.warning("The API has not returned the available store-product pairs. Redeploy the API before using this explorer.")
        chosen = None
    start = st.text_input("Start date (YYYY-MM-DD)", filters["dates"]["start"])
    end = st.text_input("End date (YYYY-MM-DD)", filters["dates"]["end"])
    params = {"limit":10000, "date_from":start, "date_to":end}
    if chosen:
        params["store_id"] = chosen["store_id"]
        params["product_id"] = chosen["product_id"]
    try:
        data = get_json("/api/v1/forecasts", params)
        f = pd.DataFrame(data["rows"])
        if f.empty:
            st.warning("No rows match this series and date range.")
        else:
            f["date"] = pd.to_datetime(f["date"])
            f = f.sort_values("date")
            st.caption(f'Showing {len(f):,} of {data["total"]:,} matching rows.')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=f["date"], y=f["p90_calibrated"], mode="lines", line={"width":0}, name="Calibrated upper", showlegend=False))
            fig.add_trace(go.Scatter(x=f["date"], y=f["p10_calibrated"], mode="lines", line={"width":0}, fill="tonexty", fillcolor="rgba(31,119,180,0.18)", name="Calibrated P10–P90 band"))
            fig.add_trace(go.Scatter(x=f["date"], y=f["actual_sale_amount"], mode="lines+markers", name="Observed sales"))
            fig.add_trace(go.Scatter(x=f["date"], y=f["p50"], mode="lines+markers", name="P50 forecast"))
            fig.update_layout(title="Observed sales, median forecast and calibrated interval",
                              xaxis_title="Date", yaxis_title="Daily observed sales units",
                              height=420, margin={"l":15,"r":15,"t":55,"b":15}, legend={"orientation":"h","y":-0.25})
            st.plotly_chart(fig, use_container_width=True)
            display_cols = [c for c in ["date","actual_sale_amount","p10","p50","p90","p10_calibrated","p90_calibrated","stock_hour6_22_cnt"] if c in f]
            compact_table(f[display_cols], 300)
    except Exception as exc:
        st.error(f"Forecast query failed: {exc}")

with tabs[2]:
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
    st.write("Conformal calibration")
    st.json({"nominal_coverage_pct":cc.get("nominal_coverage_pct"),
             "validation":cc.get("validation_calibrated_interval"),
             "evaluation":cc.get("evaluation_calibrated_interval")})
    st.warning("Evaluation coverage is below the 80% nominal target in the current run. Coverage also varies materially by stockout-exposure group; do not present these intervals as uniformly calibrated.")

with tabs[3]:
    st.subheader("Stockout exposure and observed sales")
    c1,c2,c3 = st.columns(3)
    with c1: metric("Training rows", f'{stock.get("rows",0):,}')
    with c2: metric("Rows with stockout exposure", f'{stock.get("stockout_affected_row_rate",0)*100:.1f}%')
    with c3: metric("Mean stockout hours (06–21)", f'{stock.get("mean_stockout_hours_06_22",0):.2f}')
    buckets = pd.DataFrame(get_json("/api/v1/stockout/buckets")["rows"])
    if not buckets.empty:
        buckets["zero_sales_rate_pct"] = pd.to_numeric(buckets["zero_sales_rate"], errors="coerce")*100
        fig = px.bar(buckets, x="stockout_bucket", y="mean_observed_sales", title="Mean observed sales by stockout-exposure bucket")
        fig.update_layout(height=380, margin={"l":15,"r":15,"t":55,"b":20})
        st.plotly_chart(fig, use_container_width=True)
        compact_table(buckets, 300)
    hourly = pd.DataFrame(get_json("/api/v1/stockout/hourly")["rows"])
    if not hourly.empty:
        fig = px.line(hourly, x="hour_of_day", y="out_of_stock_rate", title="Out-of-stock rate by vector index")
        fig.update_layout(height=350, margin={"l":15,"r":15,"t":55,"b":20})
        st.plotly_chart(fig, use_container_width=True)
    st.caption(stock.get("interpretation","Descriptive association only."))
    st.caption(stock.get("hour_index_note","Hourly vector positions are indices; do not infer local-time labels."))

with tabs[4]:
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
    sens = pd.DataFrame(get_json("/api/v1/inventory/sensitivity")["rows"])
    if not sens.empty:
        st.subheader("Shortage penalty sensitivity")
        cols = [c for c in ["scenario","policy","steady_state_fill_rate_pct","steady_state_total_cost","steady_state_shortage_cost"] if c in sens]
        compact_table(sens[cols], 320)
    st.warning("Costs, lead time, review cadence and starting inventory are simulation assumptions. The short evaluation horizon and observed-sales proxy limit operational interpretation.")


with tabs[5]:
    st.subheader("SQL exploratory data analysis")
    st.caption("Precomputed DuckDB SQL results over the training Parquet in R2. Select a query to inspect its SQL and result table.")
    try:
        eda = get_json("/api/v1/sql-eda")
        st.caption(f"Engine: {eda.get('engine', 'DuckDB')} · Generated: {eda.get('generated_at_utc', 'unknown')} · Queries: {eda.get('query_count', 0)}")
        query_items = eda.get("queries", [])
        if query_items:
            titles = [item["title"] for item in query_items]
            selected_query = st.selectbox("EDA query", range(len(titles)), format_func=lambda i: titles[i])
            item = query_items[selected_query]
            st.markdown(f"**{item['title']}**")
            st.code(item["sql"], language="sql")
            result = pd.DataFrame(item.get("rows", []), columns=item.get("columns", []))
            st.caption(f"{item.get('row_count_returned', len(result)):,} result rows")
            compact_table(result, 380)
            if not result.empty:
                numeric = result.select_dtypes(include="number").columns.tolist()
                if item["id"] == "daily_sales_trend" and "total_observed_sales" in numeric:
                    fig = px.line(result, x="date", y="total_observed_sales", title="Daily total observed sales")
                    fig.update_layout(height=360, margin={"l":15,"r":15,"t":55,"b":20})
                    st.plotly_chart(fig, use_container_width=True)
                elif item["id"] in ("top_stores", "top_products") and "total_observed_sales" in numeric:
                    category = "store_id" if "store_id" in result else "product_id"
                    fig = px.bar(result.sort_values("total_observed_sales"), x="total_observed_sales", y=category, orientation="h", title=item["title"])
                    fig.update_layout(height=460, margin={"l":15,"r":15,"t":55,"b":20})
                    st.plotly_chart(fig, use_container_width=True)
                elif item["id"] == "weekday_pattern" and "mean_observed_sales" in numeric:
                    fig = px.bar(result, x="weekday", y="mean_observed_sales", title="Mean observed sales by weekday")
                    fig.update_layout(height=360, margin={"l":15,"r":15,"t":55,"b":20})
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No SQL EDA results have been published yet. Run the FreshRetailNet SQL EDA workflow in GitHub Actions.")
        st.info(eda.get("source_semantics", "Observed sales may be censored by stock availability; this is descriptive EDA."))
    except requests.HTTPError as exc:
        st.warning("SQL EDA results are not published yet. Run the FreshRetailNet SQL EDA workflow in GitHub Actions, then refresh this page.")
    except Exception as exc:
        st.error(f"Could not load SQL EDA results: {exc}")
