"""Streamlit dashboard for FreshRetailNet Demand Intelligence."""
import os
import requests
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="FreshRetailNet | Demand Intelligence", page_icon="📦", layout="wide")
API = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

@st.cache_data(ttl=300, show_spinner=False)
def get_json(path, params=None):
    response = requests.get(f"{API}{path}", params=params, timeout=90)
    response.raise_for_status()
    return response.json()

def metric(label, value, help_text=None):
    st.metric(label, value, help=help_text)

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
tabs = st.tabs(["Executive overview", "Forecast explorer", "Uncertainty & calibration", "Stockout diagnostics", "Inventory optimisation"])

with tabs[0]:
    ev = point.get("evaluation", {})
    model = ev.get("model", {})
    baseline = ev.get("seasonal_naive_lag7", {})
    qeval = quant.get("evaluation", {})
    cal = quant.get("conformal_calibration", {}).get("evaluation_calibrated_interval", {})
    inv = inventory.get("base_policy_comparison", [])
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric("Evaluation MAE", f'{model.get("mae", float("nan")):.3f}')
    with c2: metric("Evaluation WAPE", f'{model.get("wape_pct", float("nan")):.1f}%')
    with c3: metric("P10–P90 calibrated coverage", f'{cal.get("coverage_pct", float("nan")):.1f}%', "Nominal target: 80%; evaluation coverage can shift.")
    with c4: metric("Training rows", f'{point.get("rows", {}).get("train", 0):,}')
    st.subheader("Point forecast vs seasonal-naive benchmark")
    compare = pd.DataFrame([{"Model":"LightGBM","MAE":model.get("mae"),"WAPE (%)":model.get("wape_pct")},
                            {"Model":"Seasonal naive lag-7","MAE":baseline.get("mae"),"WAPE (%)":baseline.get("wape_pct")}])
    st.dataframe(compare, use_container_width=True, hide_index=True)
    st.subheader("Inventory policy snapshot")
    if inv:
        inv_df = pd.DataFrame(inv)
        cols = [c for c in ["policy","fill_rate_pct","steady_state_fill_rate_pct","steady_state_total_cost"] if c in inv_df]
        st.dataframe(inv_df[cols], use_container_width=True, hide_index=True)
    st.info("All forecast targets are observed daily sales. They are not guaranteed to represent uncensored customer demand.")

with tabs[1]:
    st.subheader("Evaluation forecast explorer")
    filters = get_json("/api/v1/filters")
    a,b,c = st.columns(3)
    with a: store = st.selectbox("Store", ["All"] + filters["stores"])
    with b: product = st.selectbox("Product", ["All"] + filters["products"])
    with c:
        start = st.text_input("Start date (YYYY-MM-DD)", filters["dates"]["start"])
    end = st.text_input("End date (YYYY-MM-DD)", filters["dates"]["end"])
    params = {"limit":10000, "date_from":start, "date_to":end}
    if store != "All": params["store_id"] = store
    if product != "All": params["product_id"] = product
    try:
        data = get_json("/api/v1/forecasts", params)
        f = pd.DataFrame(data["rows"])
        if f.empty:
            st.warning("No rows match these filters.")
        else:
            f["date"] = pd.to_datetime(f["date"])
            st.caption(f'Showing {len(f):,} of {data["total"]:,} matching rows.')
            entity = f"{store if store != 'All' else 'All stores'} / {product if product != 'All' else 'All products'}"
            chart = f.groupby("date", as_index=False)[["actual_sale_amount","p10","p50","p90","p10_calibrated","p90_calibrated"]].sum()
            fig = px.line(chart, x="date", y=["actual_sale_amount","p10","p50","p90"], title=f"Observed sales and quantile forecasts — {entity}")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(f.head(500), use_container_width=True, hide_index=True)
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
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
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
        st.plotly_chart(px.bar(buckets, x="stockout_bucket", y="mean_observed_sales", title="Mean observed sales by stockout-exposure bucket"), use_container_width=True)
        st.dataframe(buckets, use_container_width=True, hide_index=True)
    hourly = pd.DataFrame(get_json("/api/v1/stockout/hourly")["rows"])
    if not hourly.empty:
        st.plotly_chart(px.line(hourly, x="hour_of_day", y="out_of_stock_rate", title="Out-of-stock rate by vector index"), use_container_width=True)
    st.caption(stock.get("interpretation","Descriptive association only."))
    st.caption(stock.get("hour_index_note","Hourly vector positions are indices; do not infer local-time labels."))

with tabs[4]:
    st.subheader("Inventory policy backtest")
    base = pd.DataFrame(inventory.get("base_policy_comparison", []))
    if not base.empty:
        st.dataframe(base, use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(base, x="policy", y="steady_state_fill_rate_pct", title="Steady-state fill rate by policy"), use_container_width=True)
        st.plotly_chart(px.bar(base, x="policy", y="steady_state_total_cost", title="Steady-state simulated cost by policy"), use_container_width=True)
    sens = pd.DataFrame(get_json("/api/v1/inventory/sensitivity")["rows"])
    if not sens.empty:
        st.subheader("Shortage penalty sensitivity")
        st.dataframe(sens, use_container_width=True, hide_index=True)
    st.warning("Costs, lead time, review cadence and starting inventory are simulation assumptions. The short evaluation horizon and observed-sales proxy limit operational interpretation.")
