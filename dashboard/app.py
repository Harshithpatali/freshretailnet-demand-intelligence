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
tabs = st.tabs(["Executive overview", "Forecast explorer", "Uncertainty & calibration", "Stockout diagnostics", "Inventory optimisation", "SQL EDA", "Experiment lab", "Wavelet benchmark", "Forecast monitoring", "Demand censoring", "Inventory scenarios"])

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
    try:
        filters = get_json("/api/v1/filters")
    except Exception as exc:
        filters = None
        st.warning(f"Forecast filters are unavailable: {exc}")
    if filters:
        with st.form("forecast_series_search_form"):
            search_store = st.text_input("Store ID contains", key="forecast_store_search_input")
            search_product = st.text_input("Product ID contains", key="forecast_product_search_input")
            search_submit = st.form_submit_button("Search store-product series")
        if search_submit:
            st.session_state["forecast_store_search"] = search_store.strip()
            st.session_state["forecast_product_search"] = search_product.strip()
            st.session_state["forecast_series_page"] = 1
        search_page = st.number_input(
            "Results page", min_value=1,
            value=int(st.session_state.get("forecast_series_page", 1)),
            step=1, key="forecast_series_page",
        )
        try:
            series_result = get_json("/api/v1/series/search", {
                "store_id": st.session_state.get("forecast_store_search", ""),
                "product_id": st.session_state.get("forecast_product_search", ""),
                "offset": (int(search_page) - 1) * 50,
                "limit": 50,
            })
            series_options = series_result.get("rows", [])
            st.caption(
                f"{series_result.get('total', 0):,} matching series · "
                f"showing page {int(search_page)} of "
                f"{max(1, (series_result.get('total', 0) + 49) // 50)}"
            )
            if series_options:
                labels = [f"Store {x['store_id']} · Product {x['product_id']}" for x in series_options]
                selected = st.selectbox("Select a result on this page", range(len(labels)),
                                        format_func=lambda i: labels[i], key="forecast_series_selection")
                chosen = series_options[selected]
                st.caption(f"Selected series: store {chosen['store_id']} / product {chosen['product_id']}")
            else:
                chosen = None
                st.info("No series on this page. Change the search filters or select an earlier page.")
        except Exception as exc:
            chosen = None
            st.warning(f"Server-side series search is unavailable: {exc}")
    else:
        chosen = None
    start = st.text_input("Start date (YYYY-MM-DD)", filters["dates"]["start"] if filters else "")
    end = st.text_input("End date (YYYY-MM-DD)", filters["dates"]["end"] if filters else "")
    params = {"limit":10000, "date_from":start, "date_to":end}
    if chosen:
        params["store_id"] = chosen["store_id"]
        params["product_id"] = chosen["product_id"]
    try:
        if not filters:
            raise RuntimeError("Forecast filter artifact is unavailable; republish quantile predictions.")
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
    except requests.HTTPError:
        st.warning("SQL EDA results are not published yet. Run the FreshRetailNet SQL EDA workflow in GitHub Actions, then refresh this page.")
    except Exception as exc:
        st.error(f"Could not load SQL EDA results: {exc}")


with tabs[6]:
    st.subheader("Experiment lab")
    st.caption("Run a configurable, reproducible what-if inventory experiment on one held-out store-product series. Results are a small-sample simulation, not causal estimates or operational savings.")
    try:
        exp_filters = get_json("/api/v1/filters")
    except Exception as exc:
        exp_filters = {}
        st.warning(f"Experiment inputs are unavailable: {exc}")
    if exp_filters:
        with st.form("experiment_series_search_form"):
            exp_store_search = st.text_input("Store ID contains", key="experiment_store_search_input")
            exp_product_search = st.text_input("Product ID contains", key="experiment_product_search_input")
            exp_search_submit = st.form_submit_button("Search experimental series")
        if exp_search_submit:
            st.session_state["experiment_store_search"] = exp_store_search.strip()
            st.session_state["experiment_product_search"] = exp_product_search.strip()
            st.session_state["experiment_series_page"] = 1
        exp_page = st.number_input(
            "Experimental series page", min_value=1,
            value=int(st.session_state.get("experiment_series_page", 1)),
            step=1, key="experiment_series_page",
        )
        try:
            exp_series_result = get_json("/api/v1/series/search", {
                "store_id": st.session_state.get("experiment_store_search", ""),
                "product_id": st.session_state.get("experiment_product_search", ""),
                "offset": (int(exp_page) - 1) * 50, "limit": 50,
            })
            exp_series = exp_series_result.get("rows", [])
        except Exception as exc:
            exp_series = []
            st.warning(f"Server-side experiment series search failed: {exc}")
    else:
        exp_series = []
    if not exp_series:
        st.warning("No matching evaluation series. Search with a different store or product ID.")
        exp_chosen = None
    else:
        exp_labels = [f"Store {x['store_id']} · Product {x['product_id']}" for x in exp_series]
        exp_idx = st.selectbox("Experimental series (current page)", range(len(exp_labels)),
                               format_func=lambda i: exp_labels[i], key="exp_series")
        exp_chosen = exp_series[exp_idx]
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


with tabs[7]:
    from wavelet_tab import render_wavelet_tab
    render_wavelet_tab(API, get_json, metric, compact_table)



with tabs[8]:
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


with tabs[9]:
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


with tabs[10]:
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
