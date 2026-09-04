"""
Page 2: Forecast — Multi-Model Forecasting & Live Scenario Simulator.

Features:
  1. Multi-Model Benchmark & Visual Curves (PyTorch LSTM, LightGBM, Seasonal-Naive)
  2. Live Interactive Scenario ML Simulator (Dual-inference comparing LSTM & LightGBM)
  3. Prediction intervals (Q10/Q90) and per-SKU evaluation accuracy
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.api_client import client as api_client
from app.shared_styles import apply_shared_theme, update_chart_theme
from src.utils import DATA_CLEANED, DATA_FORECASTS, bias, mape, wape

st.set_page_config(page_title="Forecast — FORESIGHT", page_icon="📈", layout="wide")
apply_shared_theme()

st.title("Demand Forecast & Scenario Intelligence")
st.caption("Benchmark comparison (LSTM vs LightGBM vs Baseline) & live interactive ML prediction")


@st.cache_data
def load_forecast_data():
    data = {}
    try:
        data["baseline"] = pd.read_csv(DATA_FORECASTS / "baseline_results.csv", parse_dates=["date"])
    except FileNotFoundError:
        data["baseline"] = None
    try:
        data["lgbm"] = pd.read_csv(DATA_FORECASTS / "lgbm_results.csv", parse_dates=["date"])
    except FileNotFoundError:
        data["lgbm"] = None
    try:
        data["comparison"] = pd.read_csv(DATA_FORECASTS / "model_comparison.csv")
    except FileNotFoundError:
        data["comparison"] = None
    try:
        data["sku_master"] = pd.read_csv(DATA_CLEANED / "sku_master_clean.csv")
    except FileNotFoundError:
        data["sku_master"] = None
    try:
        data["feature_importance"] = pd.read_csv(DATA_FORECASTS / "feature_importance.csv")
    except FileNotFoundError:
        data["feature_importance"] = None
    return data


data = load_forecast_data()

if data["baseline"] is None:
    st.warning("No forecast data available. Run the pipeline and models first.")
    st.stop()

baseline = data["baseline"]
lgbm_df = data["lgbm"]

# Top Tabs: Curves vs Interactive Simulator
tab_curves, tab_simulator = st.tabs([
    "📈 Multi-Model Forecasting & Historical Curves",
    "🔮 Live Dual-Model ML Simulator (LSTM vs LightGBM)",
])

# ===========================================================================
# TAB 1: Historical & Multi-Model Curves
# ===========================================================================
with tab_curves:
    st.subheader("Model Performance Comparison")

    if data["comparison"] is not None:
        comp = data["comparison"].copy().sort_values("wape")
        winner = comp.iloc[0]["name"]
        baseline_wape = comp[comp["name"].str.contains("Baseline|Naive", case=False)]["wape"].values

        col_comp1, col_comp2 = st.columns([2, 3])

        with col_comp1:
            for _, row in comp.iterrows():
                is_winner = row["name"] == winner
                border_color = "rgba(39,174,96,0.4)" if is_winner else "rgba(255,255,255,0.08)"
                wape_color = "#27AE60" if is_winner else "#e0e0ff"

                improvement = ""
                if len(baseline_wape) > 0 and not ("Baseline" in row["name"] or "Naive" in row["name"]):
                    imp_pct = ((baseline_wape[0] - row["wape"]) / baseline_wape[0]) * 100
                    if imp_pct > 0:
                        improvement = f"<div style='font-size:0.8rem; color:#27AE60; margin-top:0.2rem;'>▲ Beats baseline by {imp_pct:.1f}%</div>"

                winner_badge = '<span style="background:rgba(39,174,96,0.2); color:#27AE60; padding:3px 10px; border-radius:12px; font-size:0.75rem; font-weight:600;">WINNER</span>' if is_winner else ""

                card_html = (
                    f'<div style="background: rgba(255,255,255,0.03); border-radius:12px; padding:1rem; margin-bottom:0.6rem; border: 1px solid {border_color};">'
                    f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                    f'<span style="color:#e0e0ff; font-weight:600;">{row["name"]}</span>'
                    f'{winner_badge}'
                    f'</div>'
                    f'<div style="font-size:1.6rem; font-weight:800; color:{wape_color}; margin-top:0.3rem;">'
                    f'WAPE: {row["wape"]:.4f}'
                    f'</div>'
                    f'{improvement}'
                    f'</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)

        with col_comp2:
            fig_comp = px.bar(
                comp,
                x="name",
                y="wape",
                color="wape",
                color_continuous_scale=["#27AE60", "#F39C12", "#E74C3C"],
                labels={"name": "Model", "wape": "WAPE (lower = better)"},
            )
            fig_comp = update_chart_theme(fig_comp, height=280, show_legend=False)
            st.plotly_chart(fig_comp, use_container_width=True)

    # Merge baseline + lgbm for per-SKU visualization
    combined = baseline.copy()
    if lgbm_df is not None:
        combined = combined.merge(
            lgbm_df[["sku_id", "date", "lgbm_forecast", "lgbm_lower", "lgbm_upper"]],
            on=["sku_id", "date"],
            how="left",
        )

    st.divider()
    skus = sorted(combined["sku_id"].unique())
    selected_sku = st.selectbox("Select SKU to Visualize", skus, index=0)

    # SKU Info Header
    if data["sku_master"] is not None:
        sku_info = data["sku_master"][data["sku_master"]["sku_id"] == selected_sku]
        if len(sku_info) > 0:
            info = sku_info.iloc[0]
            st.markdown(
                f"**Category:** {info.get('category', 'N/A')} | "
                f"**Subcategory:** {info.get('subcategory', 'N/A')} | "
                f"**Unit Cost:** ₹{info.get('unit_cost', 0):,.0f} | "
                f"**List Price:** ₹{info.get('list_price', 0):,.0f}"
            )

    sku_data = combined[combined["sku_id"] == selected_sku].sort_values("date")

    if len(sku_data) > 0:
        fig = go.Figure()

        # Actuals
        fig.add_trace(go.Scatter(
            x=sku_data["date"],
            y=sku_data["actual"],
            mode="lines+markers",
            name="Actual Demand",
            line=dict(color="#e0e0ff", width=2.5),
            marker=dict(size=4),
        ))

        # Baseline
        fig.add_trace(go.Scatter(
            x=sku_data["date"],
            y=sku_data["baseline_forecast"],
            mode="lines",
            name="Seasonal Naive (Baseline)",
            line=dict(color="#e74c3c", width=2, dash="dash"),
        ))

        # LightGBM
        if "lgbm_forecast" in sku_data.columns and sku_data["lgbm_forecast"].notna().any():
            fig.add_trace(go.Scatter(
                x=sku_data["date"],
                y=sku_data["lgbm_forecast"],
                mode="lines+markers",
                name="LightGBM Point Forecast",
                line=dict(color="#27ae60", width=2.5),
                marker=dict(size=5),
            ))

            # 80% Confidence interval
            if "lgbm_upper" in sku_data.columns and "lgbm_lower" in sku_data.columns:
                fig.add_trace(go.Scatter(
                    x=pd.concat([sku_data["date"], sku_data["date"][::-1]]),
                    y=pd.concat([sku_data["lgbm_upper"], sku_data["lgbm_lower"][::-1]]),
                    fill="toself",
                    fillcolor="rgba(39,174,96,0.12)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="80% Prediction Interval (Q10 - Q90)",
                ))

        fig = update_chart_theme(fig, height=450, show_legend=True)
        fig.update_layout(
            xaxis_title="Timeline",
            yaxis_title="Weekly Demand Units",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Per-SKU Summary Table
    st.divider()
    st.subheader("Accuracy by SKU — All Models")
    sku_metrics = []
    for sku in skus[:30]:
        s = combined[combined["sku_id"] == sku]
        if len(s) > 0 and s["actual"].sum() > 0:
            row_data = {
                "SKU": sku,
                "Baseline WAPE": round(wape(s["actual"].values, s["baseline_forecast"].values), 4),
                "Baseline MAPE (%)": round(mape(s["actual"].values, s["baseline_forecast"].values), 2),
            }
            if "lgbm_forecast" in s.columns and s["lgbm_forecast"].notna().any():
                row_data["LightGBM WAPE"] = round(wape(s["actual"].values, s["lgbm_forecast"].values), 4)
                row_data["LightGBM MAPE (%)"] = round(mape(s["actual"].values, s["lgbm_forecast"].values), 2)
                row_data["Improvement"] = f"{((row_data['Baseline WAPE'] - row_data['LightGBM WAPE']) / max(row_data['Baseline WAPE'], 0.001) * 100):.1f}%"
            sku_metrics.append(row_data)

    metrics_df = pd.DataFrame(sku_metrics)
    if "LightGBM WAPE" in metrics_df.columns:
        metrics_df = metrics_df.sort_values("LightGBM WAPE")
    st.dataframe(metrics_df, use_container_width=True, height=350)


# ===========================================================================
# TAB 2: Live Dual-Model ML Simulator (LSTM vs LightGBM)
# ===========================================================================
with tab_simulator:
    st.subheader("Live Dual-Model Scenario Simulation")
    st.caption("Trigger real-time inference on both PyTorch LSTM and LightGBM models via the FastAPI backend.")

    # Status indicator
    backend_live = api_client.is_online()
    if backend_live:
        st.success(f"⚡ Connected to Live FastAPI Microservice (`{api_client.base_url}`) — Executing direct ML inference", icon="🚀")
    else:
        st.warning(f"⚠️ FastAPI Backend offline (`{api_client.base_url}`). Operating in offline emulation mode.", icon="📁")

    # Lookup default price and demand baseline for selected SKU
    default_price = 549.0
    default_4w_avg = 45.0
    if sku_data is not None:
        default_price = float(sku_data.get("unit_price", 549.0))
        default_4w_avg = float(sku_data.get("avg_weekly_forecast", 45.0))

    with st.form(key="manual_ml_simulator_form"):
        col_inp1, col_inp2, col_inp3 = st.columns(3)

        with col_inp1:
            sim_sku = st.selectbox("Target SKU", skus, index=0, key="sim_sku_select")
            sim_price = st.number_input("Unit Selling Price (₹)", min_value=10.0, max_value=50000.0, value=default_price, step=25.0)

        with col_inp2:
            sim_discount = st.slider("Promotional Discount %", min_value=0.0, max_value=60.0, value=10.0, step=1.0)
            sim_4w_avg = st.number_input("Recent 4-Week Average Demand (units/wk)", min_value=1.0, max_value=2000.0, value=default_4w_avg, step=5.0)

        with col_inp3:
            st.write("**Calendar & Campaign Overrides**")
            sim_promo = st.checkbox("Active Promotional Campaign", value=True)
            sim_holiday = st.checkbox("Holiday Week (Demand Surge Expected)", value=False)
            st.write("")
            submit_btn = st.form_submit_button("⚡ Run Dual-Model Inference", type="primary", use_container_width=True)

    # Trigger inference only upon explicit form submission
    if submit_btn:
        with st.spinner("Invoking PyTorch Deep LSTM and LightGBM models via FastAPI REST..."):
            st.session_state["manual_sim_result"] = api_client.predict_manual(
                sku_id=sim_sku,
                unit_price=sim_price,
                discount_pct=sim_discount,
                is_holiday=1 if sim_holiday else 0,
                promo_flag=1 if sim_promo else 0,
                recent_4w_avg=sim_4w_avg,
            )

    # Check if a simulation result is available in session state
    if "manual_sim_result" not in st.session_state:
        # Run default baseline once for initial display
        st.session_state["manual_sim_result"] = api_client.predict_manual(
            sku_id=skus[0],
            unit_price=default_price,
            discount_pct=10.0,
            is_holiday=0,
            promo_flag=1,
            recent_4w_avg=default_4w_avg,
        )

    res = st.session_state["manual_sim_result"]
    lgbm = res["lightgbm"]
    lstm = res["lstm"]

    # Model status badge
    is_live = res.get("is_live_model", backend_live)
    if is_live:
        model_badge = '<span style="background:#15803d; color:#bbf7d0; font-size:0.75rem; font-weight:700; padding:3px 10px; border-radius:12px;">● LIVE MODEL INFERENCE (PyTorch + LightGBM)</span>'
    else:
        model_badge = '<span style="background:#991b1b; color:#fecaca; font-size:0.75rem; font-weight:700; padding:3px 10px; border-radius:12px;">● FASTAPI OFFLINE (Run uvicorn to enable real models)</span>'

    st.markdown(
        f'<div style="background:rgba(255,255,255,0.02); border:1px solid rgba(6,182,212,0.3); border-radius:12px; padding:1rem; margin:1.2rem 0;">'
        f'<div style="display:flex; justify-content:space-between; align-items:center;">'
        f'<span style="font-size:0.95rem; font-weight:700; color:#38bdf8;">🧠 Real-Time Scenario Consensus</span>'
        f'{model_badge}'
        f'</div>'
        f'<div style="font-size:0.9rem; color:#cbd5e1; margin-top:0.5rem;">{res["explanation"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Side-by-side comparison cards
    mc1, mc2, mc3 = st.columns(3)

    with mc1:
        st.markdown(
            f'<div style="background:rgba(34,197,94,0.06); border:1px solid rgba(34,197,94,0.35); border-radius:12px; padding:1.2rem;">'
            f'<div style="display:flex; justify-content:space-between; align-items:center;">'
            f'<span style="font-size:0.9rem; font-weight:700; color:#4ade80;">PyTorch Deep LSTM</span>'
            f'<span style="background:#15803d; color:#bbf7d0; font-size:0.7rem; font-weight:800; padding:2px 8px; border-radius:8px;">★ TOP ACCURACY</span>'
            f'</div>'
            f'<div style="font-size:2.2rem; font-weight:800; color:#4ade80; margin:0.4rem 0;">{lstm["predicted_units"]:.1f} <span style="font-size:1rem; color:#94a3b8;">units/wk</span></div>'
            f'<div style="font-size:0.8rem; color:#94a3b8;">Confidence Interval: <b>{lstm["lower_bound"]:.1f} – {lstm["upper_bound"]:.1f}</b></div>'
            f'<div style="font-size:0.75rem; color:#86efac; margin-top:0.3rem;">Historical Test WAPE: {lstm["historical_wape"]:.4f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with mc2:
        st.markdown(
            f'<div style="background:rgba(14,116,144,0.06); border:1px solid rgba(14,116,144,0.35); border-radius:12px; padding:1.2rem;">'
            f'<div style="display:flex; justify-content:space-between; align-items:center;">'
            f'<span style="font-size:0.9rem; font-weight:700; color:#38bdf8;">LightGBM Regressor</span>'
            f'<span style="background:#0369a1; color:#bae6fd; font-size:0.7rem; font-weight:800; padding:2px 8px; border-radius:8px;">GBDT + QUANTILES</span>'
            f'</div>'
            f'<div style="font-size:2.2rem; font-weight:800; color:#38bdf8; margin:0.4rem 0;">{lgbm["predicted_units"]:.1f} <span style="font-size:1rem; color:#94a3b8;">units/wk</span></div>'
            f'<div style="font-size:0.8rem; color:#94a3b8;">80% Quantile Bounds: <b>{lgbm["lower_bound"]:.1f} – {lgbm["upper_bound"]:.1f}</b></div>'
            f'<div style="font-size:0.75rem; color:#7dd3fc; margin-top:0.3rem;">Historical Test WAPE: {lgbm["historical_wape"]:.4f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with mc3:
        st.markdown(
            f'<div style="background:rgba(245,158,11,0.06); border:1px solid rgba(245,158,11,0.35); border-radius:12px; padding:1.2rem;">'
            f'<div style="font-size:0.9rem; font-weight:700; color:#fbbf24;">Consensus & Divergence</div>'
            f'<div style="font-size:2.2rem; font-weight:800; color:#fbbf24; margin:0.4rem 0;">{res["consensus_prediction"]:.1f} <span style="font-size:1rem; color:#94a3b8;">units/wk</span></div>'
            f'<div style="font-size:0.8rem; color:#94a3b8;">Model Divergence: <b>{res["divergence_pct"]}%</b></div>'
            f'<div style="font-size:0.75rem; color:#fde68a; margin-top:0.3rem;">Recommended: <b>{res["recommended_model"]}</b></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Comparative Plotly Bar Chart with Error / Confidence Bounds
    sim_chart_df = pd.DataFrame([
        {
            "Model": "LightGBM",
            "Predicted Units": lgbm["predicted_units"],
            "Error Plus": max(0.0, lgbm["upper_bound"] - lgbm["predicted_units"]),
            "Error Minus": max(0.0, lgbm["predicted_units"] - lgbm["lower_bound"]),
            "Color": "#06b6d4",
        },
        {
            "Model": "PyTorch LSTM",
            "Predicted Units": lstm["predicted_units"],
            "Error Plus": max(0.0, lstm["upper_bound"] - lstm["predicted_units"]),
            "Error Minus": max(0.0, lstm["predicted_units"] - lstm["lower_bound"]),
            "Color": "#22c55e",
        },
    ])

    fig_sim = go.Figure()
    fig_sim.add_trace(go.Bar(
        name="Predicted Demand",
        x=sim_chart_df["Model"],
        y=sim_chart_df["Predicted Units"],
        marker_color=["#06b6d4", "#22c55e"],
        error_y=dict(
            type="data",
            symmetric=False,
            array=sim_chart_df["Error Plus"],
            arrayminus=sim_chart_df["Error Minus"],
            color="#e0e0ff",
            thickness=2,
            width=12,
        ),
        text=[f"{v:.1f} units" for v in sim_chart_df["Predicted Units"]],
        textposition="auto",
    ))
    fig_sim = update_chart_theme(fig_sim, height=360, show_legend=False)
    fig_sim.update_layout(
        title=dict(text="Side-by-Side Model Prediction with Confidence Envelopes", font=dict(color="#e0e0ff", size=14)),
        yaxis_title="Weekly Demand Units",
    )
    st.plotly_chart(fig_sim, use_container_width=True)
