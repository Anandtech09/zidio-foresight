"""
Page 2: Forecast — Forecast vs actual charts per SKU.

Shows all 3 models (Baseline, LightGBM, LSTM) side-by-side,
with confidence intervals from LightGBM quantile models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils import DATA_FORECASTS, DATA_CLEANED, wape, mape, bias
from app.shared_styles import apply_shared_theme, update_chart_theme

st.set_page_config(page_title="Forecast — FORESIGHT", page_icon="📈", layout="wide")
apply_shared_theme()

st.title("Demand Forecast")
st.caption("All 3 models compared: Seasonal Naive Baseline vs LightGBM vs PyTorch LSTM")


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
    st.warning("No forecast data. Run `python -m src.baseline` first.")
    st.stop()

baseline = data["baseline"]
lgbm_df = data["lgbm"]

# ─── Model Comparison Table ──────────────────────────────────────────────
st.subheader("Model Performance Comparison")

if data["comparison"] is not None:
    comp = data["comparison"].copy()
    comp = comp.sort_values("wape")
    winner = comp.iloc[0]["name"]
    baseline_wape = comp[comp["name"].str.contains("Baseline|Naive", case=False)]["wape"].values

    col_comp1, col_comp2 = st.columns([2, 3])

    with col_comp1:
        for _, row in comp.iterrows():
            is_winner = row["name"] == winner
            border_color = "rgba(39,174,96,0.4)" if is_winner else "rgba(255,255,255,0.08)"
            wape_color = "#27AE60" if is_winner else "#e0e0ff"

            improvement = ""
            if len(baseline_wape) > 0 and not row["name"].__contains__("Baseline") and not row["name"].__contains__("Naive"):
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
            comp, x="name", y="wape",
            color="wape",
            color_continuous_scale=["#27AE60", "#F39C12", "#E74C3C"],
            labels={"name": "Model", "wape": "WAPE (lower = better)"},
        )
        fig_comp.update_layout(
            height=280,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b8fa3"),
            showlegend=False,
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

# ─── Merge baseline + lgbm for per-SKU visualization ─────────────────────
# Build a combined DataFrame so we can show all models on one chart
combined = baseline.copy()
if lgbm_df is not None:
    combined = combined.merge(
        lgbm_df[["sku_id", "date", "lgbm_forecast", "lgbm_lower", "lgbm_upper"]],
        on=["sku_id", "date"],
        how="left",
    )

# ─── SKU Selector ────────────────────────────────────────────────────────
st.divider()
col_f1, col_f2 = st.columns(2)

skus = sorted(combined["sku_id"].unique())
with col_f1:
    selected_sku = st.selectbox("Select SKU", skus)

# ─── Per-SKU Forecast Chart ─────────────────────────────────────────────
sku_data = combined[combined["sku_id"] == selected_sku].sort_values("date")

if len(sku_data) > 0:
    st.subheader(f"Forecast: {selected_sku}")

    # SKU info
    if data["sku_master"] is not None:
        sku_info = data["sku_master"][data["sku_master"]["sku_id"] == selected_sku]
        if len(sku_info) > 0:
            info = sku_info.iloc[0]
            st.markdown(f"**Category:** {info.get('category', 'N/A')} | "
                        f"**Subcategory:** {info.get('subcategory', 'N/A')} | "
                        f"**List Price:** ₹{info.get('list_price', 0):,.0f}")

    # Metrics — show both baseline and LightGBM if available
    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)

    sku_wape_b = wape(sku_data["actual"].values, sku_data["baseline_forecast"].values)
    sku_mape_b = mape(sku_data["actual"].values, sku_data["baseline_forecast"].values)
    sku_bias_b = bias(sku_data["actual"].values, sku_data["baseline_forecast"].values)

    mc1.metric("Baseline WAPE", f"{sku_wape_b:.4f}")
    mc2.metric("Baseline MAPE", f"{sku_mape_b:.2f}%")
    mc3.metric("Baseline Bias", f"{sku_bias_b:.2f}")

    if "lgbm_forecast" in sku_data.columns and sku_data["lgbm_forecast"].notna().any():
        sku_wape_l = wape(sku_data["actual"].values, sku_data["lgbm_forecast"].values)
        sku_mape_l = mape(sku_data["actual"].values, sku_data["lgbm_forecast"].values)
        sku_bias_l = bias(sku_data["actual"].values, sku_data["lgbm_forecast"].values)
        mc4.metric("LightGBM WAPE", f"{sku_wape_l:.4f}")
        mc5.metric("LightGBM MAPE", f"{sku_mape_l:.2f}%")
        mc6.metric("LightGBM Bias", f"{sku_bias_l:.2f}")

    # ─── Chart: Actual vs Baseline vs LightGBM + Confidence Interval ────
    fig = go.Figure()

    # Actual demand
    fig.add_trace(go.Scatter(
        x=sku_data["date"], y=sku_data["actual"],
        name="Actual", mode="lines+markers",
        line=dict(color="#e0e0ff", width=2.5),
        marker=dict(size=6),
    ))

    # Baseline forecast
    fig.add_trace(go.Scatter(
        x=sku_data["date"], y=sku_data["baseline_forecast"],
        name="Seasonal Naive Baseline", mode="lines+markers",
        line=dict(color="#E74C3C", width=2, dash="dash"),
        marker=dict(size=5),
    ))

    # LightGBM forecast
    if "lgbm_forecast" in sku_data.columns and sku_data["lgbm_forecast"].notna().any():
        fig.add_trace(go.Scatter(
            x=sku_data["date"], y=sku_data["lgbm_forecast"],
            name="LightGBM", mode="lines+markers",
            line=dict(color="#3498DB", width=2.5),
            marker=dict(size=5),
        ))

        # 80% Confidence interval (from quantile models)
        if "lgbm_upper" in sku_data.columns and "lgbm_lower" in sku_data.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([sku_data["date"], sku_data["date"][::-1]]),
                y=pd.concat([sku_data["lgbm_upper"], sku_data["lgbm_lower"][::-1]]),
                fill="toself",
                fillcolor="rgba(52,152,219,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="80% Confidence Interval",
            ))

    fig.update_layout(
        height=480,
        xaxis_title="Date",
        yaxis_title="Weekly Demand (units)",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.18),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b8fa3"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning(f"No forecast data for SKU: {selected_sku}")

# ─── Per-SKU Accuracy Table ──────────────────────────────────────────────
st.divider()
st.subheader("Accuracy by SKU — All Models")

sku_metrics = []
for sku in skus:
    s = combined[combined["sku_id"] == sku]
    if len(s) > 0:
        row_data = {
            "SKU": sku,
            "Baseline WAPE": round(wape(s["actual"].values, s["baseline_forecast"].values), 4),
            "Baseline MAPE (%)": round(mape(s["actual"].values, s["baseline_forecast"].values), 2),
        }
        if "lgbm_forecast" in s.columns and s["lgbm_forecast"].notna().any():
            row_data["LightGBM WAPE"] = round(wape(s["actual"].values, s["lgbm_forecast"].values), 4)
            row_data["LightGBM MAPE (%)"] = round(mape(s["actual"].values, s["lgbm_forecast"].values), 2)
            row_data["Improvement"] = f"{((row_data['Baseline WAPE'] - row_data['LightGBM WAPE']) / row_data['Baseline WAPE'] * 100):.1f}%"
        sku_metrics.append(row_data)

metrics_df = pd.DataFrame(sku_metrics)
if "LightGBM WAPE" in metrics_df.columns:
    metrics_df = metrics_df.sort_values("LightGBM WAPE")
else:
    metrics_df = metrics_df.sort_values("Baseline WAPE")

st.dataframe(metrics_df, use_container_width=True, height=400)
