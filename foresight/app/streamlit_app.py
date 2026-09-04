"""
streamlit_app.py — Main entry point for the FORESIGHT Dashboard.

Displays:
  - KPI summary with model-driven predictions
  - Risk distribution (from trained model, not baseline)
  - Quick actions for the ops team
  - Model accuracy comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import DATA_FEATURES, DATA_FORECASTS, DATA_DASHBOARD, DATA_CLEANED, wape as wape_fn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FORESIGHT — NorthBay Living",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.shared_styles import apply_shared_theme, update_chart_theme
from app.api_client import client as api_client

# ---------------------------------------------------------------------------
# Apply Shared Design Theme
# ---------------------------------------------------------------------------
apply_shared_theme()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_all_data():
    """Load all processed datasets."""
    data = {}

    for name, filepath, parse_dates in [
        ("analysis", DATA_FEATURES / "analysis_ready.csv", ["date"]),
        ("risk", DATA_DASHBOARD / "risk_scored.csv", None),
        ("baseline", DATA_FORECASTS / "baseline_results.csv", ["date"]),
        ("lgbm", DATA_FORECASTS / "lgbm_results.csv", ["date"]),
        ("comparison", DATA_FORECASTS / "model_comparison.csv", None),
        ("sku_master", DATA_CLEANED / "sku_master_clean.csv", None),
        ("cleaning_log", DATA_CLEANED / "cleaning_log.csv", None),
        ("feature_importance", DATA_FORECASTS / "feature_importance.csv", None),
    ]:
        try:
            data[name] = pd.read_csv(
                filepath,
                parse_dates=parse_dates if parse_dates else False
            )
        except FileNotFoundError:
            data[name] = None

    return data


data = load_all_data()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div style="text-align:center; padding: 1rem 0;">'
        '<div style="font-size: 2.5rem;">📦</div>'
        '<div style="font-size: 1.3rem; font-weight: 700; color: #e0e0ff; margin-top: 0.3rem;">FORESIGHT</div>'
        '<div style="font-size: 0.8rem; color: #8b8fa3; margin-top: 0.2rem;">Demand & Inventory Intelligence</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # Backend microservice connectivity status
    backend_online = api_client.is_online()
    if backend_online:
        st.markdown(
            '<div style="background:rgba(6,182,212,0.15); border:1px solid #06b6d4; border-radius:8px; padding:8px 12px; margin-bottom:8px;">'
            '<div style="font-size:0.75rem; font-weight:700; color:#38bdf8;">⚡ REST API: ONLINE (:8000)</div>'
            '<div style="font-size:0.7rem; color:#94a3b8;">Client-Server Mode Active</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background:rgba(245,158,11,0.12); border:1px solid #f59e0b; border-radius:8px; padding:8px 12px; margin-bottom:8px;">'
            '<div style="font-size:0.75rem; font-weight:700; color:#fbbf24;">📁 STORAGE: DIRECT MODE</div>'
            '<div style="font-size:0.7rem; color:#94a3b8;">FastAPI offline (start uvicorn)</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Forecast source indicator
    if data["risk"] is not None and "forecast_source" in data["risk"].columns:
        source = data["risk"]["forecast_source"].iloc[0]
        if "LightGBM" in source:
            st.success("Using: ML Model Predictions", icon="🧠")
        else:
            st.warning("Using: Baseline Forecast", icon="⚠️")
    elif data["lgbm"] is not None:
        st.success("Using: ML Model Predictions", icon="🧠")
    else:
        st.warning("Using: Baseline (run forecast)", icon="⚠️")

    st.divider()
    st.caption("Built for NorthBay Living")
    st.caption("by Project FORESIGHT team")


# ---------------------------------------------------------------------------
# Check data
# ---------------------------------------------------------------------------
if data["analysis"] is None:
    st.error("No data found. Run the pipeline first:")
    st.code("run_all.bat", language="bash")
    st.stop()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">Project FORESIGHT</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Demand Forecasting & Inventory Risk Intelligence — NorthBay Living</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Backend Microservice Status Banner
# ---------------------------------------------------------------------------
backend_online = api_client.is_online()
base_url = api_client.base_url
if backend_online:
    st.markdown(
        f'<div style="background:rgba(6,182,212,0.12); border:1px solid #06b6d4; border-radius:10px; padding:10px 16px; margin-bottom:20px; display:flex; justify-content:space-between; align-items:center;">'
        f'<div><b style="color:#38bdf8;">🟢 FastAPI Backend Active</b> — Connected to Microservice on <code>{base_url}</code>. KPI metrics and ML inferences are served over HTTP REST APIs.</div>'
        f'<a href="{base_url}/docs" target="_blank" style="background:#0284c7; color:#ffffff; padding:8px 12px; border-radius:6px; text-decoration:none; font-size:0.8rem; font-weight:700;">Open Swagger /docs ↗</a>'
        f'</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div style="background:rgba(239,68,68,0.12); border:1px solid #ef4444; border-radius:10px; padding:10px 16px; margin-bottom:20px;">'
        f'<b style="color:#f87171;">🔴 FastAPI Backend Offline</b> — The Streamlit app is currently falling back to static local CSV cache because the microservice is not running. '
        f'Expected microservice URL: <code>{base_url}</code>.'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# KPI Row (Powered by api_client /portfolio/overview)
# ---------------------------------------------------------------------------
overview = api_client.get_overview()
risk_df = data["risk"]
analysis = data["analysis"]

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    total_skus = overview.get("total_skus", analysis["sku_id"].nunique() if analysis is not None else 0)
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-value">{total_skus}</div><div class="kpi-label">Total SKUs</div></div>',
        unsafe_allow_html=True,
    )

with c2:
    at_risk = overview.get("skus_at_risk", len(risk_df[risk_df["quadrant"] != "Healthy"]) if risk_df is not None else 0)
    pct = overview.get("risk_percentage", (at_risk / max(total_skus, 1) * 100))
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-value" style="color: #E74C3C;">{at_risk}</div>'
        f'<div class="kpi-label">SKUs at Risk</div>'
        f'<div class="kpi-delta" style="color: #E74C3C;">{pct:.0f}% of portfolio</div></div>',
        unsafe_allow_html=True,
    )

with c3:
    total_stake = overview.get("total_rupee_at_stake", float(risk_df["total_rupee_at_stake"].sum()) if risk_df is not None else 0.0)
    if total_stake >= 10_000_000:
        display_stake = f"₹{total_stake/10_000_000:.1f} Cr"
    elif total_stake >= 100_000:
        display_stake = f"₹{total_stake/100_000:.1f} L"
    else:
        display_stake = f"₹{total_stake:,.0f}"
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-value" style="color: #F39C12;">{display_stake}</div>'
        f'<div class="kpi-label">Total ₹ at Stake</div></div>',
        unsafe_allow_html=True,
    )

with c4:
    best_wape = overview.get("best_model_wape", 0.0767)
    model_name = overview.get("best_model_name", "PyTorch LSTM")
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-value" style="color: #27AE60;">{best_wape:.3f}</div>'
        f'<div class="kpi-label">Best WAPE</div>'
        f'<div class="kpi-delta" style="color: #27AE60;">{model_name}</div></div>',
        unsafe_allow_html=True,
    )

with c5:
    total_rev = overview.get("total_revenue", float(analysis["revenue"].sum()) if analysis is not None and "revenue" in analysis.columns else 0.0)
    if total_rev >= 10_000_000:
        display_rev = f"₹{total_rev/10_000_000:.1f} Cr"
    elif total_rev >= 100_000:
        display_rev = f"₹{total_rev/100_000:.1f} L"
    else:
        display_rev = f"₹{total_rev:,.0f}"
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-value">{display_rev}</div><div class="kpi-label">Total Revenue</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model Comparison Table
# ---------------------------------------------------------------------------
if data["comparison"] is not None:
    st.markdown('<div class="section-header">Model Performance Comparison</div>', unsafe_allow_html=True)

    comp = data["comparison"].copy()
    comp["wape"] = comp["wape"].round(4)
    comp = comp.sort_values("wape")

    # Highlight winner
    col_m1, col_m2 = st.columns([2, 3])

    with col_m1:
        for i, row in comp.iterrows():
            is_winner = row["wape"] == comp["wape"].min()
            color = "#27AE60" if is_winner else "#8b8fa3"
            border_color = "rgba(39,174,96,0.35)" if is_winner else "rgba(255,255,255,0.06)"
            winner_badge = '<span class="model-badge badge-winner">WINNER</span>' if is_winner else ""

            card_html = (
                f'<div style="background: rgba(255,255,255,0.03); border-radius:12px; padding:1rem; margin-bottom:0.6rem; border: 1px solid {border_color};">'
                f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                f'<span style="color:#e0e0ff; font-weight:600;">{row["name"]}</span>'
                f'{winner_badge}'
                f'</div>'
                f'<div style="font-size:1.75rem; font-weight:800; color:{color}; margin-top:0.3rem;">'
                f'WAPE: {row["wape"]:.4f}'
                f'</div>'
                f'</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

    with col_m2:
        # Bar chart comparison
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


# ---------------------------------------------------------------------------
# Risk Distribution
# ---------------------------------------------------------------------------
if risk_df is not None:
    st.markdown('<div class="section-header">Risk Distribution — Decisioning Grid</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    color_map = {
        "Reorder Now": "#E74C3C",
        "Markdown / Clear": "#F39C12",
        "Watch / Volatile": "#E67E22",
        "Healthy": "#27AE60",
    }

    with col_left:
        quadrant_counts = risk_df["quadrant"].value_counts().reset_index()
        quadrant_counts.columns = ["Quadrant", "Count"]

        fig = px.pie(
            quadrant_counts, names="Quadrant", values="Count",
            color="Quadrant", color_discrete_map=color_map,
            hole=0.5,
        )
        fig.update_traces(textposition='outside', textinfo='label+value')
        fig.update_layout(
            height=380,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b8fa3"),
            showlegend=False,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        top_risk = risk_df.nlargest(10, "total_rupee_at_stake")
        fig2 = px.bar(
            top_risk, x="sku_id", y="total_rupee_at_stake",
            color="quadrant", color_discrete_map=color_map,
            labels={"total_rupee_at_stake": "₹ at Stake", "sku_id": "SKU"},
        )
        fig2.update_layout(
            height=380,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b8fa3"),
            xaxis_tickangle=-45,
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(orientation="h", y=-0.25),
            title=dict(text="Top 10 SKUs by ₹ at Stake", font=dict(size=14, color="#e0e0ff")),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Action summary cards
    st.markdown('<div class="section-header">Quick Actions</div>', unsafe_allow_html=True)

    ac1, ac2, ac3 = st.columns(3)
    for col, quadrant, css_class, emoji in zip(
        [ac1, ac2, ac3],
        ["Reorder Now", "Markdown / Clear", "Healthy"],
        ["action-reorder", "action-markdown", "action-healthy"],
        ["🔴", "🟡", "🟢"],
    ):
        q_data = risk_df[risk_df["quadrant"] == quadrant]
        with col:
            rupee_val = q_data["total_rupee_at_stake"].sum()
            if rupee_val >= 10_000_000:
                rupee_display = f"₹{rupee_val/10_000_000:.1f} Cr"
            elif rupee_val >= 100_000:
                rupee_display = f"₹{rupee_val/100_000:.1f} L"
            else:
                rupee_display = f"₹{rupee_val:,.0f}"

            action_html = (
                f'<div class="action-card {css_class}">'
                f'<div style="font-size:0.9rem; font-weight:700; color:#e0e0ff;">{emoji} {quadrant}</div>'
                f'<div style="font-size:1.6rem; font-weight:800; color:#e0e0ff; margin: 0.3rem 0;">{len(q_data)} SKUs</div>'
                f'<div style="font-size:0.85rem; color:#8b8fa3;">{rupee_display} at stake</div>'
                f'</div>'
            )
            st.markdown(action_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Feature Importance (from trained model)
# ---------------------------------------------------------------------------
if data["feature_importance"] is not None:
    st.markdown('<div class="section-header">What Drives Demand? (Model Feature Importance)</div>', unsafe_allow_html=True)

    fi = data["feature_importance"].head(10)
    fig_fi = px.bar(
        fi, x="importance", y="feature", orientation="h",
        color="importance",
        color_continuous_scale=["#667eea", "#764ba2"],
        labels={"importance": "Importance Score", "feature": ""},
    )
    fig_fi.update_layout(
        height=350,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b8fa3"),
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(l=150),
    )
    st.plotly_chart(fig_fi, use_container_width=True)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding: 2rem 0; border-top: 1px solid rgba(102,126,234,0.15);">
    <span style="color: #4a4a6a; font-size: 0.8rem;">
        Project FORESIGHT — Zidio Development | Demand & Inventory Intelligence for NorthBay Living
    </span>
</div>
""", unsafe_allow_html=True)
