"""
Page 1: Overview — KPI summary and key metrics.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils import DATA_FEATURES, DATA_DASHBOARD, DATA_CLEANED
from app.shared_styles import apply_shared_theme, update_chart_theme

st.set_page_config(page_title="Overview — FORESIGHT", page_icon="📊", layout="wide")
apply_shared_theme()

st.title("Overview")
st.caption("Key performance indicators and engagement summary")


@st.cache_data
def load_data():
    data = {}
    try:
        data["analysis"] = pd.read_csv(DATA_FEATURES / "analysis_ready.csv", parse_dates=["date"])
    except FileNotFoundError:
        data["analysis"] = None
    try:
        data["risk"] = pd.read_csv(DATA_DASHBOARD / "risk_scored.csv")
    except FileNotFoundError:
        data["risk"] = None
    try:
        data["sku_master"] = pd.read_csv(DATA_CLEANED / "sku_master_clean.csv")
    except FileNotFoundError:
        data["sku_master"] = None
    return data


data = load_data()

if data["analysis"] is None:
    st.warning("No data loaded. Run `python -m src.pipeline` first.")
    st.stop()

df = data["analysis"]

# Helper formatters to prevent Streamlit metric truncation (e.g. ₹11,552,94...)
def format_inr(val: float) -> str:
    if val >= 10_000_000:
        return f"₹{val / 10_000_000:.2f} Cr"
    elif val >= 100_000:
        return f"₹{val / 100_000:.2f} L"
    return f"₹{val:,.0f}"


def format_count(val: float) -> str:
    if val >= 1_000_000:
        return f"{val / 1_000_000:.2f} M"
    elif val >= 100_000:
        return f"{val / 100_000:.1f} L"
    elif val >= 1_000:
        return f"{val / 1_000:.1f} K"
    return f"{val:,.0f}"


# --- KPIs ---
st.subheader("Business KPIs")
c1, c2, c3, c4, c5 = st.columns(5)

tot_skus = df["sku_id"].nunique()
tot_rev = float(df["revenue"].sum()) if "revenue" in df.columns else 0.0
tot_units = float(df["units_sold"].sum()) if "units_sold" in df.columns else 0.0
tot_cats = df["category"].nunique() if "category" in df.columns else 0
min_date = df["date"].min().strftime("%b %Y")
max_date = df["date"].max().strftime("%b %Y")

with c1:
    st.markdown(
        f'<div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:14px; min-height:96px;">'
        f'<div style="color:#94a3b8; font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">Total SKUs</div>'
        f'<div style="color:#f8fafc; font-size:1.55rem; font-weight:800; margin-top:4px;">{tot_skus}</div>'
        f'<div style="color:#64748b; font-size:0.72rem; margin-top:2px;">Active Catalog Items</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f'<div style="background:rgba(6,182,212,0.04); border:1px solid rgba(6,182,212,0.3); border-radius:12px; padding:14px; min-height:96px;">'
        f'<div style="color:#94a3b8; font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">Total Revenue</div>'
        f'<div style="color:#38bdf8; font-size:1.55rem; font-weight:800; margin-top:4px;">{format_inr(tot_rev)}</div>'
        f'<div style="color:#64748b; font-size:0.72rem; margin-top:2px;">₹{tot_rev/10_000_000:.1f} Cr Gross Sales</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f'<div style="background:rgba(34,197,94,0.04); border:1px solid rgba(34,197,94,0.3); border-radius:12px; padding:14px; min-height:96px;">'
        f'<div style="color:#94a3b8; font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">Total Units Sold</div>'
        f'<div style="color:#4ade80; font-size:1.55rem; font-weight:800; margin-top:4px;">{format_count(tot_units)}</div>'
        f'<div style="color:#64748b; font-size:0.72rem; margin-top:2px;">{tot_units:,.0f} Total Fulfilled</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f'<div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:14px; min-height:96px;">'
        f'<div style="color:#94a3b8; font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">Categories</div>'
        f'<div style="color:#f8fafc; font-size:1.55rem; font-weight:800; margin-top:4px;">{tot_cats}</div>'
        f'<div style="color:#64748b; font-size:0.72rem; margin-top:2px;">Core Product Verticals</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with c5:
    st.markdown(
        f'<div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:14px; min-height:96px;">'
        f'<div style="color:#94a3b8; font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">Data Period</div>'
        f'<div style="color:#f8fafc; font-size:1.05rem; font-weight:800; margin-top:6px; line-height:1.2;">{min_date} – {max_date}</div>'
        f'<div style="color:#64748b; font-size:0.72rem; margin-top:4px;">104-Week Historical Window</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.divider()

# --- Revenue trend ---
st.subheader("Revenue Trend Over Time")
if "revenue" in df.columns:
    weekly_rev = df.groupby(pd.Grouper(key="date", freq="W"))["revenue"].sum().reset_index()
    fig = px.area(weekly_rev, x="date", y="revenue",
                  labels={"revenue": "Weekly Revenue (₹)", "date": "Week"},
                  color_discrete_sequence=["#667eea"])
    fig = update_chart_theme(fig, height=350, show_legend=False)
    st.plotly_chart(fig, use_container_width=True)

# --- Category breakdown ---
st.subheader("Revenue by Category")
if "category" in df.columns and "revenue" in df.columns:
    cat_rev = df.groupby("category")["revenue"].sum().sort_values(ascending=False).reset_index()
    fig2 = px.bar(cat_rev, x="category", y="revenue",
                  color="category", title="",
                  labels={"revenue": "Total Revenue (₹)", "category": "Category"})
    fig2 = update_chart_theme(fig2, height=350, show_legend=False)
    st.plotly_chart(fig2, use_container_width=True)

# --- Demand distribution ---
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Daily Demand Distribution")
    if "units_sold" in df.columns:
        fig3 = px.histogram(df, x="units_sold", nbins=50,
                            labels={"units_sold": "Daily Units Sold"},
                            color_discrete_sequence=["#764ba2"])
        fig3 = update_chart_theme(fig3, height=300, show_legend=False)
        st.plotly_chart(fig3, use_container_width=True)

with col_b:
    st.subheader("Promo vs Non-Promo Sales")
    if "promo_flag" in df.columns and "units_sold" in df.columns:
        promo_comp = df.groupby("promo_flag")["units_sold"].mean().reset_index()
        promo_comp["promo_flag"] = promo_comp["promo_flag"].map({0: "No Promo", 1: "Promo"})
        fig4 = px.bar(promo_comp, x="promo_flag", y="units_sold",
                      color="promo_flag", title="",
                      labels={"units_sold": "Avg Daily Units", "promo_flag": ""},
                      color_discrete_map={"No Promo": "#95a5a6", "Promo": "#e74c3c"})
        fig4 = update_chart_theme(fig4, height=300, show_legend=False)
        st.plotly_chart(fig4, use_container_width=True)
