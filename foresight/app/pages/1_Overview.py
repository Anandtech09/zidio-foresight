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

# --- KPIs ---
st.subheader("Business KPIs")
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("Total SKUs", df["sku_id"].nunique())
with c2:
    if "revenue" in df.columns:
        st.metric("Total Revenue", f"₹{df['revenue'].sum():,.0f}")
with c3:
    if "units_sold" in df.columns:
        st.metric("Total Units Sold", f"{df['units_sold'].sum():,.0f}")
with c4:
    if "category" in df.columns:
        st.metric("Categories", df["category"].nunique())
with c5:
    date_range = f"{df['date'].min().strftime('%b %Y')} — {df['date'].max().strftime('%b %Y')}"
    st.metric("Data Period", date_range)

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
