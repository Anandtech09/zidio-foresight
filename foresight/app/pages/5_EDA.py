"""
Page 5: EDA — Exploratory Data Analysis insights and data quality.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils import DATA_FEATURES, DATA_CLEANED
from app.shared_styles import apply_shared_theme, update_chart_theme

st.set_page_config(page_title="EDA — FORESIGHT", page_icon="📋", layout="wide")
apply_shared_theme()

st.title("EDA & Data Quality")
st.caption("Exploratory analysis, data quality findings, and business insights")


@st.cache_data
def load_data():
    data = {}
    try:
        data["analysis"] = pd.read_csv(DATA_FEATURES / "analysis_ready.csv", parse_dates=["date"])
    except FileNotFoundError:
        data["analysis"] = None
    try:
        data["cleaning_log"] = pd.read_csv(DATA_CLEANED / "cleaning_log.csv")
    except FileNotFoundError:
        data["cleaning_log"] = None
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

# --- Data Quality Report ---
st.subheader("1. Data Quality Report")

if data["cleaning_log"] is not None:
    st.markdown("All cleaning decisions documented below:")
    st.dataframe(data["cleaning_log"], use_container_width=True)
else:
    st.info("Cleaning log not available.")

# Dataset shape
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{len(df):,}")
col2.metric("Columns", f"{len(df.columns)}")
col3.metric("Missing Values", f"{df.isnull().sum().sum():,}")
col4.metric("Duplicate Rows", f"{df.duplicated().sum()}")

# Missing value heatmap
st.markdown("**Missing Values by Column:**")
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    fig_miss = px.bar(x=missing.index, y=missing.values,
                      labels={"x": "Column", "y": "Missing Count"},
                      color_discrete_sequence=["#e74c3c"])
    fig_miss.update_layout(height=300)
    st.plotly_chart(fig_miss, use_container_width=True)
else:
    st.success("No missing values in the analysis-ready dataset!")

st.divider()

# --- Business Insights ---
st.subheader("2. Business Insights")

# Insight 1: Top movers vs Dead stock
st.markdown("### Insight 1: Top Revenue Generators vs Dead Stock")

if "sku_id" in df.columns and "revenue" in df.columns:
    sku_rev = df.groupby("sku_id")["revenue"].sum().sort_values(ascending=False)

    col_t, col_b = st.columns(2)
    with col_t:
        top10 = sku_rev.head(10).reset_index()
        fig_top = px.bar(top10, x="sku_id", y="revenue",
                         title="Top 10 SKUs by Revenue",
                         color_discrete_sequence=["#27AE60"])
        fig_top = update_chart_theme(fig_top, height=350, show_legend=False)
        st.plotly_chart(fig_top, use_container_width=True)

    with col_b:
        bottom10 = sku_rev.tail(10).reset_index()
        fig_bot = px.bar(bottom10, x="sku_id", y="revenue",
                         title="Bottom 10 SKUs (Dead Stock Candidates)",
                         color_discrete_sequence=["#E74C3C"])
        fig_bot = update_chart_theme(fig_bot, height=350, show_legend=False)
        st.plotly_chart(fig_bot, use_container_width=True)

    # Revenue concentration
    total_rev = sku_rev.sum()
    top20_pct = sku_rev.head(int(len(sku_rev) * 0.2)).sum() / total_rev * 100
    st.info(f"💡 **Pareto insight:** Top 20% of SKUs generate **{top20_pct:.0f}%** of total revenue.")

st.divider()

# Insight 2: Seasonality
st.markdown("### Insight 2: Seasonal Demand Patterns")

if "month_num" not in df.columns and "date" in df.columns:
    df["month_num"] = df["date"].dt.month

if "units_sold" in df.columns:
    monthly = df.groupby("month_num")["units_sold"].mean().reset_index()
    monthly["month_name"] = monthly["month_num"].map({
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    })
    fig_season = px.line(monthly, x="month_name", y="units_sold",
                         title="Average Daily Demand by Month",
                         markers=True, color_discrete_sequence=["#667eea"])
    fig_season = update_chart_theme(fig_season, height=350, show_legend=False)
    st.plotly_chart(fig_season, use_container_width=True)

    # Find peak months
    peak_month = monthly.loc[monthly["units_sold"].idxmax(), "month_name"]
    trough_month = monthly.loc[monthly["units_sold"].idxmin(), "month_name"]
    st.info(f"💡 **Peak demand:** {peak_month} | **Lowest demand:** {trough_month}")

st.divider()

# Insight 3: Promotional effectiveness
st.markdown("### Insight 3: Promotional Impact")

if "promo_flag" in df.columns and "units_sold" in df.columns:
    promo_stats = df.groupby("promo_flag")["units_sold"].agg(["mean", "median", "std"]).round(2)
    promo_stats.index = promo_stats.index.map({0: "No Promotion", 1: "With Promotion"})

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.dataframe(promo_stats, use_container_width=True)

    with col_p2:
        if len(promo_stats) == 2:
            uplift = ((promo_stats.loc["With Promotion", "mean"] -
                       promo_stats.loc["No Promotion", "mean"]) /
                      promo_stats.loc["No Promotion", "mean"]) * 100
            st.metric("Promo Uplift", f"{uplift:.1f}%",
                      delta="positive" if uplift > 0 else "negative")

    # Promo effect by category
    if "category" in df.columns:
        promo_by_cat = df.groupby(["category", "promo_flag"])["units_sold"].mean().unstack()
        if 0 in promo_by_cat.columns and 1 in promo_by_cat.columns:
            promo_by_cat["uplift_pct"] = ((promo_by_cat[1] - promo_by_cat[0]) / promo_by_cat[0] * 100).round(1)
            fig_promo = px.bar(promo_by_cat.reset_index(), x="category", y="uplift_pct",
                               title="Promo Uplift by Category (%)",
                               color_discrete_sequence=["#764ba2"])
            fig_promo = update_chart_theme(fig_promo, height=300, show_legend=False)
            st.plotly_chart(fig_promo, use_container_width=True)

st.divider()

# Insight 4: Day-of-week patterns
st.markdown("### Insight 4: Day-of-Week Demand Patterns")

if "date" in df.columns and "units_sold" in df.columns:
    df["dow"] = df["date"].dt.day_name()
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_demand = df.groupby("dow")["units_sold"].mean().reindex(dow_order).reset_index()

    fig_dow = px.bar(dow_demand, x="dow", y="units_sold",
                     title="Average Demand by Day of Week",
                     color_discrete_sequence=["#1abc9c"])
    fig_dow = update_chart_theme(fig_dow, height=300, show_legend=False)
    st.plotly_chart(fig_dow, use_container_width=True)

# --- Data Shape Summary ---
st.divider()
st.subheader("3. Dataset Structure")
st.markdown(f"""
| Property | Value |
|---|---|
| Date Range | {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')} |
| Total Days | {(df['date'].max() - df['date'].min()).days} |
| SKUs | {df['sku_id'].nunique()} |
| Categories | {df['category'].nunique() if 'category' in df.columns else 'N/A'} |
| Avg Daily Sales/SKU | {df['units_sold'].mean():.1f} units |
""")
