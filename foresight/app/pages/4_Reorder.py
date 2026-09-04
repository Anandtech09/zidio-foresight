"""
Page 4: Reorder — Prioritised reorder/markdown list.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils import DATA_DASHBOARD
from app.shared_styles import apply_shared_theme

st.set_page_config(page_title="Reorder — FORESIGHT", page_icon="🛒", layout="wide")
apply_shared_theme()

st.title("Reorder & Markdown List")
st.caption("Prioritised action list sorted by ₹ at stake — what the ops team acts on first")


@st.cache_data
def load_risk():
    try:
        return pd.read_csv(DATA_DASHBOARD / "risk_scored.csv")
    except FileNotFoundError:
        return None


risk_df = load_risk()

if risk_df is None:
    st.warning("No risk scores found. Run `python -m src.risk` first.")
    st.stop()

# --- Tabs for Reorder vs Markdown ---
tab_reorder, tab_markdown, tab_watch, tab_all = st.tabs([
    "🔴 Reorder Now", "🟡 Markdown / Clear", "🟠 Watch / Volatile", "📋 All Actions"
])

reorder_cols = [
    "sku_id", "category", "subcategory", "action",
    "avg_weekly_forecast", "on_hand_units", "on_order_units",
    "lead_time_days", "reorder_point",
    "stockout_risk", "stockout_rupee_at_risk", "total_rupee_at_stake",
]
reorder_cols = [c for c in reorder_cols if c in risk_df.columns]

with tab_reorder:
    st.subheader("SKUs Needing Immediate Reorder")
    st.markdown("These SKUs are at **high risk of stocking out**. Raise replenishment orders now.")

    reorder_df = risk_df[risk_df["quadrant"] == "Reorder Now"].sort_values(
        "stockout_rupee_at_risk", ascending=False
    )

    if len(reorder_df) > 0:
        st.metric("Total Reorder SKUs", len(reorder_df))
        st.metric("Revenue at Risk", f"₹{reorder_df['stockout_rupee_at_risk'].sum():,.0f}")
        st.divider()
        st.dataframe(reorder_df[reorder_cols], use_container_width=True, height=400)

        # Download button
        csv = reorder_df[reorder_cols].to_csv(index=False)
        st.download_button("📥 Download Reorder List (CSV)", csv,
                           "reorder_list.csv", "text/csv")
    else:
        st.success("No SKUs need immediate reorder. All stock levels are healthy.")

with tab_markdown:
    st.subheader("SKUs to Markdown or Clear")
    st.markdown("These SKUs are **overstocked**. Promote or discount to free up capital.")

    markdown_df = risk_df[risk_df["quadrant"] == "Markdown / Clear"].sort_values(
        "overstock_rupee_locked", ascending=False
    )

    markdown_cols = [
        "sku_id", "category", "subcategory", "action",
        "avg_weekly_forecast", "on_hand_units",
        "overstock_risk", "overstock_rupee_locked", "total_rupee_at_stake",
    ]
    markdown_cols = [c for c in markdown_cols if c in risk_df.columns]

    if len(markdown_df) > 0:
        st.metric("Total Overstock SKUs", len(markdown_df))
        st.metric("Capital Locked", f"₹{markdown_df['overstock_rupee_locked'].sum():,.0f}")
        st.divider()
        st.dataframe(markdown_df[markdown_cols], use_container_width=True, height=400)

        csv = markdown_df[markdown_cols].to_csv(index=False)
        st.download_button("📥 Download Markdown List (CSV)", csv,
                           "markdown_list.csv", "text/csv")
    else:
        st.success("No SKUs are overstocked. Inventory levels are appropriate.")

with tab_watch:
    st.subheader("SKUs to Watch (Volatile Demand)")
    st.markdown("These SKUs show **erratic demand**. Investigate manually before acting.")

    watch_df = risk_df[risk_df["quadrant"] == "Watch / Volatile"].sort_values(
        "total_rupee_at_stake", ascending=False
    )

    if len(watch_df) > 0:
        st.metric("Volatile SKUs", len(watch_df))
        st.dataframe(watch_df[reorder_cols], use_container_width=True, height=400)
    else:
        st.success("No SKUs showing volatile behaviour.")

with tab_all:
    st.subheader("Complete Action List")
    st.markdown("All SKUs ranked by total ₹ at stake.")

    all_cols = [
        "sku_id", "category", "quadrant", "action", "urgency",
        "stockout_risk", "overstock_risk",
        "avg_weekly_forecast", "on_hand_units",
        "total_rupee_at_stake",
    ]
    all_cols = [c for c in all_cols if c in risk_df.columns]

    st.dataframe(
        risk_df[all_cols].sort_values("total_rupee_at_stake", ascending=False),
        use_container_width=True,
        height=500,
    )

    csv = risk_df.to_csv(index=False)
    st.download_button("📥 Download Full Risk Report (CSV)", csv,
                       "full_risk_report.csv", "text/csv")
