"""
Page 3: Risk — Risk scoring and decisioning grid.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils import DATA_DASHBOARD
from app.shared_styles import apply_shared_theme, update_chart_theme

st.set_page_config(page_title="Risk — FORESIGHT", page_icon="⚠️", layout="wide")
apply_shared_theme()

st.title("Risk Scoring & Decisioning Grid")
st.caption("Stockout vs overstock risk for every SKU — the view the ops team acts on")


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

# --- Filters ---
col_f1, col_f2 = st.columns(2)
with col_f1:
    categories = ["All"] + sorted(risk_df["category"].dropna().unique().tolist())
    sel_cat = st.selectbox("Filter by Category", categories)
with col_f2:
    quadrants = ["All"] + sorted(risk_df["quadrant"].unique().tolist())
    sel_quad = st.selectbox("Filter by Quadrant", quadrants)

# Apply filters
filtered = risk_df.copy()
if sel_cat != "All":
    filtered = filtered[filtered["category"] == sel_cat]
if sel_quad != "All":
    filtered = filtered[filtered["quadrant"] == sel_quad]

# --- Decisioning Grid (scatter) ---
st.subheader("Decisioning Grid")
st.markdown("Each bubble is a SKU. **X-axis** = stockout risk, **Y-axis** = overstock risk, **Size** = ₹ at stake.")

color_map = {
    "Reorder Now": "#E74C3C",
    "Markdown / Clear": "#F39C12",
    "Watch / Volatile": "#E67E22",
    "Healthy": "#27AE60",
}

fig = px.scatter(
    filtered,
    x="stockout_risk",
    y="overstock_risk",
    size="total_rupee_at_stake",
    color="quadrant",
    color_discrete_map=color_map,
    hover_name="sku_id",
    hover_data={
        "category": True,
        "avg_weekly_forecast": True,
        "on_hand_units": True,
        "total_rupee_at_stake": ":,.0f",
        "action": True,
    },
    size_max=40,
    labels={
        "stockout_risk": "Stockout Risk →",
        "overstock_risk": "Overstock Risk →",
    },
)

# Add quadrant lines
fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
fig.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.5)

# Add quadrant labels
fig.add_annotation(x=0.75, y=0.25, text="REORDER NOW", showarrow=False,
                   font=dict(size=12, color="#E74C3C"), opacity=0.4)
fig.add_annotation(x=0.25, y=0.75, text="MARKDOWN/CLEAR", showarrow=False,
                   font=dict(size=12, color="#F39C12"), opacity=0.4)
fig.add_annotation(x=0.75, y=0.75, text="WATCH/VOLATILE", showarrow=False,
                   font=dict(size=12, color="#E67E22"), opacity=0.4)
fig.add_annotation(x=0.25, y=0.25, text="HEALTHY", showarrow=False,
                   font=dict(size=12, color="#27AE60"), opacity=0.4)

fig.update_layout(
    height=550,
    xaxis=dict(range=[-0.05, 1.05]),
    yaxis=dict(range=[-0.05, 1.05]),
)
st.plotly_chart(fig, use_container_width=True)

# --- Summary metrics ---
st.divider()
st.subheader("Risk Summary")

c1, c2, c3, c4 = st.columns(4)
for col, quadrant, color in zip(
    [c1, c2, c3, c4],
    ["Reorder Now", "Markdown / Clear", "Watch / Volatile", "Healthy"],
    ["🔴", "🟡", "🟠", "🟢"],
):
    q_data = filtered[filtered["quadrant"] == quadrant]
    with col:
        st.metric(
            f"{color} {quadrant}",
            f"{len(q_data)} SKUs",
            delta=f"₹{q_data['total_rupee_at_stake'].sum():,.0f}",
        )

# --- Detail table ---
st.divider()
st.subheader("Detailed Risk Scores")

display_cols = [
    "sku_id", "category", "quadrant", "action", "urgency",
    "stockout_risk", "overstock_risk",
    "avg_weekly_forecast", "on_hand_units",
    "stockout_rupee_at_risk", "overstock_rupee_locked",
    "total_rupee_at_stake",
]
display_cols = [c for c in display_cols if c in filtered.columns]

st.dataframe(
    filtered[display_cols].sort_values("total_rupee_at_stake", ascending=False),
    use_container_width=True,
    height=400,
)

# --- SKU detail card ---
st.divider()
st.subheader("SKU Detail")
sel_sku = st.selectbox("Select a SKU to inspect", sorted(filtered["sku_id"].unique()))
sku_row = filtered[filtered["sku_id"] == sel_sku]

if len(sku_row) > 0:
    row = sku_row.iloc[0]
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**SKU:** {row['sku_id']}")
        st.markdown(f"**Category:** {row.get('category', 'N/A')}")
        st.markdown(f"**Quadrant:** {row['quadrant']}")
        st.markdown(f"**Action:** {row['action']}")
    with col_b:
        st.markdown(f"**Stockout Risk:** {row['stockout_risk']:.3f}")
        st.markdown(f"**Overstock Risk:** {row['overstock_risk']:.3f}")
        st.markdown(f"**On Hand:** {row['on_hand_units']:.0f} units")
        st.markdown(f"**₹ at Stake:** ₹{row['total_rupee_at_stake']:,.0f}")
