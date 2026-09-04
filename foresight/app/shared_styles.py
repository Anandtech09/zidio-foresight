"""
shared_styles.py — Shared styling, theme tokens, and Plotly layout helper for FORESIGHT.
Ensures a unified, non-dizzy, consistent visual experience across all dashboard pages.
"""

import streamlit as st

SHARED_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(180deg, #0a0a1a 0%, #0f1422 50%, #0d1117 100%);
        color: #e0e0ff;
    }

    /* Typography */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }

    .sub-title {
        font-size: 1.05rem;
        color: #8b8fa3;
        margin-bottom: 1.8rem;
        font-weight: 400;
    }

    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e0e0ff;
        margin: 1.8rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(102,126,234,0.25);
    }

    /* Metric Cards — Robust container-based layout */
    .kpi-card {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid rgba(102, 126, 234, 0.25);
        border-radius: 14px;
        padding: 1.2rem 1rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        margin-bottom: 0.75rem;
    }

    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        color: #e0e0ff;
        line-height: 1.2;
    }

    .kpi-label {
        font-size: 0.8rem;
        color: #8b8fa3;
        margin-top: 0.35rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .kpi-delta {
        font-size: 0.8rem;
        margin-top: 0.25rem;
        font-weight: 600;
    }

    /* Action Cards */
    .action-card {
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid;
    }

    .action-reorder {
        background: rgba(231, 76, 60, 0.12);
        border-color: #E74C3C;
    }

    .action-markdown {
        background: rgba(243, 156, 18, 0.12);
        border-color: #F39C12;
    }

    .action-healthy {
        background: rgba(39, 174, 96, 0.12);
        border-color: #27AE60;
    }

    .action-watch {
        background: rgba(230, 126, 34, 0.12);
        border-color: #E67E22;
    }

    /* Model comparison badges */
    .model-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .badge-winner {
        background: rgba(39, 174, 96, 0.2);
        color: #27AE60;
        border: 1px solid rgba(39, 174, 96, 0.4);
    }

    /* Sidebar aesthetics */
    section[data-testid="stSidebar"] {
        background-color: #0c0f17 !important;
        border-right: 1px solid rgba(102, 126, 234, 0.15) !important;
    }

    /* Clean Streamlit elements */
    div[data-testid="stMetricValue"] {
        color: #e0e0ff !important;
        font-weight: 700 !important;
    }

    div[data-testid="stMetricLabel"] {
        color: #8b8fa3 !important;
    }

    /* Remove Streamlit default header decoration */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
</style>
"""


def apply_shared_theme():
    """Apply the unified dark theme across any page."""
    st.markdown(SHARED_CSS, unsafe_allow_html=True)


def update_chart_theme(fig, height=350, show_legend=True):
    """Ensure every Plotly chart shares the exact same dark theme formatting."""
    fig.update_layout(
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b8fa3", family="Inter, sans-serif"),
        showlegend=show_legend,
        margin=dict(l=30, r=20, t=30, b=30),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.08)",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.08)",
        ),
    )
    return fig
