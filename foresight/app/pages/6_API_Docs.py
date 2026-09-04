"""
Page 6: API Documentation & Developer Portal.

Provides interactive documentation for all REST API endpoints exposed by the FastAPI
microservice, including request/response schemas, live test execution, and code snippets.
"""

import json
import sys
from pathlib import Path

import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.api_client import client as api_client
from app.shared_styles import apply_shared_theme

st.set_page_config(page_title="API Portal — FORESIGHT", page_icon="🔌", layout="wide")
apply_shared_theme()

st.title("REST API Documentation & Developer Center")
st.caption("Decoupled microservice specification for external ERPs, automated supply chains, and analytics pipelines.")

# Backend Status Bar
health = api_client.get_health()
is_online = health.get("status") == "healthy"

base_url = api_client.base_url

if is_online:
    st.markdown(
        f'<div style="background:rgba(34,197,94,0.08); border:1px solid #22c55e; border-radius:10px; padding:12px 18px; margin-bottom:1.5rem; display:flex; justify-content:space-between; align-items:center;">'
        f'<div><span style="font-weight:700; color:#4ade80;">● Microservice Online</span> <span style="color:#94a3b8; font-size:0.85rem; margin-left:8px;">Base URL: <code>{base_url}</code></span></div>'
        f'<div><a href="{base_url}/docs" target="_blank" style="background:#15803d; color:#f0fdf4; padding:5px 14px; border-radius:6px; font-size:0.8rem; font-weight:700; text-decoration:none;">Open Swagger UI ↗</a></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div style="background:rgba(239,68,68,0.08); border:1px solid #ef4444; border-radius:10px; padding:12px 18px; margin-bottom:1.5rem;">'
        f'<span style="font-weight:700; color:#f87171;">● Backend Offline</span> <span style="color:#94a3b8; font-size:0.85rem; margin-left:8px;">Expected at: <code>{base_url}</code>. Launch with: <code>uvicorn service.main:app --port 8000</code></span>'
        f'</div>',
        unsafe_allow_html=True,
    )

tab_catalog, tab_tester = st.tabs(["📋 API Catalog & Integration Specs", "🧪 Live Endpoint Tester"])

# ===========================================================================
# TAB 1: API Catalog
# ===========================================================================
with tab_catalog:
    endpoints = api_client.get_api_catalog()
    if not endpoints:
        # Static fallback list
        endpoints = [
            {
                "path": "/health",
                "method": "GET",
                "summary": "System Liveness & Model Cache Status",
                "description": "Verifies backend status and returns loaded ML model weights.",
                "tags": ["System"],
                "sample_curl": f"curl -X GET '{base_url}/health'",
            },
            {
                "path": "/portfolio/overview",
                "method": "GET",
                "summary": "Executive Portfolio KPI Summary",
                "description": "Returns portfolio SKUs, revenue totals, risk distribution, and best model.",
                "tags": ["Portfolio"],
                "sample_curl": f"curl -X GET '{base_url}/portfolio/overview'",
            },
            {
                "path": "/models/comparison",
                "method": "GET",
                "summary": "Model Leaderboard & Error Metrics",
                "description": "Compares PyTorch LSTM, LightGBM, and Seasonal Naive across WAPE, MAPE, and RMSE.",
                "tags": ["Models"],
                "sample_curl": f"curl -X GET '{base_url}/models/comparison'",
            },
            {
                "path": "/sku/{sku_id}/forecast",
                "method": "GET",
                "summary": "Single SKU Demand Forecast & Risk",
                "description": "Returns weekly forecast, stockout risk, overstock risk, and recommended action.",
                "tags": ["Forecast"],
                "sample_curl": f"curl -X GET '{base_url}/sku/SKU_0001/forecast'",
            },
            {
                "path": "/predict/manual",
                "method": "POST",
                "summary": "Live Dual-Model Scenario Simulation",
                "description": "Executes real-time inference comparing PyTorch LSTM and LightGBM models.",
                "tags": ["ML Inference"],
                "sample_curl": f"curl -X POST '{base_url}/predict/manual' -H 'Content-Type: application/json' -d '{{\"sku_id\":\"SKU_0001\",\"unit_price\":549,\"discount_pct\":10,\"is_holiday\":0,\"promo_flag\":1,\"recent_4w_avg\":45}}'",
            },
            {
                "path": "/risk/catalog",
                "method": "GET",
                "summary": "Full 192-SKU Risk Decisioning Grid",
                "description": "Returns all 192 catalog items with stockout/overstock risks and financial stakes.",
                "tags": ["Risk"],
                "sample_curl": f"curl -X GET '{base_url}/risk/catalog'",
            },
        ]

    tag_filter = st.selectbox("Filter by Category", ["All"] + sorted(list(set(e["tags"][0] for e in endpoints))))

    for ep in endpoints:
        if tag_filter != "All" and ep["tags"][0] != tag_filter:
            continue

        method_color = "#22c55e" if ep["method"] == "GET" else "#38bdf8"
        method_bg = "rgba(34,197,94,0.15)" if ep["method"] == "GET" else "rgba(6,182,212,0.15)"

        st.markdown(
            f'<div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:14px; margin-bottom:12px;">'
            f'<div style="display:flex; align-items:center; gap:12px;">'
            f'<span style="background:{method_bg}; color:{method_color}; font-weight:800; font-size:0.75rem; padding:3px 8px; border-radius:5px;">{ep["method"]}</span>'
            f'<span style="font-family:monospace; font-size:0.95rem; font-weight:700; color:#f8fafc;">{ep["path"]}</span>'
            f'<span style="color:#94a3b8; font-size:0.85rem; margin-left:auto;">{ep["summary"]}</span>'
            f'</div>'
            f'<div style="font-size:0.85rem; color:#cbd5e1; margin-top:8px;">{ep["description"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        with st.expander(f"Code Snippets for `{ep['path']}`"):
            st.code(ep["sample_curl"], language="bash")
            python_snippet = (
                f"import requests\n\n"
                f"url = '{base_url}{ep['path'].replace('{sku_id}', 'SKU_0001')}'\n"
                f"{'response = requests.get(url)' if ep['method'] == 'GET' else 'response = requests.post(url, json={...})'}\n"
                f"print(response.json())"
            )
            st.code(python_snippet, language="python")

# ===========================================================================
# TAB 2: Live Endpoint Tester
# ===========================================================================
with tab_tester:
    st.subheader("Send Live Requests to Backend")
    st.caption(f"Test API responses in real time against configured backend: `{base_url}`.")

    test_endpoint = st.selectbox(
        "Choose Endpoint to Test",
        [
            "GET /health",
            "GET /portfolio/overview",
            "GET /models/comparison",
            "GET /sku/SKU_0001/forecast",
            "GET /risk/summary",
            "POST /predict/manual",
        ],
    )

    if test_endpoint == "POST /predict/manual":
        t_sku = st.text_input("SKU ID", value="SKU_0001")
        t_price = st.number_input("Unit Price (₹)", value=549.0)
        t_disc = st.slider("Discount %", 0.0, 50.0, 10.0)
        t_avg = st.number_input("Recent 4w Avg", value=45.0)
        t_promo = st.checkbox("Promo Active", value=True)
        t_hol = st.checkbox("Holiday Surge", value=False)

        if st.button("Send POST /predict/manual Request", type="primary"):
            payload = {
                "sku_id": t_sku,
                "unit_price": t_price,
                "discount_pct": t_disc,
                "is_holiday": 1 if t_hol else 0,
                "promo_flag": 1 if t_promo else 0,
                "recent_4w_avg": t_avg,
            }
            try:
                res = requests.post(f"{base_url}/predict/manual", json=payload, timeout=5)
                st.write(f"**HTTP Status:** `{res.status_code}`")
                st.json(res.json())
            except Exception as e:
                st.error(f"Failed to connect to backend at {base_url}: {e}")

    else:
        path = test_endpoint.split(" ")[1]
        if st.button(f"Send {test_endpoint} Request", type="primary"):
            try:
                res = requests.get(f"{base_url}{path}", timeout=5)
                st.write(f"**HTTP Status:** `{res.status_code}`")
                st.json(res.json())
            except Exception as e:
                st.error(f"Failed to connect to backend at {base_url}: {e}")
