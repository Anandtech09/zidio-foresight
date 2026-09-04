"""
service/routers/system.py — System health, liveness, and API documentation endpoints.
"""

from typing import List
from fastapi import APIRouter

from service.config import API_URL
from service.schemas import APIDocEndpoint, HealthResponse
from service.state import state

router = APIRouter(tags=["System"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Verify backend microservice status and in-memory model availability."""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        models_loaded={
            "lightgbm_point": state["lgbm_model"] is not None,
            "lightgbm_q10": state["lgbm_q10"] is not None,
            "lightgbm_q90": state["lgbm_q90"] is not None,
            "pytorch_lstm": state["lstm_model"] is not None,
            "risk_scores": state["risk_df"] is not None and len(state["risk_df"]) > 0,
        },
        total_skus=len(state["risk_df"]) if state["risk_df"] is not None else 0,
        architecture="Decoupled Client-Server (Streamlit UI <-> FastAPI REST <-> ML Models)",
    )


@router.get("/api/catalog", response_model=List[APIDocEndpoint])
async def get_api_catalog():
    """Retrieve structured endpoint documentation for in-dashboard integration."""
    base = API_URL.rstrip("/")
    return [
        APIDocEndpoint(
            path="/health",
            method="GET",
            summary="System Liveness & Model Cache Status",
            description="Returns status of backend service and verifies in-memory model weights.",
            tags=["System"],
            sample_curl=f"curl -X GET '{base}/health'",
        ),
        APIDocEndpoint(
            path="/portfolio/overview",
            method="GET",
            summary="Executive Portfolio KPI Summary",
            description="Aggregates total catalog SKUs, items at risk, total revenue, and capital at stake.",
            tags=["Portfolio"],
            sample_curl=f"curl -X GET '{base}/portfolio/overview'",
        ),
        APIDocEndpoint(
            path="/models/comparison",
            method="GET",
            summary="Model Leaderboard & Error Metrics",
            description="Returns WAPE, MAPE, Bias, and RMSE across PyTorch LSTM, LightGBM, and Seasonal Naive.",
            tags=["Models"],
            sample_curl=f"curl -X GET '{base}/models/comparison'",
        ),
        APIDocEndpoint(
            path="/sku/{sku_id}/forecast",
            method="GET",
            summary="Single SKU Demand Forecast & Risk",
            description="Returns average weekly demand forecast, stockout risk, overstock risk, and recommended operational action.",
            tags=["Forecast"],
            sample_curl=f"curl -X GET '{base}/sku/SKU_0001/forecast'",
        ),
        APIDocEndpoint(
            path="/sku/{sku_id}/history",
            method="GET",
            summary="Historical Weekly Demand & Curve Series",
            description="Returns weekly time series including actual units, baseline predictions, and LightGBM 80% interval bounds.",
            tags=["Forecast"],
            sample_curl=f"curl -X GET '{base}/sku/SKU_0001/history'",
        ),
        APIDocEndpoint(
            path="/predict/manual",
            method="POST",
            summary="Live Dual-Model Scenario Simulation",
            description="Simulates demand under custom user scenarios, running inference on both PyTorch LSTM and LightGBM.",
            tags=["ML Inference"],
            sample_request={
                "sku_id": "SKU_0001",
                "unit_price": 549.0,
                "discount_pct": 10.0,
                "is_holiday": 0,
                "promo_flag": 1,
                "recent_4w_avg": 45.0,
            },
            sample_curl=(
                f"curl -X POST '{base}/predict/manual' "
                "-H 'Content-Type: application/json' "
                "-d '{\"sku_id\":\"SKU_0001\",\"unit_price\":549.0,\"discount_pct\":10.0,\"is_holiday\":0,\"promo_flag\":1,\"recent_4w_avg\":45.0}'"
            ),
        ),
        APIDocEndpoint(
            path="/batch/forecast",
            method="POST",
            summary="Batch Forecast for Multiple SKUs",
            description="Score a batch of SKU IDs in a single request and compute total capital at risk.",
            tags=["Forecast"],
            sample_request={"sku_ids": ["SKU_0001", "SKU_0002", "SKU_0003"]},
            sample_curl=(
                f"curl -X POST '{base}/batch/forecast' "
                "-H 'Content-Type: application/json' "
                "-d '{\"sku_ids\":[\"SKU_0001\",\"SKU_0002\",\"SKU_0003\"]}'"
            ),
        ),
        APIDocEndpoint(
            path="/risk/catalog",
            method="GET",
            summary="Full 192-SKU Risk Decisioning Grid",
            description="Returns all 192 scored SKUs with risk quadrants, stockout/overstock risks, and ₹ at stake.",
            tags=["Risk"],
            sample_curl=f"curl -X GET '{base}/risk/catalog'",
        ),
        APIDocEndpoint(
            path="/risk/summary",
            method="GET",
            summary="Quadrant-level Capital Exposure Summary",
            description="Aggregates SKU counts and total ₹ exposure across the 4 decisioning quadrants.",
            tags=["Risk"],
            sample_curl=f"curl -X GET '{base}/risk/summary'",
        ),
    ]
