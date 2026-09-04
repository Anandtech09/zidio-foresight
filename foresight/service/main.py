"""
main.py — FastAPI scoring service for Project FORESIGHT.

Returns forecast + risk for a given SKU or batch.
Documented endpoints with Swagger UI at /docs.
Handles bad input gracefully.

Run: uvicorn service.main:app --reload
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from service.schemas import (
    ForecastResponse, BatchForecastResponse,
    SKUForecastRequest, HealthResponse, ErrorResponse,
)
from src.utils import DATA_DASHBOARD, DATA_CLEANED

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Project FORESIGHT — Scoring Service",
    description=(
        "Demand forecast and inventory risk scoring API for NorthBay Living.\n\n"
        "Returns weekly demand forecast, stockout/overstock risk, and recommended "
        "actions for any SKU."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
risk_data = None
sku_master_data = None


def load_scored_data():
    """Load risk-scored data on startup."""
    global risk_data, sku_master_data
    try:
        risk_data = pd.read_csv(DATA_DASHBOARD / "risk_scored.csv")
    except FileNotFoundError:
        risk_data = pd.DataFrame()

    try:
        sku_master_data = pd.read_csv(DATA_CLEANED / "sku_master_clean.csv")
    except FileNotFoundError:
        sku_master_data = pd.DataFrame()


@app.on_event("startup")
async def startup():
    load_scored_data()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def get_sku_forecast(sku_id: str) -> ForecastResponse:
    """Get forecast + risk for a single SKU."""
    if risk_data is None or len(risk_data) == 0:
        raise HTTPException(status_code=503, detail="Model data not loaded. Run the pipeline first.")

    sku_row = risk_data[risk_data["sku_id"] == sku_id]
    if len(sku_row) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"SKU '{sku_id}' not found. Use GET /skus to see available SKUs."
        )

    row = sku_row.iloc[0]
    return ForecastResponse(
        sku_id=row["sku_id"],
        category=row.get("category"),
        subcategory=row.get("subcategory"),
        avg_weekly_forecast=round(float(row.get("avg_weekly_forecast", 0)), 2),
        on_hand_units=float(row.get("on_hand_units", 0)),
        on_order_units=float(row.get("on_order_units", 0)),
        lead_time_days=float(row.get("lead_time_days", 14)),
        stockout_risk=round(float(row.get("stockout_risk", 0)), 3),
        overstock_risk=round(float(row.get("overstock_risk", 0)), 3),
        risk_quadrant=row.get("quadrant", "Unknown"),
        recommended_action=row.get("action", "N/A"),
        urgency=row.get("urgency", "LOW"),
        stockout_rupee_at_risk=round(float(row.get("stockout_rupee_at_risk", 0)), 2),
        overstock_rupee_locked=round(float(row.get("overstock_rupee_locked", 0)), 2),
        total_rupee_at_stake=round(float(row.get("total_rupee_at_stake", 0)), 2),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check — verify the service is running and data is loaded."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=risk_data is not None and len(risk_data) > 0,
        total_skus=len(risk_data) if risk_data is not None else 0,
    )


@app.get("/skus", tags=["SKUs"])
async def list_skus(
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(50, ge=1, le=500, description="Max results"),
):
    """List all available SKUs, optionally filtered by category."""
    if risk_data is None or len(risk_data) == 0:
        return {"skus": [], "total": 0}

    df = risk_data.copy()
    if category:
        df = df[df["category"].str.lower() == category.lower()]

    skus = df["sku_id"].unique().tolist()[:limit]
    return {
        "skus": skus,
        "total": len(skus),
        "categories": risk_data["category"].unique().tolist() if "category" in risk_data.columns else [],
    }


@app.get("/sku/{sku_id}/forecast", response_model=ForecastResponse, tags=["Forecast"])
async def get_forecast(sku_id: str):
    """
    Get forecast + risk for a single SKU.

    Returns weekly demand forecast, stockout/overstock risk scores,
    the risk quadrant, and the recommended action.
    """
    return get_sku_forecast(sku_id)


@app.post("/batch/forecast", response_model=BatchForecastResponse, tags=["Forecast"])
async def batch_forecast(request: SKUForecastRequest):
    """
    Get forecast + risk for a batch of SKUs.

    Send a list of SKU IDs and receive forecasts for all of them.
    Invalid SKU IDs are silently skipped.
    """
    results = []
    for sku_id in request.sku_ids:
        try:
            result = get_sku_forecast(sku_id)
            results.append(result)
        except HTTPException:
            continue  # Skip invalid SKUs

    total_stake = sum(r.total_rupee_at_stake for r in results)

    return BatchForecastResponse(
        results=results,
        total_skus=len(results),
        total_rupee_at_stake=round(total_stake, 2),
    )


@app.get("/risk/summary", tags=["Risk"])
async def risk_summary():
    """
    Get a summary of risk across all SKUs.

    Returns counts and rupee values per risk quadrant.
    """
    if risk_data is None or len(risk_data) == 0:
        raise HTTPException(status_code=503, detail="No risk data available.")

    summary = {}
    for quadrant in ["Reorder Now", "Markdown / Clear", "Watch / Volatile", "Healthy"]:
        q_data = risk_data[risk_data["quadrant"] == quadrant]
        summary[quadrant] = {
            "count": len(q_data),
            "total_rupee_at_stake": round(float(q_data["total_rupee_at_stake"].sum()), 2),
        }

    return {
        "total_skus": len(risk_data),
        "total_rupee_at_stake": round(float(risk_data["total_rupee_at_stake"].sum()), 2),
        "quadrants": summary,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
