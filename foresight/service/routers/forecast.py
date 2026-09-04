"""
service/routers/forecast.py — SKU demand forecasting, time-series history, and batch operations.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query
import numpy as np
import pandas as pd

from service.schemas import (
    BatchForecastResponse,
    ForecastResponse,
    SKUForecastRequest,
)
from service.state import state

router = APIRouter(tags=["Forecast"])


@router.get("/skus", tags=["Catalog"])
async def list_skus(category: Optional[str] = Query(None, description="Filter by category")):
    """List all available SKU identifiers with catalog metadata."""
    risk_df = state["risk_df"]
    if risk_df is None or len(risk_df) == 0:
        return {"skus": [], "total": 0, "categories": []}

    df = risk_df.copy()
    if category and category != "All":
        df = df[df["category"].str.lower() == category.lower()]

    categories = sorted(risk_df["category"].dropna().unique().tolist())
    skus = df["sku_id"].dropna().unique().tolist()
    return {"skus": skus, "total": len(skus), "categories": categories}


@router.get("/sku/{sku_id}/forecast", response_model=ForecastResponse)
async def get_sku_forecast(sku_id: str):
    """Retrieve demand forecast, stockout risk, overstock risk, and action recommendation for a SKU."""
    risk_df = state["risk_df"]
    if risk_df is None or len(risk_df) == 0:
        raise HTTPException(status_code=503, detail="Risk scoring data not ready. Run pipeline first.")

    row = risk_df[risk_df["sku_id"] == sku_id]
    if len(row) == 0:
        # Try alternate hyphen/underscore format
        alt_id = sku_id.replace("_", "-") if "_" in sku_id else sku_id.replace("-", "_")
        row = risk_df[risk_df["sku_id"] == alt_id]

    if len(row) == 0:
        raise HTTPException(status_code=404, detail=f"SKU '{sku_id}' not found.")

    r = row.iloc[0]
    return ForecastResponse(
        sku_id=r["sku_id"],
        category=r.get("category"),
        subcategory=r.get("subcategory"),
        avg_weekly_forecast=round(float(r.get("avg_weekly_forecast", 0)), 2),
        on_hand_units=float(r.get("on_hand_units", 0)),
        on_order_units=float(r.get("on_order_units", 0)),
        lead_time_days=float(r.get("lead_time_days", 14)),
        stockout_risk=round(float(r.get("stockout_risk", 0)), 3),
        overstock_risk=round(float(r.get("overstock_risk", 0)), 3),
        risk_quadrant=r.get("quadrant", "Healthy"),
        recommended_action=r.get("action", "Maintain current order policy"),
        urgency=r.get("urgency", "LOW"),
        stockout_rupee_at_risk=round(float(r.get("stockout_rupee_at_risk", 0)), 2),
        overstock_rupee_locked=round(float(r.get("overstock_rupee_locked", 0)), 2),
        total_rupee_at_stake=round(float(r.get("total_rupee_at_stake", 0)), 2),
    )


@router.get("/sku/{sku_id}/history")
async def get_sku_history(sku_id: str):
    """Retrieve historical weekly actuals and multi-model forecast curves for charting."""
    lgbm = state["lgbm_forecasts"]
    base = state["baseline_forecasts"]

    # Match sku_id
    records = []
    if lgbm is not None and len(lgbm) > 0 and "sku_id" in lgbm.columns:
        sub = lgbm[lgbm["sku_id"] == sku_id]
        if len(sub) == 0:
            alt_id = sku_id.replace("_", "-") if "_" in sku_id else sku_id.replace("-", "_")
            sub = lgbm[lgbm["sku_id"] == alt_id]

        sub = sub.sort_values("date")
        for _, r in sub.iterrows():
            rec = {
                "date": str(r["date"])[:10],
                "actual": float(r.get("actual", np.nan)) if pd.notna(r.get("actual")) else None,
                "lgbm_forecast": round(float(r.get("lgbm_forecast", 0)), 2),
                "lgbm_lower": round(float(r.get("lgbm_lower", 0)), 2),
                "lgbm_upper": round(float(r.get("lgbm_upper", 0)), 2),
            }
            records.append(rec)

    # Merge baseline forecast
    if base is not None and len(base) > 0 and "sku_id" in base.columns:
        b_sub = base[base["sku_id"] == sku_id]
        if len(b_sub) == 0:
            alt_id = sku_id.replace("_", "-") if "_" in sku_id else sku_id.replace("-", "_")
            b_sub = base[base["sku_id"] == alt_id]

        b_sub = b_sub.set_index("date")
        for rec in records:
            dt = pd.to_datetime(rec["date"])
            if dt in b_sub.index:
                rec["baseline_forecast"] = round(float(b_sub.loc[dt, "baseline_forecast"]), 2)
            else:
                rec["baseline_forecast"] = None

    return {"sku_id": sku_id, "series": records}


@router.post("/batch/forecast", response_model=BatchForecastResponse)
async def batch_forecast(request: SKUForecastRequest):
    """Batch-forecast multiple SKUs in a single payload."""
    results = []
    for sid in request.sku_ids:
        try:
            res = await get_sku_forecast(sid)
            results.append(res)
        except HTTPException:
            continue

    total_stake = sum(r.total_rupee_at_stake for r in results)
    return BatchForecastResponse(
        results=results,
        total_skus=len(results),
        total_rupee_at_stake=round(total_stake, 2),
    )
