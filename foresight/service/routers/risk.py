"""
service/routers/risk.py — Inventory risk categorization and capital exposure endpoints.
"""

from fastapi import APIRouter
from service.state import state

router = APIRouter(tags=["Risk"])


@router.get("/risk/catalog")
async def get_risk_catalog():
    """Retrieve full 192-SKU risk decisioning grid with financial stakes."""
    risk_df = state["risk_df"]
    if risk_df is None or len(risk_df) == 0:
        return {"total": 0, "records": []}

    records = risk_df.to_dict(orient="records")
    return {"total": len(records), "records": records}


@router.get("/risk/summary")
async def risk_summary():
    """Aggregated portfolio capital exposure partitioned by 2x2 decision quadrant."""
    risk_df = state["risk_df"]
    if risk_df is None or len(risk_df) == 0:
        return {"total_skus": 0, "total_rupee_at_stake": 0, "quadrants": {}}

    summary = {}
    for q in ["Reorder Now", "Markdown / Clear", "Watch / Volatile", "Healthy"]:
        sub = risk_df[risk_df["quadrant"] == q]
        summary[q] = {
            "count": len(sub),
            "total_rupee_at_stake": round(float(sub["total_rupee_at_stake"].sum()), 2),
        }

    return {
        "total_skus": len(risk_df),
        "total_rupee_at_stake": round(float(risk_df["total_rupee_at_stake"].sum()), 2),
        "quadrants": summary,
    }
