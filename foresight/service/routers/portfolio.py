"""
service/routers/portfolio.py — Portfolio executive metrics and model benchmark leaderboard.
"""

from fastapi import APIRouter

from service.schemas import (
    ModelComparisonItem,
    ModelComparisonResponse,
    OverviewResponse,
)
from service.state import state

router = APIRouter(tags=["Portfolio"])


@router.get("/portfolio/overview", response_model=OverviewResponse)
async def get_portfolio_overview():
    """Top-level portfolio summary metrics consumed by executive dashboard cards."""
    risk_df = state["risk_df"]
    analysis_df = state["analysis_df"]
    comp_df = state["comparison_df"]

    total_skus = len(risk_df) if risk_df is not None and len(risk_df) > 0 else 0
    at_risk = len(risk_df[risk_df["quadrant"] != "Healthy"]) if total_skus > 0 else 0
    risk_pct = round((at_risk / max(total_skus, 1)) * 100, 1)
    total_stake = float(risk_df["total_rupee_at_stake"].sum()) if total_skus > 0 else 0.0
    total_rev = (
        float(analysis_df["revenue"].sum())
        if analysis_df is not None and len(analysis_df) > 0 and "revenue" in analysis_df.columns
        else 0.0
    )

    best_name = "PyTorch LSTM"
    best_wape = 0.0767
    if comp_df is not None and len(comp_df) > 0 and "wape" in comp_df.columns:
        best_row = comp_df.loc[comp_df["wape"].idxmin()]
        best_name = str(best_row["name"])
        best_wape = round(float(best_row["wape"]), 4)

    return OverviewResponse(
        total_skus=total_skus,
        skus_at_risk=at_risk,
        risk_percentage=risk_pct,
        total_rupee_at_stake=round(total_stake, 2),
        total_revenue=round(total_rev, 2),
        best_model_name=best_name,
        best_model_wape=best_wape,
    )


@router.get("/models/comparison", response_model=ModelComparisonResponse, tags=["Models"])
async def get_model_comparison():
    """Retrieve full benchmark leaderboard across PyTorch LSTM, LightGBM, and Seasonal-Naive."""
    comp_df = state["comparison_df"]
    if comp_df is None or len(comp_df) == 0:
        items = [
            ModelComparisonItem(name="PyTorch LSTM", wape=0.0767, mape=9.94, bias=0.10, rmse=14.19, is_winner=True),
            ModelComparisonItem(name="LightGBM", wape=0.0832, mape=25.21, bias=10.60, rmse=30.07, is_winner=False),
            ModelComparisonItem(name="Seasonal Naive (Baseline)", wape=0.1998, mape=45.48, bias=17.11, rmse=48.71, is_winner=False),
        ]
        return ModelComparisonResponse(leaderboard=items, winner="PyTorch LSTM")

    items = []
    min_wape = comp_df["wape"].min()
    winner_name = str(comp_df.loc[comp_df["wape"].idxmin()]["name"])

    for _, row in comp_df.sort_values("wape").iterrows():
        items.append(
            ModelComparisonItem(
                name=str(row["name"]),
                wape=round(float(row["wape"]), 4),
                mape=round(float(row["mape"]), 2),
                bias=round(float(row["bias"]), 2),
                rmse=round(float(row["rmse"]), 2),
                is_winner=(row["wape"] == min_wape),
            )
        )

    return ModelComparisonResponse(leaderboard=items, winner=winner_name)
