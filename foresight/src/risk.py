"""Inventory risk scoring and operations decision engine.

Evaluates stockout exposure and overstock capital lockup across catalog SKUs,
mapping inventory positions to operational quadrants and financial impact figures.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.utils import DATA_CLEANED, DATA_DASHBOARD, DATA_FORECASTS, get_logger

logger = get_logger("risk")

STOCKOUT_THRESHOLD = 0.5
OVERSTOCK_THRESHOLD = 0.5
FORWARD_WINDOW_WEEKS = 8


def compute_stockout_risk(
    forecast_demand: float, on_hand: float, on_order: float, lead_time_weeks: float
) -> float:
    """Calculate normalized stockout exposure: max(0, lead_demand - total_stock) / lead_demand."""
    demand_lead = forecast_demand * lead_time_weeks
    available = on_hand + on_order

    if demand_lead <= 0:
        return 0.0

    score = max(0.0, demand_lead - available) / demand_lead
    return float(np.clip(score, 0.0, 1.0))


def compute_overstock_risk(
    on_hand: float, forecast_demand: float, forward_weeks: int = FORWARD_WINDOW_WEEKS
) -> float:
    """Calculate normalized overstock exposure: max(0, on_hand - forward_demand) / on_hand."""
    forward_demand = forecast_demand * forward_weeks

    if on_hand <= 0:
        return 0.0

    score = max(0.0, on_hand - forward_demand) / on_hand
    return float(np.clip(score, 0.0, 1.0))


def assign_quadrant(stockout_risk: float, overstock_risk: float) -> Dict[str, str]:
    """Map dual risk scores to the 4-quadrant operational decision grid."""
    is_high_stockout = stockout_risk >= STOCKOUT_THRESHOLD
    is_high_overstock = overstock_risk >= OVERSTOCK_THRESHOLD

    if is_high_stockout and not is_high_overstock:
        return {
            "quadrant": "Reorder Now",
            "action": "Raise a replenishment order before stock runs out",
            "urgency": "HIGH",
            "color": "#E74C3C",
        }
    if not is_high_stockout and is_high_overstock:
        return {
            "quadrant": "Markdown / Clear",
            "action": "Promote or discount to free up capital",
            "urgency": "MEDIUM",
            "color": "#F39C12",
        }
    if is_high_stockout and is_high_overstock:
        return {
            "quadrant": "Watch / Volatile",
            "action": "Investigate — demand is erratic; review manually",
            "urgency": "HIGH",
            "color": "#E67E22",
        }
    return {
        "quadrant": "Healthy",
        "action": "No action needed; leave as is",
        "urgency": "LOW",
        "color": "#27AE60",
    }


def compute_rupee_at_stake(
    forecast_demand: float,
    on_hand: float,
    list_price: float,
    unit_cost: float,
    lead_time_weeks: float,
) -> Dict[str, float]:
    """Quantify financial exposure in Rupees for stockout sales at risk and locked capital."""
    demand_lead = forecast_demand * lead_time_weeks
    unmet_units = max(0.0, demand_lead - on_hand)
    stockout_rupee = unmet_units * list_price

    forward_demand = forecast_demand * FORWARD_WINDOW_WEEKS
    excess_units = max(0.0, on_hand - forward_demand)
    overstock_rupee = excess_units * unit_cost

    return {
        "stockout_rupee_at_risk": round(stockout_rupee, 2),
        "overstock_rupee_locked": round(overstock_rupee, 2),
        "total_rupee_at_stake": round(stockout_rupee + overstock_rupee, 2),
    }


def score_all_skus(
    forecast_df: pd.DataFrame, inventory_df: pd.DataFrame, sku_master_df: pd.DataFrame
) -> pd.DataFrame:
    """Evaluate stockout and overstock risk positions for all catalog SKUs."""
    if "lgbm_forecast" in forecast_df.columns:
        pred_col = "lgbm_forecast"
    elif "baseline_forecast" in forecast_df.columns:
        pred_col = "baseline_forecast"
    elif "forecast" in forecast_df.columns:
        pred_col = "forecast"
    else:
        pred_col = forecast_df.columns[-1]

    avg_forecast = forecast_df.groupby("sku_id")[pred_col].mean().reset_index()
    avg_forecast.columns = ["sku_id", "avg_weekly_forecast"]

    latest_inv = (
        inventory_df.sort_values("date")
        .groupby("sku_id")
        .last()
        .reset_index()
    )

    merged = avg_forecast.merge(latest_inv, on="sku_id", how="left")
    merged = merged.merge(sku_master_df, on="sku_id", how="left")
    merged = merged.fillna({
        "avg_weekly_forecast": 0.0,
        "on_hand_units": 0.0,
        "on_order_units": 0.0,
        "lead_time_days": 14.0,
        "list_price": 0.0,
        "unit_cost": 0.0,
        "reorder_point": 20.0,
        "category": "Unknown",
        "subcategory": "Unknown",
    })

    records = []
    for _, row in merged.iterrows():
        sku_id = row["sku_id"]
        fc = max(0.0, float(row.get("avg_weekly_forecast", 0.0)))
        on_hand = max(0.0, float(row.get("on_hand_units", 0.0)))
        on_order = max(0.0, float(row.get("on_order_units", 0.0)))
        lt_days = float(row.get("lead_time_days", 14.0))
        lt_weeks = lt_days / 7.0
        list_price = float(row.get("list_price", 0.0))
        unit_cost = float(row.get("unit_cost", 0.0))

        s_risk = compute_stockout_risk(fc, on_hand, on_order, lt_weeks)
        o_risk = compute_overstock_risk(on_hand, fc, FORWARD_WINDOW_WEEKS)
        quad = assign_quadrant(s_risk, o_risk)
        rupees = compute_rupee_at_stake(fc, on_hand, list_price, unit_cost, lt_weeks)

        records.append({
            "sku_id": sku_id,
            "category": row.get("category", "Unknown"),
            "subcategory": row.get("subcategory", "Unknown"),
            "avg_weekly_forecast": round(fc, 2),
            "on_hand_units": on_hand,
            "on_order_units": on_order,
            "lead_time_days": lt_days,
            "reorder_point": float(row.get("reorder_point", 20.0)),
            "stockout_risk": round(s_risk, 3),
            "overstock_risk": round(o_risk, 3),
            "quadrant": quad["quadrant"],
            "action": quad["action"],
            "urgency": quad["urgency"],
            "color": quad["color"],
            **rupees,
        })

    scored_df = pd.DataFrame(records)
    logger.info(
        "Risk evaluation complete: %d SKUs scored across quadrants: %s",
        len(scored_df),
        dict(scored_df["quadrant"].value_counts()),
    )
    return scored_df


def run_risk_scoring() -> Optional[pd.DataFrame]:
    """Execute risk scoring and serialize results for dashboard presentation."""
    lgbm_path = DATA_FORECASTS / "lgbm_results.csv"
    baseline_path = DATA_FORECASTS / "baseline_results.csv"

    if lgbm_path.exists():
        forecast_df = pd.read_csv(lgbm_path)
        source = "LightGBM model predictions"
    elif baseline_path.exists():
        forecast_df = pd.read_csv(baseline_path)
        source = "Seasonal-naive baseline"
    else:
        logger.error("Forecast inputs not found in %s", DATA_FORECASTS)
        return None

    inventory_df = pd.read_csv(DATA_CLEANED / "inventory_clean.csv")
    sku_master_df = pd.read_csv(DATA_CLEANED / "sku_master_clean.csv")

    scored = score_all_skus(forecast_df, inventory_df, sku_master_df)
    scored["forecast_source"] = source

    out_file = DATA_DASHBOARD / "risk_scored.csv"
    scored.to_csv(out_file, index=False)
    logger.info("Saved risk scores to %s (source: %s)", out_file, source)
    return scored


if __name__ == "__main__":
    run_risk_scoring()
