"""Seasonal-naive benchmark forecasting module.

Generates seasonal-lagged demand projections and evaluates baseline performance metrics.
"""

from datetime import datetime
import json
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils import (
    DATA_FEATURES,
    DATA_FORECASTS,
    MODELS_BASELINE_DIR,
    bias,
    get_logger,
    mape,
    rmse,
    wape,
)

logger = get_logger("baseline")


def seasonal_naive_forecast(
    df: pd.DataFrame,
    target_col: str = "units_sold",
    season_period: int = 52,
    horizon: int = 6,
) -> pd.DataFrame:
    """Generate seasonal-naive forecasts using historical demand from the prior seasonal cycle."""
    data = df.sort_values(["sku_id", "date"]).copy()
    records = []

    for sku_id, sku_df in data.groupby("sku_id"):
        values = sku_df[target_col].values.astype(np.float64)
        dates = sku_df["date"].values
        total_obs = len(values)

        if total_obs < horizon + 1:
            continue

        train_vals = values[:-horizon]
        test_vals = values[-horizon:]
        test_dates = dates[-horizon:]

        preds = np.zeros(horizon)
        for i in range(horizon):
            lag_idx = len(train_vals) - season_period + i
            if 0 <= lag_idx < len(train_vals):
                preds[i] = train_vals[lag_idx]
            else:
                preds[i] = train_vals[-1] if len(train_vals) > 0 else 0.0

        for i in range(horizon):
            records.append({
                "sku_id": sku_id,
                "date": test_dates[i],
                "actual": test_vals[i],
                "baseline_forecast": max(0.0, float(preds[i])),
            })

    results = pd.DataFrame(records)
    logger.info(
        "Generated %d baseline forecast rows across %d SKUs (season_period=%dw, horizon=%dw)",
        len(results),
        results["sku_id"].nunique(),
        season_period,
        horizon,
    )
    return results


def evaluate_baseline(results_df: pd.DataFrame) -> Dict[str, object]:
    """Compute WAPE, MAPE, Bias, and RMSE for the baseline predictions."""
    actual = results_df["actual"].values
    pred = results_df["baseline_forecast"].values

    sku_wapes = [
        {"sku_id": sku_id, "wape": wape(grp["actual"].values, grp["baseline_forecast"].values)}
        for sku_id, grp in results_df.groupby("sku_id")
    ]

    metrics: Dict[str, object] = {
        "wape": wape(actual, pred),
        "mape": mape(actual, pred),
        "bias": bias(actual, pred),
        "rmse": rmse(actual, pred),
        "n_skus": results_df["sku_id"].nunique(),
        "n_forecasts": len(results_df),
        "sku_wapes": pd.DataFrame(sku_wapes),
        "median_sku_wape": float(np.nanmedian([w["wape"] for w in sku_wapes])),
    }

    logger.info(
        "Baseline evaluation: WAPE=%.4f | MAPE=%.2f%% | Bias=%.2f | RMSE=%.2f",
        metrics["wape"],
        metrics["mape"],
        metrics["bias"],
        metrics["rmse"],
    )
    return metrics


def run_baseline(df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Execute the baseline forecasting workflow and serialize evaluation artifacts."""
    if df is None:
        df = pd.read_csv(DATA_FEATURES / "featured_dataset.csv")
        df["date"] = pd.to_datetime(df["date"])

    results = seasonal_naive_forecast(df)
    metrics = evaluate_baseline(results)

    out_csv = DATA_FORECASTS / "baseline_results.csv"
    results.to_csv(out_csv, index=False)

    metadata = {
        "model_name": "Seasonal Naive Baseline",
        "season_period_weeks": 52,
        "horizon_weeks": 6,
        "created_at": datetime.now().isoformat(),
        "metrics": {
            "wape": float(metrics["wape"]),
            "mape": float(metrics["mape"]),
            "bias": float(metrics["bias"]),
            "rmse": float(metrics["rmse"]),
            "n_skus": int(metrics["n_skus"]),
            "n_forecasts": int(metrics["n_forecasts"]),
            "median_sku_wape": float(metrics["median_sku_wape"]),
        },
    }
    meta_path = MODELS_BASELINE_DIR / "baseline_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved baseline outputs to %s and %s", out_csv, meta_path)
    return results, metrics


if __name__ == "__main__":
    run_baseline()
