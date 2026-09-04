"""Feature engineering pipeline for demand forecasting.

Constructs autoregressive lag features, multi-window rolling statistics,
cyclical calendar encodings, price elasticities, and demand velocities.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from src.utils import DATA_FEATURES, get_logger, resample_to_weekly

logger = get_logger("features")


def add_lag_features(
    df: pd.DataFrame, target_col: str = "units_sold", lags: Optional[List[int]] = None
) -> pd.DataFrame:
    """Generate autoregressive lag series partitioned by SKU."""
    if lags is None:
        lags = [1, 2, 4, 6, 8]

    data = df.sort_values(["sku_id", "date"]).copy()
    grouped = data.groupby("sku_id")[target_col]

    for lag in lags:
        data[f"lag_{lag}w"] = grouped.shift(lag)

    return data


def add_rolling_features(
    df: pd.DataFrame, target_col: str = "units_sold", windows: Optional[List[int]] = None
) -> pd.DataFrame:
    """Compute moving window statistics (mean, std, median, min, max) partitioned by SKU."""
    if windows is None:
        windows = [4, 8, 12]

    data = df.sort_values(["sku_id", "date"]).copy()
    grouped = data.groupby("sku_id")[target_col]

    for w in windows:
        roll = grouped.transform(lambda s: s.rolling(window=w, min_periods=1).mean())
        data[f"rolling_mean_{w}w"] = roll
        data[f"rolling_std_{w}w"] = grouped.transform(lambda s: s.rolling(window=w, min_periods=1).std())
        data[f"rolling_median_{w}w"] = grouped.transform(lambda s: s.rolling(window=w, min_periods=1).median())
        data[f"rolling_min_{w}w"] = grouped.transform(lambda s: s.rolling(window=w, min_periods=1).min())
        data[f"rolling_max_{w}w"] = grouped.transform(lambda s: s.rolling(window=w, min_periods=1).max())

    return data


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive temporal attributes and cyclical trigonometric representations."""
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])

    data["day_of_week"] = data["date"].dt.dayofweek
    data["week_of_year"] = data["date"].dt.isocalendar().week.astype(int)
    data["month_num"] = data["date"].dt.month
    data["quarter"] = data["date"].dt.quarter
    data["year"] = data["date"].dt.year
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)

    data["month_sin"] = np.sin(2 * np.pi * data["month_num"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month_num"] / 12)
    data["week_sin"] = np.sin(2 * np.pi * data["week_of_year"] / 52)
    data["week_cos"] = np.cos(2 * np.pi * data["week_of_year"] / 52)
    data["dow_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
    data["dow_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)

    if "season" in data.columns:
        dummies = pd.get_dummies(data["season"], prefix="season", dtype=int)
        data = pd.concat([data, dummies], axis=1)

    if "is_holiday" in data.columns:
        data["is_holiday"] = data["is_holiday"].fillna(0).astype(int)
    else:
        data["is_holiday"] = 0

    if "promo_flag" in data.columns:
        data["promo_flag"] = data["promo_flag"].fillna(0).astype(int)
    else:
        data["promo_flag"] = 0

    if "promo_event" in data.columns:
        data["has_promo_event"] = (data["promo_event"].fillna("").astype(str).str.len() > 0).astype(int)
    else:
        data["has_promo_event"] = 0

    return data


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute price ratio, discount rate, unit margin, and price change velocity."""
    data = df.copy()

    if "unit_price" in data.columns and "list_price" in data.columns:
        data["price_ratio"] = np.where(data["list_price"] > 0, data["unit_price"] / data["list_price"], 1.0)
        data["discount_pct"] = np.clip(1.0 - data["price_ratio"], 0.0, 1.0) * 100

    if "unit_price" in data.columns and "unit_cost" in data.columns:
        data["margin"] = data["unit_price"] - data["unit_cost"]
        data["margin_pct"] = np.where(data["unit_price"] > 0, (data["margin"] / data["unit_price"]) * 100, 0.0)

    if "unit_price" in data.columns:
        data["price_change"] = data.groupby("sku_id")["unit_price"].pct_change().fillna(0.0)

    return data


def add_category_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute macro category-level demand indicators."""
    data = df.copy()
    if "category" in data.columns and "units_sold" in data.columns:
        data["category_avg_demand"] = data.groupby(["category", "date"])["units_sold"].transform("mean")

    if "subcategory" in data.columns and "units_sold" in data.columns:
        data["subcategory_avg_demand"] = data.groupby(["subcategory", "date"])["units_sold"].transform("mean")

    return data


def add_demand_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trajectory indicators including growth rate, acceleration, and cumulative demand."""
    data = df.sort_values(["sku_id", "date"]).copy()

    if "units_sold" in data.columns:
        grouped = data.groupby("sku_id")["units_sold"]
        data["demand_growth"] = grouped.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        data["demand_acceleration"] = data.groupby("sku_id")["demand_growth"].diff().fillna(0.0)
        data["cumulative_demand"] = data.groupby(["sku_id", "year"])["units_sold"].cumsum()

    return data


def engineer_features(df: pd.DataFrame, is_weekly: bool = True) -> pd.DataFrame:
    """Execute complete feature engineering transformations."""
    data = df.copy()
    if is_weekly:
        data = resample_to_weekly(data)

    data = add_calendar_features(data)
    data = add_lag_features(data)
    data = add_rolling_features(data)
    data = add_price_features(data)
    data = add_category_features(data)
    data = add_demand_velocity(data)

    # Trim warmup rows lacking full 8-week history
    min_warmup = 8
    data = data[data.groupby("sku_id").cumcount() >= min_warmup].reset_index(drop=True)

    logger.info("Engineered %d features across %d weekly rows", len(data.columns), len(data))
    return data


if __name__ == "__main__":
    analysis_df = pd.read_csv(DATA_FEATURES / "analysis_ready.csv")
    featured_df = engineer_features(analysis_df)
    output_path = DATA_FEATURES / "featured_dataset.csv"
    featured_df.to_csv(output_path, index=False)
    logger.info("Saved feature dataset to %s", output_path)
