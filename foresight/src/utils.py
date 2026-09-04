"""Shared utility functions, metrics, and path configurations for demand forecasting."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Path configurations
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_CLEANED = DATA_PROCESSED / "cleaned"
DATA_FEATURES = DATA_PROCESSED / "features"
DATA_FORECASTS = DATA_PROCESSED / "forecasts"
DATA_DASHBOARD = DATA_PROCESSED / "dashboard"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_BASELINE_DIR = MODELS_DIR / "baseline"
MODELS_LGBM_DIR = MODELS_DIR / "lightgbm"
MODELS_LSTM_DIR = MODELS_DIR / "lstm"
REPORTS_DIR = PROJECT_ROOT / "reports"

for directory in [
    DATA_RAW,
    DATA_PROCESSED,
    DATA_CLEANED,
    DATA_FEATURES,
    DATA_FORECASTS,
    DATA_DASHBOARD,
    MODELS_DIR,
    MODELS_BASELINE_DIR,
    MODELS_LGBM_DIR,
    MODELS_LSTM_DIR,
    REPORTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Initialize a formatted logger with stream and file handlers."""
    logger = logging.getLogger(name)
    logger.propagate = False
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s | %(name)-16s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        log_file = PROJECT_ROOT / "pipeline.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute Weighted Absolute Percentage Error: sum(|y - y_hat|) / sum(|y|)."""
    actual_arr = np.asarray(actual, dtype=np.float64)
    pred_arr = np.asarray(predicted, dtype=np.float64)
    total_actual = np.sum(np.abs(actual_arr))
    if total_actual == 0:
        return np.nan
    return float(np.sum(np.abs(actual_arr - pred_arr)) / total_actual)


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute Mean Absolute Percentage Error on non-zero actual values."""
    actual_arr = np.asarray(actual, dtype=np.float64)
    pred_arr = np.asarray(predicted, dtype=np.float64)
    non_zero = actual_arr != 0
    if not np.any(non_zero):
        return np.nan
    return float(np.mean(np.abs((actual_arr[non_zero] - pred_arr[non_zero]) / actual_arr[non_zero])) * 100)


def bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute mean forecast error (signed bias)."""
    actual_arr = np.asarray(actual, dtype=np.float64)
    pred_arr = np.asarray(predicted, dtype=np.float64)
    return float(np.mean(pred_arr - actual_arr))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    actual_arr = np.asarray(actual, dtype=np.float64)
    pred_arr = np.asarray(predicted, dtype=np.float64)
    return float(np.sqrt(np.mean((actual_arr - pred_arr) ** 2)))


def validate_dataframe(df: pd.DataFrame, required_cols: List[str], name: str = "DataFrame") -> bool:
    """Verify presence of required columns in a DataFrame."""
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing expected columns: {missing}")
    return True


def detect_outliers_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """Identify outliers using the interquartile range rule."""
    values = series.dropna().values
    q25, q75 = np.percentile(values, [25, 75])
    iqr = q75 - q25
    lower_bound = q25 - factor * iqr
    upper_bound = q75 + factor * iqr
    return (series < lower_bound) | (series > upper_bound)


def cap_outliers(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """Windsorize series outliers to 1.5x IQR boundaries."""
    values = series.dropna().values
    q25, q75 = np.percentile(values, [25, 75])
    iqr = q75 - q25
    lower_bound = q25 - factor * iqr
    upper_bound = q75 + factor * iqr
    return series.clip(lower=lower_bound, upper=upper_bound)


def test_seasonality(series: pd.Series, period: int = 7) -> Dict[str, Any]:
    """Evaluate periodicity significance using the non-parametric Kruskal-Wallis H-test."""
    values = series.dropna().values
    n_complete = (len(values) // period) * period
    if n_complete < period * 2:
        return {"has_seasonality": False, "p_value": 1.0, "statistic": 0.0}

    matrix = values[:n_complete].reshape(-1, period)
    groups = [matrix[:, i] for i in range(period)]
    stat, p_val = stats.kruskal(*groups)
    return {
        "has_seasonality": bool(p_val < 0.05),
        "p_value": float(p_val),
        "statistic": float(stat),
    }


def compute_summary_stats(series: pd.Series) -> Dict[str, float]:
    """Calculate descriptive summary distribution statistics."""
    values = series.dropna().values.astype(np.float64)
    if len(values) == 0:
        return {}
    mean_val = float(np.mean(values))
    std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": mean_val,
        "std": std_val,
        "min": float(np.min(values)),
        "q25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "q75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
        "skew": float(stats.skew(values)),
        "kurtosis": float(stats.kurtosis(values)),
        "cv": (std_val / mean_val) if mean_val != 0 else np.nan,
    }


def create_sliding_windows(
    data: np.ndarray, window_size: int, horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sequential sliding window feature-target tensors for autoregression."""
    x_windows, y_targets = [], []
    for i in range(len(data) - window_size - horizon + 1):
        x_windows.append(data[i : i + window_size])
        y_targets.append(data[i + window_size : i + window_size + horizon])
    return np.array(x_windows), np.array(y_targets)


def resample_to_weekly(
    df: pd.DataFrame, date_col: str = "date", sku_col: str = "sku_id"
) -> pd.DataFrame:
    """Aggregate daily transactional observations into weekly frequency per SKU."""
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.set_index(date_col)

    rules = {}
    if "units_sold" in data.columns:
        rules["units_sold"] = "sum"
    if "revenue" in data.columns:
        rules["revenue"] = "sum"
    if "unit_price" in data.columns:
        rules["unit_price"] = "mean"
    if "promo_flag" in data.columns:
        rules["promo_flag"] = "max"
    if "is_holiday" in data.columns:
        rules["is_holiday"] = "max"
    if "promo_event" in data.columns:
        rules["promo_event"] = "first"
    if "season" in data.columns:
        rules["season"] = "first"
    if "category" in data.columns:
        rules["category"] = "first"
    if "subcategory" in data.columns:
        rules["subcategory"] = "first"
    if "list_price" in data.columns:
        rules["list_price"] = "first"
    if "unit_cost" in data.columns:
        rules["unit_cost"] = "first"
    if "on_hand_units" in data.columns:
        rules["on_hand_units"] = "last"
    if "on_order_units" in data.columns:
        rules["on_order_units"] = "last"
    if "lead_time_days" in data.columns:
        rules["lead_time_days"] = "first"
    if "reorder_point" in data.columns:
        rules["reorder_point"] = "first"

    if not rules:
        raise ValueError("No aggregatable quantitative columns found in DataFrame")

    weekly = data.groupby(sku_col).resample("W").agg(rules).reset_index()
    return weekly

