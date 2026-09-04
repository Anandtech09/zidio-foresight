"""
service/state.py — In-memory state and artifact cache for Project FORESIGHT API.
Loads trained LightGBM, PyTorch LSTM models, scalers, and precomputed datasets.
"""

from pathlib import Path
from typing import Any, Dict
import joblib
import pandas as pd

from src.utils import (
    DATA_CLEANED,
    DATA_DASHBOARD,
    DATA_FEATURES,
    DATA_FORECASTS,
    MODELS_LGBM_DIR,
    MODELS_LSTM_DIR,
    get_logger,
)

logger = get_logger("service_state")

# Global singleton state
state: Dict[str, Any] = {
    "risk_df": None,
    "sku_master": None,
    "analysis_df": None,
    "featured_df": None,
    "lgbm_forecasts": None,
    "baseline_forecasts": None,
    "comparison_df": None,
    "lgbm_model": None,
    "lgbm_q10": None,
    "lgbm_q90": None,
    "lgbm_feature_cols": None,
    "lstm_model": None,
    "lstm_scaler_x": None,
    "lstm_scaler_y": None,
    "lstm_feature_cols": None,
    "lstm_window_size": 12,
}


def load_all_artifacts():
    """Load precomputed datasets and model weights into memory on startup."""
    # 1. Processed Data
    try:
        p = DATA_DASHBOARD / "risk_scored.csv"
        state["risk_df"] = pd.read_csv(p) if p.exists() else pd.DataFrame()
    except Exception as e:
        logger.warning(f"Error loading risk_df: {e}")
        state["risk_df"] = pd.DataFrame()

    try:
        p = DATA_CLEANED / "sku_master_clean.csv"
        state["sku_master"] = pd.read_csv(p) if p.exists() else pd.DataFrame()
    except Exception as e:
        logger.warning(f"Error loading sku_master: {e}")
        state["sku_master"] = pd.DataFrame()

    try:
        p = DATA_FEATURES / "analysis_ready.csv"
        state["analysis_df"] = pd.read_csv(p, parse_dates=["date"]) if p.exists() else pd.DataFrame()
    except Exception as e:
        logger.warning(f"Error loading analysis_df: {e}")
        state["analysis_df"] = pd.DataFrame()

    try:
        p = DATA_FEATURES / "featured_dataset.csv"
        state["featured_df"] = pd.read_csv(p) if p.exists() else pd.DataFrame()
    except Exception as e:
        logger.warning(f"Error loading featured_df: {e}")
        state["featured_df"] = pd.DataFrame()

    try:
        p = DATA_FORECASTS / "lgbm_results.csv"
        state["lgbm_forecasts"] = pd.read_csv(p, parse_dates=["date"]) if p.exists() else pd.DataFrame()
    except Exception as e:
        logger.warning(f"Error loading lgbm_forecasts: {e}")
        state["lgbm_forecasts"] = pd.DataFrame()

    try:
        p = DATA_FORECASTS / "baseline_results.csv"
        state["baseline_forecasts"] = pd.read_csv(p, parse_dates=["date"]) if p.exists() else pd.DataFrame()
    except Exception as e:
        logger.warning(f"Error loading baseline_forecasts: {e}")
        state["baseline_forecasts"] = pd.DataFrame()

    try:
        p = DATA_FORECASTS / "model_comparison.csv"
        state["comparison_df"] = pd.read_csv(p) if p.exists() else pd.DataFrame()
    except Exception as e:
        logger.warning(f"Error loading comparison_df: {e}")
        state["comparison_df"] = pd.DataFrame()

    # 2. LightGBM Models
    try:
        import lightgbm as lgb
        lgbm_file = MODELS_LGBM_DIR / "lightgbm_model.txt"
        if lgbm_file.exists():
            state["lgbm_model"] = lgb.Booster(model_file=str(lgbm_file))
        q10_file = MODELS_LGBM_DIR / "lightgbm_q10.joblib"
        if q10_file.exists():
            state["lgbm_q10"] = joblib.load(q10_file)
        q90_file = MODELS_LGBM_DIR / "lightgbm_q90.joblib"
        if q90_file.exists():
            state["lgbm_q90"] = joblib.load(q90_file)
        cols_file = MODELS_LGBM_DIR / "feature_cols.joblib"
        if cols_file.exists():
            state["lgbm_feature_cols"] = joblib.load(cols_file)
        logger.info("Loaded LightGBM models and quantile boosters.")
    except Exception as e:
        logger.warning(f"Warning loading LightGBM models: {e}")

    # 3. PyTorch LSTM Model
    try:
        import torch
        import torch.nn as nn

        class DemandLSTM(nn.Module):
            def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.2):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size,
                    hidden_size,
                    num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                )
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out, _ = self.lstm(x)
                out = self.dropout(out[:, -1, :])
                return self.fc(out)

        lstm_file = MODELS_LSTM_DIR / "lstm_model.pt"
        if lstm_file.exists():
            checkpoint = torch.load(lstm_file, map_location=torch.device("cpu"), weights_only=False)
            lstm = DemandLSTM(
                input_size=checkpoint["input_size"],
                hidden_size=checkpoint["hidden_size"],
                num_layers=checkpoint["num_layers"],
            )
            lstm.load_state_dict(checkpoint["model_state_dict"])
            lstm.eval()
            state["lstm_model"] = lstm
            state["lstm_scaler_x"] = checkpoint["scaler_X"]
            state["lstm_scaler_y"] = checkpoint["scaler_y"]
            state["lstm_feature_cols"] = checkpoint["feature_cols"]
            state["lstm_window_size"] = checkpoint.get("window_size", 12)
            logger.info("Loaded PyTorch LSTM checkpoint, scalers, and lookback window.")
    except Exception as e:
        logger.warning(f"Warning loading PyTorch LSTM model: {e}")
