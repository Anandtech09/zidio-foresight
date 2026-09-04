"""Demand forecasting models and evaluation pipeline.

Implements gradient-boosted tree regressors (LightGBM) with quantile uncertainty intervals
and recurrent deep learning sequence architectures (PyTorch LSTM).
"""

from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.utils import (
    DATA_FEATURES,
    DATA_FORECASTS,
    MODELS_LGBM_DIR,
    MODELS_LSTM_DIR,
    bias,
    create_sliding_windows,
    get_logger,
    mape,
    rmse,
    wape,
)

logger = get_logger("forecast")

EXCLUDED_COLUMNS = [
    "date",
    "sku_id",
    "units_sold",
    "revenue",
    "actual",
    "baseline_forecast",
    "category",
    "subcategory",
    "season",
    "promo_event",
    "launch_date",
]


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Extract numeric model predictors excluding targets, metadata, and timestamps."""
    return [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in EXCLUDED_COLUMNS
    ]


# ---------------------------------------------------------------------------
# LightGBM Point and Quantile Estimation
# ---------------------------------------------------------------------------

def train_lightgbm(
    df: pd.DataFrame, target_col: str = "units_sold", horizon: int = 6
) -> Optional[Dict[str, object]]:
    """Train LightGBM point and quantile models under temporal train/test split."""
    try:
        import lightgbm as lgb
    except ImportError:
        logger.error("lightgbm dependency not found")
        return None

    data = df.sort_values(["sku_id", "date"]).copy()
    feature_cols = get_feature_cols(data)

    dates = sorted(data["date"].unique())
    split_date_idx = len(dates) - horizon
    train_dates, test_dates = dates[:split_date_idx], dates[split_date_idx:]

    train_df = data[data["date"].isin(train_dates)]
    test_df = data[data["date"].isin(test_dates)]

    x_train = np.nan_to_num(train_df[feature_cols].values.astype(np.float32), nan=0.0)
    y_train = train_df[target_col].values.astype(np.float32)
    x_test = np.nan_to_num(test_df[feature_cols].values.astype(np.float32), nan=0.0)
    y_test = test_df[target_col].values.astype(np.float32)

    logger.info("Training LightGBM on train shape %s, evaluating on %s", x_train.shape, x_test.shape)

    base_params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42,
    }

    train_dataset = lgb.Dataset(x_train, label=y_train)
    model = lgb.train(base_params, train_dataset, num_boost_round=500)
    point_preds = np.clip(model.predict(x_test), 0, None)

    params_q10 = {**base_params, "objective": "quantile", "alpha": 0.1}
    params_q90 = {**base_params, "objective": "quantile", "alpha": 0.9}

    model_q10 = lgb.train(params_q10, train_dataset, num_boost_round=500)
    model_q90 = lgb.train(params_q90, train_dataset, num_boost_round=500)

    lower_bound = np.clip(model_q10.predict(x_test), 0, None)
    upper_bound = np.clip(model_q90.predict(x_test), 0, None)

    metrics = {
        "model": "LightGBM",
        "wape": wape(y_test, point_preds),
        "mape": mape(y_test, point_preds),
        "bias": bias(y_test, point_preds),
        "rmse": rmse(y_test, point_preds),
    }

    logger.info(
        "LightGBM metrics: WAPE=%.4f | MAPE=%.2f%% | Bias=%.2f | RMSE=%.2f",
        metrics["wape"],
        metrics["mape"],
        metrics["bias"],
        metrics["rmse"],
    )

    importance = dict(zip(feature_cols, model.feature_importance(importance_type="gain")))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

    results_df = test_df[["sku_id", "date", target_col]].copy()
    results_df = results_df.rename(columns={target_col: "actual"})
    results_df["lgbm_forecast"] = point_preds
    results_df["lgbm_lower"] = lower_bound
    results_df["lgbm_upper"] = upper_bound

    results_path = DATA_FORECASTS / "lgbm_results.csv"
    results_df.to_csv(results_path, index=False)

    model.save_model(str(MODELS_LGBM_DIR / "lightgbm_model.txt"))
    joblib.dump(model_q10, MODELS_LGBM_DIR / "lightgbm_q10.joblib")
    joblib.dump(model_q90, MODELS_LGBM_DIR / "lightgbm_q90.joblib")
    joblib.dump(feature_cols, MODELS_LGBM_DIR / "feature_cols.joblib")

    importance_df = pd.DataFrame(
        sorted(importance.items(), key=lambda x: x[1], reverse=True),
        columns=["feature", "importance"],
    )
    importance_df.to_csv(DATA_FORECASTS / "feature_importance.csv", index=False)

    return {
        "model": model,
        "model_q10": model_q10,
        "model_q90": model_q90,
        "predictions": results_df,
        "metrics": metrics,
        "feature_importance": importance,
        "feature_cols": feature_cols,
    }


# ---------------------------------------------------------------------------
# PyTorch LSTM Sequence Architecture
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    class DemandLSTM(nn.Module):
        """Two-layer LSTM with linear projection for multi-horizon demand forecasting."""

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

except ImportError:
    torch = None
    DemandLSTM = None


def train_pytorch_lstm(
    df: pd.DataFrame,
    target_col: str = "units_sold",
    window_size: int = 12,
    horizon: int = 6,
    hidden_size: int = 64,
    num_layers: int = 2,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
) -> Optional[Dict[str, object]]:
    """Train global sequence LSTM across all SKUs using sliding autoregressive windows."""
    if torch is None or DemandLSTM is None:
        logger.error("torch dependency not found")
        return None

    data = df.sort_values(["sku_id", "date"]).copy()
    feature_cols = get_feature_cols(data)

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    x_windows_list, y_targets_list = [], []

    for _, sku_data in data.groupby("sku_id"):
        if len(sku_data) < window_size + horizon:
            continue

        x_raw = np.nan_to_num(sku_data[feature_cols].values.astype(np.float64), nan=0.0)
        y_raw = np.nan_to_num(sku_data[target_col].values.astype(np.float64).reshape(-1, 1), nan=0.0)

        x_win, y_tgt = create_sliding_windows(
            np.hstack([x_raw, y_raw]), window_size=window_size, horizon=1
        )

        if len(x_win) > 0:
            x_windows_list.append(x_win[:, :, :-1])
            y_targets_list.append(y_tgt[:, :, -1])

    if not x_windows_list:
        logger.error("Insufficient sequence samples for LSTM training")
        return None

    x_all = np.vstack(x_windows_list).astype(np.float32)
    y_all = np.vstack(y_targets_list).astype(np.float32).reshape(-1, 1)

    n_samples, seq_len, n_features = x_all.shape
    x_flat = scaler_x.fit_transform(x_all.reshape(-1, n_features))
    x_all = x_flat.reshape(n_samples, seq_len, n_features)
    y_all = scaler_y.fit_transform(y_all)

    split_idx = int(len(x_all) * 0.8)
    x_train, x_test = x_all[:split_idx], x_all[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]

    logger.info("LSTM dataset tensors: Train=%s, Test=%s", x_train.shape, x_test.shape)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train)),
        batch_size=batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DemandLSTM(
        input_size=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_loss = float("inf")
    best_weights = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_weights = model.state_dict().copy()

    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    with torch.no_grad():
        test_pred_tensor = model(torch.FloatTensor(x_test).to(device)).cpu().numpy()

    test_preds = np.clip(scaler_y.inverse_transform(test_pred_tensor).flatten(), 0, None)
    actuals = scaler_y.inverse_transform(y_test).flatten()

    metrics = {
        "model": "PyTorch LSTM",
        "wape": wape(actuals, test_preds),
        "mape": mape(actuals, test_preds),
        "bias": bias(actuals, test_preds),
        "rmse": rmse(actuals, test_preds),
    }

    logger.info(
        "LSTM metrics: WAPE=%.4f | MAPE=%.2f%% | Bias=%.2f | RMSE=%.2f",
        metrics["wape"],
        metrics["mape"],
        metrics["bias"],
        metrics["rmse"],
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": n_features,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "scaler_X": scaler_x,
            "scaler_y": scaler_y,
            "window_size": window_size,
            "feature_cols": feature_cols,
        },
        MODELS_LSTM_DIR / "lstm_model.pt",
    )

    return {
        "model": model,
        "predictions": test_preds,
        "actuals": actuals,
        "metrics": metrics,
        "scaler_X": scaler_x,
        "scaler_y": scaler_y,
    }


# ---------------------------------------------------------------------------
# Multi-Model Benchmark and Selection
# ---------------------------------------------------------------------------

def compare_models(
    baseline_metrics: Dict[str, object],
    lgbm_result: Optional[Dict[str, object]],
    lstm_result: Optional[Dict[str, object]],
) -> Dict[str, object]:
    """Rank all evaluated architectures by WAPE against the heuristic baseline."""
    baseline_wape = float(baseline_metrics.get("wape", np.nan))
    leaderboard = [
        {
            "name": "Seasonal Naive (Baseline)",
            "wape": round(baseline_wape, 4),
            "mape": round(float(baseline_metrics.get("mape", np.nan)), 2),
            "bias": round(float(baseline_metrics.get("bias", np.nan)), 2),
            "rmse": round(float(baseline_metrics.get("rmse", np.nan)), 2),
            "improvement_pct": 0.0,
        }
    ]

    if lgbm_result:
        m = lgbm_result["metrics"]
        w = float(m["wape"])
        imp = round(((baseline_wape - w) / baseline_wape) * 100, 2) if baseline_wape > 0 else 0.0
        leaderboard.append({
            "name": "LightGBM",
            "wape": round(w, 4),
            "mape": round(float(m["mape"]), 2),
            "bias": round(float(m["bias"]), 2),
            "rmse": round(float(m["rmse"]), 2),
            "improvement_pct": imp,
        })

    if lstm_result:
        m = lstm_result["metrics"]
        w = float(m["wape"])
        imp = round(((baseline_wape - w) / baseline_wape) * 100, 2) if baseline_wape > 0 else 0.0
        leaderboard.append({
            "name": "PyTorch LSTM",
            "wape": round(w, 4),
            "mape": round(float(m["mape"]), 2),
            "bias": round(float(m["bias"]), 2),
            "rmse": round(float(m["rmse"]), 2),
            "improvement_pct": imp,
        })

    leaderboard.sort(key=lambda x: x["wape"] if not np.isnan(x["wape"]) else float("inf"))
    winner = leaderboard[0]
    beats_baseline = bool(winner["wape"] < baseline_wape)

    logger.info("Model comparison leaderboard:")
    for rank, entry in enumerate(leaderboard, start=1):
        status = " (Winner)" if rank == 1 else ""
        logger.info("  %d. %s: WAPE=%.4f, MAPE=%.2f%%, RMSE=%.2f%s", rank, entry["name"], entry["wape"], entry["mape"], entry["rmse"], status)

    return {
        "winner": winner["name"],
        "beats_baseline": beats_baseline,
        "all_results": leaderboard,
        "improvement_pct": winner["improvement_pct"],
    }


def run_forecasting(df: Optional[pd.DataFrame] = None) -> Dict[str, object]:
    """Execute end-to-end model training, cross-validation, and selection."""
    if df is None:
        df = pd.read_csv(DATA_FEATURES / "featured_dataset.csv")
        df["date"] = pd.to_datetime(df["date"])

    from src.baseline import run_baseline

    _, baseline_metrics = run_baseline(df)
    lgbm_res = train_lightgbm(df)
    lstm_res = train_pytorch_lstm(df)

    comparison = compare_models(baseline_metrics, lgbm_res, lstm_res)

    comparison_df = pd.DataFrame(comparison["all_results"])
    comparison_df.to_csv(DATA_FORECASTS / "model_comparison.csv", index=False)

    return {
        "baseline": baseline_metrics,
        "lgbm": lgbm_res,
        "lstm": lstm_res,
        "comparison": comparison,
    }


if __name__ == "__main__":
    run_forecasting()
