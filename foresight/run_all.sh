#!/usr/bin/env bash
# ===========================================================================
# Project FORESIGHT — End-to-End Pipeline Execution (Linux / Cloud / Render)
# ===========================================================================
set -e

echo "=== [1/6] Fetching Raw Data ==="
python -m src.data_fetcher

echo "=== [2/6] Cleaning and Merging Data ==="
python -m src.pipeline

echo "=== [3/6] Engineering Features ==="
python -m src.features

echo "=== [4/6] Building Seasonal-Naive Baseline Forecast ==="
python -m src.baseline

echo "=== [5/6] Training Models (LightGBM + PyTorch LSTM) ==="
python -m src.forecast

echo "=== [6/6] Scoring Stockout and Overstock Risks ==="
python -m src.risk

echo "=== Pipeline Completed Successfully! ==="
