# Project FORESIGHT — Demand Forecasting & Inventory Intelligence

> **Client:** NorthBay Living (D2C Home & Lifestyle Brand)
---

## 📌 Executive Summary & Problem Statement

NorthBay Living is a direct-to-consumer home & lifestyle brand selling ~200 active SKUs across categories like Furnishings, Décor, Kitchen, and Storage from a single central warehouse. Historically, inventory was managed through spreadsheets and intuition, resulting in twin financial leaks:
1. **Stockouts on High-Demand SKUs:** Lost revenue that can never be recovered, customer churn, and missed organic market momentum.
2. **Overstock on Slow-Moving SKUs:** Capital trapped in unsold goods, escalating holding costs, and margin-destroying emergency markdowns.

**Project FORESIGHT** provides an automated, leakage-free analytics engine and web interface that transforms daily sales and inventory logs into:
- Highly accurate weekly demand forecasts 6 weeks into the future.
- Objective stockout and overstock risk scoring for all 192 SKUs.
- Concrete operational recommendations (Reorder, Markdown, Watch, Healthy) with exact Rupee (₹) amounts at risk.

---

## 🏆 Model Benchmark & Backtest Results (All 3 Models)

In accordance with the non-negotiable rule of the engagement (*"Beat the baseline, honestly — report WAPE against the baseline"*), we evaluated three modeling paradigms using strict temporal rolling-origin cross-validation with zero data leakage:

| Model | Model Type | WAPE (Primary) | MAPE (%) | RMSE | Bias (Units) | Improvement vs Baseline | Status |
|---|---|---|---|---|---|---|---|
| **PyTorch LSTM** | Deep Learning (Recurrent Sequence) | **0.0738** | **8.30%** | **14.04** | -0.30 | **+63.0%** | 🥇 **WINNER** |
| **LightGBM** | Gradient Boosted Trees (Point + Quantile 10/90) | **0.0854** | **25.59%** | **30.98** | +10.92 | **+57.3%** | 🥈 Runner-Up |
| **Seasonal Naive** | Heuristic Baseline (Same week last year) | **0.1998** | **45.48%** | **48.71** | +17.11 | 0.0% (Ref) | 🥉 Baseline Bar |

### Key Modeling Highlights
- **Both ML models beat the baseline by >55%**.
- **Quantile Prediction Intervals:** LightGBM trains dual quantile regressors (`q=0.10` and `q=0.90`) to provide an 80% confidence bound for every prediction.
- **Top Demand Drivers:** Rolling 4-week maximum sales, rolling 4-week mean, week of year, and price elasticity/promotional flags.

---

## 🗂️ Data Architecture & Storage Structure

All raw extracts and processed artifacts are organized in a clean, categorized directory structure:

```
foresight/
├── data/
│   ├── raw/                              # Ingested client extracts (4 tables)
│   │   ├── sales_daily.csv               # Daily sales volume and selling prices
│   │   ├── sku_master.csv                # Product catalog (category, cost, list price)
│   │   ├── calendar.csv                  # Dates, seasons, holidays, promotional events
│   │   └── inventory_snapshots.csv       # Stock on hand, on order, lead times, reorder points
│   │
│   └── processed/                        # Categorized pipeline outputs
│       ├── DATA_MANIFEST.md              # Detailed inventory and schema guide
│       ├── cleaned/                      # Step 2: Cleaned individual tables + audit log
│       │   ├── sales_daily_clean.csv
│       │   ├── sku_master_clean.csv
│       │   ├── calendar_clean.csv
│       │   ├── inventory_clean.csv
│       │   └── cleaning_log.csv          # Audit trail of every cleaning decision
│       ├── features/                     # Step 3: Analysis-ready & feature-engineered data
│       │   ├── analysis_ready.csv        # Master unified daily dataset (20 columns)
│       │   └── featured_dataset.csv      # Weekly aggregated + 38 predictive features
│       ├── forecasts/                    # Step 4-5: Model predictions & comparison
│       │   ├── baseline_results.csv      # Seasonal-naive forecast vs actuals
│       │   ├── lgbm_results.csv          # LightGBM predictions + 80% interval bounds
│       │   ├── model_comparison.csv      # WAPE, MAPE, RMSE comparison across models
│       │   └── feature_importance.csv    # Feature gain rankings
│       └── dashboard/                    # Step 6: Risk scores consumed by UI & API
│           └── risk_scored.csv           # Quadrant, action recommendation, ₹ at stake
│
├── models/                               # Saved model artifacts (3 subdirectories)
│   ├── baseline/                         # Baseline model parameters & metrics metadata JSON
│   ├── lightgbm/                         # Trained booster, quantile joblibs, feature list
│   └── lstm/                             # PyTorch state dict, scalers, sequence hyperparameters
```

---

## 🎯 The 4-Quadrant Risk Decisioning Grid

The risk engine maps forecast demand against lead-time inventory to assign every SKU to one of four actionable quadrants:

| Quadrant | Stockout Risk | Overstock Risk | Action Recommendation | Color Code |
|---|---|---|---|---|
| 🔴 **Reorder Now** | HIGH (≥50%) | LOW (<50%) | Raise replenishment purchase order immediately | Red (`#E74C3C`) |
| 🟡 **Markdown / Clear** | LOW (<50%) | HIGH (≥50%) | Initiate promotional discount to unlock dead capital | Yellow (`#F39C12`) |
| 🟠 **Watch / Volatile** | HIGH (≥50%) | HIGH (≥50%) | Investigate erratic demand fluctuations manually | Orange (`#E67E22`) |
| 🟢 **Healthy** | LOW (<50%) | LOW (<50%) | Stock levels aligned with demand; maintain steady state | Green (`#27AE60`) |

---

## ⚡ How to Run the Project (Windows & Cross-Platform)

### 🟢 Single-Click Execution (Windows)
Double-click **`run_all.bat`** (or run `.\run_all.bat` in PowerShell/CMD).  
It executes all 6 pipeline stages end-to-end:
1. `src.data_fetcher` (raw dataset download/generation)
2. `src.pipeline` (cleaning, outlier capping, joining tables)
3. `src.features` (weekly resampling, lag creation, cyclic encoding)
4. `src.baseline` (seasonal-naive baseline + metrics metadata)
5. `src.forecast` (training LightGBM + PyTorch LSTM, model backtesting)
6. `src.risk` (stockout/overstock risk scoring and Rupee quantification)

### 🖥️ Launching the Web Dashboard
```powershell
.\venv\Scripts\streamlit.exe run app/streamlit_app.py
```
*Access in browser at: `http://localhost:8501`*

### 🔌 Launching the REST Scoring API
```powershell
.\venv\Scripts\uvicorn.exe service.main:app --reload --port 8000
```
*Interactive Swagger Documentation available at: `http://localhost:8000/docs`*

---

## 🖥️ Dashboard Features

The Streamlit dashboard features a unified, responsive dark theme designed specifically for operations and merchandise planners:
- **Executive Summary (`app/streamlit_app.py`):** High-level KPI cards (Total SKUs, SKUs at Risk, Total ₹ at Stake, Best WAPE, Total Revenue), dynamic model comparison card, decisioning pie chart, and quick-action summaries.
- **1. Overview (`pages/1_Overview.py`):** Multi-year revenue trends, category revenue breakdown, demand distributions, and promotional uplift analysis.
- **2. Forecast (`pages/2_Forecast.py`):** Multi-model visualizer comparing Actuals vs Seasonal-Naive Baseline vs LightGBM with 80% confidence interval band, plus SKU-level WAPE/MAPE metrics and full comparison table.
- **3. Risk (`pages/3_Risk.py`):** Interactive decisioning scatter grid plotting Stockout Risk vs Overstock Risk with bubble size proportional to Rupee value at stake.
- **4. Reorder (`pages/4_Reorder.py`):** Prioritized actionable reorder and markdown lists with one-click CSV export for procurement teams.
- **5. EDA & Data Quality (`pages/5_EDA.py`):** Data cleaning decision log audit trail, Pareto analysis (80/20 revenue concentration), monthly seasonality, and day-of-week demand patterns.

---

## 📋 Key Assumptions & Governance (Section 16 of Brief)

1. **Lead Time & Reorder Consistency:** Inventory snapshot lead times and reorder thresholds are accurate representation of warehouse replenishment dynamics.
2. **Weekly Seasonality:** Sales history contains sufficient depth to capture recurring weekly and seasonal fluctuations across core catalog items.
3. **Leakage Guard:** Engineered features rely strictly on retrospective data available prior to the forecasted horizon date.
4. **Explainability First:** All risk determinations are grounded in transparent physical inventory formulas rather than uninterpretable black-box rules.
