@echo off
echo ========================================================
echo   Project FORESIGHT - Complete Pipeline
echo ========================================================
echo.

echo [1/6] Fetching Raw Data...
.\venv\Scripts\python.exe -m src.data_fetcher
if %errorlevel% neq 0 (
    echo ERROR: Data fetching failed!
    pause
    exit /b %errorlevel%
)
echo       Done.
echo.

echo [2/6] Cleaning and Merging Data (Pipeline)...
.\venv\Scripts\python.exe -m src.pipeline
if %errorlevel% neq 0 (
    echo ERROR: Pipeline failed!
    pause
    exit /b %errorlevel%
)
echo       Done.
echo.

echo [3/6] Engineering Features (Lags, Rolling Stats, Calendar)...
.\venv\Scripts\python.exe -m src.features
if %errorlevel% neq 0 (
    echo ERROR: Feature engineering failed!
    pause
    exit /b %errorlevel%
)
echo       Done.
echo.

echo [4/6] Building Seasonal-Naive Baseline Forecast...
.\venv\Scripts\python.exe -m src.baseline
if %errorlevel% neq 0 (
    echo ERROR: Baseline forecast failed!
    pause
    exit /b %errorlevel%
)
echo       Done.
echo.

echo [5/6] Training ML Models (LightGBM + PyTorch LSTM)...
echo       This step may take a few minutes...
.\venv\Scripts\python.exe -m src.forecast
if %errorlevel% neq 0 (
    echo ERROR: Model training failed!
    pause
    exit /b %errorlevel%
)
echo       Done. Model artifacts saved to models/ folder.
echo.

echo [6/6] Scoring Stockout and Overstock Risks...
.\venv\Scripts\python.exe -m src.risk
if %errorlevel% neq 0 (
    echo ERROR: Risk scoring failed!
    pause
    exit /b %errorlevel%
)
echo       Done.
echo.

echo ========================================================
echo   SUCCESS! Full pipeline complete.
echo.
echo   Generated data:    data/processed/ (cleaned/, features/, forecasts/, dashboard/)
echo   Trained models:    models/ (baseline/, lightgbm/, lstm/)
echo   Cleaning log:      data/processed/cleaned/cleaning_log.csv
echo.
echo   Next steps:
echo     Dashboard:  .\venv\Scripts\streamlit.exe run app/streamlit_app.py
echo     API:        .\venv\Scripts\uvicorn.exe service.main:app --reload --port 8000
echo ========================================================
pause
