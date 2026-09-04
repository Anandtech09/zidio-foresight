"""
service/main.py — Main FastAPI Microservice Entry Point for Project FORESIGHT.

Assembles modular routers:
- System & API Catalog (/health, /api/catalog)
- Portfolio Analytics & Benchmarks (/portfolio/overview, /models/comparison)
- SKU Demand Forecasting & History (/skus, /sku/{id}/forecast, /sku/{id}/history, /batch/forecast)
- Inventory Risk Grid (/risk/catalog, /risk/summary)
- Dual-Model Real-Time Inference (/predict/manual)
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from service.config import API_HOST, API_PORT
from service.state import load_all_artifacts
from service.routers import forecast, inference, portfolio, risk, system


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model weights and precomputed artifacts on startup."""
    load_all_artifacts()
    yield


app = FastAPI(
    title="Project FORESIGHT — Demand & Risk Intelligence API",
    description=(
        "Enterprise REST Microservice for NorthBay Living.\n\n"
        "Provides multi-horizon demand forecasting, real-time dual-model scenario simulation "
        "(PyTorch LSTM & LightGBM), and capital-at-risk inventory prioritization."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Cross-Origin Resource Sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all modular routers
app.include_router(system.router)
app.include_router(portfolio.router)
app.include_router(forecast.router)
app.include_router(risk.router)
app.include_router(inference.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service.main:app", host=API_HOST, port=API_PORT, reload=True)
