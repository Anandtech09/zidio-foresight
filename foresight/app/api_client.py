"""
app/api_client.py — REST Client connecting Streamlit to the FastAPI Microservice.

Loads configuration from .env and communicates directly with the modular
FastAPI microservice endpoints.
"""

import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import pandas as pd
import requests

# Load environment variables from .env
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
except ImportError:
    pass

from src.utils import DATA_CLEANED, DATA_DASHBOARD, DATA_FEATURES, DATA_FORECASTS

API_URL = os.getenv("FORESIGHT_API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT_SECONDS = 4.0


class FORESIGHTApiClient:
    """Client for Project FORESIGHT REST API."""

    def __init__(self, base_url: str = API_URL):
        self.base_url = base_url.rstrip("/")

    def is_online(self) -> bool:
        """Check if FastAPI microservice is live and responsive."""
        try:
            r = requests.get(f"{self.base_url}/health", timeout=1.5)
            return r.status_code == 200
        except Exception:
            return False

    def get_health(self) -> Dict[str, Any]:
        """Fetch system liveness and model cache status."""
        try:
            r = requests.get(f"{self.base_url}/health", timeout=TIMEOUT_SECONDS)
            if r.status_code == 200:
                data = r.json()
                data["is_live"] = True
                return data
        except Exception:
            pass
        return {
            "status": "offline",
            "version": "2.0.0",
            "is_live": False,
            "models_loaded": {},
            "total_skus": 0,
            "architecture": "Offline Storage Fallback (FastAPI not running)",
        }

    def get_overview(self) -> Dict[str, Any]:
        """Fetch portfolio KPI aggregates from FastAPI /portfolio/overview."""
        try:
            r = requests.get(f"{self.base_url}/portfolio/overview", timeout=TIMEOUT_SECONDS)
            if r.status_code == 200:
                data = r.json()
                data["source"] = "FastAPI Backend"
                return data
        except Exception:
            pass

        # Offline Disk Fallback if backend is stopped
        try:
            risk = pd.read_csv(DATA_DASHBOARD / "risk_scored.csv")
            analysis = pd.read_csv(DATA_FEATURES / "analysis_ready.csv")
            comp = pd.read_csv(DATA_FORECASTS / "model_comparison.csv")
            at_risk = len(risk[risk["quadrant"] != "Healthy"])
            return {
                "source": "Local Disk Cache (Backend Offline)",
                "total_skus": len(risk),
                "skus_at_risk": at_risk,
                "risk_percentage": round((at_risk / len(risk)) * 100, 1),
                "total_rupee_at_stake": float(risk["total_rupee_at_stake"].sum()),
                "total_revenue": float(analysis["revenue"].sum()) if "revenue" in analysis.columns else 0.0,
                "best_model_name": comp.loc[comp["wape"].idxmin()]["name"] if len(comp) > 0 else "PyTorch LSTM",
                "best_model_wape": float(comp["wape"].min()) if len(comp) > 0 else 0.0767,
            }
        except Exception:
            return {}

    def get_model_comparison(self) -> Dict[str, Any]:
        """Fetch model leaderboard comparison metrics from /models/comparison."""
        try:
            r = requests.get(f"{self.base_url}/models/comparison", timeout=TIMEOUT_SECONDS)
            if r.status_code == 200:
                data = r.json()
                data["source"] = "FastAPI Backend"
                return data
        except Exception:
            pass

        # Offline Disk Fallback
        try:
            comp = pd.read_csv(DATA_FORECASTS / "model_comparison.csv")
            items = []
            min_wape = comp["wape"].min()
            for _, row in comp.sort_values("wape").iterrows():
                items.append({
                    "name": str(row["name"]),
                    "wape": float(row["wape"]),
                    "mape": float(row["mape"]),
                    "bias": float(row["bias"]),
                    "rmse": float(row["rmse"]),
                    "is_winner": row["wape"] == min_wape,
                })
            return {
                "source": "Local Disk Cache (Backend Offline)",
                "leaderboard": items,
                "winner": str(comp.loc[comp["wape"].idxmin()]["name"]),
            }
        except Exception:
            return {"leaderboard": [], "winner": "N/A", "source": "None"}

    def get_skus(self, category: Optional[str] = None) -> Dict[str, Any]:
        """Fetch list of SKU IDs and catalog categories from /skus."""
        params = {"category": category} if category and category != "All" else {}
        try:
            r = requests.get(f"{self.base_url}/skus", params=params, timeout=TIMEOUT_SECONDS)
            if r.status_code == 200:
                data = r.json()
                data["source"] = "FastAPI Backend"
                return data
        except Exception:
            pass

        # Offline Disk Fallback
        try:
            risk = pd.read_csv(DATA_DASHBOARD / "risk_scored.csv")
            cats = sorted(risk["category"].dropna().unique().tolist())
            if category and category != "All":
                risk = risk[risk["category"].str.lower() == category.lower()]
            return {
                "source": "Local Disk Cache (Backend Offline)",
                "skus": risk["sku_id"].tolist(),
                "total": len(risk),
                "categories": cats,
            }
        except Exception:
            return {"skus": [], "total": 0, "categories": [], "source": "None"}

    def get_sku_forecast(self, sku_id: str) -> Optional[Dict[str, Any]]:
        """Fetch single SKU forecast and risk details from /sku/{sku_id}/forecast."""
        try:
            r = requests.get(f"{self.base_url}/sku/{sku_id}/forecast", timeout=TIMEOUT_SECONDS)
            if r.status_code == 200:
                data = r.json()
                data["source"] = "FastAPI Backend"
                return data
        except Exception:
            pass

        # Offline Disk Fallback
        try:
            risk = pd.read_csv(DATA_DASHBOARD / "risk_scored.csv")
            row = risk[risk["sku_id"] == sku_id]
            if len(row) == 0:
                alt = sku_id.replace("_", "-") if "_" in sku_id else sku_id.replace("-", "_")
                row = risk[risk["sku_id"] == alt]
            if len(row) > 0:
                d = row.iloc[0].to_dict()
                d["source"] = "Local Disk Cache (Backend Offline)"
                return d
        except Exception:
            pass
        return None

    def get_sku_history(self, sku_id: str) -> Dict[str, Any]:
        """Fetch historical actuals and model curves for a SKU from /sku/{sku_id}/history."""
        try:
            r = requests.get(f"{self.base_url}/sku/{sku_id}/history", timeout=TIMEOUT_SECONDS)
            if r.status_code == 200:
                data = r.json()
                data["source"] = "FastAPI Backend"
                return data
        except Exception:
            pass
        return {"sku_id": sku_id, "series": [], "source": "None"}

    def get_risk_catalog(self) -> List[Dict[str, Any]]:
        """Fetch full 192-SKU risk decisioning records from /risk/catalog."""
        try:
            r = requests.get(f"{self.base_url}/risk/catalog", timeout=TIMEOUT_SECONDS)
            if r.status_code == 200:
                return r.json().get("records", [])
        except Exception:
            pass

        # Offline Disk Fallback
        try:
            risk = pd.read_csv(DATA_DASHBOARD / "risk_scored.csv")
            return risk.to_dict(orient="records")
        except Exception:
            return []

    def get_risk_summary(self) -> Dict[str, Any]:
        """Fetch risk quadrant aggregates from /risk/summary."""
        try:
            r = requests.get(f"{self.base_url}/risk/summary", timeout=TIMEOUT_SECONDS)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return {}

    def predict_manual(
        self,
        sku_id: str,
        unit_price: float,
        discount_pct: float,
        is_holiday: int,
        promo_flag: int,
        recent_4w_avg: float,
    ) -> Dict[str, Any]:
        """
        Execute live dual-model inference on PyTorch LSTM and LightGBM models
        running in the FastAPI microservice.
        """
        payload = {
            "sku_id": sku_id,
            "unit_price": float(unit_price),
            "discount_pct": float(discount_pct),
            "is_holiday": int(is_holiday),
            "promo_flag": int(promo_flag),
            "recent_4w_avg": float(recent_4w_avg),
        }
        try:
            r = requests.post(f"{self.base_url}/predict/manual", json=payload, timeout=6.0)
            if r.status_code == 200:
                data = r.json()
                data["is_live_model"] = True
                return data
        except Exception as e:
            pass

        # Offline indicator: inform user that real ML inference requires FastAPI
        return {
            "is_live_model": False,
            "sku_id": sku_id,
            "scenario_inputs": payload,
            "lightgbm": {
                "model_name": "LightGBM Regressor (Backend Offline)",
                "predicted_units": round(recent_4w_avg * (1.0 + discount_pct / 100.0) * (1.1 if promo_flag else 1.0), 2),
                "lower_bound": round(recent_4w_avg * 0.75, 2),
                "upper_bound": round(recent_4w_avg * 1.25, 2),
                "confidence_interval": "Estimated (FastAPI Offline)",
                "historical_wape": 0.0832,
                "status": "Offline Emulation",
            },
            "lstm": {
                "model_name": "PyTorch Deep LSTM (Backend Offline)",
                "predicted_units": round(recent_4w_avg * 1.02 * (1.1 if promo_flag else 1.0), 2),
                "lower_bound": round(recent_4w_avg * 0.85, 2),
                "upper_bound": round(recent_4w_avg * 1.15, 2),
                "confidence_interval": "Estimated (FastAPI Offline)",
                "historical_wape": 0.0767,
                "status": "Offline Emulation",
            },
            "consensus_prediction": round(recent_4w_avg, 2),
            "divergence_pct": 0.0,
            "recommended_model": "FastAPI Offline — Start uvicorn to run real models",
            "explanation": (
                "⚠️ FastAPI microservice is not responding at "
                f"{self.base_url}. Real PyTorch LSTM & LightGBM inference requires "
                "the backend running: 'uvicorn service.main:app --port 8000'."
            ),
        }

    def get_api_catalog(self) -> List[Dict[str, Any]]:
        """Fetch endpoint documentation catalog from /api/catalog."""
        try:
            r = requests.get(f"{self.base_url}/api/catalog", timeout=TIMEOUT_SECONDS)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return []


# Shared singleton client instance
client = FORESIGHTApiClient()
