"""
schemas.py — Pydantic models for FastAPI request/response validation.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SKUForecastRequest(BaseModel):
    """Request model for batch forecast."""
    sku_ids: List[str] = Field(..., description="List of SKU IDs to forecast")


class ForecastResponse(BaseModel):
    """Response for a single SKU forecast + risk."""
    sku_id: str
    category: Optional[str] = None
    subcategory: Optional[str] = None
    avg_weekly_forecast: float
    on_hand_units: float = 0
    on_order_units: float = 0
    lead_time_days: float = 14
    stockout_risk: float
    overstock_risk: float
    risk_quadrant: str
    recommended_action: str
    urgency: str
    stockout_rupee_at_risk: float = 0
    overstock_rupee_locked: float = 0
    total_rupee_at_stake: float = 0


class BatchForecastResponse(BaseModel):
    """Response for batch forecast."""
    results: List[ForecastResponse]
    total_skus: int
    total_rupee_at_stake: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "2.0.0"
    models_loaded: Dict[str, bool] = {}
    total_skus: int = 0
    architecture: str = "Decoupled Client-Server (Streamlit UI <-> FastAPI REST <-> ML Models)"


class ManualPredictRequest(BaseModel):
    """Parameters for live manual scenario prediction."""
    sku_id: str = Field("SKU-001", description="SKU identifier")
    unit_price: float = Field(..., gt=0, description="Hypothetical selling price per unit in ₹")
    discount_pct: float = Field(0.0, ge=0, le=100, description="Promotional discount percentage (0 to 100)")
    is_holiday: int = Field(0, ge=0, le=1, description="Holiday week flag (1 = Holiday week, 0 = Regular)")
    promo_flag: int = Field(0, ge=0, le=1, description="Active promotional campaign (1 = Active, 0 = None)")
    recent_4w_avg: float = Field(..., ge=0, description="Recent 4-week average demand baseline (units/week)")


class ModelPrediction(BaseModel):
    """Individual model forecast prediction with confidence bounds."""
    model_name: str
    predicted_units: float
    lower_bound: float
    upper_bound: float
    confidence_interval: str
    historical_wape: float
    status: str = "Active"


class DualModelPredictResponse(BaseModel):
    """Side-by-side comparison of PyTorch LSTM and LightGBM predictions."""
    sku_id: str
    scenario_inputs: Dict[str, Any]
    lightgbm: ModelPrediction
    lstm: ModelPrediction
    consensus_prediction: float
    divergence_pct: float
    recommended_model: str
    explanation: str


class OverviewResponse(BaseModel):
    """Top-level portfolio summary metrics."""
    total_skus: int
    skus_at_risk: int
    risk_percentage: float
    total_rupee_at_stake: float
    total_revenue: float
    best_model_name: str
    best_model_wape: float


class ModelComparisonItem(BaseModel):
    """Metrics for a single forecasting model."""
    name: str
    wape: float
    mape: float
    bias: float
    rmse: float
    is_winner: bool = False


class ModelComparisonResponse(BaseModel):
    """Full benchmark leaderboard."""
    leaderboard: List[ModelComparisonItem]
    winner: str


class APIDocEndpoint(BaseModel):
    """Detailed endpoint metadata for developer documentation."""
    path: str
    method: str
    summary: str
    description: str
    tags: List[str]
    sample_request: Optional[Dict[str, Any]] = None
    sample_curl: str


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: str
