"""
schemas.py — Pydantic models for FastAPI request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


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
    version: str = "1.0.0"
    model_loaded: bool = False
    total_skus: int = 0


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: str
