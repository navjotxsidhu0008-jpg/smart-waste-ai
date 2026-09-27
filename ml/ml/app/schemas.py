"""
Pydantic Schemas for FastAPI Request Validation and Response Serialization.
Provides automatic OpenAPI / Swagger documentation at /docs.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class Base64PredictRequest(BaseModel):
    """Request schema when submitting a webcam frame as a base64-encoded string."""
    image: str = Field(
        ...,
        description="Base64-encoded JPEG/PNG image frame from the webcam (with or without data:image/... prefix)."
    )
    demo_hint: Optional[str] = Field(
        None,
        description="Optional category hint when testing in interactive classroom simulation mode."
    )
    force_save: bool = Field(
        False,
        description="If true, bypasses the duplicate debounce timer and forces saving to SQLite."
    )


class PredictionResponse(BaseModel):
    """Standardized response returned by POST /api/predict."""
    waste_type: str = Field(..., examples=["Plastic"])
    confidence: float = Field(..., description="Confidence percentage (0-100)", examples=[94.7])
    recommended_bin: str = Field(..., examples=["♻️ Plastic Bin"])
    bin_short: str = Field(..., examples=["Plastic Bin"])
    icon: str = Field(..., examples=["♻️"])
    bin_color: str = Field(..., examples=["#3b82f6"])
    instruction: str = Field(
        ...,
        examples=["Place plastic waste in the plastic recycling bin. Empty liquids and rinse bottles or containers before disposal."]
    )
    eco_tip: str = Field(
        ...,
        examples=["Recycling one plastic bottle saves enough energy to power an LED lamp for over 24 hours."]
    )
    is_low_confidence: bool = Field(False, description="True if confidence is below configured threshold")
    status_message: str = Field(..., examples=["Classification successful"])
    is_demo_mode: bool = Field(False, description="True if running in fallback/demo mode without custom weights")
    model_name: str = Field(..., examples=["MobileNetV2 (Deep Learning CPU)"])
    detected_object: Optional[str] = Field(None, examples=["water bottle"])
    class_probabilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Confidence percentages across all 6 waste categories"
    )
    saved_to_db: bool = Field(..., description="True if logged to SQLite, False if debounced or low-confidence")
    detection_id: Optional[int] = Field(None, description="Database row ID if saved")
    timestamp: str = Field(..., examples=["2026-09-24T22:45:12"])


class DetectionRecord(BaseModel):
    """Single detection record from SQLite."""
    id: int
    waste_type: str
    confidence: float
    recommended_bin: str
    timestamp: str


class DetectionsListResponse(BaseModel):
    """Response for GET /api/detections."""
    count: int
    detections: List[DetectionRecord]


class CategoryDistItem(BaseModel):
    """Category count and percentage distribution."""
    count: int
    percentage: float


class StatsResponse(BaseModel):
    """Response for GET /api/stats."""
    total_detections: int = Field(..., examples=[128])
    today_detections: int = Field(..., examples=[42])
    category_counts: Dict[str, int] = Field(
        ...,
        examples=[{
            "Plastic": 41,
            "Paper": 25,
            "Metal": 18,
            "Glass": 16,
            "Organic": 17,
            "General Waste": 11
        }]
    )
    distribution: Dict[str, CategoryDistItem]
    most_frequent_category: str = Field(..., examples=["Plastic"])
    most_frequent_count: int = Field(..., examples=[41])
    categories: List[str]
    recyclable_share_pct: float = Field(default=0.0, description="Recyclable & compostable share percentage", examples=[85.5])


class WasteRuleItem(BaseModel):
    """Disposal instructions and metadata for a single waste category."""
    category: str
    recommended_bin: str
    bin_short: str
    icon: str
    bin_color: str
    badge_bg: str
    instruction: str
    eco_tip: str
    recyclable: bool
    examples: List[str]


class WasteRulesResponse(BaseModel):
    """Response for GET /api/rules."""
    count: int
    rules: List[WasteRuleItem]


class ResetDetectionsResponse(BaseModel):
    """Response for DELETE /api/detections."""
    status: str = "success"
    deleted_records: int
    message: str


class ModelInfoResponse(BaseModel):
    """Response for GET /api/model-info."""
    model_name: str
    architecture: str
    framework: str
    mode: str
    is_demo_mode: bool
    custom_model_loaded: bool
    custom_model_path: str
    input_size: List[int]
    confidence_threshold: float
    debounce_seconds: float
    supported_categories: List[str]
    device: str
    description: str


class HealthResponse(BaseModel):
    """Response for GET /health."""
    status: str = "healthy"
    app: str
    version: str
    model_ready: bool
    model_mode: str
    database_ready: bool
    timestamp: str

