"""
FastAPI Backend for the ♻️ AI Smart Waste Management System.

Provides:
- HTML Web UI routes (/ and /analytics)
- Real-time AI webcam frame classification (POST /api/predict)
- Smart bin recommendation lookup (GET /api/rules)
- Live detection statistics & SQLite logs (GET /api/stats, GET /api/detections)
- Model architecture & status diagnostics (GET /api/model-info, GET /health)
- Interactive Swagger documentation (/docs)
"""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import (
    APP_TITLE,
    APP_SUBTITLE,
    APP_VERSION,
    STATIC_DIR,
    TEMPLATES_DIR,
    DEBOUNCE_SECONDS,
    CAPTURE_INTERVAL_MS,
)
from app.database import (
    init_db,
    add_detection,
    get_recent_detections,
    get_statistics,
    clear_all_detections,
)
from app.model import classifier, decode_base64_image
from app.schemas import (
    PredictionResponse,
    Base64PredictRequest,
    StatsResponse,
    DetectionsListResponse,
    ModelInfoResponse,
    HealthResponse,
    WasteRulesResponse,
    ResetDetectionsResponse,
)
from app.waste_rules import get_all_waste_rules


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize SQLite database and ensure ML model is ready on startup."""
    init_db()
    yield


app = FastAPI(
    title=APP_TITLE,
    description=(
        "Real-Time AI Waste Classification & Smart Bin Recommendation API built with "
        "FastAPI, Deep Learning (MobileNetV2), OpenCV, and SQLite."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static assets (CSS, JS, images) and Jinja2 HTML templates
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ---------------------------------------------------------------------------
# HTML FRONTEND PAGES
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home_page(request: Request):
    """Render the main Real-Time AI Camera & Smart Waste Dashboard."""
    try:
        stats = get_statistics()
        recent = get_recent_detections(limit=10)
    except Exception:
        stats = {
            "total_detections": 0,
            "today_detections": 0,
            "category_counts": {},
            "distribution": {},
            "most_frequent_category": "None yet",
            "most_frequent_count": 0,
            "categories": [],
            "recyclable_share_pct": 0.0,
        }
        recent = []

    model_info = classifier.get_info()
    rules = get_all_waste_rules()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "app_title": APP_TITLE,
            "app_subtitle": APP_SUBTITLE,
            "stats": stats,
            "recent_detections": recent,
            "model_info": model_info,
            "waste_rules": rules,
            "capture_interval_ms": CAPTURE_INTERVAL_MS,
        },
    )


@app.get("/analytics", response_class=HTMLResponse, include_in_schema=False)
async def analytics_page(request: Request):
    """Render the dedicated Waste Analytics & Insights page."""
    try:
        stats = get_statistics()
        recent = get_recent_detections(limit=30)
    except Exception:
        stats = {
            "total_detections": 0,
            "today_detections": 0,
            "category_counts": {},
            "distribution": {},
            "most_frequent_category": "None yet",
            "most_frequent_count": 0,
            "categories": [],
            "recyclable_share_pct": 0.0,
        }
        recent = []

    model_info = classifier.get_info()
    rules = get_all_waste_rules()
    return templates.TemplateResponse(
        request=request,
        name="analytics.html",
        context={
            "app_title": APP_TITLE,
            "app_subtitle": APP_SUBTITLE,
            "stats": stats,
            "recent_detections": recent,
            "model_info": model_info,
            "waste_rules": rules,
        },
    )


# ---------------------------------------------------------------------------
# REST API ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Verify backend server, ML model, and SQLite database health."""
    db_ok = True
    try:
        get_statistics()
    except Exception:
        db_ok = False

    model_ready = classifier is not None and getattr(classifier, "mode", None) is not None
    is_healthy = db_ok and model_ready

    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        app=APP_TITLE,
        version=APP_VERSION,
        model_ready=model_ready,
        model_mode=classifier.mode if model_ready else "unavailable",
        database_ready=db_ok,
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )


@app.get("/api/model-info", response_model=ModelInfoResponse, tags=["AI Model"])
async def get_model_info():
    """Return active Deep Learning model details, mode, and supported waste categories."""
    info = classifier.get_info()
    info["debounce_seconds"] = DEBOUNCE_SECONDS
    return ModelInfoResponse(**info)


@app.get("/api/rules", response_model=WasteRulesResponse, tags=["Smart Bin Rules"])
async def get_waste_disposal_rules():
    """Return the smart bin mapping, icons, colors, and disposal instructions for all 6 waste types."""
    rules = get_all_waste_rules()
    return WasteRulesResponse(
        count=len(rules),
        rules=rules,
    )


@app.post("/api/predict", response_model=PredictionResponse, tags=["AI Classification"])
async def predict_waste(
    file: UploadFile = File(..., description="Image file or webcam JPEG frame to classify"),
    demo_hint: Optional[str] = Form(None, description="Optional category preset for classroom demo testing"),
    force_save: bool = Form(False, description="Force saving to SQLite bypassing duplicate debounce timer"),
):
    """
    Analyze a webcam frame or uploaded image and classify it into one of 6 waste categories:
    Plastic, Paper, Metal, Glass, Organic, or General Waste.

    Automatically maps the prediction to the recommended disposal bin and logs valid
    detections to the local SQLite database with duplicate debouncing.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No image file uploaded.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")

    try:
        result = classifier.predict_image(image_bytes, demo_hint=demo_hint)
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Classification error: {exc}") from exc

    # Log to SQLite if confidence meets threshold (or if force_save is requested)
    detection_id = None
    saved_to_db = False
    if not result["is_low_confidence"] or force_save:
        try:
            detection_id = add_detection(
                waste_type=result["waste_type"],
                confidence=result["confidence"],
                recommended_bin=result["recommended_bin"],
                force=force_save,
            )
            saved_to_db = detection_id is not None
        except Exception as db_exc:
            result["status_message"] += f" (Note: DB log skipped: {db_exc})"

    now_iso = datetime.now().isoformat(timespec="seconds")
    return PredictionResponse(
        **result,
        saved_to_db=saved_to_db,
        detection_id=detection_id,
        timestamp=now_iso,
    )


@app.post("/api/predict-base64", response_model=PredictionResponse, tags=["AI Classification"])
async def predict_waste_base64(payload: Base64PredictRequest):
    """
    Alternative prediction endpoint accepting a JSON payload with a base64-encoded webcam frame.
    """
    try:
        image_bytes = decode_base64_image(payload.image)
        result = classifier.predict_image(image_bytes, demo_hint=payload.demo_hint)
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Classification error: {exc}") from exc

    detection_id = None
    saved_to_db = False
    if not result["is_low_confidence"] or payload.force_save:
        try:
            detection_id = add_detection(
                waste_type=result["waste_type"],
                confidence=result["confidence"],
                recommended_bin=result["recommended_bin"],
                force=payload.force_save,
            )
            saved_to_db = detection_id is not None
        except Exception as db_exc:
            result["status_message"] += f" (Note: DB log skipped: {db_exc})"

    now_iso = datetime.now().isoformat(timespec="seconds")
    return PredictionResponse(
        **result,
        saved_to_db=saved_to_db,
        detection_id=detection_id,
        timestamp=now_iso,
    )


@app.get("/api/stats", response_model=StatsResponse, tags=["Analytics & Database"])
async def get_dashboard_stats():
    """Retrieve real-time detection totals, category counts, and distribution percentages from SQLite."""
    try:
        stats = get_statistics()
        return StatsResponse(**stats)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database query error: {exc}") from exc


@app.get("/api/detections", response_model=DetectionsListResponse, tags=["Analytics & Database"])
async def list_recent_detections(
    limit: int = Query(20, ge=1, le=200, description="Maximum number of recent detections to return")
):
    """Retrieve the most recent waste detection records stored in SQLite."""
    try:
        records = get_recent_detections(limit=limit)
        return DetectionsListResponse(count=len(records), detections=records)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch detections: {exc}") from exc


@app.delete("/api/detections", response_model=ResetDetectionsResponse, tags=["Analytics & Database"])
async def reset_detections():
    """Delete all detection records from the SQLite database (useful for resetting before a demo)."""
    try:
        deleted_count = clear_all_detections()
        return ResetDetectionsResponse(
            status="success",
            deleted_records=deleted_count,
            message=f"Cleared {deleted_count} detection record(s) from the database.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to clear detections: {exc}") from exc
