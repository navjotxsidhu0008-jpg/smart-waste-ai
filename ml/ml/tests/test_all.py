"""
Comprehensive test suite for the AI Smart Waste Management System.
Tests API routes, machine learning inference, database operations, and edge cases.
"""

import io
import base64
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.config import WASTE_CATEGORIES, CONFIDENCE_THRESHOLD
from app.model import classifier, decode_base64_image
from app.database import (
    init_db,
    add_detection,
    get_recent_detections,
    get_statistics,
    clear_all_detections,
    is_recent_duplicate,
)
from app.waste_rules import get_waste_rule, get_all_waste_rules


@pytest.fixture(scope="session")
def client():
    """Create FastAPI test client."""
    init_db()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def synthetic_image_bytes():
    """Generate a clean synthetic RGB test image in JPEG format."""
    img = Image.new("RGB", (224, 224), color=(60, 120, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def dark_image_bytes():
    """Generate an almost completely black image."""
    img = Image.new("RGB", (224, 224), color=(2, 2, 2))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ============================================================================
# 1. System Health & Metadata Routes
# ============================================================================

def test_health_endpoint(client):
    """GET /health must return status healthy, app title, version, and model mode."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["database_ready"] is True
    assert data["model_ready"] is True
    assert "version" in data
    assert "timestamp" in data


def test_model_info_endpoint(client):
    """GET /api/model-info must return model details and supported categories."""
    res = client.get("/api/model-info")
    assert res.status_code == 200
    data = res.json()
    assert data["supported_categories"] == WASTE_CATEGORIES
    assert data["input_size"] == [224, 224]
    assert data["confidence_threshold"] > 0
    assert "debounce_seconds" in data
    assert "architecture" in data


def test_waste_rules_endpoint(client):
    """GET /api/rules must return all 6 waste disposal rules."""
    res = client.get("/api/rules")
    assert res.status_code == 200
    data = res.json()
    assert data["count"] == 6
    assert len(data["rules"]) == 6
    categories = [r["category"] for r in data["rules"]]
    for cat in WASTE_CATEGORIES:
        assert cat in categories


# ============================================================================
# 2. Frontend HTML Pages
# ============================================================================

def test_home_page_renders(client):
    """GET / must render the dashboard HTML."""
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "AI Smart Waste" in res.text
    assert "Live AI Scanner" in res.text


def test_analytics_page_renders(client):
    """GET /analytics must render the analytics dashboard HTML."""
    res = client.get("/analytics")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "Waste Classification Analytics" in res.text


# ============================================================================
# 3. Model Inference & Rules Module
# ============================================================================

def test_waste_rules_lookup():
    """Verify rule lookup works for all canonical categories and handles fallback."""
    for cat in WASTE_CATEGORIES:
        rule = get_waste_rule(cat)
        assert rule["category"] == cat
        assert "recommended_bin" in rule
        assert "icon" in rule
        assert "bin_color" in rule
        assert "instruction" in rule
        assert "eco_tip" in rule

    # Unknown category falls back to General Waste
    fallback_rule = get_waste_rule("NonExistentTrashType")
    assert fallback_rule["category"] == "General Waste"


def test_predict_image_normal(synthetic_image_bytes):
    """Verify model prediction on valid synthetic image returns required keys."""
    result = classifier.predict_image(synthetic_image_bytes)
    assert result["waste_type"] in WASTE_CATEGORIES
    assert 0.0 <= result["confidence"] <= 100.0
    assert "recommended_bin" in result
    assert "class_probabilities" in result
    assert len(result["class_probabilities"]) == 6
    for cat in WASTE_CATEGORIES:
        assert cat in result["class_probabilities"]


def test_predict_image_dark_frame(dark_image_bytes):
    """Verify dark frame triggers low confidence warning and General Waste category."""
    result = classifier.predict_image(dark_image_bytes)
    assert result["waste_type"] == "General Waste"
    assert result["is_low_confidence"] is True
    assert "dark" in result["status_message"].lower()


def test_predict_image_preset_hint(synthetic_image_bytes):
    """Verify preset hint produces expected category prediction."""
    for cat in WASTE_CATEGORIES:
        result = classifier.predict_image(synthetic_image_bytes, demo_hint=cat)
        assert result["waste_type"] == cat
        assert result["confidence"] >= 80.0
        assert result["is_low_confidence"] is False


def test_decode_base64_image():
    """Verify safe base64 decoding with and without data header."""
    img = Image.new("RGB", (32, 32), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # Raw base64
    d1 = decode_base64_image(raw_b64)
    assert len(d1) > 0

    # With Data URI header
    d2 = decode_base64_image(f"data:image/png;base64,{raw_b64}")
    assert d1 == d2

    # Empty payload raises ValueError
    with pytest.raises(ValueError):
        decode_base64_image("")

    # Invalid base64 raises ValueError
    with pytest.raises(ValueError):
        decode_base64_image("!!!not_base64!!!")


# ============================================================================
# 4. Prediction API Endpoints (POST /api/predict & POST /api/predict-base64)
# ============================================================================

def test_api_predict_upload_success(client, synthetic_image_bytes):
    """POST /api/predict with valid image file."""
    files = {"file": ("test.jpg", synthetic_image_bytes, "image/jpeg")}
    data = {"force_save": "true"}
    res = client.post("/api/predict", files=files, data=data)
    assert res.status_code == 200
    body = res.json()
    assert body["waste_type"] in WASTE_CATEGORIES
    assert body["saved_to_db"] is True
    assert body["detection_id"] is not None
    assert "timestamp" in body


def test_api_predict_upload_empty_file(client):
    """POST /api/predict with empty file must return 400, not 500."""
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    res = client.post("/api/predict", files=files)
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()


def test_api_predict_upload_corrupt_data(client):
    """POST /api/predict with corrupt non-image bytes must return 400, not 500."""
    files = {"file": ("bad.jpg", b"not_an_image_binary_data", "image/jpeg")}
    res = client.post("/api/predict", files=files)
    assert res.status_code == 400
    assert "invalid image" in res.json()["detail"].lower()


def test_api_predict_base64_success(client, synthetic_image_bytes):
    """POST /api/predict-base64 with valid base64 image."""
    b64_str = base64.b64encode(synthetic_image_bytes).decode("ascii")
    payload = {
        "image": f"data:image/jpeg;base64,{b64_str}",
        "force_save": True,
    }
    res = client.post("/api/predict-base64", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["waste_type"] in WASTE_CATEGORIES
    assert body["saved_to_db"] is True
    assert body["detection_id"] is not None


def test_api_predict_base64_invalid(client):
    """POST /api/predict-base64 with invalid base64 must return 400."""
    payload = {"image": "invalid_base64_data", "force_save": False}
    res = client.post("/api/predict-base64", json=payload)
    assert res.status_code == 400


# ============================================================================
# 5. Database, Duplicate Debounce & Statistics
# ============================================================================

def test_database_crud_and_debouncing():
    """Verify database insertion, debounce filtering, and stats."""
    clear_all_detections()

    # 1. First insert should succeed
    id1 = add_detection("Plastic", 94.5, "♻️ Plastic Bin", force=False)
    assert id1 is not None

    # 2. Immediate second insert with same category should be debounced (None)
    id2 = add_detection("Plastic", 95.0, "♻️ Plastic Bin", force=False)
    assert id2 is None

    # 3. Insert with force=True should bypass debounce
    id3 = add_detection("Plastic", 93.0, "♻️ Plastic Bin", force=True)
    assert id3 is not None

    # 4. Different category should succeed without debouncing
    id4 = add_detection("Paper", 88.0, "📄 Paper Bin", force=False)
    assert id4 is not None

    # Verify recent detections
    recent = get_recent_detections(limit=10)
    assert len(recent) == 3

    # Verify statistics
    stats = get_statistics()
    assert stats["total_detections"] == 3
    assert stats["category_counts"]["Plastic"] == 2
    assert stats["category_counts"]["Paper"] == 1
    assert stats["most_frequent_category"] == "Plastic"


def test_api_stats_and_detections_endpoints(client):
    """GET /api/stats, GET /api/detections, and DELETE /api/detections."""
    # Fetch stats
    res_stats = client.get("/api/stats")
    assert res_stats.status_code == 200
    stats = res_stats.json()
    assert "total_detections" in stats
    assert "category_counts" in stats

    # Fetch detections
    res_det = client.get("/api/detections?limit=10")
    assert res_det.status_code == 200
    det_data = res_det.json()
    assert "detections" in det_data
    assert "count" in det_data

    # Test limit validation
    res_invalid_limit = client.get("/api/detections?limit=0")
    assert res_invalid_limit.status_code == 422

    # Reset detections
    res_del = client.delete("/api/detections")
    assert res_del.status_code == 200
    assert res_del.json()["status"] == "success"

    # Verify cleared stats
    cleared_stats = client.get("/api/stats").json()
    assert cleared_stats["total_detections"] == 0
    assert cleared_stats["most_frequent_count"] == 0


# ============================================================================
# 6. Advanced Edge Cases, Image Variations & Concurrency
# ============================================================================

def test_openapi_spec_valid(client):
    """GET /openapi.json must return valid OpenAPI 3.x specification."""
    res = client.get("/openapi.json")
    assert res.status_code == 200
    spec = res.json()
    assert "openapi" in spec
    assert "paths" in spec
    assert "/api/predict" in spec["paths"]
    assert "/api/rules" in spec["paths"]
    assert "/api/stats" in spec["paths"]


def test_cors_headers(client):
    """Verify CORS headers are present for cross-origin frontend requests."""
    res = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert res.status_code == 200
    assert "access-control-allow-origin" in res.headers


def test_image_format_and_channel_variations():
    """Verify PNG with alpha channel (RGBA), Grayscale (L), and various dimensions."""
    # RGBA image
    rgba_img = Image.new("RGBA", (150, 150), color=(100, 180, 50, 200))
    rgba_buf = io.BytesIO()
    rgba_img.save(rgba_buf, format="PNG")
    res_rgba = classifier.predict_image(rgba_buf.getvalue())
    assert res_rgba["waste_type"] in WASTE_CATEGORIES

    # Grayscale image
    gray_img = Image.new("L", (100, 100), color=180)
    gray_buf = io.BytesIO()
    gray_img.save(gray_buf, format="JPEG")
    res_gray = classifier.predict_image(gray_buf.getvalue())
    assert res_gray["waste_type"] in WASTE_CATEGORIES

    # Tiny image (16x16)
    tiny_img = Image.new("RGB", (16, 16), color=(200, 100, 50))
    tiny_buf = io.BytesIO()
    tiny_img.save(tiny_buf, format="JPEG")
    res_tiny = classifier.predict_image(tiny_buf.getvalue())
    assert res_tiny["waste_type"] in WASTE_CATEGORIES


def test_concurrent_database_writes():
    """Verify SQLite concurrent writes from multiple threads do not lock or crash."""
    import concurrent.futures

    def worker(idx):
        cat = WASTE_CATEGORIES[idx % len(WASTE_CATEGORIES)]
        rule = get_waste_rule(cat)
        return add_detection(
            waste_type=cat,
            confidence=85.0 + (idx % 10),
            recommended_bin=rule["recommended_bin"],
            force=True,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, i) for i in range(24)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # All 24 inserts should return a valid integer row ID
    assert len(results) == 24
    assert all(isinstance(r, int) for r in results)

