"""
Configuration settings for the AI Smart Waste Management System.
Centralizes paths, model thresholds, camera intervals, and server parameters.
"""

import os
import sys
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Base project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Optional simple .env file loader (no external dependency required)
ENV_FILE = BASE_DIR / ".env"
if ENV_FILE.exists():
    try:
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    except Exception:
        pass

# Application Metadata
APP_TITLE = "♻️ AI Smart Waste Management"
APP_SUBTITLE = "Real-Time Waste Classification Using Deep Learning"
APP_VERSION = "1.0.0"

# Server Settings
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "true").lower() in ("true", "1", "yes")

# Directory Paths
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
UPLOADS_DIR = BASE_DIR / "uploads"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Database Settings (resolve relative paths against BASE_DIR)
env_db_path = os.getenv("DB_PATH")
if env_db_path:
    _p = Path(env_db_path)
    DB_PATH = _p if _p.is_absolute() else (BASE_DIR / _p)
else:
    DB_PATH = DATA_DIR / "waste.db"

# Machine Learning / Deep Learning Model Settings
# Supports custom trained Keras (.keras / .h5) or PyTorch (.pth / .pt) or ONNX (.onnx) model
DEFAULT_MODEL_PATH = MODEL_DIR / "waste_classifier.keras"
ALT_PYTORCH_MODEL_PATH = MODEL_DIR / "waste_classifier.pth"
env_model_path = os.getenv("MODEL_PATH")
if env_model_path:
    _mp = Path(env_model_path)
    MODEL_PATH = _mp if _mp.is_absolute() else (BASE_DIR / _mp)
else:
    MODEL_PATH = DEFAULT_MODEL_PATH

# Input image dimensions expected by MobileNetV2 / standard CNNs
IMAGE_SIZE = (224, 224)

# Minimum confidence score (0.0 to 1.0) before warning about uncertain prediction
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))

# Duplicate Detection Debounce Window (in seconds)
# Prevents storing identical consecutive frames into SQLite when an object stays in front of the camera
DEBOUNCE_SECONDS = float(os.getenv("DEBOUNCE_SECONDS", "5.0"))

# Default Frontend Webcam Frame Capture Interval (in milliseconds)
CAPTURE_INTERVAL_MS = int(os.getenv("CAPTURE_INTERVAL_MS", "1200"))

# Supported Waste Categories (in strict canonical order)
WASTE_CATEGORIES = [
    "Plastic",
    "Paper",
    "Metal",
    "Glass",
    "Organic",
    "General Waste",
]
