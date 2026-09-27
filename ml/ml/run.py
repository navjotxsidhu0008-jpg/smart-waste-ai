"""
AI Smart Waste Management System - Main Execution Entry Point.

Starts the FastAPI server with Uvicorn, displays terminal status banner,
and provides direct browser links.
"""

import sys

from pathlib import Path

# Ensure project root is in sys.path for direct python execution from any directory
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import uvicorn
from app.config import HOST, PORT, DEBUG, APP_TITLE, APP_SUBTITLE, APP_VERSION


def main():
    print("=" * 70)
    print(f"  {APP_TITLE} (v{APP_VERSION})")
    print(f"  {APP_SUBTITLE}")
    print("=" * 70)
    print(f"  > Dashboard UI:   http://{HOST}:{PORT}/")
    print(f"  > Analytics:      http://{HOST}:{PORT}/analytics")
    print(f"  > API Swagger:    http://{HOST}:{PORT}/docs")
    print(f"  > Health Check:   http://{HOST}:{PORT}/health")
    print("=" * 70)
    print("  Starting server with Uvicorn (Press Ctrl+C to stop)...")
    print("=" * 70)

    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        app_dir=str(BASE_DIR),
        log_level="info",
    )


if __name__ == "__main__":
    main()
