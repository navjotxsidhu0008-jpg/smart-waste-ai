"""
SQLite Database module for AI Smart Waste Management System.
Automatically manages database creation, schema migrations, detection logging,
intelligent duplicate debouncing, and aggregate statistical queries.
"""

import sqlite3
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from app.config import DB_PATH, DEBOUNCE_SECONDS, WASTE_CATEGORIES


def get_db_connection() -> sqlite3.Connection:
    """
    Create and return a configured SQLite connection.
    Ensures directory exists and enables busy timeout for concurrent safety.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Wait up to 5 seconds if another thread/process is writing
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn


def init_db() -> None:
    """
    Initialize SQLite database tables, PRAGMA journal_mode=WAL, and indices.
    Called automatically on application startup.
    """
    conn = get_db_connection()
    try:
        with conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    waste_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    recommended_bin TEXT NOT NULL,
                    timestamp DATETIME NOT NULL
                );
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_detections_timestamp 
                ON detections (timestamp DESC);
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_detections_waste_type 
                ON detections (waste_type);
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_detections_type_id 
                ON detections (waste_type, id DESC);
            """)
    finally:
        conn.close()


def is_recent_duplicate(waste_type: str, debounce_seconds: float = DEBOUNCE_SECONDS) -> bool:
    """
    Check if the same waste category was logged within the debounce window.
    This prevents spamming the database when an object is held in front of the camera.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT timestamp FROM detections 
            WHERE waste_type = ? 
            ORDER BY id DESC LIMIT 1
        """, (waste_type,))
        row = cursor.fetchone()
        if not row:
            return False

        last_time_str = row["timestamp"]
        try:
            last_time = datetime.fromisoformat(last_time_str)
        except Exception:
            try:
                last_time = datetime.strptime(last_time_str, "%Y-%m-%d %H:%M:%S")
            except Exception:
                return False

        elapsed = (datetime.now() - last_time).total_seconds()
        return 0 <= elapsed < debounce_seconds
    except Exception:
        return False
    finally:
        conn.close()


def add_detection(
    waste_type: str,
    confidence: float,
    recommended_bin: str,
    force: bool = False
) -> Optional[int]:
    """
    Log a new waste detection to SQLite.
    Applies debounce filter unless force=True.
    Returns the new record ID if inserted, or None if debounced.
    """
    if not force and is_recent_duplicate(waste_type):
        return None

    # Normalize confidence to percentage (0.0 to 100.0)
    conf_val = float(confidence)
    if 0.0 < conf_val <= 1.0:
        conf_val = conf_val * 100.0

    now = datetime.now()
    timestamp_str = now.isoformat(timespec="seconds")

    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO detections (waste_type, confidence, recommended_bin, timestamp)
                VALUES (?, ?, ?, ?)
            """, (waste_type, round(conf_val, 4), recommended_bin, timestamp_str))
            return cursor.lastrowid
    finally:
        conn.close()


def get_recent_detections(limit: int = 20) -> List[Dict[str, Any]]:
    """
    Retrieve the most recent detection entries.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, waste_type, confidence, recommended_bin, timestamp
            FROM detections
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        results = []
        for row in rows:
            conf = float(row["confidence"])
            if 0.0 < conf <= 1.0:
                conf *= 100.0
            results.append({
                "id": row["id"],
                "waste_type": row["waste_type"],
                "confidence": round(conf, 1),
                "recommended_bin": row["recommended_bin"],
                "timestamp": row["timestamp"],
            })
        return results
    finally:
        conn.close()


def get_statistics() -> Dict[str, Any]:
    """
    Compute aggregate statistics for the dashboard and analytics page:
    - Total detections
    - Today's detections
    - Counts for each canonical waste category
    - Most frequently detected category
    - Distribution percentages
    - Recyclable & compostable share percentage
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        # Total count
        cursor.execute("SELECT COUNT(*) AS total FROM detections")
        total_detections = cursor.fetchone()["total"]

        # Today's count
        today_prefix = date.today().isoformat() + "%"
        cursor.execute("SELECT COUNT(*) AS today_count FROM detections WHERE timestamp LIKE ?", (today_prefix,))
        today_detections = cursor.fetchone()["today_count"]

        # Category counts
        cursor.execute("""
            SELECT waste_type, COUNT(*) AS count
            FROM detections
            GROUP BY waste_type
            ORDER BY count DESC
        """)
        raw_counts = {row["waste_type"]: row["count"] for row in cursor.fetchall()}

        # Ensure all 6 standard categories are represented even if 0
        category_counts: Dict[str, int] = {cat: 0 for cat in WASTE_CATEGORIES}
        for cat, cnt in raw_counts.items():
            category_counts[cat] = cnt

        # Determine most frequent category
        most_frequent_category = None
        most_frequent_count = 0
        if total_detections > 0:
            for cat, cnt in category_counts.items():
                if cnt > most_frequent_count:
                    most_frequent_count = cnt
                    most_frequent_category = cat

        # Compute percentage distribution
        distribution = {}
        for cat, cnt in category_counts.items():
            pct = round((cnt / total_detections * 100), 1) if total_detections > 0 else 0.0
            distribution[cat] = {
                "count": cnt,
                "percentage": pct
            }

        # Recyclable & Compostable share (all except General Waste)
        general_count = category_counts.get("General Waste", 0)
        recyclable_count = max(0, total_detections - general_count)
        recyclable_share_pct = round((recyclable_count / total_detections * 100), 1) if total_detections > 0 else 0.0

        return {
            "total_detections": total_detections,
            "today_detections": today_detections,
            "category_counts": category_counts,
            "distribution": distribution,
            "most_frequent_category": most_frequent_category or "None yet",
            "most_frequent_count": most_frequent_count,
            "categories": WASTE_CATEGORIES,
            "recyclable_share_pct": recyclable_share_pct,
        }
    finally:
        conn.close()


def clear_all_detections() -> int:
    """
    Delete all detection records from SQLite (useful for demonstration resets).
    Returns count of deleted rows.
    """
    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) AS cnt FROM detections")
            cnt = cursor.fetchone()["cnt"]
            cursor.execute("DELETE FROM detections")
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='detections'")
            return cnt
    finally:
        conn.close()
