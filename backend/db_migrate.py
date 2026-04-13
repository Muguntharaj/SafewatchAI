"""
db_migrate.py — SafeWatch AI — One-time database migration
Run this ONCE after deploying the fixed ai_engine.py and camera_manager_patches_v2.py.

WHAT THIS DOES:
  Adds 5 columns to the `detections` table that _save_detectionss_bulk_fixed
  writes to, but that are absent in older DB schemas:
    - person_classification  TEXT   (employee / owner / unknown)
    - person_string_id       TEXT   (PERSON_XXXXXXXX identifier)
    - person_name            TEXT   (display name)
    - face_image_url         TEXT   (full http:// URL to face image)
    - detections_uuid         TEXT   (unique UUID per detections)

  Each ALTER TABLE is wrapped in try/except so it is safe to run multiple times
  (SQLite raises "duplicate column name" if the column already exists).

USAGE:
    python db_migrate.py
    # or with a custom DB path:
    python db_migrate.py --db /path/to/safewatch.db

NOTES:
  - The script auto-discovers the DB path from core.config.settings if
    no --db flag is given.
  - Existing data is never modified — only new columns are added (nullable).
  - Run from your project root (same directory as main.py).
"""

import sys
import sqlite3
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')
logger = logging.getLogger("db_migrate")


COLUMNS_TO_ADD = [
    ("person_classification", "TEXT"),
    ("person_string_id",      "TEXT"),
    ("person_name",           "TEXT"),
    ("face_image_url",        "TEXT"),
    ("detections_uuid",        "TEXT"),
]


def discover_db_path() -> str:
    """Try to read DB path from project settings."""
    try:
        sys.path.insert(0, ".")
        from core.config import settings
        db_url = str(settings.DATABASE_URL)
        # SQLAlchemy format: sqlite+aiosqlite:///./safewatch.db
        if "///" in db_url:
            path = db_url.split("///")[-1]
            if path.startswith("./"):
                path = path[2:]
            return path
    except Exception as e:
        logger.warning(f"Could not read settings ({e}), defaulting to safewatch.db")
    return "safewatch.db"


def migrate(db_path: str):
    logger.info(f"Opening database: {db_path}")
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    # Verify the detections table exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='detections'")
    if not cur.fetchone():
        logger.error("Table 'detections' not found. Is this the right database?")
        conn.close()
        sys.exit(1)

    # Show existing columns
    cur.execute("PRAGMA table_info(detections)")
    existing = {row[1] for row in cur.fetchall()}
    logger.info(f"Existing columns in 'detections': {sorted(existing)}")

    added   = []
    skipped = []

    for col_name, col_type in COLUMNS_TO_ADD:
        if col_name in existing:
            skipped.append(col_name)
            logger.info(f"  SKIP  {col_name} (already exists)")
            continue
        try:
            cur.execute(f"ALTER TABLE detections ADD COLUMN {col_name} {col_type}")
            conn.commit()
            added.append(col_name)
            logger.info(f"  ADD   {col_name} {col_type}  ✓")
        except Exception as e:
            logger.error(f"  FAIL  {col_name}: {e}")

    # Verify final schema
    cur.execute("PRAGMA table_info(detections)")
    final_cols = sorted(row[1] for row in cur.fetchall())
    logger.info(f"\nFinal detections columns: {final_cols}")

    conn.close()

    print(f"\n{'='*50}")
    print(f"Migration complete.")
    print(f"  Added   : {added   or 'none'}")
    print(f"  Skipped : {skipped or 'none'}")
    print(f"{'='*50}")
    print("\nNext steps:")
    print("  1. Restart your FastAPI backend")
    print("  2. Reload the frontend")
    print("  3. detectionss should now appear in Time Logs within ~10 seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SafeWatch DB migration")
    parser.add_argument("--db", type=str, default=None,
                        help="Path to SQLite database file")
    args = parser.parse_args()

    db_path = args.db or discover_db_path()
    migrate(db_path)
