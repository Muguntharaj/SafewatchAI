"""
migration_add_detection_face_columns.py
Alembic migration — adds face_image_url, person_string_id, person_name
columns to the Detection table.

Run with:  alembic upgrade head
Or directly:  python migration_add_detection_face_columns.py

These columns are what make TimeLog "View Details" show the correct face
image without any cross-table lookups at render time.
"""

# ── Option A: Alembic migration ───────────────────────────────────────────────
# Place this file in alembic/versions/ with a proper revision ID.

"""
revision = 'add_detection_face_cols'
down_revision = None   # set to your current head revision
branch_labels = None
depends_on = None

from alembic import op
import sqlalchemy as sa

def upgrade():
    op.add_column('detection',
        sa.Column('face_image_url',   sa.String(512), nullable=True))
    op.add_column('detection',
        sa.Column('person_string_id', sa.String(32),  nullable=True))
    op.add_column('detection',
        sa.Column('person_name',      sa.String(128), nullable=True))
    # person_classification may already exist — wrap in try
    try:
        op.add_column('detection',
            sa.Column('person_classification', sa.String(32), nullable=True))
    except Exception:
        pass

def downgrade():
    for col in ['face_image_url', 'person_string_id',
                'person_name', 'person_classification']:
        try:
            op.drop_column('detection', col)
        except Exception:
            pass
"""


# ── Option B: Direct SQLite migration (run once, safe to re-run) ──────────────
import sqlite3
import os
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "safewatch.db"


def run_migration(db_path: Path = DB_PATH):
    if not db_path.exists():
        print(f"DB not found at {db_path} — skipping migration")
        return

    conn = sqlite3.connect(str(db_path))
    cur  = conn.cursor()

    new_columns = [
        ("face_image_url",       "TEXT"),
        ("person_string_id",     "TEXT"),
        ("person_name",          "TEXT"),
        ("person_classification","TEXT"),
    ]

    # Get existing columns
    cur.execute("PRAGMA table_info(detections)")
    existing = {row[1] for row in cur.fetchall()}

    added = []
    for col_name, col_type in new_columns:
        if col_name not in existing:
            cur.execute(f"ALTER TABLE detections ADD COLUMN {col_name} {col_type}")
            added.append(col_name)
            print(f"  + Added column: detections.{col_name} {col_type}")
        else:
            print(f"  [OK] Column already exists: detections.{col_name}")

    conn.commit()
    conn.close()

    if added:
        print(f"\nMigration complete. Added: {', '.join(added)}")
    else:
        print("\nNo changes needed — all columns already exist.")


# ── Also add to SQLAlchemy model (core/database.py) ──────────────────────────
SQLALCHEMY_PATCH = '''
# In core/database.py, add these columns to the Detection class:

class Detection(Base):
    __tablename__ = "detections"
    # ... existing columns ...
    
    # NEW — added by migration_add_detection_face_columns.py
    face_image_url        = Column(String(512), nullable=True)
    person_string_id      = Column(String(32),  nullable=True)   # "PERSON_A1B2C3"
    person_name           = Column(String(128), nullable=True)
    person_classification = Column(String(32),  nullable=True)   # employee/owner/unknown
'''

print(SQLALCHEMY_PATCH)


if __name__ == "__main__":
    print(f"Running migration on: {DB_PATH}")
    run_migration()
