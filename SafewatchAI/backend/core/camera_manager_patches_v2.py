"""
camera_manager_patches_v2.py — SafeWatch AI

ADDITIONAL PATCHES for camera_manager.py to fix:
  1. Timelogs not appearing (detections not saved to DB)
  2. False alerts from low-confidence CLIP detections

Apply these changes to your existing camera_manager.py:

═══════════════════════════════════════════════════════════════
PATCH 1: Raise _MIN_ACTION_CONF to match new CLIP thresholds
═══════════════════════════════════════════════════════════════

FIND (around line 1814 in camera_manager.py):
    _MIN_ACTION_CONF = 0.30

REPLACE WITH:
    # BUG FIX: raised from 0.30 → 0.45 to match CLIP's new stricter thresholds.
    # At 0.30, normal walking at EMA-smoothed ~0.28 regularly exceeded the gate
    # and created spurious alerts. Now requires a genuine confident detection.
    _MIN_ACTION_CONF = 0.45

═══════════════════════════════════════════════════════════════
PATCH 2: Ensure detections ARE being flushed to DB
═══════════════════════════════════════════════════════════════

The _db_flush_loop runs every 3 seconds and calls _save_detections_bulk.
If _save_detections_bulk_fixed is not properly assigned it silently does nothing.

VERIFY this line exists in the CameraStream class body (near line 2057):
    _save_detections_bulk = _save_detections_bulk_fixed

If it's missing, add it.

ALSO VERIFY _db_flush_loop is started in CameraStream.start():
    asyncio.create_task(self._db_flush_loop())

═══════════════════════════════════════════════════════════════
PATCH 3: WS broadcast includes ALL detections, not just alert ones
═══════════════════════════════════════════════════════════════

The WS broadcast (asyncio.create_task(_broadcast({...}))) already broadcasts
ALL processed detections. But Dashboard.tsx only adds to timeLogs when
handleDetectionFromWS is called, which requires `camera` to exist in camerasRef.

If cameras haven't loaded yet (camerasRef.current is empty at startup),
ALL WS detections are silently dropped because:
    const camera = camerasRef.current.find(c => c.id === cameraId);
    if (!camera) return;   ← drops the detection

The fix is in Dashboard.tsx (see Dashboard_v2_patches.ts).

═══════════════════════════════════════════════════════════════
PATCH 4: DB schema — ensure person_classification column exists
═══════════════════════════════════════════════════════════════

_save_detections_bulk_fixed stores person_classification on Detection rows.
If your DB schema doesn't have this column, the INSERT silently ignores it,
so analytics.py known count is always 0.

Run this migration on your SQLite DB:
    ALTER TABLE detection ADD COLUMN person_classification TEXT;
    ALTER TABLE detection ADD COLUMN person_string_id TEXT;
    ALTER TABLE detection ADD COLUMN person_name TEXT;
    ALTER TABLE detection ADD COLUMN face_image_url TEXT;
    ALTER TABLE detection ADD COLUMN detection_uuid TEXT;

Or add them to your SQLAlchemy model and run alembic migrate.
"""

# ─── The patches below are also provided as importable overrides ─────────────

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# These constants are unchanged from camera_manager_patches.py
# Only _THRESHOLDS are slightly adjusted downward to give the CLIP BUG-1 fix room
_THRESHOLDS = {512: 0.50, 128: 0.55, 59: 0.45}
_DEFAULT_THRESHOLD = 0.50
_AMBIGUITY_GAP     = 0.06
_DEDUP_LOOKBACK    = 60
_DEDUP_THRESHOLDS  = {512: 0.60, 128: 0.65, 59: 0.55}


def _emb_threshold(emb: np.ndarray) -> float:
    return _THRESHOLDS.get(len(emb), _DEFAULT_THRESHOLD)


def _dedup_threshold(emb: np.ndarray) -> float:
    return _DEDUP_THRESHOLDS.get(len(emb), 0.60)


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or len(a) != len(b):
        return 0.0
    n1, n2 = np.linalg.norm(a), np.linalg.norm(b)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(a, b) / (n1 * n2))


def _load_emb(raw: Optional[bytes]) -> Optional[np.ndarray]:
    if not raw:
        return None
    try:
        return np.frombuffer(raw, dtype=np.float32).copy()
    except Exception:
        return None


_MIN_MATCH_SCORE = 0.30


async def _identify_person_fixed(self, embedding, frame, bbox) -> Dict:
    if embedding is None:
        return {'classification': 'unknown', 'id': None, 'name': None,
                'confidence': 0.0, 'match_score': 0.0}

    threshold = _emb_threshold(embedding)
    known = await _load_known_cache(self)

    if not known:
        await _save_unknown_person_fixed(self, embedding, frame, bbox)
        return {'classification': 'unknown', 'id': None, 'name': None,
                'confidence': 0.5, 'match_score': 0.0}

    scores = []
    for p in known:
        p_emb = p.get('embedding')
        if p_emb is None:
            continue
        sim = _safe_cosine(embedding, p_emb)
        scores.append((sim, p))

    if not scores:
        await _save_unknown_person_fixed(self, embedding, frame, bbox)
        return {'classification': 'unknown', 'id': None, 'name': None,
                'confidence': 0.5, 'match_score': 0.0}

    scores.sort(key=lambda x: x[0], reverse=True)
    best_score, best_person = scores[0]

    if len(scores) >= 2:
        second_score = scores[1][0]
        ambiguous    = (best_score - second_score) < _AMBIGUITY_GAP
    else:
        ambiguous = False

    if best_score >= threshold and best_score >= _MIN_MATCH_SCORE and not ambiguous:
        return {
            'classification': best_person.get('classification', 'unknown'),
            'id':             best_person.get('person_id'),
            'name':           best_person.get('name'),
            'confidence':     best_score,
            'match_score':    best_score,
        }

    await _save_unknown_person_fixed(self, embedding, frame, bbox)
    return {'classification': 'unknown', 'id': None, 'name': None,
            'confidence': 1.0 - best_score, 'match_score': best_score}


async def _load_known_cache(self) -> List[Dict]:
    now = time.monotonic()
    if now - self._known_cache_ts < self._CACHE_TTL and self._known_cache:
        return self._known_cache
    try:
        from core.database import AsyncSessionLocal, Person
        from sqlalchemy import select
        async with AsyncSessionLocal() as db:
            persons = (await db.execute(select(Person))).scalars().all()
            cache = []
            for p in persons:
                emb = _load_emb(p.face_embedding)
                if emb is None:
                    continue
                cache.append({
                    'person_id':      p.person_id,
                    'name':           p.name,
                    'classification': p.classification,
                    'embedding':      emb,
                })
            type(self)._known_cache    = cache
            type(self)._known_cache_ts = now
            return cache
    except Exception as e:
        logger.error(f"[Identify] Cache load error: {e}")
        return []


async def _save_unknown_person_fixed(self, embedding, frame, bbox) -> None:
    try:
        from core.database import AsyncSessionLocal, UnknownPerson
        from sqlalchemy import select, desc

        threshold = _dedup_threshold(embedding)

        async with AsyncSessionLocal() as db:
            recent = (await db.execute(
                select(UnknownPerson)
                .order_by(desc(UnknownPerson.id))
                .limit(_DEDUP_LOOKBACK)
            )).scalars().all()

            for u in recent:
                u_emb = _load_emb(u.face_embedding)
                if u_emb is None:
                    continue
                sim = _safe_cosine(embedding, u_emb)
                if sim >= threshold:
                    logger.debug(f"[SaveUnknown] Dedup skip sim={sim:.3f}")
                    return

            h, w = frame.shape[:2]
            b = bbox
            x1 = max(0, int(b['x1'])); y1 = max(0, int(b['y1']))
            x2 = min(w, int(b['x2'])); y2 = min(h, int(b['y2']))
            crop = frame[y1:y2, x1:x2]

            face_path = None
            if crop.size > 0:
                from core.config import settings
                import cv2
                ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
                fname  = f"unknown_{ts_str}.jpg"
                fpath  = settings.FACES_UNKNOWN_DIR / fname
                cv2.imwrite(str(fpath), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                face_path = str(fpath)

            db.add(UnknownPerson(
                face_embedding  = embedding.tobytes(),
                face_image_path = face_path,
                detection_time  = datetime.utcnow(),
                cluster_id      = -1,
                is_clustered    = False,
            ))
            await db.commit()
            logger.debug(f"[SaveUnknown] Saved → {face_path}")

    except Exception as e:
        logger.error(f"[SaveUnknown] Error: {e}")


async def _save_detections_bulk_fixed(self, detections: List[Dict], camera_id: int) -> None:
    """Bulk-save detections to DB. Logs count for debugging."""
    if not detections:
        return
    try:
        from core.database import AsyncSessionLocal, Detection
        import uuid as _uuid
        async with AsyncSessionLocal() as db:
            rows = []
            for det in detections:
                bbox   = det.get('bbox', {})
                person = det.get('person', {})
                rows.append(Detection(
                    camera_id             = camera_id,
                    detection_uuid        = str(_uuid.uuid4()),
                    person_id             = None,
                    person_string_id      = person.get('id'),
                    person_name           = person.get('name'),
                    person_classification = person.get('classification', 'unknown'),
                    face_image_url        = det.get('face_image_url'),
                    bbox_x1               = float(bbox.get('x1', 0)),
                    bbox_y1               = float(bbox.get('y1', 0)),
                    bbox_x2               = float(bbox.get('x2', 0)),
                    bbox_y2               = float(bbox.get('y2', 0)),
                    confidence            = float(det.get('confidence', 0)),
                    action                = det.get('action', 'normal'),
                    action_confidence     = float(det.get('action_confidence') or 0),
                    zone                  = int(det.get('zone', 3)),
                    distance_from_camera  = float(det.get('distance', 10.0)),
                    timestamp             = datetime.utcnow(),
                ))
            db.add_all(rows)
            await db.commit()
            logger.debug(f"[SaveDetections] Saved {len(rows)} detections for camera_id={camera_id}")
    except Exception as e:
        logger.error(f"[SaveDetections] Bulk save error: {e}")
