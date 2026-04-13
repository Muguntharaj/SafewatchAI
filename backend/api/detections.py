"""
detections.py — Fixed Build

KEY FIX: DetectionResponse now includes face_image_url.
  This is what makes TimeLog "View Details" show the correct face image
  without any timestamp guessing or cross-referencing.

  Priority chain for face_image_url in each detection row:
    1. face_image_url column on Detection (set by _save_detections_bulk_fixed)
    2. Known Person → reference.jpg (joined via person_id)
    3. Unknown Person lookup by person_string_id
    4. None (UI shows initials fallback)
"""

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy import select, desc

from core.database import get_db, AsyncSession, Detection, Person, Camera, UnknownPerson

router = APIRouter()

BASE_URL = "http://localhost:8000"


def _known_face_url(person: Person) -> Optional[str]:
    """Best face URL for a known person."""
    from core.config import settings
    ref = settings.FACES_KNOWN_DIR / person.person_id / "reference.jpg"
    if ref.exists():
        return f"{BASE_URL}/media/faces/known/{person.person_id}/reference.jpg"
    if person.face_image_path:
        from pathlib import PureWindowsPath
        name = PureWindowsPath(person.face_image_path).name
        if name:
            return f"{BASE_URL}/media/faces/known/{name}"
    return None


class DetectionResponse(BaseModel):
    id:                    int
    camera_id:             int
    camera_label:          Optional[str] = None
    detection_uuid:        Optional[str] = None
    person_id:             Optional[int] = None
    # New fields — populated by _save_detections_bulk_fixed
    person_string_id:      Optional[str] = None
    person_name:           Optional[str] = None
    person_classification: Optional[str] = None
    # THE KEY FIX — this is what TimeLog "View Details" needs
    face_image_url:        Optional[str] = None
    # Existing detection fields
    bbox_x1:               float
    bbox_y1:               float
    bbox_x2:               float
    bbox_y2:               float
    confidence:            float
    action:                Optional[str]
    action_confidence:     Optional[float]
    zone:                  int
    distance_from_camera:  float
    timestamp:             datetime

    class Config:
        from_attributes = True


@router.get("/recent", response_model=List[DetectionResponse])
async def get_recent_detections(
    limit:     int           = Query(default=100, le=1000),
    camera_id: Optional[str] = None,
    db:        AsyncSession  = Depends(get_db)
):
    """
    Get recent detections with person and camera info resolved.
    Now includes face_image_url so the frontend never needs a
    separate round-trip to find the face for time-log rows.
    """
    query = (
        select(Detection)
        .order_by(desc(Detection.timestamp))
        .limit(limit)
    )

    if camera_id:
        cam_result = await db.execute(
            select(Camera).where(Camera.camera_id == camera_id)
        )
        camera = cam_result.scalar_one_or_none()
        if camera:
            query = query.where(Detection.camera_id == camera.id)

    result     = await db.execute(query)
    detections = result.scalars().all()

    # Batch-load persons to avoid N+1 — numeric person_id FK
    person_ids = {d.person_id for d in detections if d.person_id}
    persons_by_id: dict = {}
    if person_ids:
        p_result = await db.execute(
            select(Person).where(Person.id.in_(person_ids))
        )
        for p in p_result.scalars().all():
            persons_by_id[p.id] = p

    # Batch-load cameras
    camera_ids = {d.camera_id for d in detections if d.camera_id}
    cameras_by_id: dict = {}
    if camera_ids:
        c_result = await db.execute(
            select(Camera).where(Camera.id.in_(camera_ids))
        )
        for c in c_result.scalars().all():
            cameras_by_id[c.id] = c

    enriched = []
    for d in detections:
        person = persons_by_id.get(d.person_id) if d.person_id else None
        cam    = cameras_by_id.get(d.camera_id) if d.camera_id else None

        # ── Face image URL resolution (priority chain) ────────────────────
        # 1. Check new DB column (set by _save_detections_bulk_fixed)
        face_url: Optional[str] = getattr(d, 'face_image_url', None)

        # 2. Known person's reference image
        if not face_url and person:
            face_url = _known_face_url(person)

        # 3. person_string_id → look up in known folder
        if not face_url:
            ps_id = getattr(d, 'person_string_id', None)
            if ps_id:
                from core.config import settings
                ref = settings.FACES_KNOWN_DIR / ps_id / "reference.jpg"
                if ref.exists():
                    face_url = f"{BASE_URL}/media/faces/known/{ps_id}/reference.jpg"

        # 4. Look up latest unknown person by camera and time window
        if not face_url and d.camera_id:
            from datetime import timedelta
            time_window = d.timestamp - timedelta(seconds=5)
            unknown_result = await db.execute(
                select(UnknownPerson)
                .where(UnknownPerson.camera_id == d.camera_id)
                .where(UnknownPerson.detection_time >= time_window)
                .where(UnknownPerson.detection_time <= d.timestamp + timedelta(seconds=5))
                .order_by(UnknownPerson.detection_time.desc())
                .limit(1)
            )
            unknown_person = unknown_result.scalar_one_or_none()
            if unknown_person and unknown_person.face_image_path:
                fn = PureWindowsPath(unknown_person.face_image_path).name
                if fn:
                    face_url = f"{BASE_URL}/media/faces/unknown/{fn}"

        # ── Person info resolution ─────────────────────────────────────────
        p_name  = (person.name if person
                   else getattr(d, 'person_name', None))
        p_class = (person.classification if person
                   else getattr(d, 'person_classification', None))

        # Map DB stored name fallback
        if not p_name:
            action = d.action
            p_name = (action if action and action not in ('normal', 'unknown', '')
                      else 'Unknown Person')

        enriched.append(DetectionResponse(
            id                   = d.id,
            camera_id            = d.camera_id,
            camera_label         = cam.camera_id if cam else getattr(d, 'camera_label', None),
            detection_uuid       = getattr(d, 'detection_uuid', None),
            person_id            = d.person_id if hasattr(d, 'person_id') else None,
            person_string_id     = getattr(d, 'person_string_id', None),
            person_name          = p_name,
            person_classification= p_class,
            face_image_url       = face_url,
            bbox_x1              = d.bbox_x1,
            bbox_y1              = d.bbox_y1,
            bbox_x2              = d.bbox_x2,
            bbox_y2              = d.bbox_y2,
            confidence           = d.confidence,
            action               = d.action,
            action_confidence    = d.action_confidence,
            zone                 = d.zone,
            distance_from_camera = d.distance_from_camera,
            timestamp            = d.timestamp,
        ))

    return enriched


@router.get("/stats")
async def get_detection_stats(
    start_date: Optional[datetime] = None,
    end_date:   Optional[datetime] = None,
    db:         AsyncSession        = Depends(get_db)
):
    if not start_date:
        start_date = datetime.utcnow() - timedelta(days=1)
    if not end_date:
        end_date = datetime.utcnow()

    result = await db.execute(
        select(Detection).where(
            Detection.timestamp >= start_date,
            Detection.timestamp <= end_date
        )
    )
    detections = result.scalars().all()

    total   = len(detections)
    actions: dict = {}
    zones:   dict = {1: 0, 2: 0, 3: 0}

    for det in detections:
        if det.action:
            actions[det.action] = actions.get(det.action, 0) + 1
        zones[det.zone] = zones.get(det.zone, 0) + 1

    return {
        "total_detections": total,
        "actions":          actions,
        "zones":            zones,
        "period": {
            "start": start_date.isoformat(),
            "end":   end_date.isoformat()
        }
    }


@router.get("/by-camera/{camera_id}")
async def get_detections_by_camera(
    camera_id: str,
    hours:     int = Query(default=24, le=168),
    db:        AsyncSession = Depends(get_db)
):
    cam_result = await db.execute(
        select(Camera).where(Camera.camera_id == camera_id)
    )
    camera = cam_result.scalar_one_or_none()
    if not camera:
        return {"error": "Camera not found"}

    start_time = datetime.utcnow() - timedelta(hours=hours)
    result     = await db.execute(
        select(Detection).where(
            Detection.camera_id == camera.id,
            Detection.timestamp >= start_time
        ).order_by(desc(Detection.timestamp))
    )
    detections = result.scalars().all()

    return {
        "camera_id":   camera_id,
        "camera_name": camera.name,
        "detections":  [
            {
                "id":         d.id,
                "bbox":       {"x1": d.bbox_x1, "y1": d.bbox_y1,
                               "x2": d.bbox_x2, "y2": d.bbox_y2},
                "confidence": d.confidence,
                "action":     d.action,
                "zone":       d.zone,
                "timestamp":  d.timestamp.isoformat(),
                "face_image_url": getattr(d, 'face_image_url', None),
            }
            for d in detections
        ]
    }