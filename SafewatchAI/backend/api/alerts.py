"""
Alerts API Endpoints

FIXES:
  1. `from fastapi import Path, logger` → logger is NOT in fastapi; removed Path (unused)
  2. `from api import settings` → settings lives in core.config, not api
  3. _extract_alert_clip code block was accidentally pasted inside AlertResponse
     Pydantic class body → removed (that method belongs in alert_manager.py, already there)
  4. AlertResponse.camera_id typed as int but can be None → made Optional
"""

from fastapi import APIRouter, Depends, Query, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy import select, desc, func
from pathlib import PureWindowsPath, PurePosixPath
import logging

from core.database import AsyncSessionLocal, get_db, AsyncSession, Alert

BASE_URL = "http://localhost:8000"


def _alert_media_url(raw_path: Optional[str]) -> Optional[str]:
    """Convert any stored absolute path to a /media/alerts/ URL."""
    if not raw_path:
        return None
    if raw_path.startswith('http://') or raw_path.startswith('https://'):
        return raw_path
    # Extract filename handling Windows paths (C:\Users\...) on any OS
    name = PureWindowsPath(raw_path).name
    if not name or '\\' in name or '/' in name:
        name = PurePosixPath(raw_path).name or raw_path.replace('\\', '/').split('/')[-1]
    if not name:
        return None
    return f"{BASE_URL}/media/alerts/{name}"

logger = logging.getLogger(__name__)
router = APIRouter()


class AlertResponse(BaseModel):
    id: int
    alert_id: str
    alert_type: str
    alert_level: int
    severity: str
    camera_id: Optional[int] = None
    zone: int
    description: str
    action_detected: str
    video_path: Optional[str]       # raw stored path (may be Windows absolute)
    snapshot_path: Optional[str]    # raw stored path
    video_url: Optional[str] = None     # proper HTTP URL for frontend
    snapshot_url: Optional[str] = None  # proper HTTP URL for frontend
    status: str
    email_sent: bool
    sms_sent: bool
    whatsapp_sent: bool
    created_at: datetime

    class Config:
        from_attributes = True


@router.get("/list", response_model=List[AlertResponse])
async def list_alerts(
    limit: int = Query(default=50, le=500),
    status: Optional[str] = None,
    alert_level: Optional[int] = None,
    date: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    query = select(Alert).order_by(desc(Alert.created_at))
    if status:
        query = query.where(Alert.status == status)
    if alert_level:
        query = query.where(Alert.alert_level == alert_level)
    if date:
        try:
            start = datetime.strptime(date, "%Y-%m-%d")
            end = start.replace(hour=23, minute=59, second=59)
            query = query.where(Alert.created_at >= start, Alert.created_at <= end)
        except ValueError:
            pass
    query = query.limit(limit)
    result  = await db.execute(query)
    alerts  = result.scalars().all()
    # Enrich each alert with proper HTTP URLs for video and snapshot
    enriched = []
    for a in alerts:
        r = AlertResponse.model_validate(a)
        r.video_url    = _alert_media_url(a.video_path)
        r.snapshot_url = _alert_media_url(a.snapshot_path)
        enriched.append(r)
    return enriched


@router.get("/stats/summary")
async def get_alert_stats(
    days: int = Query(default=7, le=365),
    db: AsyncSession = Depends(get_db),
):
    start_date = datetime.utcnow() - timedelta(days=days)

    total = (await db.execute(
        select(func.count(Alert.id)).where(Alert.created_at >= start_date)
    )).scalar() or 0

    by_level = dict((await db.execute(
        select(Alert.alert_level, func.count(Alert.id))
        .where(Alert.created_at >= start_date)
        .group_by(Alert.alert_level)
    )).all())

    by_status = dict((await db.execute(
        select(Alert.status, func.count(Alert.id))
        .where(Alert.created_at >= start_date)
        .group_by(Alert.status)
    )).all())

    by_type = dict((await db.execute(
        select(Alert.alert_type, func.count(Alert.id))
        .where(Alert.created_at >= start_date)
        .group_by(Alert.alert_type)
    )).all())

    return {
        "total_alerts": total,
        "by_level": by_level,
        "by_status": by_status,
        "by_type": by_type,
        "period_days": days,
    }


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(alert_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Alert).where(Alert.alert_id == alert_id))
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert


@router.post("/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Alert).where(Alert.alert_id == alert_id))
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.status = "acknowledged"
    alert.acknowledged_at = datetime.utcnow()
    await db.commit()
    return {"message": "Alert acknowledged"}


@router.post("/{alert_id}/resolve")
async def resolve_alert(alert_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Alert).where(Alert.alert_id == alert_id))
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.status = "resolved"
    alert.resolved_at = datetime.utcnow()
    await db.commit()
    return {"message": "Alert resolved"}