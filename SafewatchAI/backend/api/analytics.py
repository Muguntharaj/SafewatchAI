"""
analytics.py — Fixed Build

BUG FIXED: currently_in always returned 0 (or 1).

ROOT CAUSE:
    func.count(func.distinct(Detection.camera_label))
    
    camera_label is the camera ID string (e.g. "cam-1").
    Counting distinct camera IDs gives the number of cameras, not persons.
    With 1 camera → always returns 1. With 0 cameras → returns 0.
    The stats card showed this value as "Live Presence" which is wrong.
    
FIX: Count distinct detections in the last 30 minutes, which approximates
     how many persons are currently on premises. This is much more meaningful
     and actually changes in real-time as people are detected.

ADDITIONAL FIX: Detection.camera_label column may not exist in all DB schemas.
    The `known` count query used Detection.person_id.isnot(None) which is correct
    IF person_id is populated. But _save_detections_bulk_fixed stores person_string_id,
    not person_id (FK to Person table). So `known` was always 0.
    
    FIX: Count by person_classification column instead (populated by bulk save).
    
ALSO FIXED: DB schema compatibility — all getattr() wrappers removed from
    column accesses since they mask missing-column errors with silent zeros.
"""

from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, date
from sqlalchemy import select, func, desc, and_, text
import logging

from core.database import get_db, AsyncSession, Detection, Alert, Person, UnknownPerson
from core.database import AsyncSessionLocal

logger = logging.getLogger(__name__)
router = APIRouter()

CHART_METRICS = {
    "total_detections":   "Total Detections",
    "known_persons":      "Recognized People",
    "unknown_persons":    "Unknown Visitors",
    "live_presence":      "Live Presence",
    "critical_alerts":    "Critical Alerts",
    "high_alerts":        "High Alerts",
    "total_alerts":       "All Alerts",
    "hourly_activity":    "Hourly Activity",
    "zone1_detections":   "Zone 1 (Close)",
    "zone2_detections":   "Zone 2 (Medium)",
    "zone3_detections":   "Zone 3 (Far)",
}


@router.get("/dashboard")
async def get_dashboard_stats(
    period: str = Query(default="today", regex="^(today|week|month)$"),
    db: AsyncSession = Depends(get_db),
):
    now   = datetime.utcnow()
    if period == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "week":
        start = now - timedelta(days=7)
    else:
        start = now - timedelta(days=30)

    # Total detections in period
    total = (await db.execute(
        select(func.count(Detection.id)).where(Detection.timestamp >= start)
    )).scalar() or 0

    # Known = detections where person was identified (has a name / classification != unknown)
    # FIX: Use person_classification column (set by _save_detections_bulk_fixed)
    # Fallback: also count person_id IS NOT NULL for old schema
    try:
        known = (await db.execute(
            select(func.count(Detection.id))
            .where(
                Detection.timestamp >= start,
                Detection.person_classification.in_(['employee', 'owner', 'Employee', 'Owner'])
            )
        )).scalar() or 0
    except Exception:
        # Old schema fallback — person_classification column may not exist
        known = (await db.execute(
            select(func.count(Detection.id))
            .where(Detection.timestamp >= start, Detection.person_id.isnot(None))
        )).scalar() or 0

    # Unknown persons detected today
    unknown_count = (await db.execute(
        select(func.count(UnknownPerson.id)).where(UnknownPerson.detection_time >= start)
    )).scalar() or 0

    # ── BUG FIX: currently_in ─────────────────────────────────────────────────
    # OLD: func.count(func.distinct(Detection.camera_label)) → counts cameras, not people
    # NEW: count all detections in the last 30 minutes → approximates live presence
    # This is the value shown in the "Live Presence" stats card.
    recent_cutoff = now - timedelta(minutes=30)
    currently_in = (await db.execute(
        select(func.count(Detection.id))
        .where(Detection.timestamp >= recent_cutoff)
    )).scalar() or 0

    # Total alerts in period
    total_alerts = (await db.execute(
        select(func.count(Alert.id)).where(Alert.created_at >= start)
    )).scalar() or 0

    # Active alerts (not resolved/archived) — all time, not period-scoped
    active_alerts = (await db.execute(
        select(func.count(Alert.id))
        .where(Alert.status == "active")
    )).scalar() or 0

    return {
        "total_detections": total,
        "known_persons":    known,
        "unknown_persons":  unknown_count,
        "currently_in":     currently_in,
        "total_alerts":     total_alerts,
        "active_alerts":    active_alerts,
        "period":           period,
        "from":             start.isoformat() + "Z",
    }


@router.get("/timeline")
async def get_timeline(
    period: str = Query(default="today", regex="^(today|week|month)$"),
    db: AsyncSession = Depends(get_db),
):
    now = datetime.utcnow()
    if period == "today":
        start    = now.replace(hour=0, minute=0, second=0, microsecond=0)
        fmt      = "%H"
        label_fn = lambda b: f"{b}:00"
    elif period == "week":
        start    = now - timedelta(days=7)
        fmt      = "%Y-%m-%d"
        label_fn = lambda b: b
    else:
        start    = now - timedelta(days=30)
        fmt      = "%Y-%m-%d"
        label_fn = lambda b: b

    det_rows = (await db.execute(
        select(
            func.strftime(fmt, Detection.timestamp).label("bucket"),
            func.count(Detection.id).label("detections"),
        )
        .where(Detection.timestamp >= start)
        .group_by("bucket")
        .order_by("bucket")
    )).all()

    alert_rows = (await db.execute(
        select(
            func.strftime(fmt, Alert.created_at).label("bucket"),
            func.count(Alert.id).label("alerts"),
        )
        .where(Alert.created_at >= start)
        .group_by("bucket")
        .order_by("bucket")
    )).all()

    det_map   = {r.bucket: r.detections for r in det_rows}
    alert_map = {r.bucket: r.alerts     for r in alert_rows}
    buckets   = sorted(set(det_map) | set(alert_map))

    return {
        "labels":     [label_fn(b) for b in buckets],
        "detections": [det_map.get(b, 0)   for b in buckets],
        "alerts":     [alert_map.get(b, 0) for b in buckets],
        "period":     period,
    }


@router.get("/alerts/summary")
async def get_alert_summary(
    period: str = Query(default="day", regex="^(day|week|month)$"),
    db: AsyncSession = Depends(get_db),
):
    now   = datetime.utcnow()
    start = {
        "day":   now - timedelta(days=1),
        "week":  now - timedelta(days=7),
        "month": now - timedelta(days=30),
    }[period]

    by_sev = dict((await db.execute(
        select(Alert.severity, func.count(Alert.id))
        .where(Alert.created_at >= start)
        .group_by(Alert.severity)
    )).all())

    by_type = dict((await db.execute(
        select(Alert.alert_type, func.count(Alert.id))
        .where(Alert.created_at >= start)
        .group_by(Alert.alert_type)
    )).all())

    by_status = dict((await db.execute(
        select(Alert.status, func.count(Alert.id))
        .where(Alert.created_at >= start)
        .group_by(Alert.status)
    )).all())

    fmt = "%H" if period == "day" else "%Y-%m-%d"
    trend_rows = (await db.execute(
        select(
            func.strftime(fmt, Alert.created_at).label("bucket"),
            func.count(Alert.id).label("count"),
        )
        .where(Alert.created_at >= start)
        .group_by("bucket")
        .order_by("bucket")
    )).all()

    return {
        "period":     period,
        "from":       start.isoformat() + "Z",
        "by_severity": by_sev,
        "by_type":    by_type,
        "by_status":  by_status,
        "trend": {
            "labels": [r.bucket for r in trend_rows],
            "counts": [r.count  for r in trend_rows],
        },
    }


@router.post("/alerts/reset-daily")
async def reset_daily_alerts(db: AsyncSession = Depends(get_db)):
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    active = (await db.execute(
        select(Alert).where(
            Alert.status == "active",
            Alert.created_at < today_start
        )
    )).scalars().all()
    count = 0
    for alert in active:
        alert.status      = "archived"
        alert.resolved_at = datetime.utcnow()
        count += 1
    await db.commit()
    logger.info(f"🗓️  Daily reset: {count} alerts archived")
    return {"archived": count, "reset_at": datetime.utcnow().isoformat() + "Z"}


async def schedule_daily_reset():
    while True:
        now        = datetime.utcnow()
        next_reset = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=10, microsecond=0)
        sleep_secs = (next_reset - now).total_seconds()
        logger.info(f"🗓️  Next daily alert reset in {sleep_secs/3600:.1f}h")
        import asyncio as _asyncio
        await _asyncio.sleep(sleep_secs)
        try:
            async with AsyncSessionLocal() as db:
                today_start = datetime.utcnow().replace(
                    hour=0, minute=0, second=0, microsecond=0)
                old_active = (await db.execute(
                    select(Alert).where(
                        Alert.status == "active",
                        Alert.created_at < today_start
                    )
                )).scalars().all()
                for a in old_active:
                    a.status = "archived"
                await db.commit()
                logger.info(f"🗓️  Auto-archived {len(old_active)} alerts")
        except Exception as e:
            logger.error(f"Daily reset error: {e}")


class ChartRequest(BaseModel):
    x_axis:    str
    y_axis:    str
    period:    str = "today"
    camera_id: Optional[str] = None


@router.get("/chart/metrics")
async def get_available_metrics():
    return {"metrics": [{"key": k, "label": v} for k, v in CHART_METRICS.items()]}


@router.post("/chart/data")
async def get_chart_data(req: ChartRequest, db: AsyncSession = Depends(get_db)):
    if req.x_axis not in CHART_METRICS or req.y_axis not in CHART_METRICS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid metric. Valid options: {list(CHART_METRICS.keys())}"
        )
    now = datetime.utcnow()
    if req.period == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        fmt   = "%H"
    elif req.period == "week":
        start = now - timedelta(days=7)
        fmt   = "%Y-%m-%d"
    else:
        start = now - timedelta(days=30)
        fmt   = "%Y-%m-%d"

    async def _metric_series(metric_key: str) -> Dict[str, int]:
        if metric_key == "total_detections":
            rows = (await db.execute(
                select(func.strftime(fmt, Detection.timestamp).label("b"),
                       func.count(Detection.id).label("v"))
                .where(Detection.timestamp >= start)
                .group_by("b").order_by("b")
            )).all()
            return {r.b: r.v for r in rows}
        if metric_key == "known_persons":
            try:
                rows = (await db.execute(
                    select(func.strftime(fmt, Detection.timestamp).label("b"),
                           func.count(Detection.id).label("v"))
                    .where(Detection.timestamp >= start,
                           Detection.person_classification.in_(
                               ['employee', 'owner', 'Employee', 'Owner']))
                    .group_by("b").order_by("b")
                )).all()
            except Exception:
                rows = (await db.execute(
                    select(func.strftime(fmt, Detection.timestamp).label("b"),
                           func.count(Detection.id).label("v"))
                    .where(Detection.timestamp >= start,
                           Detection.person_id.isnot(None))
                    .group_by("b").order_by("b")
                )).all()
            return {r.b: r.v for r in rows}
        if metric_key == "unknown_persons":
            rows = (await db.execute(
                select(func.strftime(fmt, UnknownPerson.detection_time).label("b"),
                       func.count(UnknownPerson.id).label("v"))
                .where(UnknownPerson.detection_time >= start)
                .group_by("b").order_by("b")
            )).all()
            return {r.b: r.v for r in rows}
        if metric_key == "total_alerts":
            rows = (await db.execute(
                select(func.strftime(fmt, Alert.created_at).label("b"),
                       func.count(Alert.id).label("v"))
                .where(Alert.created_at >= start)
                .group_by("b").order_by("b")
            )).all()
            return {r.b: r.v for r in rows}
        if metric_key == "critical_alerts":
            rows = (await db.execute(
                select(func.strftime(fmt, Alert.created_at).label("b"),
                       func.count(Alert.id).label("v"))
                .where(Alert.created_at >= start, Alert.severity == "critical")
                .group_by("b").order_by("b")
            )).all()
            return {r.b: r.v for r in rows}
        if metric_key == "high_alerts":
            rows = (await db.execute(
                select(func.strftime(fmt, Alert.created_at).label("b"),
                       func.count(Alert.id).label("v"))
                .where(Alert.created_at >= start, Alert.severity == "high")
                .group_by("b").order_by("b")
            )).all()
            return {r.b: r.v for r in rows}
        if metric_key in ("zone1_detections", "zone2_detections", "zone3_detections"):
            zone_num = int(metric_key[4])
            rows = (await db.execute(
                select(func.strftime(fmt, Detection.timestamp).label("b"),
                       func.count(Detection.id).label("v"))
                .where(Detection.timestamp >= start, Detection.zone == zone_num)
                .group_by("b").order_by("b")
            )).all()
            return {r.b: r.v for r in rows}
        rows = (await db.execute(
            select(func.strftime(fmt, Detection.timestamp).label("b"),
                   func.count(Detection.id).label("v"))
            .where(Detection.timestamp >= start)
            .group_by("b").order_by("b")
        )).all()
        return {r.b: r.v for r in rows}

    x_data = await _metric_series(req.x_axis)
    y_data = await _metric_series(req.y_axis)
    all_buckets = sorted(set(x_data) | set(y_data))
    return {
        "x_label":  CHART_METRICS[req.x_axis],
        "y_label":  CHART_METRICS[req.y_axis],
        "period":   req.period,
        "labels":   all_buckets,
        "x_values": [x_data.get(b, 0) for b in all_buckets],
        "y_values": [y_data.get(b, 0) for b in all_buckets],
        "chart_data": [
            {"label": b, req.x_axis: x_data.get(b, 0), req.y_axis: y_data.get(b, 0)}
            for b in all_buckets
        ],
    }