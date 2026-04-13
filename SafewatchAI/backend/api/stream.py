"""
Stream API — serves MJPEG live feed for each camera.

FIXES:
  1. `from fastapi import ... logger` — logger is NOT exported by fastapi.
     Replaced with standard logging.getLogger(__name__).
  2. HEAD request support added for CCTVFeed health-check pings.
  3. MJPEG boundary matches what camera_manager.stream_camera yields ("frame").
"""

import logging
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
from core import camera_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.head("/{camera_id}")
async def stream_mjpeg_head(camera_id: str):
    """
    Health-check endpoint used by CCTVFeed.tsx ping loop.
    Returns 200 if the camera is running, 404 if not.
    """
    manager = camera_service.camera_manager
    if manager is None or camera_id not in manager.cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not running")
    return Response(status_code=200)


@router.get("/{camera_id}")
async def stream_mjpeg(camera_id: str):
    """MJPEG live stream for the given camera_id."""
    manager = camera_service.camera_manager

    if manager is None:
        raise HTTPException(status_code=503, detail="Camera manager not initialised")

    if camera_id not in manager.cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not running")

    return StreamingResponse(
        manager.stream_camera(camera_id),
        media_type='multipart/x-mixed-replace; boundary=frame',
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma":        "no-cache",
            "Expires":       "0",
            "Connection":    "keep-alive",
            "X-Accel-Buffering": "no",     # disable nginx buffering if behind proxy
        },
    )