"""
Camera Management API

FIX: True camera deletion — DELETE /api/cameras/{camera_id} now:
  1. Stops the stream
  2. Removes the Camera row from the database (so camera_id is freed)
  3. Returns 200 so the frontend can use the same camera_id again immediately

Previously the UI was calling POST /stop (which keeps the DB record) when
the user clicked "Delete", so the camera_id remained in the DB and trying to
add the same ID returned "already exists".

Also:  cameras.py imported camera_manager at MODULE LOAD TIME via
         from core.camera_service import camera_manager
       At startup that variable is None (set later by lifespan).
       Fix: import camera_service and call camera_service.camera_manager at
       request time so we always get the live instance.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy import select

from core.database import get_db, AsyncSession, Camera
from core import camera_service          # access .camera_manager at request time

mgr = camera_service.camera_manager

router = APIRouter()


class CameraCreate(BaseModel):
    camera_id:   str
    name:        str
    location:    str
    stream_url:  str
    camera_type: str              # IP / WiFi / USB
    fps:         Optional[int]   = 30
    resolution:  Optional[str]   = "1920x1080"


class CameraResponse(BaseModel):
    id:          int
    camera_id:   str
    name:        str
    location:    str
    stream_url:  str
    camera_type: str
    status:      str
    fps:         int
    resolution:  str

    class Config:
        from_attributes = True


@router.post("/add", response_model=CameraResponse)
async def add_camera(camera: CameraCreate, db: AsyncSession = Depends(get_db)):
    mgr = camera_service.camera_manager
    if mgr is None:
        raise HTTPException(status_code=503, detail="Camera manager not ready yet")

    # Check DB for duplicate camera_id
    existing = (await db.execute(
        select(Camera).where(Camera.camera_id == camera.camera_id)
    )).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=400,
                            detail=f"Camera ID '{camera.camera_id}' already exists in DB. "
                                   f"Use a different ID or delete the existing record first.")

    # Check in-memory manager
    if camera.camera_id in mgr.cameras:
        raise HTTPException(status_code=400,
                            detail=f"Camera '{camera.camera_id}' is currently running.")

    db_camera = Camera(
        camera_id   = camera.camera_id,
        name        = camera.name,
        location    = camera.location,
        stream_url  = camera.stream_url,
        camera_type = camera.camera_type,
        fps         = camera.fps,
        resolution  = camera.resolution,
        status      = "active",
    )
    db.add(db_camera)
    await db.commit()
    await db.refresh(db_camera)

    # Start stream
    source  = int(camera.stream_url) if camera.camera_type == "USB" else camera.stream_url
    started = await mgr.add_camera(camera.camera_id, source, db_camera.id)
    if not started:
        # DELETE the DB record so the user can retry with the same camera_id.
        # Keeping it as 'error' causes "already exists" on the next add attempt.
        await db.delete(db_camera)
        await db.commit()
        raise HTTPException(
            status_code=503,
            detail=(
                f"Could not open stream for '{camera.camera_id}'. "
                f"The camera was NOT saved — you can try again with the same ID.\n"
                f"URL tried: {camera.stream_url}\n"
                f"Tips: ✓ Check the URL/IP is correct  "
                f"✓ Camera must be on the same network  "
                f"✓ USB index: 0 for first USB camera"
            ),
        )

    return db_camera


@router.get("/list", response_model=List[CameraResponse])
async def list_cameras(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Camera))
    return result.scalars().all()


@router.get("/{camera_id}", response_model=CameraResponse)
async def get_camera(camera_id: str, db: AsyncSession = Depends(get_db)):
    camera = (await db.execute(
        select(Camera).where(Camera.camera_id == camera_id)
    )).scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


@router.delete("/{camera_id}")
async def delete_camera(camera_id: str, db: AsyncSession = Depends(get_db)):
    """
    FULL DELETE — stops stream AND removes the DB record so the
    camera_id can be reused immediately.
    """
    mgr    = camera_service.camera_manager
    camera = (await db.execute(
        select(Camera).where(Camera.camera_id == camera_id)
    )).scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    # 1. Stop stream safely
    if mgr:
        try:
            await mgr.remove_camera(camera_id)
        except Exception as e:
            pass  # already stopped — continue with DB delete

    # 2. Delete DB record — camera_id is now free to reuse
    await db.delete(camera)
    await db.commit()

    return {"message": f"Camera '{camera_id}' deleted. ID is now free to reuse."}


@router.post("/{camera_id}/stop")
async def stop_camera(camera_id: str, db: AsyncSession = Depends(get_db)):
    """
    Stop stream only — keeps the DB record.
    Use this when you want to pause a camera temporarily.
    Use DELETE to completely remove and free the camera_id.
    """
    mgr    = camera_service.camera_manager
    camera = (await db.execute(
        select(Camera).where(Camera.camera_id == camera_id)
    )).scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    if mgr:
        try:
            await mgr.remove_camera(camera_id)
        except Exception:
            pass

    camera.status = "inactive"
    await db.commit()
    return {"message": f"Camera '{camera_id}' stream stopped (DB record kept)"}


@router.post("/{camera_id}/start")
async def start_camera(camera_id: str, db: AsyncSession = Depends(get_db)):
    """Re-start a stopped camera stream."""
    mgr = camera_service.camera_manager
    if mgr is None:
        raise HTTPException(status_code=503, detail="Camera manager not ready")

    camera = (await db.execute(
        select(Camera).where(Camera.camera_id == camera_id)
    )).scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    if camera_id in mgr.cameras:
        return {"message": f"Camera '{camera_id}' is already running"}

    source  = int(camera.stream_url) if camera.camera_type == "USB" else camera.stream_url
    started = await mgr.add_camera(camera.camera_id, source, camera.id)
    if not started:
        raise HTTPException(status_code=503, detail="Failed to open camera stream")

    camera.status = "active"
    await db.commit()
    return {"message": f"Camera '{camera_id}' started"}