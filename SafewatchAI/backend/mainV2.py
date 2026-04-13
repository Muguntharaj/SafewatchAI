"""
SafeWatch AI — Main Backend Entry Point
Powered by Neelaminds Private Limited

FIXES:
  1. Duplicate StaticFiles mounts for /media/alerts caused startup crash (name conflict)
  2. faces_router was imported TWICE (from api.stream AND api.faces) — name collision
  3. CORSMiddleware added TWICE — second one silently overrode the first
  4. alerts_media_router imported from stream but stream.py router is the MJPEG router
     — removed; StaticFiles mounts handle /media/* correctly
"""
import os
import sys


os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|timeout;5000000'

# Fix Windows ProactorEventLoop MJPEG buffering — must be before asyncio import
if sys.platform == 'win32':
    import asyncio as _asyncio
    _asyncio.set_event_loop_policy(_asyncio.WindowsSelectorEventLoopPolicy())

import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from core.config import settings as config
from core.database import init_db, AsyncSessionLocal, Camera
from core.ai_engine import AIEngine
from core.camera_manager import CameraManager
from core.alert_manager import AlertManager
from core import camera_service
from sqlalchemy import select

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# ── WebSocket registry ────────────────────────────────────────────────────────
_ws_clients: Set[WebSocket] = set()


async def broadcast_to_frontend(payload: dict):
    dead: Set[WebSocket] = set()
    for ws in _ws_clients.copy():
        try:
            await ws.send_json(payload)
        except Exception:
            dead.add(ws)
    _ws_clients -= dead


# ── Seed / load cameras ───────────────────────────────────────────────────────
async def _load_cameras(mgr: CameraManager):
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(select(Camera))).scalars().all()

        if not rows:
            logger.info("⚠️  No cameras — seeding defaults")
            defaults = [
                dict(camera_id="cam-1", name="Main Entrance",
                     location="Office", stream_url="0", camera_type="USB"),
                dict(camera_id="cam-2", name="Parking Lot",
                     location="Parking",
                     stream_url="rtsp://admin:Admin_NM001@192.168.29.84:554/video/live?channel=1&subtype=0",
                     camera_type="IP"),
            ]
            for d in defaults:
                db.add(Camera(**d, status="active", fps=30, resolution="1920x1080"))
            await db.commit()
            rows = (await db.execute(select(Camera))).scalars().all()

        for cam in rows:
            if cam.status != "active":
                continue
            try:
                src = int(cam.stream_url) if cam.camera_type == "USB" else cam.stream_url
                await mgr.add_camera(cam.camera_id, src, cam.id)
                logger.info(f"✅ Camera started: {cam.camera_id}")
            except Exception as e:
                logger.error(f"❌ Failed to start {cam.camera_id}: {e}")


# ── Lifespan ──────────────────────────────────────────────────────────────────
# main.py - Update the lifespan and camera loading

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"🚀 Starting SafeWatch AI | device={config.DEVICE}")
    await init_db()

    ai_engine = AIEngine()
    await ai_engine.initialize()

    mgr = CameraManager(ai_engine)
    camera_service.camera_manager = mgr

    # Load cameras with performance logging
    await _load_cameras(mgr)
    
    # Print performance config on startup
    logger.info(f"⚙️ Performance Config:")
    logger.info(f"   AI Skip: DETECTION={config.DETECTION_FRAME_SKIP}, "
                f"ACTION={config.ACTION_FRAME_SKIP}, FACE={config.FACE_FRAME_SKIP}")
    logger.info(f"   Stream FPS: {config.STREAM_FPS}, JPEG Quality: {config.JPEG_QUALITY}")
    logger.info(f"   AI Workers: {config.AI_POOL_WORKERS}, IO Workers: {config.IO_POOL_WORKERS}")
    
    from api.analytics import schedule_daily_reset
    daily_reset_task = asyncio.create_task(schedule_daily_reset())
    logger.info("🎉 SafeWatch AI is live!")
    yield

    daily_reset_task.cancel()
    logger.info("🛑 Shutting down…")
    await camera_service.camera_manager.stop_all_cameras()
    await ai_engine.cleanup()
    logger.info("👋 Goodbye")

async def _load_cameras(mgr: CameraManager):
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(select(Camera))).scalars().all()

        if not rows:
            logger.info("⚠️  No cameras — seeding defaults")
            defaults = [
                dict(camera_id="cam-1", name="Main Entrance",
                     location="Office", stream_url="0", camera_type="USB"),
                dict(camera_id="cam-2", name="Parking Lot",
                     location="Parking",
                     stream_url="rtsp://192.168.1.64:554/stream",
                     camera_type="IP"),
            ]
            for d in defaults:
                db.add(Camera(**d, status="active", fps=12, resolution="640x480"))
            await db.commit()
            rows = (await db.execute(select(Camera))).scalars().all()

        for cam in rows:
            if cam.status != "active":
                continue
            try:
                src = int(cam.stream_url) if cam.camera_type == "USB" else cam.stream_url
                await mgr.add_camera(cam.camera_id, src, cam.id)
                logger.info(f"✅ Camera started: {cam.camera_id}")
            except Exception as e:
                logger.error(f"❌ Failed to start {cam.camera_id}: {e}")

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     logger.info(f"🚀 Starting SafeWatch AI | device={config.DEVICE}")
#     await init_db()

#     ai_engine = AIEngine()
#     await ai_engine.initialize()

#     mgr = CameraManager(ai_engine)
#     camera_service.camera_manager = mgr

#     # Initialise the alert_manager singleton
#     from core.alert_manager import alert_manager
#     _ = alert_manager

#     await _load_cameras(mgr)

#     # Start background tasks
#     from api.analytics import schedule_daily_reset
#     daily_reset_task = asyncio.create_task(schedule_daily_reset())
#     logger.info("🎉 SafeWatch AI is live!")
#     yield

#     daily_reset_task.cancel()
#     logger.info("🛑 Shutting down…")
#     await camera_service.camera_manager.stop_all_cameras()
#     await ai_engine.cleanup()
#     logger.info("👋 Goodbye")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SafeWatch AI API",
    description="Smart Security CCTV — Powered by Neelaminds",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — single middleware only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static mounts (order matters: more-specific paths first) ──────────────────
# /media/faces/unknown  → face crop images saved by camera_manager
# /media/faces/known    → registered face images
# /media/alerts         → alert snapshots + video clips
# /media/recordings     → continuous recording segments
app.mount("/media/faces/unknown",
          StaticFiles(directory=str(config.FACES_UNKNOWN_DIR), check_dir=True),
          name="faces_unknown")
app.mount("/media/faces/known",
          StaticFiles(directory=str(config.FACES_KNOWN_DIR), check_dir=True),
          name="faces_known")
app.mount("/media/alerts",
          StaticFiles(directory=str(config.ALERTS_DIR), check_dir=True),
          name="alerts_media")
app.mount("/media/recordings",
          StaticFiles(directory=str(config.RECORDINGS_DIR), check_dir=True),
          name="recordings")


# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)
    logger.info(f"🔌 WS connect total={len(_ws_clients)}")
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws)
        logger.info(f"🔌 WS disconnect total={len(_ws_clients)}")


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "operational", "version": "1.0.0", "device": config.DEVICE}


@app.get("/health")
async def health():
    mgr = camera_service.camera_manager
    return {"status": "healthy",
            "active_cameras": mgr.get_active_count() if mgr else 0,
            "database": "connected"}


# ── API Routers ───────────────────────────────────────────────────────────────
from api.cameras    import router as cameras_router
from api.alerts     import router as alerts_router
from api.detections import router as detections_router
from api.analytics  import router as analytics_router
from api.faces      import router as faces_router
from api.settings   import router as settings_router
from api.stream     import router as stream_router

app.include_router(cameras_router,    prefix="/api/cameras",    tags=["Cameras"])
app.include_router(alerts_router,     prefix="/api/alerts",     tags=["Alerts"])
app.include_router(detections_router, prefix="/api/detections", tags=["Detections"])
app.include_router(analytics_router,  prefix="/api/analytics",  tags=["Analytics"])
app.include_router(faces_router,      prefix="/api/faces",      tags=["Faces"])
app.include_router(settings_router,   prefix="/api/settings",   tags=["Settings"])
app.include_router(stream_router,     prefix="/api/stream",     tags=["Stream"])


# ── Global exception handler to prevent crashes from propagating ──────────────
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=config.HOST, port=config.PORT,
                reload=False, log_level="info", loop="asyncio")