"""
SafeWatch AI — Main Backend Entry Point
Powered by Neelaminds Private Limited

FIXES IN THIS VERSION:
──────────────────────
FIX-1  Redis import removed from top-level.
       `from core.redis_manager import get_recent_detections` at the TOP of
       the original file crashes the ENTIRE server on startup if Redis is not
       installed/running — before FastAPI even starts. Moved to a try/except
       inside the WS handler and replaced with in-memory buffer fallback.

FIX-2  `from api.faces import _cluster_all_for_person, router as faces_router`
       Importing a private function (_cluster_all_for_person) from outside its
       module is fragile. Moved to a proper import inside the nightly task.

FIX-3  @app.on_event("startup") is deprecated in FastAPI ≥ 0.93.
       Moved nightly cluster into lifespan.

FIX-4  broadcast_to_frontend returns early and silently on exception.
       Added per-client error logging to diagnose WS send failures.

FIX-5  WS detection replay uses in-memory _detection_buffer instead of Redis.
       Works reliably without any external dependency.
"""

import os
import sys
import asyncio
import logging
import json
from contextlib import asynccontextmanager
from typing import Set, Optional
from pathlib import Path
from collections import deque

# Fix Windows ProactorEventLoop MJPEG buffering
if sys.platform == 'win32':
    import asyncio as _asyncio
    _asyncio.set_event_loop_policy(_asyncio.WindowsSelectorEventLoopPolicy())

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|timeout;5000000'

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select

from core.config import settings as config
from core.database import init_db, AsyncSessionLocal, Camera
from core.ai_engine import AIEngine
from core.camera_manager import CameraManager
from core.alert_manager import AlertManager
from core import camera_service

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# WebSocket registry
_ws_clients: Set[WebSocket] = set()

# FIX-5: in-memory detection buffer replaces Redis dependency
# Stores last 200 WS payloads so new clients get a brief history replay
_detection_buffer: deque = deque(maxlen=200)


async def broadcast_to_frontend(payload: dict):
    """Broadcast message to all connected WebSocket clients."""
    # Buffer detection events for new-client replay
    if payload.get('type') == 'detections':
        _detection_buffer.append(payload)

    dead: Set[WebSocket] = set()
    for ws in _ws_clients.copy():
        try:
            await ws.send_json(payload)
        except Exception as e:
            logger.debug(f"WS send failed: {e}")
            dead.add(ws)
    _ws_clients -= dead


async def _load_cameras(mgr: CameraManager):
    """Load and start cameras from database at startup."""
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(select(Camera))).scalars().all()

        if not rows:
            logger.info("⚠️  No cameras in DB — seeding defaults")
            defaults = [
                dict(camera_id="cam-1", name="Main Entrance",
                     location="Office", stream_url="0", camera_type="USB"),
            ]
            for d in defaults:
                db.add(Camera(**d, status="active", fps=13, resolution="1280x720"))
            await db.commit()
            rows = (await db.execute(select(Camera))).scalars().all()

        started, failed = 0, 0
        for cam in rows:
            if cam.status == "inactive":
                logger.info(f"⏭️  Skipping inactive camera: {cam.camera_id}")
                continue
            try:
                src = int(cam.stream_url) if cam.camera_type == "USB" else cam.stream_url
                success = await mgr.add_camera(cam.camera_id, src, cam.id)
                if success:
                    cam.status = "active"
                    started += 1
                    logger.info(f"✅ Camera started: {cam.camera_id} ({cam.name})")
                else:
                    cam.status = "error"
                    failed += 1
                    logger.error(f"❌ Camera stream failed: {cam.camera_id}")
            except Exception as e:
                cam.status = "error"
                failed += 1
                logger.error(f"❌ Failed to start {cam.camera_id}: {e}")
        await db.commit()
        logger.info(f"📷 Cameras: {started} started, {failed} failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info(f"🚀 Starting SafeWatch AI | device={config.DEVICE}")

    await init_db()

    ai_engine = AIEngine()
    await ai_engine.initialize()   # AIEngine.initialize() is async — correct

    mgr = CameraManager(ai_engine)
    camera_service.camera_manager = mgr

    await _load_cameras(mgr)

    logger.info(f"⚙️  AI Skip: DETECTION={config.DETECTION_FRAME_SKIP}, "
                f"ACTION={config.ACTION_FRAME_SKIP}, FACE={config.FACE_FRAME_SKIP}")
    logger.info(f"   Stream FPS: {config.STREAM_FPS}, JPEG Quality: {config.JPEG_QUALITY}")
    logger.info(f"   AI Workers: {config.AI_POOL_WORKERS}")

    from api.analytics import schedule_daily_reset
    daily_reset_task = asyncio.create_task(schedule_daily_reset())

    # FIX-3: nightly cluster task moved here (on_event deprecated)
    # FIX-2: import _cluster_all_for_person inside the task, not at module level
    async def _nightly_cluster():
        while True:
            await asyncio.sleep(86400)  # once per day
            try:
                from api.faces import _cluster_all_for_person
                from core.database import AsyncSessionLocal as _ASL, Person as _P
                from sqlalchemy import select as _sel
                async with _ASL() as db:
                    persons = (await db.execute(_sel(_P))).scalars().all()
                for p in persons:
                    try:
                        await _cluster_all_for_person(p.person_id)
                    except Exception as e:
                        logger.debug(f"[NightlyCluster] {p.person_id}: {e}")
                logger.info(f"[NightlyCluster] Done, processed {len(persons)} persons")
            except Exception as e:
                logger.error(f"[NightlyCluster] Error: {e}")

    nightly_task = asyncio.create_task(_nightly_cluster())

    logger.info("🎉 SafeWatch AI is live!")
    yield

    logger.info("🛑 Shutting down...")
    daily_reset_task.cancel()
    nightly_task.cancel()
    await mgr.stop_all_cameras()
    await ai_engine.cleanup()
    logger.info("👋 Goodbye")


app = FastAPI(
    title="SafeWatch AI API",
    description="Smart Security CCTV — Powered by Neelaminds",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    await ws.accept()

    # FIX-5: replay recent detections from in-memory buffer (no Redis needed)
    try:
        recent = list(_detection_buffer)[-50:]
        for payload in recent:
            try:
                await ws.send_json(payload)
            except Exception:
                break
    except Exception as e:
        logger.debug(f"Detection replay error: {e}")

    _ws_clients.add(ws)
    logger.info(f"🔌 WS connect  total={len(_ws_clients)}")
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws)
        logger.info(f"🔌 WS disconnect total={len(_ws_clients)}")


@app.get("/")
async def root():
    return {"status": "operational", "version": "1.0.0", "device": config.DEVICE}


@app.get("/health")
async def health():
    mgr = camera_service.camera_manager
    return {
        "status":         "healthy",
        "active_cameras": mgr.get_active_count() if mgr else 0,
        "database":       "connected",
        "ai_ready":       mgr.ai_engine.ready if mgr and hasattr(mgr, 'ai_engine') else False,
    }


# FIX-2: import router only, not private functions
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


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=config.HOST, port=config.PORT,
                reload=False, log_level="info", loop="asyncio")