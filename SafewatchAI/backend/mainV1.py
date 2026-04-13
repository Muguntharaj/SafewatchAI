"""
main.py  — full file (replaces camera_service.py as the entry point)

Paste this as your main.py.  The old camera_service.py is now just a
one-liner stub that holds the camera_manager reference.
"""
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from core.config import settings
from core.database import init_db
from core.ai_engine import AIEngine
from core.camera_manager import CameraManager
from core import camera_service
from core.database import AsyncSessionLocal, Camera
from sqlalchemy import select

logger = logging.getLogger(__name__)

# ── WebSocket connection registry ─────────────────────────────────────────────
_ws_clients: Set[WebSocket] = set()

async def broadcast_to_frontend(payload: dict):
    """Broadcast a JSON payload to every connected WebSocket client."""
    dead = set()
    for ws in _ws_clients.copy():
        try:
            await ws.send_json(payload)
        except Exception:
            dead.add(ws)
    _ws_clients -= dead


# ── Lifespan: startup / shutdown ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    await init_db()

    ai_engine = AIEngine()
    await ai_engine.initialize()

    mgr = CameraManager(ai_engine)
    camera_service.camera_manager = mgr          # inject for cameras.py / faces.py

    # Re-start cameras that were active at last shutdown
    async with AsyncSessionLocal() as db:
        cams = (await db.execute(
            select(Camera).where(Camera.status == 'active')
        )).scalars().all()
        for cam in cams:
            src = int(cam.stream_url) if cam.camera_type == 'USB' else cam.stream_url
            await mgr.add_camera(cam.camera_id, src, cam.id)

    logger.info("✅ SafeWatch AI started")

    yield

    # SHUTDOWN
    await camera_service.camera_manager.stop_all_cameras()
    await ai_engine.cleanup()
    logger.info("🛑 SafeWatch AI stopped")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="SafeWatch AI", lifespan=lifespan)

# CORS — allow the React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file mounts — lets the browser load face images and alert clips
app.mount(
    "/media/faces",
    StaticFiles(directory=str(settings.FACES_UNKNOWN_DIR.parent)),
    name="faces"
)
app.mount(
    "/media/alerts",
    StaticFiles(directory=str(settings.ALERTS_DIR)),
    name="alerts"
)


# ── WebSocket endpoint ────────────────────────────────────────────────────────
@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)
    logger.info(f"WS client connected — total: {len(_ws_clients)}")
    try:
        while True:
            # Keep the connection alive; actual data is pushed via broadcast_to_frontend()
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws)
        logger.info(f"WS client disconnected — total: {len(_ws_clients)}")


# ── API routers ───────────────────────────────────────────────────────────────
from api.cameras    import router as cameras_router
from api.alerts     import router as alerts_router
from api.detections import router as detections_router
from api.analytics  import router as analytics_router
from api.faces      import router as faces_router
from api.settings   import router as settings_router
from api.stream     import router as stream_router

app.include_router(cameras_router,    prefix="/api/cameras",    tags=["cameras"])
app.include_router(alerts_router,     prefix="/api/alerts",     tags=["alerts"])
app.include_router(detections_router, prefix="/api/detections", tags=["detections"])
app.include_router(analytics_router,  prefix="/api/analytics",  tags=["analytics"])
app.include_router(faces_router,      prefix="/api/faces",      tags=["faces"])
app.include_router(settings_router,   prefix="/api/settings",   tags=["settings"])
app.include_router(stream_router,     prefix="/api/stream",     tags=["stream"])

# """
# SafeWatch AI - Main Backend Server
# Powered by Neelaminds Private Limited
# """
# import json
# import numpy as np
# from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import StreamingResponse, FileResponse
# from contextlib import asynccontextmanager
# import uvicorn
# import asyncio
# from typing import List, Dict
# import logging


# from api import cameras, detections, alerts, faces, analytics, settings,stream
# from core.database import init_db
# from core.ai_engine import AIEngine
# from core.camera_manager import CameraManager
# from core.alert_manager import AlertManager
# from core.config import settings as config
# from core import camera_service

# from sqlalchemy import select
# from core.database import AsyncSessionLocal, Camera

# active_ws_clients: list[WebSocket] = []

# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # Global instances
# ai_engine: AIEngine = None
# camera_manager: CameraManager = None
# alert_manager: AlertManager = None






# def serialize_detections(data):
#     """Convert detection data to JSON-safe format"""
#     result = {}
#     for cam_id, dets in data.items():
#         result[cam_id] = []
#         for d in dets:
#             safe = {
#                 'bbox': d.get('bbox', {}),
#                 'confidence': float(d.get('confidence', 0)),
#                 'action': d.get('action', 'unknown'),
#                 'action_confidence': float(d.get('action_confidence', 0) or 0),
#                 'zone': int(d.get('zone', 3)),
#                 'distance': float(d.get('distance', 0)),
#                 'person': d.get('person', {}),
#             }
#             result[cam_id].append(safe)
#     return result


# async def load_cameras_from_db():
#     async with AsyncSessionLocal() as db:

#         result = await db.execute(select(Camera))
#         cameras = result.scalars().all()

#         # If database empty → create default cameras
#         if len(cameras) == 0:
#             logger.info("⚠️ No cameras found in DB. Creating default cameras...")

#             default_cameras = [
#                 {
#                     "camera_id": "cam-1",
#                     "name": "Main Entrance",
#                     "location": "Office",
#                     "stream_url": "0",
#                     "camera_type": "USB"
#                 },
#                 {
#                     "camera_id": "cam-2",
#                     "name": "Parking Lot",
#                     "location": "Parking",
#                     "stream_url": "rtsp://192.168.1.64:554/stream",
#                     "camera_type": "IP"
#                 }
#             ]

#             for cam in default_cameras:
#                 new_cam = Camera(
#                     camera_id=cam["camera_id"],
#                     name=cam["name"],
#                     location=cam["location"],
#                     stream_url=cam["stream_url"],
#                     camera_type=cam["camera_type"],
#                     status="active",
#                     fps=30,
#                     resolution="1920x1080"
#                 )

#                 db.add(new_cam)

#             await db.commit()

#             result = await db.execute(select(Camera))
#             cameras = result.scalars().all()

#         # Start cameras
#         for cam in cameras:

#             try:

#                 if cam.camera_type == "USB":
#                     source = int(cam.stream_url)
#                 else:
#                     source = cam.stream_url

#                 await camera_manager.add_camera(
#                     cam.camera_id,
#                     source,
#                     cam.id
#                 )

#                 logger.info(f"✅ Auto started camera: {cam.camera_id}")

#             except Exception as e:
#                 logger.error(f"❌ Failed to start camera {cam.camera_id}: {e}")
                
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Initialize and cleanup resources"""
#     global ai_engine, camera_manager, alert_manager
    
#     logger.info("🚀 Starting SafeWatch AI Backend...")
#     logger.info(f"🖥️  Device: {config.DEVICE}")
    
#     # Initialize database
#     await init_db()
#     logger.info("✅ Database initialized")
    
#     # Initialize AI Engine (YOLO, CLIP, FaceNet)
#     ai_engine = AIEngine()
#     await ai_engine.initialize()
#     logger.info("✅ AI Models loaded")
    
#     # Initialize Camera Manager
#     camera_manager = CameraManager(ai_engine)
#     camera_service.camera_manager = camera_manager
#     logger.info("✅ Camera Manager ready")
    
#     # Auto load cameras from database
#     await load_cameras_from_db()
#     logger.info("✅ Cameras loaded from database")
    
#     # Initialize Alert Manager
#     alert_manager = AlertManager()
#     logger.info("✅ Alert Manager initialized")
    
#     logger.info("🎉 SafeWatch AI Backend is ready!")
    
#     yield
    
#     # Cleanup
#     logger.info("🛑 Shutting down...")
#     await camera_manager.stop_all_cameras()
#     await ai_engine.cleanup()
#     logger.info("👋 Shutdown complete")

# # Create FastAPI app
# app = FastAPI(
#     title="SafeWatch AI API",
#     description="Smart Security CCTV System with AI-Powered Detection",
#     version="1.0.0",
#     lifespan=lifespan
# )


# from fastapi.staticfiles import StaticFiles

# app.mount('/media/faces',
#     StaticFiles(directory=str(settings.DATA_DIR / 'faces')), name='faces')
# app.mount('/media/alerts',
#     StaticFiles(directory=str(settings.ALERTS_DIR)), name='alerts')

# # Frontend image URL then becomes:
# # http://localhost:8000/media/faces/unknown/unknown_20240101_120000.jpg
# # http://localhost:8000/media/alerts/ALERT_xxx_snapshot.jpg
# # CORS Middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Update for production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Include routers
# app.include_router(cameras.router, prefix="/api/cameras", tags=["Cameras"])
# app.include_router(detections.router, prefix="/api/detections", tags=["Detections"])
# app.include_router(alerts.router, prefix="/api/alerts", tags=["Alerts"])
# app.include_router(faces.router, prefix="/api/faces", tags=["Face Recognition"])
# app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])
# app.include_router(settings.router, prefix="/api/settings", tags=["Settings"])
# app.include_router(stream.router, prefix="/api/stream",tags =["stream"])
# @app.get("/")



# async def root():
#     """Health check endpoint"""
#     return {
#         "status": "operational",
#         "message": "SafeWatch AI Backend - Powered by Neelaminds",
#         "version": "1.0.0",
#         "device": config.DEVICE,
#         "models_loaded": ai_engine.is_ready() if ai_engine else False
#     }

# @app.get("/health")
# async def health_check():
#     """Detailed health check"""
#     return {
#         "status": "healthy",
#         "ai_engine": ai_engine.get_status() if ai_engine else {},
#         "active_cameras": camera_manager.get_active_count() if camera_manager else 0,
#         "database": "connected"
#     }

# @app.websocket("/ws/detections")
# async def websocket_detections(websocket: WebSocket):
#     """WebSocket for real-time detection streaming"""
#     await websocket.accept()
#     logger.info("🔌 WebSocket client connected")
    
    
    
#     try:
#         while True:
#             # Stream real-time detections
#             if camera_manager:
#                 detections = await camera_manager.get_latest_detections()
#                 await websocket.send_json(serialize_detections(detections))
            
#             await asyncio.sleep(0.1)  # 10 FPS update rate
            
#     except WebSocketDisconnect:
#         logger.info("🔌 WebSocket client disconnected")
#     except Exception as e:
#         logger.error(f"WebSocket error: {e}")

# @app.websocket("/ws/camera/{camera_id}")
# async def websocket_camera_stream(websocket: WebSocket, camera_id: str):
#     """WebSocket for individual camera video streaming"""
#     await websocket.accept()
#     logger.info(f"🎥 Camera stream started: {camera_id}")
    
#     if camera_id not in camera_manager.cameras:
#         await websocket.send_json({"error": f"Camera {camera_id} not running"})
#         return
    
#     try:
#         async for frame in camera_manager.stream_camera(camera_id):
#             await websocket.send_bytes(frame)
            
#     except WebSocketDisconnect:
#         logger.info(f"🎥 Camera stream stopped: {camera_id}")
#     except Exception as e:
#         logger.error(f"Camera stream error: {e}")
        

# @app.websocket('/ws/live')
# async def websocket_live(ws: WebSocket):
#     await ws.accept()
#     active_ws_clients.append(ws)
#     try:
#         while True:
#             await asyncio.sleep(30)  # keep-alive ping
#     except WebSocketDisconnect:
#         active_ws_clients.remove(ws)

# async def broadcast_to_frontend(payload: dict):
#     dead = []
#     for ws in active_ws_clients:
#         try:
#             await ws.send_text(json.dumps(payload))
#         except:
#             dead.append(ws)
#     for d in dead:
#         active_ws_clients.remove(d)

# # @app.on_event("startup")
# # async def startup_event():
# #     await ai_engine.initialize()

# if __name__ == "__main__":
#     uvicorn.run(
#         "main:app",
#         host=config.HOST,
#         port=config.PORT,
#         reload=False,
#         log_level="info"
#     )
