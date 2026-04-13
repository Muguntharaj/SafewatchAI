"""
camera_service.py
─────────────────
This module has ONE job: hold the live CameraManager singleton so that
api/cameras.py (and any other module) can import it without circular deps.
 
    from core.camera_service import camera_manager
 
The actual object is injected at startup by main.py's lifespan handler.
"""
 
camera_manager = None   # injected by main.py lifespan