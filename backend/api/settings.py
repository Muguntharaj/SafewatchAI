"""
Settings API Endpoints
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

from core.config import settings

router = APIRouter()

class SystemSettings(BaseModel):
    device: str
    yolo_model: str
    clip_model: str
    face_recognition_threshold: float
    alert_level_1_actions: List[str]
    alert_level_2_actions: List[str]
    alert_level_3_actions: List[str]
    zone_1_distance: float
    zone_2_distance: float
    zone_3_distance: float

@router.get("/system", response_model=SystemSettings)
async def get_system_settings():
    """Get current system settings"""
    return {
        "device": settings.DEVICE,
        "yolo_model": settings.YOLO_MODEL,
        "clip_model": settings.CLIP_MODEL,
        "face_recognition_threshold": settings.FACE_RECOGNITION_THRESHOLD,
        "alert_level_1_actions": settings.ALERT_LEVEL_1_ACTIONS,
        "alert_level_2_actions": settings.ALERT_LEVEL_2_ACTIONS,
        "alert_level_3_actions": settings.ALERT_LEVEL_3_ACTIONS,
        "zone_1_distance": settings.ZONE_1_DISTANCE,
        "zone_2_distance": settings.ZONE_2_DISTANCE,
        "zone_3_distance": settings.ZONE_3_DISTANCE
    }

@router.get("/notification-config")
async def get_notification_config():
    """Get notification configuration status"""
    return {
        "email": {
            "configured": bool(settings.SMTP_USERNAME and settings.SMTP_PASSWORD),
            "recipients": settings.ALERT_EMAIL_RECIPIENTS
        },
        "sms": {
            "configured": bool(settings.TWILIO_ACCOUNT_SID and settings.TWILIO_AUTH_TOKEN),
            "recipients": settings.ALERT_SMS_RECIPIENTS
        },
        "whatsapp": {
            "configured": True,  # pywhatkit is always available
            "recipients": settings.WHATSAPP_RECIPIENTS
        }
    }

@router.get("/storage-info")
async def get_storage_info():
    """Get storage information"""
    import shutil
    
    total, used, free = shutil.disk_usage(settings.DATA_DIR)
    
    # Count files in each directory
    recordings_count = len(list(settings.RECORDINGS_DIR.glob('*.mp4')))
    alerts_count = len(list(settings.ALERTS_DIR.glob('*')))
    known_faces = len(list(settings.FACES_KNOWN_DIR.glob('*.jpg')))
    unknown_faces = len(list(settings.FACES_UNKNOWN_DIR.glob('*.jpg')))
    
    return {
        "disk": {
            "total_gb": round(total / (1024**3), 2),
            "used_gb": round(used / (1024**3), 2),
            "free_gb": round(free / (1024**3), 2),
            "usage_percent": round((used / total) * 100, 2)
        },
        "files": {
            "recordings": recordings_count,
            "alerts": alerts_count,
            "known_faces": known_faces,
            "unknown_faces": unknown_faces
        }
    }
