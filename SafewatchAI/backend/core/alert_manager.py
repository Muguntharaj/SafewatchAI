"""
Alert Manager — SafeWatch AI

FIXES:
  1. PRE-ALERT CLIP uses ring buffer (10 s before event) from CameraStream.
     Uses mp4v → MJPG → avi codec chain so it works without libavcodec.
  2. Broadcasts alert with video_url (HTTP path) so frontend gets the clip URL.
  3. alert_manager is a module-level singleton.
  4. Snapshot saved with alert overlay.
  5. Email/SMS notifications gated on SMTP/Twilio config availability.
"""

import asyncio
import concurrent.futures as _cf
import smtplib

# FIX: dedicated IO pool for alert clip writes — avoids blocking the asyncio
# default thread pool (None) which can interact with the event loop.
_ALERT_IO_POOL = _cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix='sw_alert_io')
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from typing import Optional, Dict
from datetime import datetime
import logging
from pathlib import Path
import cv2
import numpy as np
import subprocess

from core.config import settings
from core.database import AsyncSessionLocal, Alert

logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"


class AlertManager:
    def __init__(self):
        self.last_alert_time: Dict[str, datetime] = {}
        self._recent_alerts: list = []  # For similarity dedup
        self._DEDUP_WINDOW = 60  # seconds to group similar alerts

    async def create_alert(
        self,
        alert_type:      str,
        camera_id:       str,
        zone:            int,
        description:     str,
        action_detected: str,
        frame:           Optional[np.ndarray] = None,
        video_path:      Optional[str]        = None,
    ) -> Dict:
        now = datetime.utcnow()
        self._recent_alerts = [
            a for a in self._recent_alerts
            if (now - a['time']).total_seconds() < self._DEDUP_WINDOW
        ]
        similar = [
            a for a in self._recent_alerts
            if a['camera_id'] == camera_id and a['action'] == action_detected
        ]
        if similar:
            logger.debug(f"Duplicate alert suppressed: {action_detected} on {camera_id}")
            return {'alert_id': similar[0]['alert_id'], 'duplicate': True}

        alert_level = self._determine_alert_level(action_detected)
        severity    = self._get_severity(alert_level)
        alert_id    = f"ALERT_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"

        snapshot_path   = await self._save_snapshot(alert_id, frame) if frame is not None else None
        # PRIMARY: write 10-second pre-alert clip from ring buffer
        video_clip_path = await self._write_pre_alert_clip(alert_id, camera_id)
        # FALLBACK: ffmpeg on current recording segment
        if not video_clip_path:
            video_clip_path = await self._extract_via_ffmpeg(alert_id, camera_id)

        async with AsyncSessionLocal() as db:
            from sqlalchemy import select
            from core.database import Camera
            cam_obj   = (await db.execute(
                select(Camera).where(Camera.camera_id == camera_id)
            )).scalar_one_or_none()
            cam_db_id = cam_obj.id if cam_obj else None
            cam_name  = cam_obj.name if cam_obj else camera_id

            db_alert = Alert(
                alert_id        = alert_id,
                alert_type      = alert_type,
                alert_level     = alert_level,
                severity        = severity,
                camera_id       = cam_db_id,
                zone            = zone,
                description     = description,
                action_detected = action_detected,
                video_path      = video_clip_path,
                snapshot_path   = snapshot_path,
                status          = 'active',
                created_at      = datetime.utcnow(),
            )
            db.add(db_alert)
            await db.commit()
            await db.refresh(db_alert)

        await self._send_notifications(db_alert, snapshot_path, video_clip_path)

        # Build proper HTTP URLs for the frontend
        video_url    = self._media_url(video_clip_path, "alerts")
        snapshot_url = self._media_url(snapshot_path,   "alerts")

        logger.info(
            f"🚨 Alert: {alert_id} level={alert_level} "
            f"clip={'✅ ' + Path(video_clip_path).name if video_clip_path else '❌'}"
        )

        self._recent_alerts.append({
            'alert_id': alert_id, 'camera_id': camera_id,
            'action': action_detected, 'time': now
        })

        # Broadcast to all WebSocket clients
        try:
            from main import broadcast_to_frontend
            await broadcast_to_frontend({
                'type': 'alert',
                'alert': {
                    'alert_id':        alert_id,
                    'alert_type':      alert_type,
                    'alert_level':     alert_level,
                    'severity':        severity,
                    'camera_id':       camera_id,
                    'camera_name':     cam_name,
                    'zone':            zone,
                    'description':     description,
                    'action_detected': action_detected,
                    'video_path':      video_clip_path,
                    'snapshot_path':   snapshot_path,
                    'video_url':       video_url,       # FIX 2: proper HTTP URL
                    'snapshot_url':    snapshot_url,    # FIX 2: proper HTTP URL
                    'status':          'active',
                    'created_at':      datetime.utcnow().isoformat() + 'Z',
                },
            })
        except Exception as e:
            logger.debug(f"WS alert broadcast skipped: {e}")

        return {
            'alert_id':    alert_id,
            'alert_level': alert_level,
            'severity':    severity,
            'video_url':   video_url,
            'timestamp':   datetime.utcnow().isoformat(),
        }

    def _media_url(self, path: Optional[str], subdir: str) -> Optional[str]:
        """Convert an absolute path to a /media/{subdir}/{filename} URL."""
        if not path:
            return None
        if str(path).startswith('http://') or str(path).startswith('https://'):
            return path
        name = Path(path).name
        return f"{BASE_URL}/media/{subdir}/{name}" if name else None

    def _determine_alert_level(self, action: str) -> int:
        a = action.lower()
        # Level 1 — CRITICAL: immediate danger, weapon, fire, physical violence
        CRITICAL_ACTIONS = {
            "fighting", "weapon_detected", "weapon_grip", "violence",
            "fire", "falling", "break_in", "weapon",
        }
        if any(w in a for w in CRITICAL_ACTIONS):
            return 1
        # Level 2 — HIGH: theft, trespassing, running, vandalism
        HIGH_ACTIONS = {
            "running", "shouting", "vandalism", "theft", "trespassing",
            "stealing", "crowding", "distress",
        }
        if any(w in a for w in HIGH_ACTIONS):
            return 2
        # Level 3 — WATCH: suspicious, loitering, possible_weapon (CLIP not confirmed)
        return 3

    def _get_severity(self, level: int) -> str:
        return {1: 'critical', 2: 'high', 3: 'medium'}.get(level, 'low')

    async def _save_snapshot(self, alert_id: str, frame: np.ndarray) -> Optional[str]:
        try:
            filepath = settings.ALERTS_DIR / f"{alert_id}_snapshot.jpg"
            overlay  = self._add_alert_overlay(frame.copy(), alert_id)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                _ALERT_IO_POOL, lambda: cv2.imwrite(str(filepath), overlay,
                                                   [cv2.IMWRITE_JPEG_QUALITY, 90]))
            return str(filepath)
        except Exception as e:
            logger.error(f"Snapshot error: {e}")
            return None

    def _add_alert_overlay(self, frame: np.ndarray, alert_id: str) -> np.ndarray:
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
        cv2.putText(frame, "SECURITY ALERT", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(frame, f"ID: {alert_id}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {ts}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return frame

    async def _write_pre_alert_clip(self, alert_id: str, camera_id: str) -> Optional[str]:
        """
        Write last N seconds from ring buffer to a browser-compatible H264 MP4.

        FIXES vs old version:
          1. Codec: uses avc1/H264 (browser-native) instead of mp4v/MPEG4 which
             Chrome/Safari refuse to play inline.  Falls back to MJPG/avi only if
             H264 writer fails to open.
          2. Speed: uses _measured_fps (actual ~10fps) not _camera_fps (reported 30fps).
             900 frames at 10fps = 90s clip. 900 frames at 30fps = 30s clip → 3x fast.
          3. Rotation: each frame is corrected to portrait/landscape using its actual
             shape before writing, so the video is never sideways.
        """
        try:
            from core import camera_service
            mgr = camera_service.camera_manager
            if not mgr or camera_id not in mgr.cameras:
                return None

            cam_stream = mgr.cameras[camera_id]
            frames     = cam_stream.get_ring_buffer_frames()
            if not frames:
                return None

            # ── Correct FPS — measured actual rate, not hardware-reported ─────
            fps = float(getattr(cam_stream, "_measured_fps", None) or
                        getattr(cam_stream, "_camera_fps", 10))
            fps = max(1.0, min(60.0, fps))

            # ── Determine output dimensions from actual frame data ─────────────
            # Handles sideways cameras: if every frame is taller than wide, rotate.
            sample_frame = frames[len(frames)//2][1]   # middle frame
            sh, sw = sample_frame.shape[:2]
            # If USB camera is rotated 90°, correct it
            rotate_90 = (sh > sw * 1.5)  # portrait → rotate to landscape
            if rotate_90:
                out_w, out_h = sh, sw   # after rotation, dimensions swap
            else:
                out_w, out_h = sw, sh

            # ── Try H264 first (browser-playable), then fallbacks ─────────────
            clip_path = None
            writer    = None
            candidates = [
                ('avc1', '.mp4'),   # H264 → works in Chrome, Safari, Firefox
                ('H264', '.mp4'),   # alias on some systems
                ('MJPG', '.avi'),   # last resort — not browser-playable but downloadable
            ]
            for fc_str, ext in candidates:
                _p = settings.ALERTS_DIR / f"{alert_id}_clip{ext}"
                try:
                    fourcc = cv2.VideoWriter_fourcc(*fc_str.ljust(4)[:4])
                    _w = cv2.VideoWriter(str(_p), fourcc, fps, (out_w, out_h))
                    if _w.isOpened():
                        clip_path = _p
                        writer    = _w
                        logger.info(f"Alert clip codec: {fc_str} @ {fps:.1f}fps  {out_w}x{out_h}")
                        break
                    _w.release()
                except Exception:
                    pass

            if writer is None or clip_path is None:
                logger.warning("Could not open any VideoWriter for alert clip")
                return None

            loop = asyncio.get_event_loop()

            def _write_all_frames():
                for _ts, frame in frames:
                    f = frame.copy()
                    # Rotate if camera was sideways
                    if rotate_90:
                        f = cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE)
                    fh, fw = f.shape[:2]
                    if fw != out_w or fh != out_h:
                        f = cv2.resize(f, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                    writer.write(f)
                writer.release()

            await loop.run_in_executor(_ALERT_IO_POOL, _write_all_frames)

            if clip_path.exists() and clip_path.stat().st_size > 2000:
                duration = len(frames) / fps
                logger.info(
                    f"✅ Pre-alert clip: {clip_path.name} "
                    f"({len(frames)} frames @ {fps:.1f}fps = {duration:.0f}s)"
                )
                return str(clip_path)

            logger.warning(f"Alert clip too small or missing: {clip_path}")
            return None
        except Exception as e:
            logger.error(f"Ring buffer clip error: {e}")
            return None

    async def _extract_via_ffmpeg(self, alert_id: str, camera_id: str) -> Optional[str]:
        """Fallback: use ffmpeg to trim the current continuous recording."""
        clip_path   = settings.ALERTS_DIR / f"{alert_id}_clip.mp4"
        source_file = None

        try:
            from core import camera_service
            mgr = camera_service.camera_manager
            if mgr and camera_id in mgr.cameras:
                p = mgr.cameras[camera_id].recording_file_path
                if p and Path(str(p)).exists():
                    source_file = str(p)
        except Exception:
            pass

        if not source_file:
            try:
                async with AsyncSessionLocal() as db:
                    from sqlalchemy import select, desc
                    from core.database import Recording, Camera
                    cam_obj = (await db.execute(
                        select(Camera).where(Camera.camera_id == camera_id)
                    )).scalar_one_or_none()
                    if cam_obj:
                        rec = (await db.execute(
                            select(Recording)
                            .where(Recording.camera_id == cam_obj.id)
                            .order_by(desc(Recording.start_time)).limit(1)
                        )).scalar_one_or_none()
                        if rec and Path(rec.file_path).exists():
                            source_file = rec.file_path
            except Exception:
                pass

        if not source_file:
            return None
        try:
            duration = getattr(settings, 'ALERT_VIDEO_DURATION', 30)
            subprocess.run(
                ['ffmpeg', '-sseof', f'-{duration}',
                 '-i', source_file, '-c', 'copy', str(clip_path), '-y'],
                capture_output=True, timeout=30)
            if clip_path.exists() and clip_path.stat().st_size > 1000:
                return str(clip_path)
        except Exception as e:
            logger.error(f"ffmpeg clip error: {e}")
        return None

    async def _send_notifications(self, alert: Alert,
                                   snapshot_path: Optional[str],
                                   video_path:    Optional[str]):
        action = (alert.action_detected or "").lower()
        is_weapon = any(w in action for w in ("weapon", "weapon_grip", "break_in"))
        is_fire   = "fire" in action

        if alert.alert_level == 1:
            # Weapon/fire: send ALL channels simultaneously + urgent subject
            await asyncio.gather(
                self._send_email(alert, snapshot_path, video_path,
                                 urgent=is_weapon or is_fire),
                self._send_sms(alert, urgent=is_weapon or is_fire),
            )
        elif alert.alert_level == 2:
            await self._send_email(alert, snapshot_path, video_path)

    async def _send_email(self, alert, snapshot_path, video_path, urgent: bool = False):
        try:
            if not settings.SMTP_USERNAME or not settings.SMTP_PASSWORD:
                return
            msg = MIMEMultipart()
            msg['From'] = settings.SMTP_USERNAME
            msg['To']   = ', '.join(settings.ALERT_EMAIL_RECIPIENTS)
            prefix      = "🚨🚨 URGENT" if urgent else "🚨 SafeWatch"
            msg['Subject'] = (
                f"{prefix} L{alert.alert_level} — "
                f"{alert.alert_type.upper().replace('_',' ')}"
            )
            body = (
                f"<h2 style='color:{'#cc0000' if urgent else '#ff6600'}'>"
                f"{'⚠️ WEAPON / FIRE DETECTED' if urgent else 'Security Alert'}: "
                f"{alert.alert_id}</h2>"
                f"<p><strong>Type:</strong> {alert.alert_type}</p>"
                f"<p><strong>Severity:</strong> {alert.severity.upper()}</p>"
                f"<p><strong>Description:</strong> {alert.description}</p>"
                f"<p><strong>Time:</strong> {alert.created_at}</p>"
                f"{'<p style=color:red><b>CALL EMERGENCY SERVICES IMMEDIATELY</b></p>' if urgent else ''}"
            )
            msg.attach(MIMEText(body, 'html'))
            if snapshot_path and Path(snapshot_path).exists():
                with open(snapshot_path, 'rb') as f:
                    img = MIMEImage(f.read())
                    img.add_header('Content-Disposition', 'attachment',
                                   filename='snapshot.jpg')
                    msg.attach(img)

            def _smtp_send():
                with smtplib.SMTP(settings.SMTP_SERVER, settings.SMTP_PORT) as s:
                    s.starttls()
                    s.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
                    s.send_message(msg)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(_ALERT_IO_POOL, _smtp_send)
            logger.info(f"📧 Email sent for {alert.alert_id} (urgent={urgent})")
        except Exception as e:
            logger.error(f"Email error: {e}")

    async def _send_sms(self, alert, urgent: bool = False):
        try:
            if not settings.TWILIO_ACCOUNT_SID or not settings.TWILIO_AUTH_TOKEN:
                return
            from twilio.rest import Client
            prefix = "🚨🚨 WEAPON/FIRE ALERT" if urgent else f"🚨 SafeWatch L{alert.alert_level}"
            body = (
                f"{prefix}: {alert.alert_type.upper().replace('_',' ')}\n"
                f"Cam: {alert.camera_id}  Zone: {alert.zone}  "
                f"@ {alert.created_at.strftime('%H:%M:%S')}"
                f"{chr(10) + 'CALL EMERGENCY SERVICES' if urgent else ''}"
            )

            def _twilio_send():
                client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
                for r in settings.ALERT_SMS_RECIPIENTS:
                    client.messages.create(
                        body=body,
                        from_=settings.TWILIO_PHONE_NUMBER,
                        to=r)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(_ALERT_IO_POOL, _twilio_send)
            logger.info(f"📱 SMS sent for {alert.alert_id} (urgent={urgent})")
        except Exception as e:
            logger.error(f"SMS error: {e}")


# Module-level singleton
alert_manager = AlertManager()