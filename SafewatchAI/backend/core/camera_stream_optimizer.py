"""
camera_stream_optimizer.py — SafeWatch AI

Utility classes for MJPEG stream optimisation.
Imported by camera_manager.py for StreamOptimizer and MJPEGStreamResponse.

NOTE: This file was accidentally replaced with config.py content in a previous
edit. Restored to correct content here.
"""

import cv2
import asyncio
import numpy as np
from typing import Optional, Tuple
from collections import deque
import time
import logging

logger = logging.getLogger(__name__)


class StreamOptimizer:
    """Frame rate limiter and JPEG encoder for MJPEG streaming."""

    def __init__(self, target_fps: int = 10, buffer_size: int = 2):
        self.target_fps     = target_fps
        self.frame_interval = 1.0 / target_fps
        self.buffer_size    = buffer_size
        self.frame_buffer   = deque(maxlen=buffer_size)
        self.last_frame_time = 0.0
        self.encoded_cache: Optional[bytes] = None
        self.cache_timestamp = 0.0
        self.cache_duration  = 1.0 / target_fps

    def should_send_frame(self) -> bool:
        current = time.monotonic()
        if current - self.last_frame_time >= self.frame_interval:
            self.last_frame_time = current
            return True
        return False

    def optimize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Downscale if too large to save bandwidth."""
        h, w = frame.shape[:2]
        if w > 1280 or h > 720:
            scale = min(1280 / w, 720 / h)
            frame = cv2.resize(frame,
                               (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        return frame

    def encode_frame(self, frame: np.ndarray, quality: int = 60) -> bytes:
        """Encode frame to JPEG with simple caching."""
        current = time.monotonic()
        if (self.encoded_cache is not None
                and current - self.cache_timestamp < self.cache_duration):
            return self.encoded_cache

        optimized = self.optimize_frame(frame)
        ok, buf = cv2.imencode('.jpg', optimized,
                               [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("JPEG encode failed")

        self.encoded_cache   = buf.tobytes()
        self.cache_timestamp = current
        return self.encoded_cache

    def get_optimal_quality(self, bandwidth_estimate: float = 1.0) -> int:
        return max(50, min(90, int(50 + 40 * bandwidth_estimate)))


class MJPEGStreamResponse:
    """Async generator for multipart/x-mixed-replace MJPEG streaming."""

    def __init__(self, camera_stream,
                 optimizer: Optional[StreamOptimizer] = None):
        self.camera_stream = camera_stream
        self.optimizer     = optimizer or StreamOptimizer(target_fps=10)
        self.boundary      = "frame"
        self.running       = False

    async def generate(self):
        self.running = True
        try:
            while self.running and self.camera_stream.running:
                if not self.optimizer.should_send_frame():
                    await asyncio.sleep(0.001)
                    continue

                frame = (self.camera_stream.annotated_frame
                         or self.camera_stream.last_frame)
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue

                try:
                    jpeg = self.optimizer.encode_frame(frame, quality=60)
                    yield (b"--" + self.boundary.encode() + b"\r\n"
                           b"Content-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n"
                           b"\r\n" + jpeg + b"\r\n")
                except Exception as e:
                    logger.debug(f"Frame encode error: {e}")
                    await asyncio.sleep(0.05)
                    continue

                await asyncio.sleep(0.001)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"MJPEG stream error: {e}")
        finally:
            self.running = False


def get_optimized_camera_settings(source_type: str = "IP") -> dict:
    """Return optimal cv2.VideoCapture settings for the given source type."""
    base = {
        cv2.CAP_PROP_BUFFERSIZE: 1,
        cv2.CAP_PROP_FPS: 12,
    }
    if source_type == "USB":
        return {
            **base,
            cv2.CAP_PROP_FRAME_WIDTH:  1280,
            cv2.CAP_PROP_FRAME_HEIGHT: 720,
            cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
        }
    return base


def apply_camera_settings(cap: cv2.VideoCapture, settings: dict) -> None:
    """Apply a dict of cv2.CAP_PROP_* settings to a VideoCapture."""
    for prop, value in settings.items():
        try:
            cap.set(prop, value)
        except Exception as e:
            logger.warning(f"Could not set {prop}={value}: {e}")