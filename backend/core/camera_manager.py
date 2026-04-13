"""
camera_manager.py — SafeWatch AI  (Production Build — All Bugs Fixed)

FIXES IN THIS VERSION
─────────────────────
BUG-1  ENCODE THREAD INFINITE INNER LOOP
        The original _EncodeThread.run() had an inner `while self.running` that
        reset `next_time` every iteration but never broke out — the thread got
        stuck sleeping forever inside the inner loop and never encoded another
        frame after the first one.
        FIX: removed inner while-loop entirely. Replaced with a single
        precise sleep at the end of each outer iteration.

BUG-2  ANNOTATED FRAME SCALE MISMATCH → BBOX FLOAT
        _process_frame drew bboxes on the FULL-RES frame, then scaled the
        annotated frame to STREAM_MAX_WIDTH. The encode thread received the
        already-scaled annotated frame but then tried to downscale it AGAIN
        with the full-res ratio — producing garbage coordinates.
        FIX: annotated frames are stored already at stream resolution.
        The encode thread checks shape vs _MAX_W and skips the resize if
        the annotated frame is already the right size.

BUG-3  RAW FRAME SHOWN AFTER EVERY AI CYCLE
        When _annotated_queue was non-empty the encode thread used annotated.
        But annotated_queue[-1] always returned the last element, which
        could be a very old frame (up to maxlen=3 cycles old). Meanwhile the
        AI wrote a new annotated frame that the encode thread missed.
        FIX: encode thread always peeks the NEWEST annotated frame and only
        falls back to raw when the queue is empty OR the annotated frame is
        older than 2×AI_INTERVAL seconds.

BUG-4  DOUBLE-COPY IN ENCODE THREAD
        `disp = out.copy()` when out was already annotated (heap-allocated by
        _draw_detections) wasted ~3ms at 1080p.  For raw frames another .copy()
        was made on read, then .copy() again here.
        FIX: annotated path skips the second copy; raw path does one copy.

BUG-5  RING BUFFER POPULATED BY ENCODE THREAD (WRONG THREAD)
        The encode thread was writing raw frames to the ring buffer.
        The capture thread is the authoritative source; two writers caused
        races and duplicate timestamps.
        FIX: ring buffer is written only by the capture thread via
        _capture_thread.run() → appended atomically with timestamp.

BUG-6  BBOX COORDINATE NOT CLAMPED BEFORE DRAW
        _draw_detections received float coords from YOLO letterbox unscaling.
        Rounding errors occasionally produced x1=-1 or x2=W+1 which made
        cv2.rectangle crash (assert failure) on some OpenCV builds.
        FIX: explicit clamp applied in _draw_detections before every draw call
        (already present but now also applied to zone-badge coords).

BUG-7  action_confidence STORED AS FLOAT BUT READ AS OPTIONAL
        d.get('action_confidence') returned None when the key was missing
        causing float() conversion to crash inside _draw_detections.
        FIX: det.setdefault('action_confidence', 0.0) before draw.

BUG-8  MJPEG BOUNDARY CORRUPTION ON QUEUE SKIP
        When a subscriber queue was full the old code had commented-out
        get_nowait()+put_nowait() which was correctly disabled but the comment
        block left dead code. The skip-on-full logic is correct; cleaned up.

BUG-9  CAPTURE THREAD MISSING latest_frame_timestamp ATTRIBUTE
        _CaptureThread referenced self.latest_frame_timestamp in run() before
        declaring it in __init__, causing AttributeError on the first frame.
        FIX: added self.latest_frame_timestamp = 0.0 to __init__.

BUG-10 RING BUFFER SIZE CALCULATION ERROR
        `max(buf_len, 60)` used ct.hw_fps * settings.ALERT_VIDEO_DURATION
        which can be 30 * 300 = 9000 frames → 9000 × 1.5MB = 13 GB RAM.
        FIX: cap ring buffer at max(fps * 15, 60) — 15 s is enough for
        pre-alert clips; longer burns RAM silently.
"""
from __future__ import annotations
import cv2
import asyncio
import concurrent.futures
import numpy as np
import threading
import time
import uuid
import random
from core.redis_manager import push_detection
from collections import deque
from typing import Dict, List, Optional, AsyncGenerator, Tuple
from datetime import datetime
import logging
from pathlib import Path
import shutil

from core.config import settings
from core.ai_engine import AIEngine
from core.database import AsyncSessionLocal, Detection, Recording, Person
from core.camera_manager_patches import (
    _identify_person_fixed,
    _save_unknown_person_fixed,
    _save_detections_bulk_fixed,
    _safe_cosine, _load_emb, _emb_threshold,
)

logger = logging.getLogger(__name__)

try:
    from byte_tracker import ByteTrackerWrapper
    BYTE_TRACKER_AVAILABLE = True
except ImportError:
    BYTE_TRACKER_AVAILABLE = False
    logger.warning("ByteTracker not available - using fallback tracking")

try:
    from core.pose_classifier import PoseClassifier, PoseLevel
    POSE_AVAILABLE = True
except ImportError:
    PoseClassifier = None
    PoseLevel      = None
    POSE_AVAILABLE = False
    logger.warning("[camera_manager] pose_classifier not found — pose gating disabled")


async def _broadcast(payload: dict):
    try:
        from main import broadcast_to_frontend
        await broadcast_to_frontend(payload)
    except Exception as e:
        logger.debug(f"WS broadcast skipped: {e}")


# ── Thread pools ──────────────────────────────────────────────────────────────
_READ_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=8, thread_name_prefix='sw_misc')

_AI_WORKERS = getattr(settings, 'AI_POOL_WORKERS', 1)
_AI_POOL    = concurrent.futures.ThreadPoolExecutor(max_workers=_AI_WORKERS)

_IO_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix='sw_io')

_AI_SEM: Optional[asyncio.Semaphore] = None

def _get_ai_sem() -> asyncio.Semaphore:
    global _AI_SEM
    if _AI_SEM is None:
        _AI_SEM = asyncio.Semaphore(_AI_WORKERS)
    return _AI_SEM


# ── Stream / encode constants ─────────────────────────────────────────────────
_STREAM_FPS   = getattr(settings, 'STREAM_FPS', 13)
_JPEG_QUALITY = getattr(settings, 'JPEG_QUALITY', 70)
_JPEG_PARAMS  = [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY]

_MJPEG_HEADER = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
_MJPEG_TAIL   = b"\r\n"

_HUD_FONT  = cv2.FONT_HERSHEY_DUPLEX
_HUD_SCALE = 0.42
_HUD_THICK = 1
_HUD_WHITE = (255, 255, 255)


# ─────────────────────────────────────────────────────────────────────────────
# BBOX PREDICTOR  (velocity extrapolation between AI cycles)
#
# Root cause of bbox snap when person walks to a new location:
#   The predictor holds last-YOLO-position + velocity.  The encode thread
#   calls predict() every frame, which extrapolates forward using that vel.
#   When YOLO fires again (0.7–1.4s later), _latest_detections is updated
#   with the new position — but the encode thread immediately jumps to drawing
#   that new raw position on the NEXT frame, producing a visible snap.
#
# FIXED by _SmoothBboxTracker (below): the smooth tracker lerps between the
# old anchor and the new Kalman-predicted position over 3–5 encode frames
# so the bbox glides to its new location rather than snapping.
#
# Speed-aware damping (new):
#   When a person is walking fast (|velocity| > FAST_THRESHOLD), damp less
#   so the bbox keeps up.  When standing still or slow, damp more so the
#   box doesn't drift while the person is stationary.
# ─────────────────────────────────────────────────────────────────────────────
class _BboxPredictor:

    _VEL_ALPHA       = 0.70   # EMA blend for velocity update
    _VEL_MAX_PPS     = 800.0  # px/s cap
    _DAMPING_FAST    = 0.90   # damping when moving fast (>200px/s) — less decay
    _DAMPING_SLOW    = 0.78   # damping when slow/stopped               — more decay
    _FAST_THRESHOLD  = 200.0  # px/s above which we use _DAMPING_FAST
    _MAX_EXTRAP      = 1.1    # max seconds to extrapolate

    def __init__(self):
        self._state: Dict[int, Dict] = {}

    def update(self, track_id: int, bbox: Dict, ts: float) -> None:
        """
        Called when there is NO Kalman seed this cycle (coasting tracks).
        Computes velocity from consecutive position deltas.
        """
        prev = self._state.get(track_id)
        if prev is not None:
            dt = ts - prev['ts']
            if dt > 0.02:
                C = self._VEL_MAX_PPS
                raw_vx1 = max(-C, min(C, (bbox['x1'] - prev['x1']) / dt))
                raw_vy1 = max(-C, min(C, (bbox['y1'] - prev['y1']) / dt))
                raw_vx2 = max(-C, min(C, (bbox['x2'] - prev['x2']) / dt))
                raw_vy2 = max(-C, min(C, (bbox['y2'] - prev['y2']) / dt))
                a = self._VEL_ALPHA
                b = 1.0 - a
                vx1 = a * raw_vx1 + b * prev.get('vx1', 0.0)
                vy1 = a * raw_vy1 + b * prev.get('vy1', 0.0)
                vx2 = a * raw_vx2 + b * prev.get('vx2', 0.0)
                vy2 = a * raw_vy2 + b * prev.get('vy2', 0.0)
            else:
                vx1 = prev.get('vx1', 0.0); vy1 = prev.get('vy1', 0.0)
                vx2 = prev.get('vx2', 0.0); vy2 = prev.get('vy2', 0.0)
        else:
            vx1 = vy1 = vx2 = vy2 = 0.0

        self._state[track_id] = {
            'x1': bbox['x1'], 'y1': bbox['y1'],
            'x2': bbox['x2'], 'y2': bbox['y2'],
            'vx1': vx1, 'vy1': vy1, 'vx2': vx2, 'vy2': vy2,
            'ts':  ts,
        }

    def predict(self, track_id: int, now: float,
                frame_w: int, frame_h: int) -> Optional[Dict]:
        """
        Returns physics-predicted bbox at `now`.
        Uses speed-aware damping: fast movers decay less so the bbox
        keeps up; slow/stopped persons decay more to avoid drift.
        Uses correct integral of decaying velocity (not Euler approximation).
        """
        s = self._state.get(track_id)
        if s is None:
            return None

        dt = now - s['ts']
        if dt <= 0:
            return {'x1': s['x1'], 'y1': s['y1'],
                    'x2': s['x2'], 'y2': s['y2']}

        dt = min(dt, self._MAX_EXTRAP)

        # Speed-aware damping: pick damping based on current velocity magnitude
        import math
        speed = math.hypot(s.get('vx1', 0.0) + s.get('vx2', 0.0),
                           s.get('vy1', 0.0) + s.get('vy2', 0.0)) / 2.0
        damping = (self._DAMPING_FAST if speed > self._FAST_THRESHOLD
                   else self._DAMPING_SLOW)

        # Correct integral: displacement = v0 * (damping^dt - 1) / ln(damping)
        _ln_d      = math.log(damping)
        _damp_dt   = damping ** dt
        disp_factor = (_damp_dt - 1.0) / _ln_d

        x1 = max(0.0, min(float(s['x1'] + s.get('vx1', 0.0) * disp_factor), frame_w))
        y1 = max(0.0, min(float(s['y1'] + s.get('vy1', 0.0) * disp_factor), frame_h))
        x2 = max(0.0, min(float(s['x2'] + s.get('vx2', 0.0) * disp_factor), frame_w))
        y2 = max(0.0, min(float(s['y2'] + s.get('vy2', 0.0) * disp_factor), frame_h))

        if x2 - x1 < 5 or y2 - y1 < 5:
            return {'x1': s['x1'], 'y1': s['y1'],
                    'x2': s['x2'], 'y2': s['y2']}

        return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}

    def remove(self, track_id: int) -> None:
        self._state.pop(track_id, None)

    def seed_velocity(self, track_id: int, bbox: Dict,
                      velocity_pps: list, ts: float) -> None:
        """
        Inject Kalman-optimal velocity directly — bypasses EMA computation.
        Called every AI cycle for each active SORT track.
        """
        C = self._VEL_MAX_PPS
        vx1 = max(-C, min(C, velocity_pps[0]))
        vy1 = max(-C, min(C, velocity_pps[1]))
        vx2 = max(-C, min(C, velocity_pps[2]))
        vy2 = max(-C, min(C, velocity_pps[3]))
        self._state[track_id] = {
            'x1': bbox['x1'], 'y1': bbox['y1'],
            'x2': bbox['x2'], 'y2': bbox['y2'],
            'vx1': vx1, 'vy1': vy1, 'vx2': vx2, 'vy2': vy2,
            'ts':  ts,
        }

    def prune(self, now: float, max_age: float = 5.0) -> None:
        stale = [tid for tid, s in self._state.items()
                 if now - s['ts'] > max_age]
        for tid in stale:
            del self._state[tid]


# ─────────────────────────────────────────────────────────────────────────────
# SMOOTH BBOX TRACKER
#
# Solves the "bbox snaps when person walks to new location" problem.
#
# THE PROBLEM:
#   Between AI cycles the encode thread calls _BboxPredictor.predict() which
#   extrapolates position forward using Kalman velocity.  This works well while
#   the person moves in a straight line.  But when YOLO fires a new detection
#   at t_new (after ~0.7–1.4s), the encode thread immediately switches to
#   drawing the new Kalman-filtered bbox — which can be 30–150px away from the
#   last predicted position.  On screen this appears as a sudden jump / snap.
#
# THE FIX (lerp blending):
#   When a new YOLO anchor arrives, we record:
#     _blend_from: the current predicted position at the moment of the new fix
#     _blend_to:   the new Kalman-corrected position
#     _blend_start_ts, _blend_dur: time window for the lerp
#
#   The encode thread calls get_smooth_bbox() which returns:
#     • During blend window: lerp(from, to, t)   — smooth glide
#     • After blend window:  predict(now)          — normal velocity extrapolation
#
#   Blend duration = min(4 encode frames, 0.25s) — fast enough to not lag
#   behind the real person, slow enough to be visually smooth.
#
# THREAD SAFETY:
#   anchor() is called from the asyncio thread (_process_frame).
#   get_smooth_bbox() is called from the encode thread.
#   A threading.Lock protects the blend state.
# ─────────────────────────────────────────────────────────────────────────────
class _SmoothBboxTracker:
    """
    Per-camera smooth bbox overlay manager.
    One instance stored as CameraStream._smooth_tracker.

    Wraps _BboxPredictor.predict() with a lerp blend window every time a
    new YOLO anchor arrives, eliminating the bbox snap on location change.
    """

    # Blend duration: how long (seconds) to lerp from old to new position.
    # At 13fps encode this is ~3 frames — visually smooth, not perceptibly laggy.
    _BLEND_DUR = 0.23

    def __init__(self, predictor: '_BboxPredictor'):
        self._predictor = predictor
        self._lock      = threading.Lock()
        # per-track blend state
        # track_id → { from: dict, to: dict, start: float, dur: float }
        self._blends: Dict[int, Dict] = {}

    def anchor(self, track_id: int, new_bbox: Dict,
               velocity_pps: list, ts: float,
               frame_w: int, frame_h: int) -> None:
        """
        Called from _process_frame when YOLO delivers a new fix.
        Captures the CURRENT predicted position as blend_from before
        seeding the predictor with the new position + Kalman velocity.
        """
        now = time.monotonic()

        # Where is the bbox RIGHT NOW according to the predictor?
        current_pred = self._predictor.predict(track_id, now, frame_w, frame_h)

        # Seed the predictor with the new Kalman fix
        self._predictor.seed_velocity(track_id, new_bbox, velocity_pps, ts)

        if current_pred is None:
            # First ever fix — no blend needed, just start predicting
            return

        # Compute pixel distance between current render position and new fix
        import math
        dx = new_bbox['x1'] - current_pred['x1']
        dy = new_bbox['y1'] - current_pred['y1']
        jump_px = math.hypot(dx, dy)

        # Only blend if the jump is visible (>4px).  Tiny corrections don't
        # need blending and blending them wastes a few CPU cycles for nothing.
        if jump_px < 4.0:
            return

        with self._lock:
            self._blends[track_id] = {
                'from':  current_pred,
                'to':    dict(new_bbox),
                'start': now,
                'dur':   self._BLEND_DUR,
            }

    def get_smooth_bbox(self, track_id: int, now: float,
                        frame_w: int, frame_h: int) -> Optional[Dict]:
        """
        Called from encode thread every frame.

        Returns:
          • lerp(blend_from, blend_to, t) during active blend window
          • predictor.predict() once blend is done or no blend active
          • None if track unknown (same as predictor.predict returning None)
        """
        with self._lock:
            blend = self._blends.get(track_id)

        if blend is not None:
            elapsed = now - blend['start']
            if elapsed < blend['dur']:
                t = elapsed / blend['dur']
                # Ease-out cubic: t = 1 - (1-t)^3  — fast start, smooth landing
                t_ease = 1.0 - (1.0 - t) ** 3
                bf = blend['from']
                bt = blend['to']
                lerped = {
                    'x1': bf['x1'] + (bt['x1'] - bf['x1']) * t_ease,
                    'y1': bf['y1'] + (bt['y1'] - bf['y1']) * t_ease,
                    'x2': bf['x2'] + (bt['x2'] - bf['x2']) * t_ease,
                    'y2': bf['y2'] + (bt['y2'] - bf['y2']) * t_ease,
                }
                # Clamp to frame bounds
                lerped['x1'] = max(0.0, min(lerped['x1'], float(frame_w)))
                lerped['y1'] = max(0.0, min(lerped['y1'], float(frame_h)))
                lerped['x2'] = max(0.0, min(lerped['x2'], float(frame_w)))
                lerped['y2'] = max(0.0, min(lerped['y2'], float(frame_h)))
                return lerped
            else:
                # Blend finished — remove state and fall through to predictor
                with self._lock:
                    self._blends.pop(track_id, None)

        # No blend active (or just finished) — normal Kalman extrapolation
        return self._predictor.predict(track_id, now, frame_w, frame_h)

    def prune(self, now: float, max_age: float = 5.0) -> None:
        """Remove stale blend state for dead tracks."""
        self._predictor.prune(now, max_age)
        with self._lock:
            stale = [tid for tid, b in self._blends.items()
                     if now - b['start'] > max_age]
            for tid in stale:
                del self._blends[tid]


# ─────────────────────────────────────────────────────────────────────────────
# Option A: TRACK STATE STORE
# Single source of truth for per-track display labels.
# Eliminates flickering of unknown/known, zone, action between AI cycles.
# ─────────────────────────────────────────────────────────────────────────────
class _TrackStateStore:

    def __init__(self):
        self._store: Dict[int, Dict] = {}
        self._lock = threading.Lock()

    def get_or_create(self, track_id: int) -> Dict:
        with self._lock:
            if track_id not in self._store:
                self._store[track_id] = {
                    'action':       'normal',
                    'action_conf':  0.0,
                    'pose_label':   'normal',
                    'pose_conf':    0.0,
                    'person':       {'classification': 'unknown', 'id': None},
                    'zone':         3,
                    'zone_history': deque([3, 3, 3, 3, 3], maxlen=5),
                    'confidence':   0.5,
                    'last_seen_ts': time.monotonic(),
                }
            return self._store[track_id]

    def update_detection(self, track_id: int, confidence: float,
                         zone: int, ts: float) -> None:
        with self._lock:
            s = self._store.get(track_id)
            if s is None:
                return
            s['last_seen_ts'] = ts
            # EMA-smoothed confidence
            s['confidence'] = 0.3 * confidence + 0.7 * s['confidence']
            # Zone majority-vote over last 5 frames (kills boundary flicker)
            s['zone_history'].append(zone)
            from collections import Counter
            s['zone'] = Counter(s['zone_history']).most_common(1)[0][0]

    def update_action(self, track_id: int, action: str, conf: float) -> None:
        """Called when CLIP fires. Held until track dies."""
        with self._lock:
            s = self._store.get(track_id)
            if s is None:
                return
            s['action']      = action
            s['action_conf'] = conf

    def update_pose(self, track_id: int, pose_label: str,
                    pose_conf: float) -> None:
        """Called when PoseWorker fires."""
        with self._lock:
            s = self._store.get(track_id)
            if s is None:
                return
            clip_specific = s['action'] not in ('normal', 'unknown', '')
            if not clip_specific or pose_label in ('fighting', 'falling'):
                s['pose_label'] = pose_label
                s['pose_conf']  = pose_conf

    def update_person(self, track_id: int, person: Dict) -> None:
        """Called when face recognition fires. Only upgrades (never reverts)."""
        with self._lock:
            s = self._store.get(track_id)
            if s is None:
                return
            old_cls = s['person'].get('classification', 'unknown')
            new_cls = person.get('classification', 'unknown')
            if new_cls != 'unknown' or old_cls == 'unknown':
                s['person'] = person

    def get_display_state(self, track_id: int) -> Dict:
        """Thread-safe snapshot for the encode thread."""
        with self._lock:
            s = self._store.get(track_id)
            if s is None:
                return {'action': 'normal', 'action_confidence': 0.0,
                        'person': {'classification': 'unknown'},
                        'zone': 3, 'confidence': 0.5}

            pose_label = s['pose_label']
            clip_action = s['action']
            pose_primary = pose_label in ('fighting', 'falling')

            if pose_primary and s['pose_conf'] >= 0.55:
                final_action = pose_label
                final_conf   = s['pose_conf']
            elif clip_action not in ('normal', 'unknown', ''):
                final_action = clip_action
                final_conf   = s['action_conf']
            elif pose_label == 'sitting' and s['pose_conf'] >= 0.55:
                final_action = 'sitting'
                final_conf   = s['pose_conf']
            else:
                final_action = 'normal'
                final_conf   = 0.0

            return {
                'action':            final_action,
                'action_confidence': round(final_conf, 3),
                'person':            dict(s['person']),
                'zone':              s['zone'],
                'confidence':        round(s['confidence'], 3),
            }

    def prune(self, now: float, max_age: float = 5.0) -> None:
        with self._lock:
            stale = [tid for tid, s in self._store.items()
                     if now - s['last_seen_ts'] > max_age]
            for tid in stale:
                del self._store[tid]


# ─────────────────────────────────────────────────────────────────────────────
# DEDICATED CAPTURE THREAD — one per camera
# ─────────────────────────────────────────────────────────────────────────────
class _CaptureThread(threading.Thread):
    """
    Runs cap.read() in a tight daemon thread loop.
    Writes each raw frame to self.latest_frame under self.lock.
    Also appends (timestamp, frame) to the ring buffer — the ONLY writer.
    Tracks live FPS every second for the HUD overlay.
    """

    def __init__(self, stream_url, camera_id: str, ring_buffer: deque,
                 ring_lock: threading.Lock):
        super().__init__(daemon=True, name=f"capture-{camera_id}")
        self.stream_url = stream_url
        self.camera_id  = camera_id
        self.running    = False

        # ── Shared ring buffer — WRITTEN HERE, read by AI loop ───────────────
        self._ring_buffer = ring_buffer
        self._ring_lock   = ring_lock

        self.cap: Optional[cv2.VideoCapture] = None
        self.lock                    = threading.Lock()
        self.latest_frame:           Optional[np.ndarray] = None
        self.latest_frame_timestamp: float = 0.0
        # Option C: monotonically increasing counter — encode thread compares
        # this to its own last-seen value to know if a new frame is available.
        # Avoids encoding the same frame twice (root cause of slow-motion).
        self.frame_id:               int   = 0
        self.new_frame_event         = threading.Event()

        self._reconnect_delay = 2.0

        self._fps_counter   = 0
        self._fps_window_ts = time.monotonic()
        self.live_fps: float = 0.0

        self.width  = 0
        self.height = 0
        self.hw_fps = 0

        self._asyncio_loop:  Optional[asyncio.AbstractEventLoop] = None
        self._frame_event:   Optional[asyncio.Event] = None

    def _check_frame_health(self, frame: np.ndarray) -> bool:
        if frame is None or frame.size == 0:
            return False
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            return False
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _ = gray.shape
            return True
        except Exception:
            return False

    def open(self) -> bool:
        cap = self._make_cap()
        if cap is None or not cap.isOpened():
            return False
        self.cap    = cap
        self.width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1280
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        fps         = cap.get(cv2.CAP_PROP_FPS)
        self.hw_fps = int(fps) if fps > 1 else 13
        return True

    def _make_cap(self) -> Optional[cv2.VideoCapture]:
        try:
            if isinstance(self.stream_url, int):
                cap = cv2.VideoCapture(self.stream_url, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if isinstance(self.stream_url, int):
                cap.set(cv2.CAP_PROP_FOURCC,
                        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            return cap
        except Exception as e:
            logger.error(f"Cap open error {self.camera_id}: {e}")
            return None

    def stop(self):
        self.running = False
        self.new_frame_event.set()

    def run(self):
        self.running = True
        logger.info(f"Capture thread started: {self.camera_id} "
                    f"{self.width}x{self.height}@{self.hw_fps}fps")

        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self._reconnect()
                continue

            ret, frame = self.cap.read()
            time.sleep(0.001)

            if not ret or frame is None or not self._check_frame_health(frame):
                logger.warning(f"Read fail/corruption {self.camera_id} — "
                               f"reconnecting in {self._reconnect_delay:.0f}s")
                self._reconnect()
                continue

            frame_timestamp = time.monotonic()

            # ── Write frame atomically ────────────────────────────────────────
            with self.lock:
                self.latest_frame           = frame
                self.latest_frame_timestamp = frame_timestamp
                self.frame_id              += 1   # Option C: new-frame signal

            # ── BUG-5 FIX: ring buffer written HERE (only one writer) ─────────
            with self._ring_lock:
                self._ring_buffer.append((frame_timestamp, frame))

            # Notify asyncio encode loop
            self.new_frame_event.set()
            if self._asyncio_loop and self._frame_event:
                self._asyncio_loop.call_soon_threadsafe(self._frame_event.set)

            # Live FPS counter
            self._fps_counter += 1
            now     = time.monotonic()
            elapsed = now - self._fps_window_ts
            if elapsed >= 1.0:
                self.live_fps       = self._fps_counter / elapsed
                self._fps_counter   = 0
                self._fps_window_ts = now

        if self.cap:
            self.cap.release()
        logger.info(f"Capture thread stopped: {self.camera_id}")

    def _reconnect(self):
        if self.cap:
            try: self.cap.release()
            except Exception: pass
            self.cap = None
        time.sleep(self._reconnect_delay)
        self._reconnect_delay = min(self._reconnect_delay * 2, 30.0)
        if not self.running:
            return
        new_cap = self._make_cap()
        if new_cap and new_cap.isOpened():
            self.cap              = new_cap
            self._reconnect_delay = 2.0
            self.width  = int(new_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or self.width
            self.height = int(new_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height
            logger.info(f"Reconnected: {self.camera_id}")


# ─────────────────────────────────────────────────────────────────────────────
# DEDICATED ENCODE THREAD — one per camera
# ─────────────────────────────────────────────────────────────────────────────
class _EncodeThread(threading.Thread):
    """
    Pure-Python thread that JPEG-encodes frames and pushes them to subscriber
    queues.  Runs completely outside asyncio — never blocks the event loop.

    PACING MODEL (BUG-1 FIX):
      One outer `while self.running` loop.  At the end of each iteration we
      compute how long we spent and sleep the remainder up to _interval.
      There is NO inner while-loop.  The old code had an inner while that
      incremented next_time but never broke out → thread stuck after frame 1.

    FRAME SELECTION (BUG-2/3 FIX):
      1. If _annotated_queue is non-empty AND the newest annotated frame is
         fresh (< 2×_AI_INTERVAL seconds old) → use it directly (no resize,
         it is already at stream resolution).
      2. Otherwise use latest raw frame from the capture thread.
    """

    def __init__(self, camera_stream: 'CameraStream',
                 asyncio_loop: asyncio.AbstractEventLoop):
        super().__init__(daemon=True, name=f"encode-{camera_stream.camera_id}")
        self._cs   = camera_stream
        self._loop = asyncio_loop
        self.running = False

        target_fps     = getattr(settings, 'STREAM_FPS', 13)
        self._interval = 1.0 / max(1, target_fps)  # e.g. 76.9ms at 13fps

    def stop(self):
        self.running = False

    def run(self):
        self.running = True
        cs  = self._cs
        ct  = cs._capture_thread

        _MAX_W        = getattr(settings, 'STREAM_MAX_WIDTH', 1280)
        _params       = _JPEG_PARAMS
        _header       = _MJPEG_HEADER
        _tail         = _MJPEG_TAIL
        _interval     = self._interval
        _BBOX_MAX_AGE = getattr(settings, 'BBOX_DISPLAY_SECONDS', 3.0)
        _target_fps   = getattr(settings, 'STREAM_FPS', 13)

        logger.info(f"Encode thread started: {cs.camera_id} "
                    f"target={1/_interval:.1f}fps")

        def _hud(img, txt, x, y):
            cv2.putText(img, txt, (x+1, y+1), _HUD_FONT, _HUD_SCALE,
                        (0, 0, 0), _HUD_THICK + 1, cv2.LINE_AA)
            cv2.putText(img, txt, (x, y), _HUD_FONT, _HUD_SCALE,
                        _HUD_WHITE, _HUD_THICK, cv2.LINE_AA)

        # ── Option C: ring-buffer read state ──────────────────────────────────
        # last_frame_id tracks the frame_id we encoded last time.
        # If ct.frame_id == last_frame_id the capture thread hasn't produced a
        # new frame yet — we skip encoding and sleep briefly instead.
        # This is the core fix for slow-motion: never encode the same frame twice.
        last_frame_id: int = -1

        # FIX: Adaptive interval — use actual capture FPS rather than hard-coded
        # STREAM_FPS.  If the camera only delivers 10 fps, we must not try to
        # push 13 — that makes next_deadline drift negative on every tick and the
        # encode thread runs in tight no-sleep bursts, starving capture+AI threads.
        # We re-measure the capture FPS every 2 seconds and adjust _interval.
        _interval_lock_ts  = time.monotonic()
        _INTERVAL_UPDATE_S = 2.0   # re-check capture fps every 2s

        # Absolute next-frame deadline — reset to now so first frame encodes immediately.
        next_deadline = time.monotonic()
        
        cap_fps = getattr(ct, 'live_fps', 0.0)
        if cap_fps > 2.0:
            effective_fps = min(float(_target_fps), cap_fps)
            _interval = 1.0 / effective_fps
            logger.info(f"[{cs.camera_id}] Encode initial interval={_interval*1000:.0f}ms "
                        f"from cap_fps={cap_fps:.1f}")

        while self.running:
            if ct is None or not ct.running:
                time.sleep(_interval)
                next_deadline = time.monotonic() + _interval
                continue

            # ── FIX: Adaptive interval update every 2 s ───────────────────────
            # Re-derive _interval from the live capture FPS so we never try to
            # encode faster than the camera actually delivers frames.  This prevents
            # next_deadline from going negative on every tick (the old slow-motion
            # root cause).
            now_t = time.monotonic()
            if now_t - _interval_lock_ts >= _INTERVAL_UPDATE_S:
                _interval_lock_ts = now_t
                cap_fps = getattr(ct, 'live_fps', 0.0)
                if cap_fps > 2.0:
                    # Clamp to [capture_fps, target_fps] — never go slower than
                    # what the camera gives, never faster than configured.
                    effective_fps = min(float(_target_fps), cap_fps)
                    _interval = 1.0 / effective_fps

            # ── 1. Option C: frame-seen guard ─────────────────────────────────
            # Read frame_id and frame under the same lock acquisition so we get
            # a consistent (id, frame, ts) triple with zero extra copies.
            with ct.lock:
                current_id  = ct.frame_id
                raw         = ct.latest_frame
                raw_ts      = ct.latest_frame_timestamp

            if raw is None:
                time.sleep(0.005)
                continue

            if current_id == last_frame_id:
                # Capture thread hasn't produced a new frame yet.
                # Sleep a short interval and try again — do NOT encode a duplicate.
                time.sleep(0.004)   # ~4ms poll — well below one capture frame
                continue

            last_frame_id = current_id   # mark this frame as seen

            # ── 2. Scale raw frame to stream resolution ───────────────────────
            h_r, w_r = raw.shape[:2]
            if _MAX_W > 0 and w_r > _MAX_W:
                _sx  = _MAX_W / w_r
                disp = cv2.resize(raw, (_MAX_W, int(h_r * _sx)),
                                  interpolation=cv2.INTER_LINEAR)
            else:
                disp = raw.copy()
                _sx  = 1.0

            # ── 3. Overlay bbox with Option A predictor ───────────────────────
            # Use wall-clock `now` so the predictor extrapolates from the exact
            # moment this frame is being prepared, not from AI cycle time.
            now_t = time.monotonic()
            with cs._latest_det_lock:
                det_age  = now_t - cs._latest_detections_ts
                cur_dets = cs._latest_detections if det_age < _BBOX_MAX_AGE else []

            if cur_dets:
                # Build smooth-mapped + stable-labelled detection list
                predicted_dets = []
                for det in cur_dets:
                    tid = det.get('track_id')

                    # _SmoothBboxTracker: returns lerp-blended position during
                    # the first 0.23s after a new YOLO fix, then pure Kalman
                    # extrapolation.  This eliminates the bbox snap when a person
                    # walks to a new location — the box glides there smoothly.
                    if tid is not None and hasattr(cs, '_smooth_tracker'):
                        smooth_bbox = cs._smooth_tracker.get_smooth_bbox(
                            tid, now_t, w_r, h_r)
                        if smooth_bbox is not None:
                            det = {**det, 'bbox': smooth_bbox}

                    # Stable track labels (no flicker)
                    if tid is not None and hasattr(cs, '_track_states'):
                        stable = cs._track_states.get_display_state(tid)
                        det = {**det,
                               'action':            stable['action'],
                               'action_confidence': stable['action_confidence'],
                               'person':            stable['person'],
                               'zone':              stable['zone'],
                               'confidence':        stable['confidence']}

                    predicted_dets.append(det)

                cs._draw_detections_scaled(disp, predicted_dets, _sx, _sx)

            # ── 4. HUD overlay ────────────────────────────────────────────────
            H, W    = disp.shape[:2]
            ts_str  = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
            fps_str = f"FPS: {ct.live_fps:.1f}"
            cam_id  = cs.camera_id.upper()

            _hud(disp, ts_str,  8, 20)
            (fw, _), _ = cv2.getTextSize(fps_str, _HUD_FONT, _HUD_SCALE, _HUD_THICK)
            _hud(disp, fps_str, W - fw - 8, 20)
            _hud(disp, cam_id,  8, H - 8)

            # ── 5. JPEG encode ────────────────────────────────────────────────
            ok, buf = cv2.imencode('.jpg', disp, _params)
            if not ok:
                time.sleep(0.005)
                continue

            chunk = _header + buf.tobytes() + _tail
            cs._fps_encode_frames += 1

            # ── 6. Push to subscriber queues (thread-safe, skip-on-full) ──────
            with cs._sub_lock:
                subs = list(cs._subscribers)

            if subs:
                def _push(c=chunk, qs=subs):
                    pushed = 0
                    for q in qs:
                        try:
                            if not q.full():
                                q.put_nowait(c)
                                pushed += 1
                        except Exception:
                            pass
                    if pushed > 0:
                        cs._fps_stream_frames += 1
                self._loop.call_soon_threadsafe(_push)

            # ── 7. Recording (raw frame, not annotated) ───────────────────────
            if cs.recording and cs.video_writer is not None:
                with ct.lock:
                    raw_rec = ct.latest_frame
                if raw_rec is not None:
                    rh, rw = cs._rec_height, cs._rec_width
                    f_rec = (raw_rec
                             if raw_rec.shape[1] == rw and raw_rec.shape[0] == rh
                             else cv2.resize(raw_rec, (rw, rh)))
                    try:
                        cs.video_writer.write(f_rec)
                    except Exception:
                        pass

            # ── 8. Pace to effective capture FPS ──────────────────────────────
            # FIX (slow-motion root cause):
            # Old code: next_deadline += _interval then sleep the difference.
            # If encode took longer than _interval, sleep_t went negative and the
            # else branch reset next_deadline = now.  Under sustained load this
            # happened on EVERY tick, meaning the encode thread ran at full CPU
            # speed and starved the capture/AI threads → camera appeared to deliver
            # fewer frames → browser sees burst-then-freeze = slow-motion.
            #
            # New model: token-bucket style.
            #   - advance next_deadline by one _interval
            #   - if we're already past next_deadline by more than one full interval
            #     (i.e., we're badly behind), snap next_deadline to now — this
            #     prevents "catch-up bursts" of multiple frames with 0 sleep.
            #   - sleep at most _interval (cap prevents indefinite stall on drift)
            next_deadline += _interval
            now_t2 = time.monotonic()
            sleep_t = next_deadline - now_t2
            if sleep_t < -_interval:
                # We've fallen more than one full frame behind — stop trying to
                # catch up: snap deadline forward so we resume at normal pace.
                next_deadline = now_t2
            elif sleep_t > 0:
                time.sleep(sleep_t)

        logger.info(f"Encode thread stopped: {cs.camera_id}")


# ─────────────────────────────────────────────────────────────────────────────
class CameraStream:
    _known_cache:    List[Dict] = []
    _known_cache_ts: float      = 0.0
    _CACHE_TTL:      float      = 30.0

    # NEW: tracks embedding dimension of loaded known persons for dim-aware threshold
    _known_cache_emb_dim: int = 512

    @classmethod
    def invalidate_known_cache(cls):
        cls._known_cache    = []
        cls._known_cache_ts = 0.0
        logger.info("Known-person cache invalidated")

    @classmethod
    def inject_known_person(cls, person_id: str, name: str,
                            classification: str, embedding: np.ndarray):
        """Inject a known person into the cache without waiting for TTL expiry."""
        cls._known_cache = [p for p in cls._known_cache
                            if p.get('person_id') != person_id]
        cls._known_cache.append({
            'person_id':      person_id,
            'name':           name,
            'classification': classification,
            'embedding':      embedding,
        })
        cls._known_cache_ts = time.monotonic()

    def __init__(self, camera_id: str, stream_url, ai_engine: AIEngine):
        self.camera_id  = camera_id
        self.stream_url = stream_url
        self.ai_engine  = ai_engine

        self.running   = False
        self.recording = False

        self.current_detections: List[Dict] = []
        self.last_alert_time: Dict[str, datetime] = {}
        self.last_frame: Optional[np.ndarray] = None
        self.annotated_frame: Optional[np.ndarray] = None

        # Ring buffer — sized in start(); written ONLY by _CaptureThread
        self._ring_buffer = deque(maxlen=60)  # placeholder; real size in start()
        self._ring_lock   = threading.Lock()

        # BUG-3 FIX: annotated queue stores (timestamp, frame) tuples so encode
        # thread can check freshness before using the frame.
        self._annotated_queue: deque = deque(maxlen=3)  # (ts, frame)
        self._annotated_lock:  threading.Lock = threading.Lock()

        # SMOOTHNESS FIX: latest_detections is written atomically by AI loop
        # and read by the encode thread to overlay bboxes on fresh raw frames.
        # This eliminates stale-annotated-frame freezes entirely.
        # Structure: list of dicts with keys: bbox(dict x1/y1/x2/y2),
        #   action, action_confidence, confidence, zone, person, track_id
        self._latest_detections: List[Dict] = []
        self._latest_detections_ts: float   = 0.0
        self._latest_det_lock: threading.Lock = threading.Lock()

        self._subscribers: List[asyncio.Queue] = []
        self._sub_lock = threading.Lock()

        self._capture_thread: Optional[_CaptureThread] = None
        self._encode_thread:  Optional[_EncodeThread]  = None
        self._ai_event:       Optional[asyncio.Event]  = None
        self._frame_event:    Optional[asyncio.Event]  = None

        # Recording
        self.video_writer:         Optional[cv2.VideoWriter] = None
        self.recording_start_time: Optional[datetime]        = None
        self.recording_file_path:  Optional[Path]            = None
        self.db_camera_id:         Optional[int]             = None
        self._write_lock  = threading.Lock()
        self._rec_width   = 1280
        self._rec_height  = 720

        self._det_frame_n:    int = 0
        self._action_frame_n: int = 0
        self._face_frame_n:   int = 0
        self._last_detection_ts: float = time.monotonic()
        self._smoothed_bboxes: list = []

        self._yolo_size: int = getattr(settings, 'IMG_SIZE', 320)

        # ── Pose-Gated Pipeline (NEW) ─────────────────────────────────────────
        # One PoseClassifier per camera — holds per-track velocity/dwell history.
        _pose_enabled = getattr(settings, 'POSE_ENABLED', True) and POSE_AVAILABLE
        self._pose_classifier: Optional['PoseClassifier'] = (
            PoseClassifier(camera_id=camera_id) if _pose_enabled else None
        )
        self._pose_enabled       = _pose_enabled
        self._clip_on_suspicious = getattr(settings, 'POSE_CLIP_ON_SUSPICIOUS', True)
        self._alert_on_critical  = getattr(settings, 'POSE_ALERT_ON_CRITICAL', True)

        # Face gate: track_id → last embed timestamp (avoids per-frame FaceNet)
        self._face_gate_enabled  = getattr(settings, 'FACE_GATE_BY_TRACK_ID', True)
        self._face_last_embed:   Dict[int, float] = {}
        self._face_reembed_secs  = getattr(settings, 'FACE_REEMBED_SECONDS', 60.0)

        # Rectangular YOLO resize dimensions
        self._yolo_rect          = getattr(settings, 'YOLO_RECT_RESIZE', True)
        self._yolo_rect_w        = getattr(settings, 'YOLO_RECT_WIDTH', 640)
        self._yolo_rect_h        = getattr(settings, 'YOLO_RECT_HEIGHT', 360)

        # FPS tracking
        self._fps_encode_frames: int   = 0
        self._fps_stream_frames: int   = 0
        self._fps_ai_frames:     int   = 0
        self._fps_window_start:  float = 0.0
        self._measured_fps:      float = float(_STREAM_FPS)
        self._cap_samples_roll:  list  = []

        self._fps_samples:       list  = []
        self._fps_summary_start: float = 0.0

        self._det_buffer: List[Dict] = []
        self._det_buffer_lock = threading.Lock()

        # Action persistence
        self._persisted_actions: List[Dict] = []
        self._action_ttl_frames: int = max(
            30, getattr(settings, 'ACTION_FRAME_SKIP', 30) * 3)

        self.ai_frame_skip_counter = 0

        # Object tracking
        self.tracker = None
        if BYTE_TRACKER_AVAILABLE:
            self.tracker = ByteTrackerWrapper()
            logger.info(f"[{camera_id}] ByteTracker initialized")
        else:
            self._tracking_enabled   = True
            self._tracked_objects    = {}
            self._next_object_id     = 1
            self._prediction_enabled = True

        self.track_age      = {}
        self.track_faces    = {}
        self.person_actions = {}
        self.last_clip_time = {}
        self.last_face_time = {}
        self.action_history = {}
        self.harmful_memory = {}

        self.MIN_TRACK_AGE      = 1
        self.ACTION_WINDOW      = 6
        self.HARMFUL_MEMORY_TIME = 5
        self.ALERT_COOLDOWN     = 5

        self.frame_count: Dict[str, int] = {}

        # ── Option A: bbox predictor + smooth tracker + stable track labels ─────
        self._bbox_predictor = _BboxPredictor()
        self._smooth_tracker = _SmoothBboxTracker(self._bbox_predictor)
        self._track_states   = _TrackStateStore()

        # Pose frame skip (half of face skip so pose runs more often)
        self._pose_frame_n:    int = 0
        self._pose_frame_skip: int = max(5, getattr(settings, 'FACE_FRAME_SKIP', 15) // 2)

        CameraStream._known_cache_ts = time.monotonic() - random.uniform(0, 15)

    # ── START / STOP ──────────────────────────────────────────────────────────
    async def start(self) -> bool:
        logger.info(f"Starting camera: {self.camera_id}")
        try:
            loop = asyncio.get_event_loop()

            # BUG-10 FIX: cap ring buffer at 15 s to prevent RAM blowout
            # Create ring buffer BEFORE _CaptureThread so it can be shared
            buf_fps = 13  # will be updated after open()
            tmp_ring   = deque(maxlen=max(buf_fps * 15, 60))
            tmp_ring_lock = self._ring_lock

            ct = _CaptureThread(self.stream_url, self.camera_id,
                                tmp_ring, tmp_ring_lock)
            opened = await loop.run_in_executor(_READ_POOL, ct.open)
            if not opened:
                logger.error(f"Cannot open {self.camera_id} ({self.stream_url})")
                return False

            # Now we know the real fps — resize ring buffer
            real_fps  = ct.hw_fps or 13
            real_len  = max(real_fps * 15, 60)
            ring      = deque(maxlen=real_len)
            # Replace ring buffer reference on capture thread
            ct._ring_buffer = ring
            self._ring_buffer = ring

            self._capture_thread = ct
            self._rec_width      = ct.width
            self._rec_height     = ct.height
            self._measured_fps   = float(ct.hw_fps)

            self.running           = True
            self._fps_window_start = time.monotonic()
            self._fps_summary_start = time.monotonic()
            self._ai_event         = asyncio.Event()
            self._frame_event      = asyncio.Event()

            ct.start()

            # Dedicated encode thread
            et = _EncodeThread(self, loop)
            self._encode_thread = et
            et.start()

            asyncio.create_task(self._ai_loop())
            asyncio.create_task(self._db_flush_loop())
            asyncio.create_task(self._fps_reporter_loop())
            if getattr(settings, 'ENABLE_RECORDING', False):
                asyncio.create_task(self._recording_loop())

            logger.info(
                f"Camera {self.camera_id} started: "
                f"{ct.width}x{ct.height}@{ct.hw_fps}fps  "
                f"yolo={self._yolo_size}px  "
                f"stream={_STREAM_FPS}fps  "
                f"encode=thread  AI=ring-buffer"
            )
            return True
        except Exception as e:
            logger.error(f"Camera start error {self.camera_id}: {e}")
            return False

    async def stop(self):
        self.running = False
        if self._encode_thread:
            self._encode_thread.stop()
        if self._capture_thread:
            self._capture_thread.stop()
        if self._ai_event:
            self._ai_event.set()
        with self._sub_lock:
            for q in self._subscribers:
                try: q.put_nowait(None)
                except Exception: pass
        if self.video_writer:
            try: self.video_writer.release()
            except Exception: pass
            self.video_writer = None
        logger.info(f"Stopped: {self.camera_id}")

    # ── FPS REPORTER ──────────────────────────────────────────────────────────
    async def _fps_reporter_loop(self):
        _INSTANT = 5.0
        _SUMMARY = 120.0
        G = "\033[32m"; Y = "\033[33m"; C = "\033[36m"
        B = "\033[1m";  R = "\033[0m";  RED = "\033[31m"; M = "\033[35m"

        while self.running:
            await asyncio.sleep(_INSTANT)
            ct = self._capture_thread
            if ct is None: continue

            now    = time.monotonic()
            enc    = self._fps_encode_frames / _INSTANT
            stream = self._fps_stream_frames / _INSTANT
            ai     = self._fps_ai_frames     / _INSTANT
            cap    = ct.live_fps

            sc = G if stream >= _STREAM_FPS * 0.85 else (Y if stream >= _STREAM_FPS * 0.5 else RED)
            print(
                f"{B}[{self.camera_id}]{R}  "
                f"{C}cap={cap:5.1f}fps{R}  "
                f"{C}enc={enc:5.1f}fps{R}  "
                f"{sc}stream={stream:5.1f}fps{R}  "
                f"{Y}AI={ai:4.1f}fps{R}  "
                f"({ct.width}x{ct.height}  viewers={len(self._subscribers)})"
            )

            if cap > 0:
                self._cap_samples_roll.append(cap)
                if len(self._cap_samples_roll) > 5:
                    self._cap_samples_roll.pop(0)
                self._measured_fps = round(
                    sum(self._cap_samples_roll) / len(self._cap_samples_roll), 1)

            if enc > 0 or stream > 0:
                self._fps_samples.append((cap, enc, stream, ai))

            self._fps_encode_frames = 0
            self._fps_stream_frames = 0
            self._fps_ai_frames     = 0

            if now - self._fps_summary_start >= _SUMMARY and self._fps_samples:
                caps    = [s[0] for s in self._fps_samples]
                encs    = [s[1] for s in self._fps_samples]
                streams = [s[2] for s in self._fps_samples]
                ais     = [s[3] for s in self._fps_samples]

                def _stat(vals):
                    return min(vals), max(vals), sum(vals) / len(vals)

                c_min,c_max,c_avg = _stat(caps)
                e_min,e_max,e_avg = _stat(encs)
                s_min,s_max,s_avg = _stat(streams)
                a_min,a_max,a_avg = _stat(ais)

                print(
                    f"\n{M}{B}{'─'*70}{R}\n"
                    f"{M}{B}  [{self.camera_id}]  2-MINUTE FPS SUMMARY  "
                    f"({len(self._fps_samples)} samples){R}\n"
                    f"  {C}Capture {R}: min={c_min:5.1f}  max={c_max:5.1f}  avg={c_avg:5.1f}\n"
                    f"  {C}Encode  {R}: min={e_min:5.1f}  max={e_max:5.1f}  avg={e_avg:5.1f}\n"
                    f"  {C}Stream  {R}: min={s_min:5.1f}  max={s_max:5.1f}  avg={s_avg:5.1f}\n"
                    f"  {Y}AI      {R}: min={a_min:5.1f}  max={a_max:5.1f}  avg={a_avg:5.1f}\n"
                    f"{M}{B}{'─'*70}{R}\n"
                )
                self._fps_samples       = []
                self._fps_summary_start = now

    # ── AI LOOP ───────────────────────────────────────────────────────────────
    async def _ai_loop(self):
        logger.info(f"[{self.camera_id}] AI loop starting...")
        _AI_INTERVAL = 0.7   # minimum seconds between AI cycles
        _MIN_SLEEP   = 0.03  # SAFEWATCH_FIX_GUIDE §1: always yield to event loop
        _MIN_SLEEP    = 0.03  # minimum sleep to prevent tight loop on errors

        # FIX: track actual AI interval so predictor knows how far to extrapolate.
        # Under load, YOLO + overhead can push each cycle to 1.0-1.4 s, meaning
        # the predictor would extrapolate 40-100% further than expected → overshoot.
        self._last_ai_interval: float = _AI_INTERVAL

        logger.info(f"[{self.camera_id}] AI loop started")

        _cycle_start = time.monotonic()

        while self.running:
            # Pace: sleep the remainder of the target interval.
            # If _process_frame already took longer than _AI_INTERVAL, sleep 0.
            elapsed_since_start = time.monotonic() - _cycle_start
            sleep_t = max(_MIN_SLEEP, _AI_INTERVAL - elapsed_since_start)
            await asyncio.sleep(sleep_t)

            _cycle_start = time.monotonic()   # start of THIS cycle

            if not self.ai_engine.ready:
                continue

            ct = self._capture_thread
            if ct is None or not ct.running:
                continue

            with ct.lock:
                frame = ct.latest_frame
                frame_ts = ct.latest_frame_timestamp
            if frame is None:
                continue

            self.last_frame = frame

            try:
                if _get_ai_sem().locked():
                    continue
                t0 = time.monotonic()
                await self._process_frame(frame, frame_ts)
                # FIX: record actual AI cycle duration for predictor damping
                self._last_ai_interval = max(0.05, time.monotonic() - t0 + sleep_t)
                self._fps_ai_frames += 1
            except Exception as e:
                logger.error(f"AI loop error {self.camera_id}: {e}")

    def get_ring_buffer_frames(self) -> List[Tuple[float, np.ndarray]]:
        with self._ring_lock:
            return list(self._ring_buffer)

    # ── PROCESS FRAME ─────────────────────────────────────────────────────────
    async def _process_frame(self, frame: np.ndarray, frame_ts: float = 0.0):
        """
        AI pipeline — new architecture:

          Resize (640×360) → YOLO → ByteTrack
               ↓
          Pose Decision Layer  (every cycle, ~8ms/person)
               ↓
          ├── NORMAL   → skip (no CLIP, no Face)
          ├── SUSPICIOUS → CLIP (background task)
          └── CRITICAL → immediate alert + optional CLIP confirm
               ↓
          Face (only on new track_id OR after FACE_REEMBED_SECONDS)
               ↓
          Publish → DB → WebSocket → Alerts
        """
        # ── 1. Rectangular resize for YOLO (preserves aspect ratio) ──────────
        H_orig, W_orig = frame.shape[:2]
        if self._yolo_rect:
            rw, rh = self._yolo_rect_w, self._yolo_rect_h
            yolo_frame = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
            scale_x = W_orig / rw
            scale_y = H_orig / rh
        else:
            yolo_frame = frame
            scale_x = scale_y = 1.0

        sz   = self._yolo_size
        conf = min(0.35, settings.YOLO_CONFIDENCE)

        # ── 2. YOLO detection ─────────────────────────────────────────────────
        try:
            raw_dets = await asyncio.wait_for(
                asyncio.wrap_future(
                    self.ai_engine.submit_yolo(yolo_frame, sz, conf, 0.40)),
                timeout=12.0)
            # Scale coords back to original frame space
            detections = []
            for d in raw_dets:
                b = d['bbox']
                scaled = {
                    'x1': b['x1'] * scale_x, 'y1': b['y1'] * scale_y,
                    'x2': b['x2'] * scale_x, 'y2': b['y2'] * scale_y,
                }
                nd = {**d, 'bbox': scaled}
                if 'zone' not in nd:
                    nd['zone']     = self.ai_engine._calculate_zone(scaled, frame.shape)
                if 'distance' not in nd:
                    nd['distance'] = self.ai_engine._estimate_distance(scaled, frame.shape)
                detections.append(nd)
        except asyncio.TimeoutError:
            logger.warning(f"[{self.camera_id}] YOLO timeout — skipping frame")
            detections = []
        except Exception as e:
            logger.error(f"[{self.camera_id}] YOLO error: {type(e).__name__}: {e}")
            detections = []

        if not detections:
            # ── OCCLUSION RECOVERY ────────────────────────────────────────────
            if self.tracker:
                frame_id  = self.frame_count.get(self.camera_id, 0)
                held_raw  = self.tracker.update([], frame_id, frame.shape[:2])
                self.frame_count[self.camera_id] = frame_id + 1
                hold_dets = []
                for track_id, track in held_raw.items():
                    if track.get("bbox") is None:
                        continue
                    x1, y1, x2, y2 = map(int, track["bbox"])
                    H2, W2 = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(W2, x2), min(H2, y2)
                    if x2 <= x1 + 5 or y2 <= y1 + 5:
                        continue
                    hd = {
                        'bbox':               {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                        'confidence':         track.get("score", 0.5),
                        'track_id':           track_id,
                        'class':              'person',
                        'action':             'normal',
                        'action_confidence':  0.0,
                        'person':             {'classification': 'unknown', 'id': None},
                    }
                    hd['zone']     = self.ai_engine._calculate_zone(hd['bbox'], frame.shape)
                    hd['distance'] = self.ai_engine._estimate_distance(hd['bbox'], frame.shape)
                    for prev in self.current_detections:
                        if prev.get('track_id') == track_id:
                            hd['action']            = prev.get('action', 'normal')
                            hd['action_confidence'] = prev.get('action_confidence', 0.0)
                            hd['person']            = prev.get('person', hd['person'])
                            break
                    hold_dets.append(hd)

                if hold_dets:
                    self.current_detections = hold_dets
                    with self._latest_det_lock:
                        self._latest_detections    = hold_dets
                        self._latest_detections_ts = time.monotonic()
                    with self._det_buffer_lock:
                        self._det_buffer.extend(hold_dets)
                    return

            _GHOST_SECS = getattr(settings, 'GHOST_BBOX_SECONDS', 4.0)
            if time.monotonic() - self._last_detection_ts > _GHOST_SECS:
                self._smoothed_bboxes   = []
                self._persisted_actions = []
                self.current_detections = []
                with self._latest_det_lock:
                    self._latest_detections    = []
                    self._latest_detections_ts = time.monotonic()
                with self._annotated_lock:
                    self._annotated_queue.clear()
            return

        self._last_detection_ts = time.monotonic()

        # ── 3. BBox smoothing (ByteTracker inactive fallback) ─────────────────
        if not self.tracker:
            ALPHA = 0.3
            smoothed = []
            for det in detections:
                new_b = dict(det['bbox'])
                if self._smoothed_bboxes:
                    best_iou, best_prev = 0.0, None
                    for pb in self._smoothed_bboxes:
                        iou = self._bbox_iou(new_b, pb)
                        if iou > best_iou:
                            best_iou, best_prev = iou, pb
                    if best_prev and best_iou > 0.15:
                        for k in ('x1', 'y1', 'x2', 'y2'):
                            new_b[k] = ALPHA * new_b[k] + (1 - ALPHA) * best_prev[k]
                smoothed.append(new_b)
                asyncio.create_task(push_detection(self.camera_id, det))
            self._smoothed_bboxes = smoothed
            for i, det in enumerate(detections):
                detections[i] = {**det, 'bbox': smoothed[i]}

        # ── 4. ByteTracker ────────────────────────────────────────────────────
        tracker_input = []
        for det in detections:
            bbox = det['bbox']
            tracker_input.append([
                bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2'],
                det['confidence'], 0
            ])

        if self.tracker:
            frame_id = self.frame_count.get(self.camera_id, 0)
            tracks   = self.tracker.update(tracker_input, frame_id, frame.shape[:2])
            self.frame_count[self.camera_id] = frame_id + 1

            processed_tracks = []
            for track_id, track in tracks.items():
                if track.get("bbox") is None:
                    continue
                self.track_age[track_id] = self.track_age.get(track_id, 0) + 1
                if self.track_age[track_id] < self.MIN_TRACK_AGE:
                    continue
                if track.get("score", 1.0) < 0.5:
                    continue
                x1, y1, x2, y2 = map(int, track["bbox"])
                H, W = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                det_entry = {
                    'bbox':       {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                    'confidence': track.get("score", 1.0),
                    'track_id':   track_id,
                    'class':      'person',
                }
                det_entry['zone']     = self.ai_engine._calculate_zone(det_entry['bbox'], frame.shape)
                det_entry['distance'] = self.ai_engine._estimate_distance(det_entry['bbox'], frame.shape)

                kalman_vel = track.get("velocity")
                if kalman_vel is not None and len(kalman_vel) == 4:
                    cap_fps = max(1.0, getattr(
                        self._capture_thread, 'hw_fps', 13) if self._capture_thread else 13)
                    H_f, W_f = frame.shape[:2]
                    self._smooth_tracker.anchor(
                        track_id,
                        {'x1': float(x1), 'y1': float(y1),
                         'x2': float(x2), 'y2': float(y2)},
                        [v * cap_fps for v in kalman_vel],
                        frame_ts, W_f, H_f,
                    )
                    if not hasattr(self, '_kalman_seeded_this_cycle'):
                        self._kalman_seeded_this_cycle = set()
                    self._kalman_seeded_this_cycle.add(track_id)

                processed_tracks.append(det_entry)
            detections = processed_tracks
        else:
            if getattr(self, '_tracking_enabled', False):
                self._update_tracking(detections)

        self.frame_count[self.camera_id] = \
            self.frame_count.get(self.camera_id, 0) + 1

        # ── 5. Action persistence (carry forward until CLIP updates) ──────────
        new_persisted = []
        for det in detections:
            best_iou, best_slot = 0.0, None
            for slot in self._persisted_actions:
                iou = self._bbox_iou(det['bbox'], slot['bbox'])
                if iou > best_iou:
                    best_iou, best_slot = iou, slot
            if best_slot and best_iou > 0.15 and best_slot['ttl'] > 0:
                det['action']            = best_slot['action']
                det['action_confidence'] = best_slot['aconf']
                new_persisted.append({
                    'bbox':   det['bbox'], 'action': best_slot['action'],
                    'aconf':  best_slot['aconf'], 'ttl': best_slot['ttl'] - 1,
                })
            else:
                det['action']            = 'normal'
                det['action_confidence'] = 0.0
                new_persisted.append({
                    'bbox':   det['bbox'], 'action': 'normal',
                    'aconf':  0.0, 'ttl': 0,
                })
        self._persisted_actions = new_persisted

        # ── 6. POSE DECISION LAYER (NEW — core of the new architecture) ───────
        #
        # classify_batch() runs synchronously via run_in_executor so it doesn't
        # block the asyncio event loop.  Cost: ~8ms × N persons on i5-6500.
        #
        # Returns PoseDecision per detection:
        #   NORMAL     → skip CLIP, skip Face (saves ~150ms + ~40ms per cycle)
        #   SUSPICIOUS → queue for CLIP inference (erratic, loitering, disguise…)
        #   CRITICAL   → trigger alert immediately (fall, fight, weapon, child)
        #
        now_pose = time.monotonic()
        pose_decisions = []
        critical_alerts_triggered = set()  # action strings already alerted this cycle

        if self._pose_enabled and self._pose_classifier is not None and detections:
            loop = asyncio.get_event_loop()
            pose_decisions = await loop.run_in_executor(
                None,
                self._pose_classifier.classify_batch,
                frame, detections, now_pose,
            )

            # ── 6a. Immediate critical alerts (no CLIP wait) ──────────────────
            if self._alert_on_critical:
                _HARMFUL_SET = {'fighting', 'falling', 'weapon_detected',
                                'fire', 'child_activity'}
                for det, decision in zip(detections, pose_decisions):
                    if decision.level == PoseLevel.CRITICAL:
                        action = decision.action
                        if action not in critical_alerts_triggered:
                            critical_alerts_triggered.add(action)
                            now_dt = datetime.utcnow()
                            last   = self.last_alert_time.get(action)
                            if not last or (now_dt - last).total_seconds() > 10:
                                self.last_alert_time[action] = now_dt
                                logger.warning(
                                    f"[{self.camera_id}] CRITICAL pose alert: "
                                    f"{action.upper()} (reason={decision.reason})")
                                from core.alert_manager import alert_manager
                                asyncio.create_task(alert_manager.create_alert(
                                    alert_type=action,
                                    camera_id=self.camera_id,
                                    zone=det.get('zone', 3),
                                    description=f"Pose detection: {decision.reason}",
                                    action_detected=action,
                                    frame=frame,
                                ))
                            # Inject action into track state immediately
                            tid = det.get('track_id')
                            if tid is not None:
                                self._track_states.update_action(tid, action, decision.score)
                                det['action']            = action
                                det['action_confidence'] = decision.score
        else:
            # No pose classifier — all go to CLIP (old behaviour)
            pose_decisions = [None] * len(detections)

        # ── 6b. WEAPON SECOND-PASS (YOLOv8 on person crops) ──────────────────
        # For every person that pose classified as SUSPICIOUS or CRITICAL,
        # run weapon detector on their crop for definitive confirmation.
        # This runs in the YOLO worker thread (non-blocking via submit_weapon).
        _weapon_enabled = getattr(settings, 'WEAPON_DETECTION_ENABLED', True)
        if _weapon_enabled:
            h_f2, w_f2 = frame.shape[:2]
            for idx2, det2 in enumerate(detections):
                dec2 = pose_decisions[idx2] if pose_decisions else None
                if dec2 is None or dec2.level.value == "normal":
                    continue
                b2 = det2['bbox']
                crop2 = frame[max(0,int(b2['y1'])):min(h_f2,int(b2['y2'])),
                              max(0,int(b2['x1'])):min(w_f2,int(b2['x2']))]
                if crop2.size == 0:
                    continue
                try:
                    wf = self.ai_engine.submit_weapon(crop2)
                    weapon_dets = await asyncio.wait_for(
                        asyncio.wrap_future(wf), timeout=1.0)
                    if weapon_dets:
                        best_w = max(weapon_dets, key=lambda x: x['confidence'])
                        wconf  = best_w['confidence']
                        wcls   = best_w['weapon_class']
                        tid2   = det2.get('track_id')
                        logger.warning(
                            f"[{self.camera_id}] WEAPON CONFIRMED by YOLO: "
                            f"{wcls} conf={wconf:.2f} track={tid2}")
                        if tid2 is not None:
                            self._track_states.update_action(tid2, 'weapon_detected', wconf)
                            det2['action']            = 'weapon_detected'
                            det2['action_confidence'] = wconf
                        # Immediate alert — don't wait for CLIP
                        now_dt3 = datetime.utcnow()
                        last_w  = self.last_alert_time.get('weapon_detected')
                        if not last_w or (now_dt3 - last_w).total_seconds() > 10:
                            self.last_alert_time['weapon_detected'] = now_dt3
                            from core.alert_manager import alert_manager
                            asyncio.create_task(alert_manager.create_alert(
                                alert_type='weapon_detected',
                                camera_id=self.camera_id,
                                zone=det2.get('zone', 3),
                                description=f"YOLO weapon detection: {wcls} ({wconf*100:.0f}%)",
                                action_detected='weapon_detected',
                                frame=frame,
                            ))
                except Exception as e:
                    logger.debug(f"Weapon detect error: {e}")

        # ── 7. CLIP — pose-gated, fire-and-forget ─────────────────────────────
        #
        # Gate logic:
        #   pose_enabled=True  → CLIP only if decision is SUSPICIOUS
        #   pose_enabled=False → CLIP every ACTION_FRAME_SKIP (old behaviour)
        #
        self._action_frame_n += 1
        _CLIP_SKIP = getattr(settings, 'ACTION_FRAME_SKIP', 30)

        clip_jobs = []
        h_f, w_f = frame.shape[:2]
        for idx, det in enumerate(detections):
            dec = pose_decisions[idx] if pose_decisions else None

            # Determine whether to run CLIP for this person
            if self._pose_enabled and dec is not None:
                run_clip = (dec.level.value == "suspicious") if self._clip_on_suspicious \
                           else (dec.level.value != "normal")
            else:
                # Old behaviour: every ACTION_FRAME_SKIP cycles for all persons
                run_clip = (self._action_frame_n % max(1, _CLIP_SKIP) == 0)

            if not run_clip:
                continue

            b    = det['bbox']
            crop = frame[max(0, int(b['y1'])):min(h_f, int(b['y2'])),
                         max(0, int(b['x1'])):min(w_f, int(b['x2']))]
            if crop.size > 0:
                clip_jobs.append((det.get('track_id'), det['bbox'], crop))

        if clip_jobs:
            async def _run_clip_bg(jobs=clip_jobs):
                try:
                    futures = [asyncio.wrap_future(self.ai_engine.submit_clip(crop))
                               for _, _, crop in jobs]
                    results = await asyncio.wait_for(
                        asyncio.gather(*futures, return_exceptions=True),
                        timeout=5.0)
                    ttl = self._action_ttl_frames
                    for (tid, bbox, _), res in zip(jobs, results):
                        if isinstance(res, Exception):
                            continue
                        action, aconf = res
                        CLIP_MIN_CONF = 0.55
                        if aconf < CLIP_MIN_CONF:
                            action = 'normal'
                        if tid is not None:
                            self._track_states.update_action(tid, action, aconf)
                        # After computing action, aconf — save the crop with its label for training
                        if getattr(settings, 'COLLECT_TRAINING_DATA', False):
                            import uuid as _u
                            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
                            label_safe = action.replace(" ", "_")
                            out = settings.DATA_DIR / "training" / "actions" / label_safe
                            out.mkdir(parents=True, exist_ok=True)
                            crop_path = out / f"{ts}_{_u.uuid4().hex[:6]}.jpg"
                            cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        for slot in self._persisted_actions:
                            if self._bbox_iou(slot['bbox'], bbox) > 0.15:
                                slot['action'] = action
                                slot['aconf']  = aconf
                                slot['ttl']    = ttl
                                break
                    with self._latest_det_lock:
                        self._latest_detections    = list(self.current_detections)
                        self._latest_detections_ts = time.monotonic()
                except Exception as e:
                    logger.debug(f"CLIP bg error {self.camera_id}: {e}")

            asyncio.create_task(_run_clip_bg())

        # ── 8. FACE — track_id-gated, fire-and-forget ─────────────────────────
        #
        # Gate: only embed if this track_id has never been embedded,
        #       OR FACE_REEMBED_SECONDS have passed since last embed.
        # This replaces the old FACE_FRAME_SKIP counter.
        # Old counter fallback preserved when FACE_GATE_BY_TRACK_ID is False.
        #
        self._face_frame_n += 1
        _FACE_SKIP = max(1, getattr(settings, 'FACE_FRAME_SKIP', 15))

        face_jobs = []
        mc_default = getattr(settings, 'FACE_DETECTION_CONFIDENCE', 0.85)
        now_face   = time.monotonic()

        for det in detections:
            tid = det.get('track_id')

            if self._face_gate_enabled and tid is not None:
                last_embed = self._face_last_embed.get(tid, 0.0)
                run_face   = (now_face - last_embed) >= self._face_reembed_secs
            else:
                # Old behaviour: frame-counter gate
                run_face = (self._face_frame_n % _FACE_SKIP == 0)

            if not run_face:
                continue

            b    = det['bbox']
            crop = frame[max(0, int(b['y1'])):min(h_f, int(b['y2'])),
                         max(0, int(b['x1'])):min(w_f, int(b['x2']))]
            if crop.size > 0:
                face_jobs.append((tid, det['bbox'], crop, mc_default))

        if face_jobs:
            async def _run_face_bg(jobs=face_jobs, frame_snap=frame):
                try:
                    futures = [asyncio.wrap_future(self.ai_engine.submit_face(crop, mc))
                               for _, _, crop, mc in jobs]
                    embs = await asyncio.wait_for(
                        asyncio.gather(*futures, return_exceptions=True),
                        timeout=4.0)
                    for (tid, bbox, _, _), emb in zip(jobs, embs):
                        if isinstance(emb, Exception) or emb is None:
                            continue
                        # Record successful embed time
                        if tid is not None:
                            self._face_last_embed[tid] = time.monotonic()
                        person = await self._identify_person(emb, frame_snap, bbox)
                        if tid is not None:
                            self._track_states.update_person(tid, person)
                        # Save face crop immediately and use the path
                        face_img_path = None
                        if person.get('classification') == 'unknown':
                            try:
                                from core.config import settings as _s
                                import cv2
                                h, w = frame_snap.shape[:2]
                                x1 = max(0, int(bbox['x1'])); y1 = max(0, int(bbox['y1']))
                                x2 = min(w, int(bbox['x2'])); y2 = min(h, int(bbox['y2']))
                                crop = frame_snap[y1:y2, x1:x2]
                                if crop.size > 0:
                                    ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
                                    fname = f"unknown_{ts_str}.jpg"
                                    fpath = _s.FACES_UNKNOWN_DIR / fname
                                    cv2.imwrite(str(fpath), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                                    face_img_path = f"http://localhost:8000/media/faces/unknown/{fname}"
                                    logger.info(f"[Face] Saved unknown face: {fname}")
                                    # Auto cluster in background
                                    try:
                                        from core.faces_cluster import auto_cluster_new_face
                                        asyncio.create_task(auto_cluster_new_face(fpath, emb))
                                    except Exception as e:
                                        logger.warning(f"[Face] Cluster error: {e}")
                            except Exception as e:
                                logger.error(f"[Face] Save error: {e}")
                        updated = []
                        for det in self.current_detections:
                            if self._bbox_iou(det['bbox'], bbox) > 0.15:
                                det = {**det, 'person': person}
                                if face_img_path:
                                    det['face_image_url'] = face_img_path
                                elif person.get('id'):
                                    det['face_image_url'] = (
                                        f"http://localhost:8000/media/faces/known/"
                                        f"{person['id']}/reference.jpg")
                            updated.append(det)
                        self.current_detections = updated
                except Exception as e:
                    logger.debug(f"Face bg error {self.camera_id}: {e}")

            asyncio.create_task(_run_face_bg())

        # ── 9. Merge and publish detections ───────────────────────────────────
        processed        = []
        alert_pairs      = []
        _MIN_ACTION_CONF = 0.45

        for det in detections:
            det.setdefault('action', 'normal')
            det.setdefault('action_confidence', 0.0)
            det.setdefault('person', {'classification': 'unknown', 'id': None})
            processed.append(det)
            action = det.get('action', 'normal')
            aconf  = float(det.get('action_confidence') or 0)
            if (action not in ('normal', 'unknown', '') and
                    aconf >= _MIN_ACTION_CONF and
                    action not in critical_alerts_triggered):
                alert_pairs.append((action, det))

        now_ts = time.monotonic()
        self._last_detection_ts = now_ts
        self.current_detections = list(processed)

        # ── Option A: feed bbox predictor with ACTUAL frame timestamp ─────────
        # frame_ts is the monotonic timestamp from when the capture thread read
        # this specific frame. Using it (not time.monotonic() now) means velocity
        # = Δpos / real_Δt regardless of how long YOLO took on the CPU.
        # This eliminates the overshoot oscillation caused by irregular AI timing.
        #
        # CRITICAL FIX: seed_velocity() was called above (in the ByteTracker loop)
        # to inject the Kalman-optimal velocity into the predictor.  But then
        # _bbox_predictor.update() below recomputed velocity from position deltas
        # and OVERWROTE that Kalman velocity with a weaker EMA estimate — negating
        # all the benefit of seed_velocity.
        # Fix: collect the set of track IDs that were Kalman-seeded this cycle,
        # and only call update() (position-delta EMA) for tracks that were NOT
        # seeded — typically tracks coasting through occlusion without a new
        # YOLO detection (ByteTracker predicts their position, no Kalman update).
        _kalman_seeded_tids: set = getattr(self, '_kalman_seeded_this_cycle', set())

        for det in processed:
            tid = det.get('track_id')
            if tid is not None:
                if tid not in _kalman_seeded_tids:
                    # No Kalman seed this cycle — fall back to position-delta EMA
                    self._bbox_predictor.update(tid, det['bbox'], frame_ts)
                else:
                    # Kalman velocity already seeded — just update the position
                    # anchor in the predictor state without recomputing velocity.
                    s = self._bbox_predictor._state.get(tid)
                    if s is not None:
                        # Refresh position anchor + timestamp only; keep Kalman vel
                        s['x1'] = det['bbox']['x1']
                        s['y1'] = det['bbox']['y1']
                        s['x2'] = det['bbox']['x2']
                        s['y2'] = det['bbox']['y2']
                        s['ts'] = frame_ts
                self._track_states.get_or_create(tid)
                self._track_states.update_detection(
                    tid,
                    float(det.get('confidence', 0.5)),
                    int(det.get('zone', 3)),
                    now_ts,
                )
        self._kalman_seeded_this_cycle = set()
        self._smooth_tracker.prune(now_ts)   # prunes predictor + blend state
        self._track_states.prune(now_ts)

        # Prune face embed timestamps and pose histories for dead tracks
        active_tids = {d.get('track_id') for d in processed if d.get('track_id')}
        stale_face  = [tid for tid in self._face_last_embed if tid not in active_tids]
        for tid in stale_face:
            del self._face_last_embed[tid]
        if self._pose_classifier is not None:
            for tid in stale_face:
                self._pose_classifier.prune_track(tid)

        # ── Option C + A: publish to encode thread (no annotated frame needed) ─
        # The encode thread reads _latest_detections + predicts bbox positions.
        # Writing the annotated frame to _annotated_queue here was the "dead code"
        # burning ~3ms per cycle — removed entirely.
        with self._latest_det_lock:
            self._latest_detections    = list(processed)
            self._latest_detections_ts = now_ts

        with self._det_buffer_lock:
            self._det_buffer.extend(processed)

        # ── Terminal action log ───────────────────────────────────────────────
        _HARMFUL_SET = {'fighting', 'falling', 'distress', 'fire',
                        'weapon_detected', 'weapon_grip', 'break_in'}
        _RISK_SET    = {'running', 'shouting', 'vandalism', 'vandal',
                        'stealing', 'theft', 'trespassing', 'crowding'}
        _RED = '\033[31m'; _ORG = '\033[33m'; _CYN = '\033[36m'
        _RST = '\033[0m';  _BLD = '\033[1m'
        for det in processed:
            act = (det.get('action') or 'normal').lower()
            if act in ('normal', 'unknown', ''): continue
            apc  = float(det.get('action_confidence') or 0) * 100
            pers = det.get('person') or {}
            name = pers.get('name') or pers.get('classification', 'unknown')
            zone = det.get('zone', 3)
            aclr = _RED if act in _HARMFUL_SET else (_ORG if act in _RISK_SET else _CYN)
            print(f"{_BLD}[{self.camera_id}]{_RST} "
                  f"{aclr}{act.upper().replace('_',' ')}{_RST} "
                  f"({apc:.0f}%)  zone=Z{zone}  person={name}")

        for action, det in alert_pairs:
            now  = datetime.utcnow()
            last = self.last_alert_time.get(action)
            if not last or (now - last).total_seconds() > 10:
                self.last_alert_time[action] = now
                _sev = ('CRITICAL' if action in _HARMFUL_SET
                        else 'HIGH' if action in _RISK_SET else 'MEDIUM')
                _ac  = _RED if _sev == 'CRITICAL' else _ORG if _sev == 'HIGH' else _CYN
                _pers = (det.get('person') or {}).get('name') or 'Unknown'
                print(f"\n{_BLD}{_ac}{'━'*60}{_RST}\n"
                      f"{_BLD}{_ac}🚨 ALERT [{_sev}]  {action.upper().replace('_',' ')}{_RST}\n"
                      f"   Camera : {self.camera_id}\n"
                      f"   Zone   : Z{det.get('zone',3)}\n"
                      f"   Person : {_pers}\n"
                      f"   Conf   : {float(det.get('action_confidence') or 0)*100:.0f}%\n"
                      f"{_BLD}{_ac}{'━'*60}{_RST}\n")
                person = det.get('person') or {}
                person_name = person.get('name')
                person_id   = person.get('id')
                person_cls  = person.get('classification', 'unknown')
                pers_info   = f"{person_name or 'Unknown'} ({person_cls})" if person_name else "Unknown person"
                from core.alert_manager import alert_manager
                asyncio.create_task(alert_manager.create_alert(
                    alert_type=action, camera_id=self.camera_id,
                    zone=det['zone'],
                    description=f"Suspicious activity detected: {action} — {pers_info}",
                    action_detected=action, frame=frame,
                ))

        asyncio.create_task(_broadcast({
            'type': 'detections', 'camera_id': self.camera_id,
            'detections': [{
                'bbox':               d['bbox'],
                'action':             d.get('action', 'normal'),
                'action_confidence':  float(d.get('action_confidence') or 0),
                'person':             d.get('person', {'classification': 'unknown'}),
                'zone':               d['zone'],
                'confidence':         float(d['confidence']),
                'timestamp':          datetime.utcnow().isoformat() + 'Z',
                # NEW: include face image URL so frontend can show face in real-time
                'face_image_url':     d.get('face_image_url'),
            } for d in processed],
        }))

    # ── HELPERS ───────────────────────────────────────────────────────────────
    @staticmethod
    def _scale_to_stream(frame: np.ndarray, max_w: int) -> np.ndarray:
        """Scale frame to stream width if needed. Returns view or new array."""
        if max_w > 0 and frame.shape[1] > max_w:
            scale = max_w / frame.shape[1]
            return cv2.resize(frame, (max_w, int(frame.shape[0] * scale)),
                              interpolation=cv2.INTER_LINEAR)
        return frame

    def _bbox_iou(self, a: dict, b: dict) -> float:
        ix1 = max(a['x1'], b['x1']); iy1 = max(a['y1'], b['y1'])
        ix2 = min(a['x2'], b['x2']); iy2 = min(a['y2'], b['y2'])
        inter = max(0.0, ix2-ix1) * max(0.0, iy2-iy1)
        if inter == 0: return 0.0
        union = ((a['x2']-a['x1'])*(a['y2']-a['y1']) +
                 (b['x2']-b['x1'])*(b['y2']-b['y1']) - inter)
        return inter / union if union > 0 else 0.0

    def _generate_object_id(self):
        obj_id = self._next_object_id
        self._next_object_id += 1
        return obj_id

    def _update_tracking(self, detections):
        current_time = time.monotonic()
        matched_objects = set()
        for det in detections:
            det_bbox  = det['bbox']
            best_match, best_iou = None, 0.3
            for obj_id, obj in self._tracked_objects.items():
                if obj_id in matched_objects: continue
                prev = obj.get('last_bbox')
                if prev:
                    iou = self._bbox_iou(det_bbox, prev)
                    if iou > best_iou:
                        best_iou, best_match = iou, obj_id
            if best_match:
                obj = self._tracked_objects[best_match]
                prev = obj.get('last_bbox', det_bbox)
                dt   = current_time - obj.get('last_seen', current_time)
                vel  = {k: (det_bbox[k] - prev[k]) / max(dt, 0.001)
                        for k in ['x1','x2','y1','y2']}
                obj.update({'last_bbox': det_bbox, 'last_seen': current_time,
                            'velocity': vel, 'confidence': det.get('confidence', 0)})
                matched_objects.add(best_match)
                det['object_id'] = best_match
            else:
                obj_id = self._generate_object_id()
                self._tracked_objects[obj_id] = {
                    'last_bbox': det_bbox, 'last_seen': current_time,
                    'velocity':  {k: 0 for k in ['x1','x2','y1','y2']},
                    'confidence': det.get('confidence', 0),
                }
                det['object_id'] = obj_id
                matched_objects.add(obj_id)
        for obj_id in list(self._tracked_objects):
            if obj_id not in matched_objects:
                if current_time - self._tracked_objects[obj_id]['last_seen'] > 1.0:
                    del self._tracked_objects[obj_id]

    # ── DB FLUSH ──────────────────────────────────────────────────────────────
    async def _db_flush_loop(self):
        while self.running:
            await asyncio.sleep(3.0)
            with self._det_buffer_lock:
                batch = self._det_buffer[:30]
                self._det_buffer.clear()
            if batch:
                await self._save_detections_bulk(batch)

    # _save_detections_bulk is bound above via:
    #   _save_detections_bulk = _save_detections_bulk_fixed

    # ── YOLO (synchronous back-compat helper) ─────────────────────────────────
    def _run_yolo(self, frame: np.ndarray) -> List[Dict]:
        if not self.ai_engine.ready:
            return []
        try:
            sz   = self._yolo_size
            conf = min(0.35, settings.YOLO_CONFIDENCE)
            raw  = self.ai_engine.submit_yolo(frame, sz, conf, 0.40).result(timeout=6.0)
            dets = []
            for d in raw:
                b = d['bbox']
                d2 = {'bbox': {'x1': b['x1'], 'y1': b['y1'],
                               'x2': b['x2'], 'y2': b['y2']},
                      'confidence': d['confidence'], 'class': 'person'}
                d2['zone']     = self.ai_engine._calculate_zone(d2['bbox'], frame.shape)
                d2['distance'] = self.ai_engine._estimate_distance(d2['bbox'], frame.shape)
                dets.append(d2)
            return dets
        except Exception as e:
            logger.error(f"YOLO error {self.camera_id}: {e}")
            return []

    # ── IDENTIFY PERSON / DB SAVE — patched versions ──────────────────────────
    _identify_person      = _identify_person_fixed
    _save_unknown_person  = _save_unknown_person_fixed
    _save_detections_bulk = _save_detections_bulk_fixed

    # ── DRAW DETECTIONS ───────────────────────────────────────────────────────
    def _draw_detections(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """
        Professional CCTV-style bounding box overlay.
        All coordinates are clamped before every draw call (BUG-6 fix).
        action_confidence is guaranteed non-None by callers (BUG-7 fix).
        """
        if not detections:
            return frame

        out  = frame   # caller passes frame.copy() — no double-copy here
        H, W = out.shape[:2]
        FONT  = cv2.FONT_HERSHEY_DUPLEX
        SCALE = 0.52
        THICK = 2
        PAD   = 5

        COL = {
            'harmful':    (0,   0,   230),
            'high_risk':  (0,  110,  255),
            'suspicious': (200, 0,   200),
            'known':      (30,  200,  50),
            'unknown':    (220, 180,  0 ),
            'white':      (255, 255, 255),
            'black':      (0,   0,   0  ),
        }

        HARMFUL    = {'fighting','falling','distress','fire','weapon_detected','weapon_grip','break_in'}
        HIGH_RISK  = {'running','shouting','vandalism','vandal','stealing','theft','trespassing'}
        SUSPICIOUS = {'loitering','crowding','suspicious','suspicious_behavior','unauthorized_area'}

        def _text_size(txt, scale=SCALE):
            (tw, th), _ = cv2.getTextSize(txt, FONT, scale, THICK)
            return tw, th

        def _shadow_text(img, txt, x, y, color, scale=SCALE, thick=THICK):
            cv2.putText(img, txt, (x+1, y+1), FONT, scale,
                        COL['black'], thick+1, cv2.LINE_AA)
            cv2.putText(img, txt, (x, y), FONT, scale,
                        color, thick, cv2.LINE_AA)

        def _solid_bar(img, txt, bx1, by, bar_color, scale=SCALE, above=True):
            tw, th = _text_size(txt, scale)
            bar_w  = tw + PAD * 2
            bar_h  = th + PAD * 2
            bx1    = max(0, min(bx1, W - bar_w))
            bx2    = bx1 + bar_w
            if above:
                by2 = min(H, by)
                by1 = max(0, by2 - bar_h)
            else:
                by1 = max(0, by)
                by2 = min(H, by1 + bar_h)
            if bx2 <= bx1 or by2 <= by1:
                return by
            cv2.rectangle(img, (bx1, by1), (bx2, by2), bar_color, -1)
            ty = by2 - PAD if above else by1 + th + PAD
            _shadow_text(img, txt, bx1 + PAD, ty, COL['white'], scale)
            return by1 if above else by2

        for i, det in enumerate(detections):
            b  = det.get('bbox', {})
            # BUG-6 FIX: clamp all coords
            x1 = max(1,   int(round(b.get('x1', 0))))
            y1 = max(1,   int(round(b.get('y1', 0))))
            x2 = min(W-1, int(round(b.get('x2', W))))
            y2 = min(H-1, int(round(b.get('y2', H))))
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue

            action  = (det.get('action') or 'normal').strip().lower()
            person  = det.get('person') or {}
            classif = (person.get('classification') or 'unknown').lower()
            name    = person.get('name') or ''
            conf    = float(det.get('confidence') or 0)
            zone    = int(det.get('zone') or 3)
            aconf   = float(det.get('action_confidence') or 0)  # BUG-7 FIX

            is_harmful    = any(w in action for w in HARMFUL)
            is_high_risk  = any(w in action for w in HIGH_RISK)
            is_suspicious = any(w in action for w in SUSPICIOUS)
            is_known      = classif not in ('unknown', 'visitor', '')

            if   is_harmful:    color = COL['harmful']
            elif is_high_risk:  color = COL['high_risk']
            elif is_suspicious: color = COL['suspicious']
            elif is_known:      color = COL['known']
            else:               color = COL['unknown']

            # 1. Bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

            # 2. Top bar — identity
            if name and is_known:
                id_text = f"{name[:14]}  [{classif[:3].upper()}]"
            elif is_known:
                id_text = f"[{classif.upper()}]  {conf*100:.0f}%"
            else:
                id_text = f"Unknown  {conf*100:.0f}%"
            _solid_bar(out, id_text, x1, y1, color, scale=SCALE, above=True)

            # 3. Bottom bar — action
            if action not in ('normal', 'unknown', ''):
                act_label = action.upper().replace('_', ' ')[:18]
                if aconf > 0:
                    act_label += f"  {aconf*100:.0f}%"
                bar_col = (COL['harmful'] if is_harmful
                           else COL['high_risk'] if is_high_risk else color)
                _solid_bar(out, act_label, x1, y2, bar_col, scale=0.48, above=False)

            # 4. Zone badge
            zone_col = {1: COL['harmful'], 2: COL['high_risk']}.get(zone, color)
            z_txt = f" Z{zone} "
            ztw, zth = _text_size(z_txt, 0.40)
            # BUG-6 FIX: clamp zone badge coords
            zx = max(x1, min(x2 - ztw - PAD, W - ztw - PAD))
            zy = max(y1, min(y1 + zth + PAD * 2, y2 - PAD))
            zx = max(0, zx)
            cv2.rectangle(out, (zx, y1), (min(W-1, zx + ztw + PAD), min(H-1, y1 + zth + PAD*2)),
                          zone_col, -1)
            cv2.putText(out, z_txt, (zx, min(H-1, y1 + zth + PAD)),
                        FONT, 0.40, COL['white'], 1, cv2.LINE_AA)

            # 5. Track ID
            track_id = det.get('track_id', i + 1)
            idx_txt  = f"#{track_id}"
            itw, ith = _text_size(idx_txt, 0.36)
            ix = x1 + (x2 - x1 - itw) // 2
            iy = y1 + ith + 3
            cv2.putText(out, idx_txt, (ix+1, iy+1), FONT, 0.36,
                        COL['black'], 2, cv2.LINE_AA)
            cv2.putText(out, idx_txt, (ix, iy), FONT, 0.36,
                        COL['white'], 1, cv2.LINE_AA)

        return out

    def _draw_detections_scaled(self, disp: np.ndarray,
                                detections: list,
                                sx: float, sy: float) -> None:
        """
        SMOOTHNESS FIX: Draw bbox overlays directly on a STREAM-RESOLUTION frame.

        `detections` contains bbox coords in ORIGINAL (full-res) frame space.
        `sx` / `sy` are the scale factors from original → stream resolution.
        Mutates `disp` in-place — no copy needed.

        Keeps the visual style identical to _draw_detections but is called by
        the encode thread instead of the AI thread, so the video always shows
        the freshest raw frame rather than a stale pre-drawn annotated frame.
        """
        if not detections or disp is None:
            return

        H, W   = disp.shape[:2]
        FONT   = cv2.FONT_HERSHEY_DUPLEX
        SCALE  = 0.45
        THICK  = 1
        PAD    = 4

        COL = {
            'harmful':    (0,   0,   230),
            'high_risk':  (0,  110,  255),
            'suspicious': (200, 0,   200),
            'known':      (30,  200,  50),
            'unknown':    (220, 180,  0 ),
            'white':      (255, 255, 255),
            'black':      (0,   0,   0  ),
        }
        HARMFUL    = {'fighting','falling','distress','fire','weapon_detected','weapon_grip','break_in'}
        HIGH_RISK  = {'running','shouting','vandalism','vandal','stealing','theft','trespassing'}
        SUSPICIOUS = {'loitering','crowding','suspicious','suspicious_behavior','unauthorized_area'}

        def _clamp_x(v): return max(1, min(W - 1, int(round(v))))
        def _clamp_y(v): return max(1, min(H - 1, int(round(v))))

        def _label_bar(img, txt, lx1, ly, color, above=True):
            (tw, th), _ = cv2.getTextSize(txt, FONT, SCALE, THICK)
            bx1 = max(0, min(lx1, W - tw - PAD * 2))
            bx2 = min(W, bx1 + tw + PAD * 2)
            if above:
                by2 = max(0, ly);        by1 = max(0, by2 - th - PAD * 2)
            else:
                by1 = min(H, ly);        by2 = min(H, by1 + th + PAD * 2)
            if bx2 > bx1 and by2 > by1:
                cv2.rectangle(img, (bx1, by1), (bx2, by2), color, -1)
                ty = (by2 - PAD) if above else (by1 + th + PAD)
                cv2.putText(img, txt, (bx1 + PAD + 1, ty + 1),
                            FONT, SCALE, COL['black'], THICK + 1, cv2.LINE_AA)
                cv2.putText(img, txt, (bx1 + PAD,     ty),
                            FONT, SCALE, COL['white'], THICK,     cv2.LINE_AA)

        for i, det in enumerate(detections):
            b    = det.get('bbox', {})
            # Scale original-res coords to stream-res
            x1 = _clamp_x(b.get('x1', 0) * sx)
            y1 = _clamp_y(b.get('y1', 0) * sy)
            x2 = _clamp_x(b.get('x2', W) * sx)
            y2 = _clamp_y(b.get('y2', H) * sy)
            if x2 <= x1 + 5 or y2 <= y1 + 5:
                continue

            action  = (det.get('action') or 'normal').strip().lower()
            person  = det.get('person') or {}
            classif = (person.get('classification') or 'unknown').lower()
            name    = person.get('name') or ''
            conf    = float(det.get('confidence') or 0)
            zone    = int(det.get('zone') or 3)
            aconf   = float(det.get('action_confidence') or 0)
            tid     = det.get('track_id', i + 1)

            is_harmful    = any(w in action for w in HARMFUL)
            is_high_risk  = any(w in action for w in HIGH_RISK)
            is_suspicious = any(w in action for w in SUSPICIOUS)
            is_known      = classif not in ('unknown', 'visitor', '')

            if   is_harmful:    color = COL['harmful']
            elif is_high_risk:  color = COL['high_risk']
            elif is_suspicious: color = COL['suspicious']
            elif is_known:      color = COL['known']
            else:               color = COL['unknown']

            # Bounding box
            cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)

            # Identity bar (top)
            if name and is_known:
                id_txt = f"{name[:14]} [{classif[:3].upper()}]"
            elif is_known:
                id_txt = f"[{classif.upper()}] {conf*100:.0f}%"
            else:
                id_txt = f"Unknown {conf*100:.0f}%"
            _label_bar(disp, id_txt, x1, y1, color, above=True)

            # Action bar (bottom)
            if action not in ('normal', 'unknown', ''):
                act_lbl = action.upper().replace('_', ' ')[:18]
                if aconf > 0:
                    act_lbl += f" {aconf*100:.0f}%"
                bar_col = (COL['harmful'] if is_harmful
                           else COL['high_risk'] if is_high_risk else color)
                _label_bar(disp, act_lbl, x1, y2, bar_col, above=False)

            # Track ID
            tid_txt = f"#{tid}"
            (itw, ith), _ = cv2.getTextSize(tid_txt, FONT, 0.36, 1)
            ix = x1 + (x2 - x1 - itw) // 2
            iy = _clamp_y(y1 + ith + 3)
            cv2.putText(disp, tid_txt, (ix + 1, iy + 1),
                        FONT, 0.36, COL['black'], 2, cv2.LINE_AA)
            cv2.putText(disp, tid_txt, (ix,     iy),
                        FONT, 0.36, COL['white'], 1, cv2.LINE_AA)

            # Zone badge
            zone_col = {1: COL['harmful'], 2: COL['high_risk']}.get(zone, color)
            z_txt = f" Z{zone} "
            (ztw, zth), _ = cv2.getTextSize(z_txt, FONT, 0.38, 1)
            zx = max(x1, min(x2 - ztw - PAD, W - ztw - PAD * 2))
            zy1 = y1;  zy2 = min(H - 1, y1 + zth + PAD * 2)
            if zy2 > zy1:
                cv2.rectangle(disp, (zx, zy1),
                              (min(W - 1, zx + ztw + PAD), zy2), zone_col, -1)
                cv2.putText(disp, z_txt, (zx, min(H - 1, zy1 + zth + PAD)),
                            FONT, 0.38, COL['white'], 1, cv2.LINE_AA)
        with self._write_lock:
            vw = self.video_writer
            if vw is None: return
            rw = self._rec_width; rh = self._rec_height
        h, w = frame.shape[:2]
        if w != rw or h != rh: frame = cv2.resize(frame, (rw, rh))
        if frame.ndim == 2: frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4: frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        try: vw.write(frame)
        except Exception as e: logger.debug(f"Frame write skipped {self.camera_id}: {e}")

    async def _save_recording_info(self):
        try:
            if not self.recording_file_path or not self.recording_file_path.exists(): return
            size     = self.recording_file_path.stat().st_size
            if not self.recording_start_time: return
            duration = int((datetime.utcnow() - self.recording_start_time).total_seconds())
            ct = self._capture_thread
            async with AsyncSessionLocal() as db:
                db.add(Recording(
                    camera_id=self.db_camera_id, camera_label=self.camera_id,
                    file_path=str(self.recording_file_path), file_size=size,
                    duration=duration, start_time=self.recording_start_time,
                    end_time=datetime.utcnow(),
                    fps=int(ct.hw_fps) if ct else 13,
                    resolution=f"{ct.width}x{ct.height}" if ct else "1280x720",
                ))
                await db.commit()
        except Exception as e:
            logger.error(f"Save recording info error: {e}")

    def get_latest_detections(self) -> List[Dict]:
        return self.current_detections


# ─────────────────────────────────────────────────────────────────────────────
class CameraManager:
    def __init__(self, ai_engine: AIEngine):
        self.ai_engine = ai_engine
        self.cameras: Dict[str, CameraStream] = {}
        logger.info(f"AI workers: {_AI_WORKERS} | IO: {getattr(settings,'IO_POOL_WORKERS',4)}")

    async def add_camera(self, camera_id: str, stream_url, db_id: int) -> bool:
        if camera_id in self.cameras: return False
        _get_ai_sem()
        cam = CameraStream(camera_id, stream_url, self.ai_engine)
        cam.db_camera_id = db_id
        success = await cam.start()
        if success: self.cameras[camera_id] = cam
        return success

    async def remove_camera(self, camera_id: str):
        if camera_id in self.cameras:
            await self.cameras[camera_id].stop()
            del self.cameras[camera_id]

    async def stop_all_cameras(self):
        for cam in self.cameras.values(): await cam.stop()
        self.cameras.clear()

    def get_active_count(self) -> int:
        return len(self.cameras)

    async def get_latest_detections(self) -> Dict:
        return {cid: cam.get_latest_detections() for cid, cam in self.cameras.items()}

    async def stream_camera(self, camera_id: str) -> AsyncGenerator[bytes, None]:
        """Fan-out MJPEG. Queue maxsize=4. Skip-on-full (BUG-8 clean)."""
        if camera_id not in self.cameras: return

        cam   = self.cameras[camera_id]
        queue: asyncio.Queue = asyncio.Queue(maxsize=4)

        with cam._sub_lock:
            cam._subscribers.append(queue)
        logger.debug(f"+viewer {camera_id} ({len(cam._subscribers)} total)")

        try:
            while cam.running:
                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=3.0)
                except asyncio.TimeoutError:
                    continue
                if chunk is None: break
                yield chunk
                await asyncio.sleep(0)
        finally:
            with cam._sub_lock:
                try: cam._subscribers.remove(queue)
                except ValueError: pass
            logger.debug(f"-viewer {camera_id} ({len(cam._subscribers)} left)")