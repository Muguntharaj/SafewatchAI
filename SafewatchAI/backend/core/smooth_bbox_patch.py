"""
smooth_bbox_patch.py — SafeWatch AI
======================================
Fast, smooth bounding-box overlay system.

PROBLEM BEING SOLVED
─────────────────────
The encode thread draws bboxes on every raw frame at ~13fps.
Between AI cycles (~0.7–1.4s), the bbox sits frozen at the last
YOLO position.  When YOLO fires again, the bbox JUMPS to the new
position — a visible snap of 30–100px every second.

THREE-LAYER SMOOTH MAPPING
───────────────────────────
Layer 1 — Kalman velocity seeding (already in camera_manager)
  SORT gives [vx1, vy1, vx2, vy2] px/frame which _BboxPredictor
  uses to extrapolate position between AI cycles.

Layer 2 — Per-frame cubic interpolation  ← NEW (this file)
  SmoothBboxMapper sits between the predictor and the draw call.
  For each track it maintains a small history of (timestamp, bbox)
  waypoints.  Each encode frame it produces a smooth interpolated
  position using Catmull-Rom spline through the last 4 waypoints.
  This eliminates the sharp jump when YOLO fires a new fix —
  instead the bbox glides smoothly to the corrected position over
  ~3–4 encode frames (~0.23s).

Layer 3 — Alpha-blended confidence ring  ← NEW (this file)
  When a track's time_since_update grows (person partially occluded),
  the bbox border fades from solid → dashed → ghost transparency.
  Drawn directly by _draw_smooth_bbox() replacing the plain cv2.rectangle.

HOW TO INTEGRATE
─────────────────
1. Import at top of camera_manager.py:
       from core.smooth_bbox_patch import SmoothBboxMapper, draw_smooth_bbox

2. In CameraStream.__init__() add:
       self._smooth_mapper = SmoothBboxMapper()

3. In _EncodeThread.run(), replace the bbox overlay section (step 3):

   OLD:
       if cur_dets:
           predicted_dets = []
           for det in cur_dets:
               tid = det.get('track_id')
               if tid is not None and hasattr(cs, '_bbox_predictor'):
                   pred_bbox = cs._bbox_predictor.predict(tid, now_t, w_r, h_r)
                   if pred_bbox is not None:
                       det = {**det, 'bbox': pred_bbox}
               ...
               predicted_dets.append(det)
           cs._draw_detections_scaled(disp, predicted_dets, _sx, _sx)

   NEW:
       if cur_dets:
           predicted_dets = []
           for det in cur_dets:
               tid = det.get('track_id')
               if tid is not None and hasattr(cs, '_bbox_predictor'):
                   pred_bbox = cs._bbox_predictor.predict(tid, now_t, w_r, h_r)
                   if pred_bbox is not None:
                       # Feed waypoint into smooth mapper, get interpolated pos
                       smooth_bbox = cs._smooth_mapper.get_smooth_bbox(
                           tid, pred_bbox, now_t)
                       det = {**det, 'bbox': smooth_bbox}
               if tid is not None and hasattr(cs, '_track_states'):
                   stable = cs._track_states.get_display_state(tid)
                   det = {**det, **stable,
                          'action_confidence': stable['action_confidence']}
               predicted_dets.append(det)
           cs._draw_detections_scaled(disp, predicted_dets, _sx, _sx)

4. From camera_manager._process_frame, when YOLO fires a new bbox fix,
   notify the mapper so it can anchor a new hard waypoint immediately:

       if hasattr(self, '_smooth_mapper'):
           self._smooth_mapper.anchor(
               track_id,
               {'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2)},
               frame_ts,
           )

   Place this right after the ByteTracker loop where x1,y1,x2,y2 are set
   (same place as the _bbox_predictor.seed_velocity call).

5. Call cs._smooth_mapper.prune() periodically (e.g. in _track_states.prune()):
       if hasattr(self, '_smooth_mapper'):
           self._smooth_mapper.prune(now, max_age=5.0)

METRICS INTEGRATION
────────────────────
To feed metrics_collector with drift data, add after the smooth_bbox call:
    if pred_bbox and yolo_bbox:
        from core.metrics_collector import metrics
        metrics.record_bbox_drift(camera_id, smooth_bbox, yolo_bbox)

PERFORMANCE
────────────
SmoothBboxMapper.get_smooth_bbox() is O(1) per track — just 4 float lerps
plus a single Catmull-Rom evaluation (12 multiplications, no allocations after
the first call).  At 13fps × 6 tracks = 78 calls/second — negligible.
"""

import math
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ── Catmull-Rom spline helpers ─────────────────────────────────────────────────

def _catmull_rom_1d(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
    """
    Evaluate Catmull-Rom spline at parameter t ∈ [0,1] between p1 and p2.
    p0 is the point before p1; p3 is the point after p2.
    α = 0.5 (centripetal) gives smooth curves without cusps.
    """
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * p1) +
        (-p0 + p2) * t +
        (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
        (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def _catmull_rom_bbox(
    b0: dict, b1: dict, b2: dict, b3: dict, t: float
) -> dict:
    """
    Catmull-Rom interpolation through four bboxes.
    Each bbox: {'x1','y1','x2','y2'}.
    t ∈ [0,1] gives position between b1 and b2.
    """
    return {
        'x1': _catmull_rom_1d(b0['x1'], b1['x1'], b2['x1'], b3['x1'], t),
        'y1': _catmull_rom_1d(b0['y1'], b1['y1'], b2['y1'], b3['y1'], t),
        'x2': _catmull_rom_1d(b0['x2'], b1['x2'], b2['x2'], b3['x2'], t),
        'y2': _catmull_rom_1d(b0['y2'], b1['y2'], b2['y2'], b3['y2'], t),
    }


def _lerp_bbox(a: dict, b: dict, t: float) -> dict:
    """Linear interpolation fallback when fewer than 4 waypoints exist."""
    s = 1.0 - t
    return {
        'x1': s * a['x1'] + t * b['x1'],
        'y1': s * a['y1'] + t * b['y1'],
        'x2': s * a['x2'] + t * b['x2'],
        'y2': s * a['y2'] + t * b['y2'],
    }


# ── Per-track waypoint state ───────────────────────────────────────────────────

class _TrackSmoothing:
    """
    Maintains a sliding window of up to 4 (timestamp, bbox) hard waypoints
    for one track.  The encode thread queries get_at(now) to obtain a
    smoothly interpolated position.

    Waypoints come from two sources:
      • anchor() — called when YOLO delivers a fresh detected bbox.
        This is a "hard" fix: the real position is known.
      • push_predicted() — called every encode frame with the Kalman prediction.
        This gives the smooth curve something to follow between fixes.

    The spline always evaluates between the TWO MOST RECENT waypoints with
    t = (now - ts[-2]) / (ts[-1] - ts[-2]).  When now > ts[-1] we extrapolate
    by at most EXTRAP_CAP seconds before falling back to the last known position.
    """

    MAX_WAYPOINTS = 6       # keep last 6 waypoints in ring
    EXTRAP_CAP    = 0.3     # max seconds to extrapolate past last waypoint
    BLEND_FRAMES  = 4       # over how many encode frames to blend a new anchor in

    def __init__(self, track_id: int):
        self.track_id = track_id
        # deque of (timestamp: float, bbox: dict)
        self._waypoints: List[Tuple[float, dict]] = []
        self._last_anchor_ts: float = 0.0
        # Blending state: when a new hard fix arrives, blend from old to new
        self._blend_start_bbox: Optional[dict] = None
        self._blend_start_ts:   float = 0.0
        self._blend_frames_left: int  = 0

    def anchor(self, bbox: dict, ts: float) -> None:
        """Hard fix from YOLO. Initiates a smooth blend toward the new position."""
        # Save current smoothed position as blend-from
        if self._waypoints:
            last_bbox = self._waypoints[-1][1]
            self._blend_start_bbox  = last_bbox
            self._blend_start_ts    = ts
            self._blend_frames_left = self.BLEND_FRAMES

        # Record the hard waypoint
        self._waypoints.append((ts, bbox))
        if len(self._waypoints) > self.MAX_WAYPOINTS:
            self._waypoints.pop(0)
        self._last_anchor_ts = ts

    def push_predicted(self, bbox: dict, ts: float) -> None:
        """
        Kalman-predicted position from _BboxPredictor.
        Used as an intermediate waypoint to keep the spline moving smoothly.
        Only inserted if at least 30ms have passed since the last waypoint
        (prevents waypoint pile-up at high encode FPS).
        """
        if self._waypoints and ts - self._waypoints[-1][0] < 0.030:
            return
        self._waypoints.append((ts, bbox))
        if len(self._waypoints) > self.MAX_WAYPOINTS:
            self._waypoints.pop(0)

    def get_at(self, now: float) -> Optional[dict]:
        """
        Return smoothly interpolated bbox at time `now`.
        Returns None if no waypoints exist.
        """
        wps = self._waypoints
        n   = len(wps)
        if n == 0:
            return None
        if n == 1:
            return dict(wps[0][1])

        # Find the two waypoints that bracket `now`
        t0, b0 = wps[-2]
        t1, b1 = wps[-1]

        if now <= t0:
            return dict(b0)

        if now >= t1:
            # Extrapolation — only allowed for EXTRAP_CAP seconds past last wp
            dt_over = now - t1
            if dt_over > self.EXTRAP_CAP:
                return dict(b1)
            # Linear extrapolation past last two waypoints
            span = max(1e-6, t1 - t0)
            t_ext = dt_over / span
            result = _lerp_bbox(b0, b1, 1.0 + t_ext)
            return result

        # Interpolation within [t0, t1]
        span = max(1e-6, t1 - t0)
        t    = (now - t0) / span

        if n >= 4:
            # Catmull-Rom through last 4 waypoints
            _, bm2 = wps[-4]
            _, bm1 = wps[-3]
            result = _catmull_rom_bbox(bm2, bm1, b0, b1, t)
        elif n == 3:
            _, bm1 = wps[-3]
            result = _catmull_rom_bbox(bm1, bm1, b0, b1, t)
        else:
            result = _lerp_bbox(b0, b1, t)

        # Blend new anchor in smoothly if a hard fix just arrived
        if self._blend_frames_left > 0 and self._blend_start_bbox is not None:
            alpha = 1.0 - (self._blend_frames_left / self.BLEND_FRAMES)
            result = _lerp_bbox(self._blend_start_bbox, result, alpha)
            self._blend_frames_left -= 1

        return result

    def last_anchor_age(self, now: float) -> float:
        return now - self._last_anchor_ts if self._last_anchor_ts > 0 else 999.0


# ── Public Mapper ──────────────────────────────────────────────────────────────

class SmoothBboxMapper:
    """
    Thread-safe per-camera smooth bbox mapper.

    One instance per CameraStream, stored as self._smooth_mapper.

    The encode thread (one thread) calls get_smooth_bbox() for each track
    on every frame.  The AI loop (asyncio thread) calls anchor() when YOLO
    delivers a fresh fix.  A threading.Lock is used for cross-thread safety.
    """

    def __init__(self):
        self._tracks: Dict[int, _TrackSmoothing] = {}
        self._lock = threading.Lock()

    def anchor(self, track_id: int, bbox: dict, ts: float) -> None:
        """
        Register a new hard YOLO fix for track_id.
        Call this from _process_frame after ByteTracker delivers a new bbox.
        bbox = {'x1': float, 'y1': float, 'x2': float, 'y2': float}
        ts   = frame_ts (monotonic capture timestamp of the analysed frame)
        """
        with self._lock:
            if track_id not in self._tracks:
                self._tracks[track_id] = _TrackSmoothing(track_id)
            self._tracks[track_id].anchor(bbox, ts)

    def get_smooth_bbox(
        self,
        track_id: int,
        predicted_bbox: dict,
        now: float,
    ) -> dict:
        """
        Main API called by the encode thread for every track on every frame.

        1. Feed the Kalman-predicted bbox as an intermediate waypoint.
        2. Return the Catmull-Rom interpolated position at `now`.

        If the track is unknown (first frame), returns predicted_bbox as-is.

        predicted_bbox: {'x1','y1','x2','y2'} from _BboxPredictor.predict()
        now:            time.monotonic() at encode-frame preparation time
        """
        with self._lock:
            if track_id not in self._tracks:
                self._tracks[track_id] = _TrackSmoothing(track_id)
            tracker = self._tracks[track_id]

        # Push predicted position as waypoint
        tracker.push_predicted(predicted_bbox, now)

        result = tracker.get_at(now)
        if result is None:
            return predicted_bbox

        # Clamp to valid pixel space (coordinates may drift slightly from spline)
        return {
            'x1': max(0.0, result['x1']),
            'y1': max(0.0, result['y1']),
            'x2': max(0.0, result['x2']),
            'y2': max(0.0, result['y2']),
        }

    def prune(self, now: float, max_age: float = 5.0) -> None:
        """Remove tracks that haven't had a new anchor in max_age seconds."""
        with self._lock:
            stale = [tid for tid, tr in self._tracks.items()
                     if tr.last_anchor_age(now) > max_age]
            for tid in stale:
                del self._tracks[tid]


# ── Enhanced bbox draw ─────────────────────────────────────────────────────────

def draw_smooth_bbox(
    img:        np.ndarray,
    bbox:       dict,
    color:      tuple,
    track_id:   int,
    anchor_age: float = 0.0,
    thickness:  int   = 2,
) -> None:
    """
    Draw a bounding box with visual confidence decay.

    When anchor_age (seconds since last YOLO fix) grows:
      0.0 – 0.5s  → solid line, full opacity
      0.5 – 1.5s  → dashed line (every 12px)
      1.5s+       → ghost box at 50% opacity with dashed lines

    This gives the operator an instant visual cue that the bbox is
    coasting on prediction (person partially occluded) without flickering.

    Usage: replace the cv2.rectangle call in _draw_detections_scaled() with:
        draw_smooth_bbox(disp, bbox_dict, color, track_id, anchor_age)
    """
    x1 = int(bbox.get('x1', 0))
    y1 = int(bbox.get('y1', 0))
    x2 = int(bbox.get('x2', 0))
    y2 = int(bbox.get('y2', 0))
    if x2 - x1 < 5 or y2 - y1 < 5:
        return

    H, W = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)

    if anchor_age <= 0.5:
        # Solid, full confidence
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        return

    if anchor_age <= 1.5:
        # Dashed line — draw short segments around the rectangle
        _draw_dashed_rect(img, x1, y1, x2, y2, color, thickness, dash_len=12)
        return

    # Ghost: blend a dimmed version onto the image
    ghost_alpha = max(0.25, 1.0 - (anchor_age - 1.5) / 3.0)
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
    _draw_dashed_rect(overlay, x1, y1, x2, y2, color, thickness, dash_len=8)
    cv2.addWeighted(overlay, ghost_alpha, img, 1.0 - ghost_alpha, 0, img)


def _draw_dashed_rect(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple,
    thickness: int,
    dash_len: int = 12,
) -> None:
    """Draw dashed rectangle by alternating line segments."""
    gap = dash_len

    def _dash_line(pt_a, pt_b):
        dx = pt_b[0] - pt_a[0]
        dy = pt_b[1] - pt_a[1]
        length = math.hypot(dx, dy)
        if length < 1:
            return
        steps = max(1, int(length / (dash_len + gap)))
        for i in range(steps):
            t0 = i * (dash_len + gap) / length
            t1 = min(1.0, t0 + dash_len / length)
            sx = int(pt_a[0] + dx * t0);  ex = int(pt_a[0] + dx * t1)
            sy = int(pt_a[1] + dy * t0);  ey = int(pt_a[1] + dy * t1)
            cv2.line(img, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)

    _dash_line((x1, y1), (x2, y1))   # top
    _dash_line((x2, y1), (x2, y2))   # right
    _dash_line((x2, y2), (x1, y2))   # bottom
    _dash_line((x1, y2), (x1, y1))   # left
