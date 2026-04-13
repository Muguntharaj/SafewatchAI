"""
byte_tracker.py — SafeWatch AI
================================
Wraps the SORT Kalman tracker (sort.py) with the same interface that
camera_manager.py expects from ByteTrackerWrapper.update().

WHY WE SWITCHED FROM SimpleTracker TO SORT
───────────────────────────────────────────
SimpleTracker assigned a brand-new track ID to every detection on every
YOLO call.  This made both _BboxPredictor and _TrackStateStore in
camera_manager.py completely ineffective:

  _BboxPredictor: needs to see the SAME track_id in two consecutive calls
    to compute velocity = Δpos / Δt.  With a fresh ID each time,
    _state.get(tid) always returns None → velocity = 0 → no extrapolation.

  _TrackStateStore: stores CLIP labels, face identity, zone history per tid.
    A new ID every 0.7s means these are discarded every cycle → constant
    "Unknown / normal" flicker on screen regardless of what CLIP/Face found.

SORT fixes both by:
  1. Kalman filter maintains velocity state → accurate prediction between
     YOLO cycles → bbox follows person even during 0.7s AI gaps
  2. IoU-based Hungarian matching → same person keeps the same track_id
     across ALL YOLO cycles → labels and face identity persist

INTERFACE (unchanged from old ByteTrackerWrapper)
──────────────────────────────────────────────────
  wrapper = ByteTrackerWrapper()
  tracks  = wrapper.update(detections, frame_id, shape)
  # tracks: dict { track_id (int) → {"bbox": [x1,y1,x2,y2], "score": float,
  #                                   "velocity": [vx1,vy1,vx2,vy2]} }

The "velocity" key is NEW — camera_manager._BboxPredictor can read it
directly instead of computing velocity from consecutive positions, giving
the most accurate prediction possible (Kalman already has optimal velocity).

ALSO: camera_manager._process_frame should call predictor.seed_from_kalman()
if available, to initialise velocity from the Kalman estimate rather than
re-computing it from raw YOLO bbox positions.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from core.sort import Sort
    _SORT_OK = True
except ImportError:
    try:
        from sort import Sort
        _SORT_OK = True
    except ImportError:
        _SORT_OK = False
        logger.warning("sort.py not found — falling back to SimpleTracker (no Kalman)")


# ─────────────────────────────────────────────────────────────────────────────
# SimpleTracker fallback (only used if sort.py is missing)
# ─────────────────────────────────────────────────────────────────────────────
class _SimpleTracker:
    """Last-resort fallback. No Kalman, no ID persistence."""
    def __init__(self):
        self.next_id = 1

    def update(self, detections):
        results = {}
        for det in np.atleast_2d(detections):
            x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            score = float(det[4]) if det.shape[0] > 4 else 0.9
            results[self.next_id] = {
                "bbox":     [x1, y1, x2, y2],
                "score":    score,
                "velocity": [0.0, 0.0, 0.0, 0.0],
            }
            self.next_id += 1
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Public wrapper (camera_manager uses this)
# ─────────────────────────────────────────────────────────────────────────────
class ByteTrackerWrapper:
    """
    Drop-in replacement for the old ByteTrackerWrapper.
    Internally uses SORT (Kalman + Hungarian matching).

    Parameters tuned for SafeWatch CPU deployment:
      max_age    = 8   — keep track alive for 8 missed frames (~0.6s at 13fps)
                         handles brief occlusions without spawning new IDs
      min_hits   = 1   — show bbox immediately on first detection (no warm-up delay)
      iou_thresh = 0.15 — very permissive matching to handle fast movement
                         between 0.7s AI cycles (person may have moved ~100px)
    """

    def __init__(self):
        if _SORT_OK:
            self._sort = Sort(max_age=12, min_hits=1, iou_thresh=0.15)
            self._mode = "sort_kalman"
            logger.info("[ByteTrackerWrapper] SORT Kalman tracker initialised (max_age=12)")
        else:
            self._sort = _SimpleTracker()
            self._mode = "simple_fallback"
            logger.warning("[ByteTrackerWrapper] Using SimpleTracker fallback — "
                           "install filterpy for Kalman tracking: pip install filterpy")

    def update(self, detections: list, frame_id: int, shape: tuple) -> dict:
        """
        Args:
            detections: list of [x1, y1, x2, y2, score] or [x1..x2, score, class_id]
            frame_id:   current frame index (unused by SORT but kept for API compat)
            shape:      (height, width) of the full frame (unused by SORT)

        Returns:
            dict { track_id (int) → {
                "bbox":               [x1, y1, x2, y2],   # Kalman-filtered position
                "score":              float,
                "velocity":           [vx1, vy1, vx2, vy2], # px/frame
                "hit_streak":         int,   # consecutive matched frames (confidence proxy)
                "time_since_update":  int,   # frames since last YOLO match (0 = matched this frame)
            }}
        """
        if not detections:
            if _SORT_OK:
                self._sort.update([])   # let Kalman coast existing tracks
            return {}

        det = np.array(detections, dtype=np.float32)
        if det.ndim == 1:
            det = det.reshape(1, -1)
        # Strip class_id column if present — keep only first 5 cols
        if det.shape[1] > 5:
            det = det[:, :5]

        if self._mode == "sort_kalman":
            tracks = self._sort.update(det.tolist())
            results = {}
            for t in tracks:
                x1, y1, x2, y2 = t.bbox
                results[t.id] = {
                    "bbox":              [int(x1), int(y1), int(x2), int(y2)],
                    "score":             float(t.score),
                    "velocity":          list(t.velocity),
                    "hit_streak":        int(t.hit_streak),
                    "time_since_update": int(t.time_since_update),
                }
            return results
        else:
            # SimpleTracker fallback
            return self._sort.update(det)

    @property
    def mode(self) -> str:
        return self._mode