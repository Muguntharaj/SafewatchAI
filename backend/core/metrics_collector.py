"""
metrics_collector.py — SafeWatch AI
=====================================
Lightweight ring-buffer of per-camera performance metrics.

PURPOSE
───────
Feeds adaptive_tuner.py with real-time data about:
  bbox_drift_px    — distance (px) between Kalman-predicted and next YOLO bbox
  track_churn_rate — how often a person's track ID changes per minute
  yolo_ms          — YOLO inference latency
  clip_ms          — CLIP inference latency
  encode_fps       — encode thread actual output rate
  ai_interval_ms   — wall-clock gap between successive AI cycles
  yolo_miss_rate   — fraction of AI cycles returning 0 detections

DESIGN DECISIONS
───────────────
• Zero disk I/O — everything lives in deque ring buffers per camera.
• Thread-safe — a single lock guards all per-camera rings.
  _process_frame (asyncio thread) and adaptive_tuner (asyncio task) only
  read/write through the public API.
• Singleton: `from core.metrics_collector import metrics` anywhere.

HOW TO EMIT FROM camera_manager.py
────────────────────────────────────
  from core.metrics_collector import metrics

  # after each YOLO call:
  metrics.record_yolo(camera_id, ms=elapsed * 1000)

  # after each CLIP call:
  metrics.record_clip(camera_id, ms=elapsed * 1000)

  # when a Kalman-predicted bbox is compared to the incoming YOLO bbox:
  metrics.record_bbox_drift(camera_id, predicted_bbox, actual_bbox)

  # when a track_id flip is detected (new ID on same person):
  metrics.record_id_churn(camera_id)

  # from encode thread after each JPEG push:
  metrics.record_encode_frame(camera_id)

  # at the start of each _ai_loop cycle:
  metrics.record_ai_cycle(camera_id, interval_ms)

  # when YOLO returns 0 detections:
  metrics.record_yolo_miss(camera_id)
  # when YOLO returns ≥1 detections:
  metrics.record_yolo_hit(camera_id)
"""

import threading
import time
import math
from collections import deque
from typing import Dict, List, Optional


_YOLO_RING    = 60
_CLIP_RING    = 30
_DRIFT_RING   = 120
_CHURN_WINDOW = 60.0   # seconds for churn-rate sliding window
_ENCODE_RING  = 100
_AI_RING      = 50
_MISS_RING    = 50


class _CameraMetrics:
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self._lock = threading.Lock()
        self._yolo_ms:        deque = deque(maxlen=_YOLO_RING)
        self._clip_ms:        deque = deque(maxlen=_CLIP_RING)
        self._drift_px:       deque = deque(maxlen=_DRIFT_RING)
        self._churn_events:   deque = deque(maxlen=200)
        self._encode_ts:      deque = deque(maxlen=_ENCODE_RING)
        self._ai_interval_ms: deque = deque(maxlen=_AI_RING)
        self._yolo_miss:      deque = deque(maxlen=_MISS_RING)

    # ── Emitters ─────────────────────────────────────────────────────────────
    def record_yolo(self, ms: float) -> None:
        with self._lock: self._yolo_ms.append(ms)

    def record_clip(self, ms: float) -> None:
        with self._lock: self._clip_ms.append(ms)

    def record_bbox_drift(self, pred: dict, actual: dict) -> None:
        try:
            d = (math.hypot(actual['x1'] - pred['x1'], actual['y1'] - pred['y1']) +
                 math.hypot(actual['x2'] - pred['x2'], actual['y2'] - pred['y2'])) / 2.0
            with self._lock: self._drift_px.append(d)
        except (KeyError, TypeError):
            pass

    def record_id_churn(self) -> None:
        with self._lock: self._churn_events.append(time.monotonic())

    def record_encode_frame(self) -> None:
        with self._lock: self._encode_ts.append(time.monotonic())

    def record_ai_cycle(self, interval_ms: float) -> None:
        with self._lock: self._ai_interval_ms.append(interval_ms)

    def record_yolo_miss(self) -> None:
        with self._lock: self._yolo_miss.append(True)

    def record_yolo_hit(self) -> None:
        with self._lock: self._yolo_miss.append(False)

    # ── Snapshot ──────────────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        with self._lock:
            yolo_ms  = list(self._yolo_ms)
            clip_ms  = list(self._clip_ms)
            drift_px = list(self._drift_px)
            churn    = list(self._churn_events)
            enc_ts   = list(self._encode_ts)
            ai_ivl   = list(self._ai_interval_ms)
            misses   = list(self._yolo_miss)

        now = time.monotonic()

        def avg(lst):   return sum(lst) / len(lst) if lst else 0.0
        def pct(lst, p):
            if not lst: return 0.0
            s = sorted(lst)
            return s[max(0, int(len(s) * p / 100) - 1)]

        recent_churn = [t for t in churn if now - t <= _CHURN_WINDOW]
        enc_recent   = [t for t in enc_ts if now - t <= 5.0]

        return {
            "camera_id":           self.camera_id,
            "yolo_avg_ms":         round(avg(yolo_ms),   1),
            "yolo_p95_ms":         round(pct(yolo_ms, 95), 1),
            "clip_avg_ms":         round(avg(clip_ms),   1),
            "bbox_drift_avg_px":   round(avg(drift_px),  1),
            "bbox_drift_p95_px":   round(pct(drift_px, 95), 1),
            "track_churn_per_min": round(len(recent_churn) / (_CHURN_WINDOW / 60.0), 2),
            "encode_fps":          round(len(enc_recent) / 5.0, 1),
            "ai_interval_avg_ms":  round(avg(ai_ivl),   1),
            "yolo_miss_rate":      round(
                sum(1 for m in misses if m) / len(misses) if misses else 0.0, 3),
            "sample_counts": {
                "yolo":   len(yolo_ms),
                "clip":   len(clip_ms),
                "drift":  len(drift_px),
                "churn":  len(churn),
                "encode": len(enc_ts),
                "ai_ivl": len(ai_ivl),
            },
        }


class MetricsCollector:
    """
    Singleton registry of _CameraMetrics objects.

    Usage:
        from core.metrics_collector import metrics
        metrics.record_yolo("cam_1", ms=23.4)
        snap = metrics.snapshot("cam_1")
        all_snaps = metrics.all_snapshots()
    """

    def __init__(self):
        self._cameras: Dict[str, _CameraMetrics] = {}
        self._lock = threading.Lock()

    def _get(self, camera_id: str) -> _CameraMetrics:
        with self._lock:
            if camera_id not in self._cameras:
                self._cameras[camera_id] = _CameraMetrics(camera_id)
            return self._cameras[camera_id]

    def remove_camera(self, camera_id: str) -> None:
        with self._lock: self._cameras.pop(camera_id, None)

    def record_yolo(self, camera_id: str, ms: float) -> None:
        self._get(camera_id).record_yolo(ms)

    def record_clip(self, camera_id: str, ms: float) -> None:
        self._get(camera_id).record_clip(ms)

    def record_bbox_drift(self, camera_id: str, pred: dict, actual: dict) -> None:
        self._get(camera_id).record_bbox_drift(pred, actual)

    def record_id_churn(self, camera_id: str) -> None:
        self._get(camera_id).record_id_churn()

    def record_encode_frame(self, camera_id: str) -> None:
        self._get(camera_id).record_encode_frame()

    def record_ai_cycle(self, camera_id: str, interval_ms: float) -> None:
        self._get(camera_id).record_ai_cycle(interval_ms)

    def record_yolo_miss(self, camera_id: str) -> None:
        self._get(camera_id).record_yolo_miss()

    def record_yolo_hit(self, camera_id: str) -> None:
        self._get(camera_id).record_yolo_hit()

    def snapshot(self, camera_id: str) -> Optional[dict]:
        with self._lock: cam = self._cameras.get(camera_id)
        return cam.snapshot() if cam else None

    def all_snapshots(self) -> List[dict]:
        with self._lock: cams = list(self._cameras.values())
        return [c.snapshot() for c in cams]


# Module-level singleton
metrics = MetricsCollector()
