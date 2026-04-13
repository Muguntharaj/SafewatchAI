"""
adaptive_tuner.py — SafeWatch AI
==================================
Live control loop that watches MetricsCollector and adjusts runtime
constants every TUNE_INTERVAL seconds — zero model changes, zero retraining.

WHAT IT TUNES (in priority order)
───────────────────────────────────
1. _BboxPredictor._VEL_ALPHA
   Controls how fast velocity adapts to new Kalman data.
   High drift  → raise alpha (trust new vel faster).
   Low drift + oscillating → lower alpha (smooth out noise).
   Range: [0.50, 0.88].

2. _SmoothBboxTracker._BLEND_DUR
   Controls how long (seconds) the lerp blend runs when a new YOLO fix
   arrives.  High track churn (IDs flipping) often correlates with large
   position jumps — increase blend duration to hide them visually.
   Very low churn → reduce blend (less latency between real and drawn pos).
   Range: [0.10, 0.40].

3. ByteTrackerWrapper iou_thresh (SORT.iou_threshold)
   Controls how permissive the Hungarian matching is.
   High churn  → lower threshold (easier re-match = fewer ID flips).
   Zero churn  → raise threshold slightly (tighter = fewer ID merges).
   Range: [0.08, 0.30].

4. settings.DETECTION_FRAME_SKIP
   How often YOLO runs relative to capture frames.
   High YOLO timeout rate (>5%) → raise skip (less load).
   Range: [2, 6].

5. settings.JPEG_QUALITY
   Stream bandwidth vs encode CPU cost.
   encode_fps drops below 80% of target → lower quality.
   Range: [55, 80].

6. settings.AI_LATENCY_SECONDS
   How far behind live feed the AI analyses frames.
   AI is fast → reduce latency for tighter overlay.
   AI is slow → increase buffer.
   Range: [0.1, 1.5].

HOW IT WORKS — HOT RELOAD
───────────────────────────
The tuner writes adjustments into a LiveSettings dict that shadows
config.py.  camera_manager.py already reads settings through getattr(settings, ...)
which means any mutation to the settings object is immediately visible
to all running camera loops — no restart required.

The tuner ALSO directly patches _BboxPredictor and _SmoothBboxTracker class
attributes which are read on every predict()/get_smooth_bbox() call — safe
because Python float assignments are atomic at the CPython level.

SORT's iou_threshold is patched in-place on each camera's
ByteTrackerWrapper._sort object.

STARTUP
────────
Call this once from CameraManager.start() or from main.py lifespan:

    from core.adaptive_tuner import AdaptiveTuner
    tuner = AdaptiveTuner(camera_manager)
    asyncio.create_task(tuner.run())

The tuner task runs for the lifetime of the process.
It catches all exceptions internally and never crashes the main loop.
"""

import asyncio
import logging
import time
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.camera_manager import CameraManager

from core.metrics_collector import metrics

logger = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────────
TUNE_INTERVAL       = 60.0     # seconds between tuning passes
WARMUP_SAMPLES      = 20       # minimum samples before tuning starts

# _BboxPredictor._VEL_ALPHA bounds
VEL_ALPHA_MIN       = 0.50
VEL_ALPHA_MAX       = 0.88
VEL_ALPHA_STEP      = 0.05
VEL_ALPHA_DEFAULT   = 0.70

# _SmoothBboxTracker._BLEND_DUR bounds (seconds)
# This controls how long the lerp glide runs after each new YOLO fix.
# Longer = smoother but bbox lags more behind reality.
# Shorter = snappier but can still show micro-jumps on direction changes.
BLEND_DUR_MIN       = 0.10
BLEND_DUR_MAX       = 0.40
BLEND_DUR_STEP      = 0.05
BLEND_DUR_DEFAULT   = 0.23

# SORT iou_threshold bounds
IOU_MIN             = 0.08
IOU_MAX             = 0.30
IOU_STEP            = 0.02
IOU_DEFAULT         = 0.15

# DETECTION_FRAME_SKIP bounds
DFS_MIN             = 2
DFS_MAX             = 6
DFS_DEFAULT         = 3

# JPEG_QUALITY bounds
JPEG_MIN            = 55
JPEG_MAX            = 80
JPEG_STEP           = 5
JPEG_DEFAULT        = 70

# AI_LATENCY_SECONDS bounds
LAT_MIN             = 0.10
LAT_MAX             = 1.50
LAT_STEP            = 0.10
LAT_DEFAULT         = 0.30

# Trigger thresholds
DRIFT_HIGH_PX       = 25.0    # drift above → increase vel_alpha
DRIFT_LOW_PX        = 8.0     # drift below (+ oscillating) → decrease vel_alpha
CHURN_HIGH_PER_MIN  = 2.0     # churn above → loosen iou_thresh + extend blend
CHURN_LOW_PER_MIN   = 0.3     # churn below this → tighten iou_thresh + shorten blend
YOLO_TIMEOUT_RATE   = 0.05    # miss_rate above → raise frame skip
ENCODE_RATIO_LOW    = 0.80    # encode_fps below target * 0.80 → lower JPEG quality


class AdaptiveTuner:
    """
    Asyncio control loop. Instantiate once and call asyncio.create_task(tuner.run()).

    Args:
        camera_manager: the live CameraManager instance.
        interval:       seconds between tuning passes (default TUNE_INTERVAL).
    """

    def __init__(self, camera_manager: "CameraManager",
                 interval: float = TUNE_INTERVAL):
        self._cm       = camera_manager
        self._interval = interval
        self._running  = False

        # Current live parameter values
        self._vel_alpha  = VEL_ALPHA_DEFAULT
        self._blend_dur  = BLEND_DUR_DEFAULT
        self._iou_thresh = IOU_DEFAULT
        self._dfs        = DFS_DEFAULT
        self._jpeg_q     = JPEG_DEFAULT
        self._latency    = LAT_DEFAULT

        # Oscillation detection for vel_alpha: track last 3 drift values
        self._drift_history: list = []

    # ── Main loop ─────────────────────────────────────────────────────────────
    async def run(self) -> None:
        self._running = True
        logger.info("[AdaptiveTuner] Starting — first pass in %.0fs", self._interval)
        while self._running:
            try:
                await asyncio.sleep(self._interval)
                await self._tune_pass()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[AdaptiveTuner] Unexpected error: %s", e, exc_info=True)

    def stop(self) -> None:
        self._running = False

    # ── Single tuning pass ────────────────────────────────────────────────────
    async def _tune_pass(self) -> None:
        snaps = metrics.all_snapshots()
        if not snaps:
            return

        # Aggregate across cameras (average)
        def _avg_field(field):
            vals = [s[field] for s in snaps if s.get(field) is not None]
            return sum(vals) / len(vals) if vals else 0.0

        drift_avg   = _avg_field("bbox_drift_avg_px")
        churn_rate  = _avg_field("track_churn_per_min")
        yolo_miss   = _avg_field("yolo_miss_rate")
        encode_fps  = _avg_field("encode_fps")
        ai_ivl_ms   = _avg_field("ai_interval_avg_ms")
        yolo_avg_ms = _avg_field("yolo_avg_ms")

        # Check we have enough samples to be meaningful
        min_yolo_samples = min(s["sample_counts"]["yolo"] for s in snaps)
        if min_yolo_samples < WARMUP_SAMPLES:
            logger.debug("[AdaptiveTuner] Warmup: %d/%d YOLO samples",
                         min_yolo_samples, WARMUP_SAMPLES)
            return

        changed = []

        # ── 1. vel_alpha ───────────────────────────────────────────────────────
        self._drift_history.append(drift_avg)
        if len(self._drift_history) > 3:
            self._drift_history.pop(0)

        oscillating = (len(self._drift_history) == 3 and
                       self._drift_history[0] < self._drift_history[1] >
                       self._drift_history[2])

        if drift_avg > DRIFT_HIGH_PX:
            new_alpha = min(VEL_ALPHA_MAX,
                            self._vel_alpha + VEL_ALPHA_STEP)
            if new_alpha != self._vel_alpha:
                self._vel_alpha = new_alpha
                self._apply_vel_alpha(new_alpha)
                changed.append(f"vel_alpha↑ {new_alpha:.2f} (drift={drift_avg:.1f}px)")

        elif drift_avg < DRIFT_LOW_PX and oscillating:
            new_alpha = max(VEL_ALPHA_MIN,
                            self._vel_alpha - VEL_ALPHA_STEP)
            if new_alpha != self._vel_alpha:
                self._vel_alpha = new_alpha
                self._apply_vel_alpha(new_alpha)
                changed.append(f"vel_alpha↓ {new_alpha:.2f} (low drift, oscillating)")

        # ── 2. blend_dur (smooth bbox glide window) ────────────────────────────
        # High churn means lots of ID flips, which often coincide with large
        # position jumps → extend blend so the snap is hidden by the glide.
        # Very low churn means tracking is stable → shorten blend to keep the
        # bbox as close to the real person position as possible.
        if churn_rate > CHURN_HIGH_PER_MIN:
            new_blend = min(BLEND_DUR_MAX, self._blend_dur + BLEND_DUR_STEP)
            if new_blend != self._blend_dur:
                self._blend_dur = new_blend
                self._apply_blend_dur(new_blend)
                changed.append(
                    f"blend_dur↑ {new_blend:.2f}s (churn={churn_rate:.1f}/min)")
        elif churn_rate < CHURN_LOW_PER_MIN and self._blend_dur > BLEND_DUR_DEFAULT:
            new_blend = max(BLEND_DUR_DEFAULT, self._blend_dur - BLEND_DUR_STEP)
            if new_blend != self._blend_dur:
                self._blend_dur = new_blend
                self._apply_blend_dur(new_blend)
                changed.append(
                    f"blend_dur↓ {new_blend:.2f}s (churn low, tightening)")

        # ── 3. iou_thresh ──────────────────────────────────────────────────────
        if churn_rate > CHURN_HIGH_PER_MIN:
            new_iou = max(IOU_MIN, self._iou_thresh - IOU_STEP)
            if new_iou != self._iou_thresh:
                self._iou_thresh = new_iou
                self._apply_iou_thresh(new_iou)
                changed.append(f"iou_thresh↓ {new_iou:.2f} (churn={churn_rate:.1f}/min)")

        elif churn_rate < CHURN_LOW_PER_MIN and self._iou_thresh < IOU_DEFAULT:
            new_iou = min(IOU_DEFAULT, self._iou_thresh + IOU_STEP)
            if new_iou != self._iou_thresh:
                self._iou_thresh = new_iou
                self._apply_iou_thresh(new_iou)
                changed.append(f"iou_thresh↑ {new_iou:.2f} (churn low, restoring default)")

        # ── 4. DETECTION_FRAME_SKIP ────────────────────────────────────────────
        if yolo_miss > YOLO_TIMEOUT_RATE:
            new_dfs = min(DFS_MAX, self._dfs + 1)
            if new_dfs != self._dfs:
                self._dfs = new_dfs
                self._apply_setting("DETECTION_FRAME_SKIP", new_dfs)
                changed.append(f"frame_skip↑ {new_dfs} (miss_rate={yolo_miss:.1%})")

        elif yolo_miss < 0.01 and yolo_avg_ms < 30.0 and self._dfs > DFS_DEFAULT:
            new_dfs = max(DFS_DEFAULT, self._dfs - 1)
            if new_dfs != self._dfs:
                self._dfs = new_dfs
                self._apply_setting("DETECTION_FRAME_SKIP", new_dfs)
                changed.append(f"frame_skip↓ {new_dfs} (system healthy)")

        # ── 5. JPEG_QUALITY ────────────────────────────────────────────────────
        target_fps = getattr(self._get_settings(), 'STREAM_FPS', 13)
        if encode_fps > 0 and encode_fps < target_fps * ENCODE_RATIO_LOW:
            new_q = max(JPEG_MIN, self._jpeg_q - JPEG_STEP)
            if new_q != self._jpeg_q:
                self._jpeg_q = new_q
                self._apply_setting("JPEG_QUALITY", new_q)
                changed.append(
                    f"jpeg_quality↓ {new_q} (encode_fps={encode_fps:.1f}<{target_fps * ENCODE_RATIO_LOW:.1f})")

        elif encode_fps >= target_fps * 0.95 and self._jpeg_q < JPEG_DEFAULT:
            new_q = min(JPEG_DEFAULT, self._jpeg_q + JPEG_STEP)
            if new_q != self._jpeg_q:
                self._jpeg_q = new_q
                self._apply_setting("JPEG_QUALITY", new_q)
                changed.append(f"jpeg_quality↑ {new_q} (encode healthy)")

        # ── 6. AI_LATENCY_SECONDS ──────────────────────────────────────────────
        if ai_ivl_ms > 0:
            ideal_latency = round(max(LAT_MIN,
                                      min(LAT_MAX, ai_ivl_ms / 1000.0 * 0.5)), 2)
            current_lat   = getattr(self._get_settings(),
                                    'AI_LATENCY_SECONDS', self._latency)
            if abs(ideal_latency - current_lat) > LAT_STEP:
                self._latency = ideal_latency
                self._apply_setting("AI_LATENCY_SECONDS", ideal_latency)
                changed.append(f"ai_latency→ {ideal_latency:.2f}s "
                                f"(ai_interval={ai_ivl_ms:.0f}ms)")

        if changed:
            logger.info("[AdaptiveTuner] Adjustments: %s", " | ".join(changed))
        else:
            logger.debug(
                "[AdaptiveTuner] No changes (drift=%.1fpx churn=%.2f/min "
                "miss=%.1f%% enc=%.1ffps blend=%.2fs)",
                drift_avg, churn_rate, yolo_miss * 100, encode_fps, self._blend_dur)

    # ── Appliers ──────────────────────────────────────────────────────────────
    def _get_settings(self):
        try:
            from core.config import settings
            return settings
        except Exception:
            return None

    def _apply_setting(self, key: str, value) -> None:
        """Hot-patch the settings singleton. Immediately visible to all readers."""
        s = self._get_settings()
        if s is not None:
            try:
                object.__setattr__(s, key, value)
                logger.debug("[AdaptiveTuner] settings.%s = %s", key, value)
            except Exception as e:
                logger.warning("[AdaptiveTuner] Could not set %s: %s", key, e)

    def _apply_vel_alpha(self, alpha: float) -> None:
        """
        Patch _BboxPredictor._VEL_ALPHA class attribute.
        Python float assignments are atomic at CPython level — safe without lock.
        All running CameraStream instances share the same class attribute.
        """
        try:
            from core.camera_manager import _BboxPredictor
            _BboxPredictor._VEL_ALPHA = alpha
            logger.debug("[AdaptiveTuner] _BboxPredictor._VEL_ALPHA = %.2f", alpha)
        except Exception as e:
            logger.warning("[AdaptiveTuner] Could not patch _VEL_ALPHA: %s", e)

    def _apply_blend_dur(self, dur: float) -> None:
        """
        Patch _SmoothBboxTracker._BLEND_DUR class attribute.
        New blend windows started after this call will use the new duration.
        In-progress blends keep their original duration (stored per-track).
        """
        try:
            from core.camera_manager import _SmoothBboxTracker
            _SmoothBboxTracker._BLEND_DUR = dur
            logger.debug("[AdaptiveTuner] _SmoothBboxTracker._BLEND_DUR = %.2f", dur)
        except Exception as e:
            logger.warning("[AdaptiveTuner] Could not patch _BLEND_DUR: %s", e)

    def _apply_iou_thresh(self, thresh: float) -> None:
        """
        Patch iou_threshold on every live camera's SORT tracker instance.
        ByteTracker's SORT object stores iou_threshold as a plain float attribute
        that's read at the start of each update() call — live update is safe.
        """
        for cam in self._cm.cameras.values():
            try:
                tracker = getattr(cam, 'tracker', None)
                if tracker is None:
                    continue
                sort_obj = getattr(tracker, '_sort', None)
                if sort_obj is None:
                    continue
                sort_obj.iou_threshold = thresh
                logger.debug("[AdaptiveTuner] cam %s iou_threshold = %.2f",
                             cam.camera_id, thresh)
            except Exception as e:
                logger.warning("[AdaptiveTuner] iou_thresh patch error %s: %s",
                               getattr(cam, 'camera_id', '?'), e)

    # ── Manual override API ────────────────────────────────────────────────────
    def force_vel_alpha(self, alpha: float) -> None:
        """Override vel_alpha manually (e.g. from a debug API endpoint)."""
        alpha = max(VEL_ALPHA_MIN, min(VEL_ALPHA_MAX, alpha))
        self._vel_alpha = alpha
        self._apply_vel_alpha(alpha)

    def force_blend_dur(self, dur: float) -> None:
        """Override blend duration manually."""
        dur = max(BLEND_DUR_MIN, min(BLEND_DUR_MAX, dur))
        self._blend_dur = dur
        self._apply_blend_dur(dur)

    def force_iou_thresh(self, thresh: float) -> None:
        thresh = max(IOU_MIN, min(IOU_MAX, thresh))
        self._iou_thresh = thresh
        self._apply_iou_thresh(thresh)

    def current_params(self) -> dict:
        return {
            "vel_alpha":            self._vel_alpha,
            "blend_dur_s":          self._blend_dur,
            "iou_thresh":           self._iou_thresh,
            "detection_frame_skip": self._dfs,
            "jpeg_quality":         self._jpeg_q,
            "ai_latency_seconds":   self._latency,
        }