"""
sort.py — SORT (Simple Online and Realtime Tracking) with Kalman Filter
========================================================================

WHY THIS REPLACES SimpleTracker
────────────────────────────────
SimpleTracker assigned a NEW track ID to every detection on every call.
This broke both the _BboxPredictor and _TrackStateStore in camera_manager.py:

  • _BboxPredictor.update(tid, bbox, ts) needs to see the SAME tid twice
    to compute velocity = Δpos / Δt.  With a new tid every cycle, prev is
    always None → velocity is always 0 → prediction = frozen last position.

  • _TrackStateStore stores action labels and person identity per track_id.
    A new id every cycle means CLIP labels and face recognition results are
    discarded every 0.7s → "Unknown / normal" flicker on screen.

This implementation gives STABLE IDs across frames via IoU-based Hungarian
matching, plus a Kalman filter that:
  (1) Predicts bbox position forward between YOLO cycles (velocity-based)
  (2) Smooths noisy YOLO bbox coordinates (measurement noise covariance R)
  (3) Handles missing detections gracefully (track coast for MAX_AGE frames)

KALMAN STATE VECTOR
───────────────────
  [x1, y1, x2, y2,  vx1, vy1, vx2, vy2]
   ─── position ───   ────── velocity ──────

  State transition F: constant-velocity model
    x1(t+1) = x1(t) + vx1(t)
    vx1(t+1) = vx1(t)          (velocity unchanged between measurements)

  Measurement H: we observe only [x1, y1, x2, y2] from YOLO.

NOISE TUNING (tuned for CPU-deployed YOLOv8n @ 320px)
───────────────────────────────────────────────────────
  R (measurement noise) — YOLO bbox jitter ±8px → R diagonal = 1.0 for
    position coords, 10.0 for width/height (larger uncertainty on box size).

  Q (process noise) — person motion is smooth walking, not erratic.
    Small Q on velocity = smoother prediction but slower to adapt to turns.
    Q[4:,4:] = 0.01 gives good balance for walking/running persons.

  P (initial uncertainty) — large for velocity (we don't know it yet)
    P[4:,4:] = 1000 → filter converges within 2-3 frames.

HUNGARIAN MATCHING
──────────────────
  scipy.optimize.linear_sum_assignment solves the assignment problem in O(n³).
  Cost matrix = 1 - IoU(predicted_box, detected_box).
  Matches with IoU < IOU_MIN are treated as unmatched (new track).

LIFECYCLE
─────────
  • New track:     detection with no matching prediction → KalmanBoxTracker created
  • Active track:  matched → Kalman update → output
  • Coasting:      unmatched → Kalman predict only → output if hit_streak > 0
  • Dead:          time_since_update > MAX_AGE → removed
"""

import numpy as np
from filterpy.kalman import KalmanFilter

try:
    from scipy.optimize import linear_sum_assignment
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Kalman-based single-object tracker
# ─────────────────────────────────────────────────────────────────────────────
class KalmanBoxTracker:
    """
    One Kalman filter per tracked object.
    State: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
    """
    _id_counter = 0

    def __init__(self, bbox: list):
        """
        bbox: [x1, y1, x2, y2]
        """
        self.kf = KalmanFilter(dim_x=8, dim_z=4)

        # ── State transition (constant-velocity) ─────────────────────────────
        self.kf.F = np.array([
            [1, 0, 0, 0,  1, 0, 0, 0],   # x1 += vx1
            [0, 1, 0, 0,  0, 1, 0, 0],   # y1 += vy1
            [0, 0, 1, 0,  0, 0, 1, 0],   # x2 += vx2
            [0, 0, 0, 1,  0, 0, 0, 1],   # y2 += vy2
            [0, 0, 0, 0,  1, 0, 0, 0],   # vx1 constant
            [0, 0, 0, 0,  0, 1, 0, 0],   # vy1 constant
            [0, 0, 0, 0,  0, 0, 1, 0],   # vx2 constant
            [0, 0, 0, 0,  0, 0, 0, 1],   # vy2 constant
        ], dtype=np.float64)

        # ── Measurement (we observe x1,y1,x2,y2 only) ───────────────────────
        self.kf.H = np.array([
            [1, 0, 0, 0,  0, 0, 0, 0],
            [0, 1, 0, 0,  0, 0, 0, 0],
            [0, 0, 1, 0,  0, 0, 0, 0],
            [0, 0, 0, 1,  0, 0, 0, 0],
        ], dtype=np.float64)

        # ── Measurement noise R: YOLO bbox jitter ────────────────────────────
        # Position corners: tighter noise so Kalman trusts YOLO position more.
        # This keeps the filtered bbox from "floating" away from the real person.
        # x1/y1: ±0.5px noise  x2/y2: ±2px (size is noisier than corner position)
        self.kf.R = np.diag([0.5, 0.5, 4.0, 4.0]).astype(np.float64)

        # ── Process noise Q: motion uncertainty per frame ────────────────────
        # Position elements: small Q — we trust Kalman prediction between frames.
        # Velocity elements: slightly larger Q — allows adaptation to direction
        #   changes without waiting 4+ frames. 0.04 = ~twice the old 0.01.
        #   This is the key fix for the "bbox runs ahead of person on turns" bug:
        #   larger Q_vel lets the filter accept the new velocity faster.
        self.kf.Q = np.eye(8, dtype=np.float64) * 0.5
        self.kf.Q[:4, :4] *= 0.1    # position process noise — low (Kalman trusted)
        self.kf.Q[4:, 4:] *= 0.04   # velocity process noise — allows turn adaptation

        # ── Initial covariance P ─────────────────────────────────────────────
        # Lower initial P than before — we trust the first YOLO detection position.
        # High velocity uncertainty (1000) stays so the filter converges in 2-3 frames.
        self.kf.P = np.eye(8, dtype=np.float64) * 5.0
        self.kf.P[4:, 4:] *= 200.0  # high uncertainty on initial velocity only

        # ── Initialise state from first detection ────────────────────────────
        self.kf.x[:4] = np.array(bbox, dtype=np.float64).reshape(4, 1)
        self.kf.x[4:] = 0.0          # initial velocity = 0

        KalmanBoxTracker._id_counter += 1
        self.id               = KalmanBoxTracker._id_counter
        self.time_since_update = 0   # frames since last matched detection
        self.hit_streak        = 0   # consecutive matched frames
        self.age               = 0   # total frames alive

    def predict(self) -> np.ndarray:
        """
        Advance Kalman state by one frame.
        Called every frame whether or not a detection was matched.
        Returns current state estimate [x1, y1, x2, y2].
        """
        self.kf.predict()
        self.age               += 1
        self.time_since_update += 1
        return self.kf.x[:4].flatten()

    def update(self, bbox: list) -> None:
        """
        Correct Kalman state with a matched YOLO detection.
        bbox: [x1, y1, x2, y2]
        """
        self.kf.update(np.array(bbox, dtype=np.float64).reshape(4, 1))
        self.hit_streak        += 1
        self.time_since_update  = 0

    def get_state(self) -> np.ndarray:
        """Return current position estimate [x1, y1, x2, y2]."""
        return self.kf.x[:4].flatten()

    def get_velocity(self) -> np.ndarray:
        """Return current velocity estimate [vx1, vy1, vx2, vy2]."""
        return self.kf.x[4:].flatten()


# ─────────────────────────────────────────────────────────────────────────────
# IoU helper
# ─────────────────────────────────────────────────────────────────────────────
def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _iou_matrix(dets: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """
    Compute IoU matrix between N detections and M predictions.
    Returns (N, M) matrix.
    """
    N, M = len(dets), len(preds)
    mat  = np.zeros((N, M), dtype=np.float64)
    for i, d in enumerate(dets):
        for j, p in enumerate(preds):
            mat[i, j] = _iou(d, p)
    return mat


# ─────────────────────────────────────────────────────────────────────────────
# SORT multi-object tracker
# ─────────────────────────────────────────────────────────────────────────────
class Sort:
    """
    SORT (Simple Online and Realtime Tracking) with Kalman filter.

    Usage (identical interface to SimpleTracker):
        tracker = Sort()
        tracks  = tracker.update(detections)
        # tracks: list of Track objects with .id and .bbox [x1,y1,x2,y2]

    Or via ByteTrackerWrapper.update() which calls this internally.
    """

    def __init__(self,
                 max_age:    int   = 12,   # FIX: was 5 — 5 frames ≈ 0.38s too short.
                                           # 12 frames ≈ 0.9s handles brief occlusions.
                 min_hits:   int   = 1,
                 iou_thresh: float = 0.15):
        """
        max_age:    remove a track after this many consecutive missed detections
        min_hits:   require this many consecutive hits before showing a track
                    (set to 1 to show immediately — avoids 'missing box on entry')
        iou_thresh: minimum IoU to match a detection to a prediction
                    (0.15 is permissive — handles fast movement between YOLO cycles)
        """
        self.max_age    = max_age
        self.min_hits   = min_hits
        self.iou_thresh = iou_thresh
        self.trackers:  list = []    # list of KalmanBoxTracker
        self.frame_count = 0

    def update(self, detections: list) -> list:
        """
        detections: list of [x1, y1, x2, y2, score] (or [x1..x2, score, class])
        Returns: list of Track namedtuples with .id and .bbox
        """
        self.frame_count += 1
        dets = np.array(detections, dtype=np.float64) if detections else np.empty((0, 5))
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)
        # Use only [x1,y1,x2,y2] columns
        det_boxes = dets[:, :4] if len(dets) > 0 else np.empty((0, 4))

        # ── Predict all existing trackers one step forward ────────────────────
        pred_boxes = np.array([t.predict() for t in self.trackers],
                              dtype=np.float64)

        # ── Match detections to predictions via IoU ───────────────────────────
        matched_det_idx  = set()
        matched_trk_idx  = set()

        if len(det_boxes) > 0 and len(pred_boxes) > 0:
            iou_mat = _iou_matrix(det_boxes, pred_boxes)   # (N_det, N_trk)

            if _SCIPY_OK:
                # Hungarian assignment minimises cost = 1 - IoU
                cost = 1.0 - iou_mat
                row_ind, col_ind = linear_sum_assignment(cost)
                for r, c in zip(row_ind, col_ind):
                    if iou_mat[r, c] >= self.iou_thresh:
                        matched_det_idx.add(r)
                        matched_trk_idx.add(c)
                        self.trackers[c].update(det_boxes[r].tolist())
            else:
                # Greedy fallback (no scipy): iterate by best IoU
                used_trk = set()
                for d_i in range(len(det_boxes)):
                    best_iou, best_t = 0.0, None
                    for t_i in range(len(pred_boxes)):
                        if t_i in used_trk:
                            continue
                        v = iou_mat[d_i, t_i]
                        if v > best_iou:
                            best_iou, best_t = v, t_i
                    if best_t is not None and best_iou >= self.iou_thresh:
                        matched_det_idx.add(d_i)
                        matched_trk_idx.add(best_t)
                        used_trk.add(best_t)
                        self.trackers[best_t].update(det_boxes[d_i].tolist())

        # ── Create new tracks for unmatched detections ────────────────────────
        for d_i in range(len(det_boxes)):
            if d_i not in matched_det_idx:
                new_trk = KalmanBoxTracker(det_boxes[d_i].tolist())
                # Store matched detection score on the tracker so output can use it
                new_trk._last_score = float(dets[d_i, 4]) if len(dets) > d_i else 0.9
                self.trackers.append(new_trk)

        # ── Remove dead tracks (too many missed frames) ───────────────────────
        self.trackers = [t for t in self.trackers
                         if t.time_since_update <= self.max_age]

        # ── Collect output ────────────────────────────────────────────────────
        class Track:
            __slots__ = ('id', 'bbox', 'score', 'velocity', 'hit_streak', 'time_since_update')
            def __init__(self, tid, bbox, score, vel, hit_streak, time_since_update):
                self.id                 = tid
                self.bbox               = bbox
                self.score              = score
                self.velocity           = vel
                self.hit_streak         = hit_streak
                self.time_since_update  = time_since_update

        # Build tracker_id → matched_det_score map for matched tracks
        # (matched tracks already called .update() above; we need their det score)
        # We stored _last_score on new tracks; for existing matched tracks update it now.
        # Rebuild the match map from row_ind/col_ind is expensive here, so we use
        # time_since_update==0 as a proxy: a track updated this frame was matched.
        # For matched trackers, find the best-IoU detection score.
        if len(det_boxes) > 0 and len(self.trackers) > 0:
            iou_out = _iou_matrix(det_boxes, np.array([t.get_state() for t in self.trackers]))
            for t_i, t in enumerate(self.trackers):
                if t.time_since_update == 0:   # matched this frame
                    best_d = int(np.argmax(iou_out[:, t_i]))
                    t._last_score = float(dets[best_d, 4]) if len(dets) > best_d else 0.9

        results = []
        for t in self.trackers:
            if t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                state = t.get_state()
                vel   = t.get_velocity()
                results.append(Track(
                    tid                = t.id,
                    bbox               = [float(state[0]), float(state[1]),
                                          float(state[2]), float(state[3])],
                    score              = getattr(t, '_last_score', 0.9),
                    vel                = vel.tolist(),
                    hit_streak         = t.hit_streak,
                    time_since_update  = t.time_since_update,
                ))

        return results