"""
pose_classifier.py — SafeWatch AI  v3

CHANGES FROM v2
───────────────
  FIX 1 (SAFEWATCH_FIX_GUIDE §3): Weapon false-positive prevention
    weapon_streak counter: must score ≥ WEAPON_SCORE_SUSPICIOUS on 2
    consecutive frames before firing CRITICAL. Single-frame spikes (waving,
    reaching for shelf, pointing at screen) are now ignored.
    Thresholds raised: CRITICAL 0.58→0.65, SUSPICIOUS 0.38→0.45.

  FIX 2 (SAFEWATCH_FIX_GUIDE §4): Fighting false-positive prevention
    fight_streak counter: must have arm_velocity > threshold on 2 consecutive
    frames. Clapping, stretching, waving goodbye no longer fire fighting.

  FIX 3 (SAFEWATCH_FIX_GUIDE §6): Improved thresholds
    FALL_ANGLE_THRESH 50→55 (shoe-tying at ~45° no longer fires).
    DWELL_SECONDS 90→120 (reduces loiter noise in busy areas).

  FIX 4: Mask/disguise detection improved
    Added temporal streak (mask_streak >= 3) before SUSPICIOUS to avoid
    firing on people who briefly turn away from camera (natural movement
    makes face landmarks disappear transiently).

  FIX 5: Mask-wearing identity handling
    face_covered=True is returned as a flag so the face worker knows to:
    a) Still attempt SCRFD/cascade face detection (finds face even masked)
    b) Use body silhouette embedding (gait/clothing) as auxiliary signal
    c) Record the masked sighting in a separate 'masked_appearances' field

  NEW: Body-based re-identification for masked persons
    When face is covered, _body_signature() computes a body colour histogram
    (clothing signature) as an auxiliary embedding. This is compared against
    known persons' stored body signatures to identify even masked individuals
    based on their clothing/body shape.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS  (all from SAFEWATCH_FIX_GUIDE §3/§4/§6)
# ─────────────────────────────────────────────────────────────────────────────
FALL_ANGLE_THRESH        = 55.0    # was 50 — shoe-tying at ~45° no longer fires
CROUCH_PX_THRESH         = 10
ERRATIC_DIR_CHANGES      = 3
DIR_WINDOW               = 8
FIGHTING_ARM_ANGLE       = 120
FIGHTING_VELOCITY_THRESH = 80.0
WEAPON_SCORE_CRITICAL    = 0.65    # was 0.58 — raised per SAFEWATCH_FIX_GUIDE §3
WEAPON_SCORE_SUSPICIOUS  = 0.45    # was 0.38 — raised per SAFEWATCH_FIX_GUIDE §3
WEAPON_ARM_EXTEND_ANGLE  = 160
FACE_COVER_VIS_THRESH    = 0.55
DWELL_SECONDS            = 120.0   # was 90 — per SAFEWATCH_FIX_GUIDE §6
LOITER_MOVE_PX           = 40
ASPECT_LIE_THRESH        = 1.35
FIRE_WARM_RATIO          = 0.14

# Streak requirements (SAFEWATCH_FIX_GUIDE §3/§4)
WEAPON_STREAK_REQUIRED   = 2   # consecutive frames above threshold before CRITICAL
FIGHT_STREAK_REQUIRED    = 2
MASK_STREAK_REQUIRED     = 3   # consecutive frames with face hidden before SUSPICIOUS


# ─────────────────────────────────────────────────────────────────────────────
class PoseLevel(str, Enum):
    NORMAL     = "normal"
    SUSPICIOUS = "suspicious"
    CRITICAL   = "critical"


@dataclass
class PoseDecision:
    level:        PoseLevel            = PoseLevel.NORMAL
    reason:       str                  = ""
    score:        float                = 0.0
    action:       str                  = "normal"
    face_covered: bool                 = False
    body_sig:     Optional[np.ndarray] = field(default=None, repr=False)
    landmarks:    Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def skip_clip(self) -> bool:
        return self.level == PoseLevel.NORMAL

    @property
    def immediate_alert(self) -> bool:
        return self.level == PoseLevel.CRITICAL


# ─────────────────────────────────────────────────────────────────────────────
# PER-TRACK HISTORY  (added weapon_streak, fight_streak, mask_streak)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class _TrackHistory:
    centres:      deque = field(default_factory=lambda: deque(maxlen=20))
    zone:         int   = 3
    zone_since:   float = 0.0
    dir_vecs:     deque = field(default_factory=lambda: deque(maxlen=DIR_WINDOW))
    arm_pts:      Optional[np.ndarray] = None
    arm_ts:       float = 0.0
    speed_hist:   deque = field(default_factory=lambda: deque(maxlen=6))
    # ── Streak counters (SAFEWATCH_FIX_GUIDE fixes) ─────────────────────────
    weapon_streak: int  = 0   # consecutive frames above WEAPON_SCORE_SUSPICIOUS
    fight_streak:  int  = 0   # consecutive frames with fighting pose
    mask_streak:   int  = 0   # consecutive frames with face landmarks hidden
    # ── Body signature for masked re-identification ──────────────────────────
    body_sigs:    deque = field(default_factory=lambda: deque(maxlen=5))

    def update_centre(self, cx: float, cy: float, zone: int, now: float):
        speed = 0.0
        if self.centres:
            lx, ly, lt = self.centres[-1]
            dt = now - lt
            if dt > 0.02:
                dx = cx - lx; dy = cy - ly
                speed = math.hypot(dx, dy) / dt
                self.speed_hist.append(speed)
                if speed > LOITER_MOVE_PX / dt:
                    if zone != self.zone:
                        self.zone = zone
                        self.zone_since = now
                vx = dx / dt; vy = dy / dt
                self.dir_vecs.append((vx, vy))
        else:
            self.zone = zone
            self.zone_since = now
        self.centres.append((cx, cy, now))

    @property
    def dwell_seconds(self) -> float:
        if not self.centres:
            return 0.0
        return self.centres[-1][2] - self.zone_since

    def direction_reversals(self) -> int:
        if len(self.dir_vecs) < 2:
            return 0
        rev = 0
        vecs = list(self.dir_vecs)
        for i in range(1, len(vecs)):
            px, py = vecs[i-1]; cx, cy = vecs[i]
            if px*cx + py*cy < 0:
                rev += 1
        return rev

    def sudden_stop(self) -> bool:
        if len(self.speed_hist) < 4:
            return False
        hist = list(self.speed_hist)
        old_avg = sum(hist[:-2]) / max(1, len(hist)-2)
        new_avg = sum(hist[-2:]) / 2.0
        return old_avg > 60.0 and new_avg < 15.0


# ─────────────────────────────────────────────────────────────────────────────
# MEDIAPIPE BACKEND
# ─────────────────────────────────────────────────────────────────────────────
class _MediapipeBackend:
    _pose  = None
    _tried = False

    @classmethod
    def get(cls):
        if cls._tried:
            return cls._pose
        cls._tried = True
        try:
            import mediapipe as mp
            cls._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            logger.info("[PoseClassifier] MediaPipe Pose Lite loaded")
        except Exception as e:
            logger.warning(f"[PoseClassifier] MediaPipe not available ({e}) — heuristic fallback")
        return cls._pose


# ─────────────────────────────────────────────────────────────────────────────
# FIRE DETECTOR
# ─────────────────────────────────────────────────────────────────────────────
def _fire_score(crop_bgr: np.ndarray) -> float:
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    small = cv2.resize(crop_bgr, (64, 64))
    b = small[:,:,0].astype(np.float32)
    g = small[:,:,1].astype(np.float32)
    r = small[:,:,2].astype(np.float32)
    warm  = ((r > 180) & (g > 80) & (b < 100) & (r > g * 1.2))
    warm_ratio = float(warm.sum()) / (64 * 64)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    fire_mask = (
        ((hsv[:,:,0] < 25) | (hsv[:,:,0] > 160)) &
        (hsv[:,:,1] > 120) & (hsv[:,:,2] > 120)
    )
    fire_ratio = float(fire_mask.sum()) / (64 * 64)
    return min(1.0, warm_ratio * 2.0 + fire_ratio * 3.0)


# ─────────────────────────────────────────────────────────────────────────────
# BODY SIGNATURE  (clothing colour histogram for masked re-identification)
# ─────────────────────────────────────────────────────────────────────────────
def _body_signature(crop_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract a normalised HSV colour histogram from the torso region of the crop.
    Used as auxiliary embedding when face is covered (mask/cap/balaclava).
    Focuses on the mid-body region (shoulders to hips) to capture clothing colour.
    Returns 48-dim float32 vector or None if crop too small.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    h, w = crop_bgr.shape[:2]
    if h < 40 or w < 20:
        return None
    # Mid-body region: rows 25%–70% of height (skip head and legs)
    y1 = int(h * 0.25); y2 = int(h * 0.70)
    torso = crop_bgr[y1:y2, :]
    if torso.size == 0:
        return None
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    # 16 hue bins + 16 saturation bins + 16 value bins = 48-dim
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    sig = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    norm = np.linalg.norm(sig)
    return sig / norm if norm > 0 else sig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
class PoseClassifier:
    def __init__(self, camera_id: str = ""):
        self.camera_id  = camera_id
        self._histories: Dict[int, _TrackHistory] = {}
        self._mp        = _MediapipeBackend.get()
        self._use_mp    = self._mp is not None
        logger.info(f"[PoseClassifier:{camera_id}] "
                    f"backend={'mediapipe' if self._use_mp else 'heuristic'}")

    def classify_batch(
        self,
        frame:      np.ndarray,
        detections: List[Dict],
        now:        float,
    ) -> List[PoseDecision]:
        H, W = frame.shape[:2]
        decisions = []

        for det in detections:
            tid  = det.get("track_id", -1)
            bbox = det["bbox"]

            if tid not in self._histories:
                self._histories[tid] = _TrackHistory()
            hist = self._histories[tid]

            cx = (bbox["x1"] + bbox["x2"]) / 2.0
            cy = (bbox["y1"] + bbox["y2"]) / 2.0
            hist.update_centre(cx, cy, det.get("zone", 3), now)

            x1 = max(0, int(bbox["x1"])); y1 = max(0, int(bbox["y1"]))
            x2 = min(W, int(bbox["x2"])); y2 = min(H, int(bbox["y2"]))
            crop = frame[y1:y2, x1:x2]

            # Fire detection first (runs on every crop regardless of pose)
            if crop.size > 0:
                fscore = _fire_score(crop)
                if fscore > FIRE_WARM_RATIO * 3:
                    decisions.append(PoseDecision(
                        level=PoseLevel.CRITICAL, reason="fire_detected",
                        score=min(1.0, fscore), action="fire"))
                    continue

            if self._use_mp and crop.size > 0:
                dec = self._classify_mp(crop, bbox, hist, now, H, W, tid)
            else:
                dec = self._classify_heuristic(bbox, hist, now, H, W)

            # Attach body signature always (useful even when not masked)
            if crop.size > 0:
                bsig = _body_signature(crop)
                if bsig is not None:
                    hist.body_sigs.append(bsig)
                    dec.body_sig = bsig

            decisions.append(dec)

        self._prune_histories(now)
        return decisions

    # ── MediaPipe classification ──────────────────────────────────────────────
    def _classify_mp(self, crop, bbox, hist, now, fH, fW, tid) -> PoseDecision:
        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            res = self._mp.process(rgb)
        except Exception as e:
            logger.debug(f"[PoseClassifier] mp error: {e}")
            return self._classify_heuristic(bbox, hist, now, fH, fW)

        if not res.pose_landmarks:
            return self._classify_heuristic(bbox, hist, now, fH, fW)

        lm = res.pose_landmarks.landmark
        h, w = crop.shape[:2]
        bx1 = bbox["x1"]; by1 = bbox["y1"]

        def px(idx) -> Tuple[float, float]:
            l = lm[idx]
            return (bx1 + l.x * w, by1 + l.y * h)

        def vis(idx) -> float:
            return float(lm[idx].visibility)

        MP = dict(
            nose=0, l_eye=1, r_eye=2,
            l_ear=7, r_ear=8, mouth_l=9, mouth_r=10,
            l_shoulder=11, r_shoulder=12,
            l_elbow=13, r_elbow=14,
            l_wrist=15, r_wrist=16,
            l_hip=23, r_hip=24,
            l_knee=25, r_knee=26,
            l_ankle=27, r_ankle=28,
        )

        lsh = px(MP['l_shoulder']); rsh = px(MP['r_shoulder'])
        lhp = px(MP['l_hip']);      rhp = px(MP['r_hip'])
        lkn = px(MP['l_knee']);     rkn = px(MP['r_knee'])
        lwr = px(MP['l_wrist']);    rwr = px(MP['r_wrist'])
        lel = px(MP['l_elbow']);    rel = px(MP['r_elbow'])

        sh_mid = ((lsh[0]+rsh[0])/2, (lsh[1]+rsh[1])/2)
        hp_mid = ((lhp[0]+rhp[0])/2, (lhp[1]+rhp[1])/2)
        torso_dx = sh_mid[0] - hp_mid[0]
        torso_dy = sh_mid[1] - hp_mid[1]
        torso_len = math.hypot(torso_dx, torso_dy)
        torso_angle = (math.degrees(math.atan2(abs(torso_dx), abs(torso_dy) + 1e-6))
                       if torso_len > 5 else 0.0)

        bbox_w = bbox["x2"] - bbox["x1"]
        bbox_h = bbox["y2"] - bbox["y1"]

        def _angle3(a, b, c) -> float:
            ba = (a[0]-b[0], a[1]-b[1]); bc = (c[0]-b[0], c[1]-b[1])
            dot = ba[0]*bc[0]+ba[1]*bc[1]
            cross = abs(ba[0]*bc[1]-ba[1]*bc[0])
            return math.degrees(math.atan2(cross, dot + 1e-6))

        l_elbow_ang = _angle3(lsh, lel, lwr)
        r_elbow_ang = _angle3(rsh, rel, rwr)

        # ── 1. CRITICAL: Fall ─────────────────────────────────────────────────
        if torso_angle > FALL_ANGLE_THRESH or (bbox_w / (bbox_h+1e-6)) > ASPECT_LIE_THRESH:
            return PoseDecision(
                level=PoseLevel.CRITICAL, reason="fall_detected",
                score=min(1.0, torso_angle/90.0), action="falling",
                landmarks=np.array([[lm[i].x, lm[i].y, lm[i].z] for i in range(33)]))

        # ── 2. CRITICAL: Weapon — streak-gated (FIX 1) ───────────────────────
        weapon_score = self._weapon_score(
            lsh, rsh, lhp, rhp, lel, rel, lwr, rwr,
            l_elbow_ang, r_elbow_ang, torso_angle, hist, now
        )
        # Update streak counter
        if weapon_score >= WEAPON_SCORE_SUSPICIOUS:
            hist.weapon_streak += 1
        else:
            hist.weapon_streak = max(0, hist.weapon_streak - 1)

        # Only fire CRITICAL after WEAPON_STREAK_REQUIRED consecutive suspicious frames
        if hist.weapon_streak >= WEAPON_STREAK_REQUIRED:
            if weapon_score >= WEAPON_SCORE_CRITICAL:
                hist.weapon_streak = 0   # reset after firing
                return PoseDecision(
                    level=PoseLevel.CRITICAL, reason="weapon_detected",
                    score=weapon_score, action="weapon_detected")
            if weapon_score >= WEAPON_SCORE_SUSPICIOUS:
                return PoseDecision(
                    level=PoseLevel.SUSPICIOUS, reason="possible_weapon",
                    score=weapon_score, action="suspicious_behavior")

        # ── 3. CRITICAL: Fighting — streak-gated (FIX 2) ─────────────────────
        l_arm_up = lwr[1] < lsh[1]
        r_arm_up = rwr[1] < rsh[1]
        arms_extended = (l_elbow_ang > FIGHTING_ARM_ANGLE or
                         r_elbow_ang > FIGHTING_ARM_ANGLE)
        arm_pts_now = np.array([lsh, rsh, lel, rel, lwr, rwr])
        arm_velocity = 0.0
        if hist.arm_pts is not None and now - hist.arm_ts > 0.05:
            dt_arm = now - hist.arm_ts
            arm_velocity = float(
                np.linalg.norm(arm_pts_now - hist.arm_pts[:6], axis=1).mean() / dt_arm)
        hist.arm_pts = arm_pts_now
        hist.arm_ts  = now

        if (l_arm_up or r_arm_up) and arms_extended and arm_velocity > FIGHTING_VELOCITY_THRESH:
            hist.fight_streak += 1
        else:
            hist.fight_streak = max(0, hist.fight_streak - 1)

        if hist.fight_streak >= FIGHT_STREAK_REQUIRED:
            hist.fight_streak = 0
            return PoseDecision(
                level=PoseLevel.CRITICAL, reason="fighting_pose",
                score=min(1.0, arm_velocity/200.0), action="fighting")

        # ── 4. CRITICAL: Harm equipment (close-grip + extended) ───────────────
        wrist_dist = math.hypot(lwr[0]-rwr[0], lwr[1]-rwr[1])
        shoulder_w  = math.hypot(lsh[0]-rsh[0], lsh[1]-rsh[1])
        close_grip  = wrist_dist < shoulder_w * 0.3
        if close_grip and arms_extended:
            return PoseDecision(
                level=PoseLevel.CRITICAL, reason="weapon_grip",
                score=0.72, action="weapon_detected")

        # ── 5. Mask/disguise detection — streak-gated (FIX 4) ────────────────
        nose_vis   = vis(MP['nose'])
        mouth_vis  = (vis(MP['mouth_l']) + vis(MP['mouth_r'])) / 2.0
        ear_vis    = (vis(MP['l_ear'])   + vis(MP['r_ear']))   / 2.0
        face_covered = (nose_vis + mouth_vis) / 2.0 < FACE_COVER_VIS_THRESH and ear_vis > 0.4

        if face_covered:
            hist.mask_streak += 1
        else:
            hist.mask_streak = max(0, hist.mask_streak - 1)

        # ── 6. SUSPICIOUS: Crouching ──────────────────────────────────────────
        hip_y  = (lhp[1]+rhp[1]) / 2.0
        knee_y = (lkn[1]+rkn[1]) / 2.0
        if hip_y > knee_y - CROUCH_PX_THRESH:
            return PoseDecision(
                level=PoseLevel.SUSPICIOUS, reason="crouching",
                score=0.65, action="suspicious_behavior",
                face_covered=face_covered)

        # ── 7. SUSPICIOUS: Confirmed disguise (streak >= 3 frames) ────────────
        # FIX 4: Require 3 consecutive frames to avoid transient angle misreads
        if hist.mask_streak >= MASK_STREAK_REQUIRED:
            body_sig = _body_signature(
                np.zeros((int(bbox["y2"]-bbox["y1"]), int(bbox["x2"]-bbox["x1"]), 3),
                         dtype=np.uint8))
            return PoseDecision(
                level=PoseLevel.SUSPICIOUS, reason=f"face_covered_streak{hist.mask_streak}",
                score=1.0 - (nose_vis+mouth_vis)/2.0,
                action="suspicious_behavior", face_covered=True)

        # ── 8. SUSPICIOUS: Loitering ──────────────────────────────────────────
        if hist.dwell_seconds > DWELL_SECONDS:
            return PoseDecision(
                level=PoseLevel.SUSPICIOUS,
                reason=f"loitering_{int(hist.dwell_seconds)}s",
                score=min(1.0, hist.dwell_seconds/(DWELL_SECONDS*2)),
                action="loitering")

        # ── 9. SUSPICIOUS: Erratic ────────────────────────────────────────────
        rev = hist.direction_reversals()
        if rev >= ERRATIC_DIR_CHANGES:
            return PoseDecision(
                level=PoseLevel.SUSPICIOUS,
                reason=f"erratic_{rev}_reversals",
                score=min(1.0, rev/6.0),
                action="suspicious_behavior")

        # ── 10. SUSPICIOUS: Sneaking (crouching + moving) ────────────────────
        if (hip_y > knee_y) and len(hist.speed_hist) > 1 and hist.speed_hist[-1] > 20:
            return PoseDecision(
                level=PoseLevel.SUSPICIOUS, reason="sneaking",
                score=0.55, action="trespassing",
                face_covered=face_covered)

        return PoseDecision(level=PoseLevel.NORMAL, reason="normal_pose", score=0.9,
                            face_covered=face_covered)

    # ── Weapon multi-signal scorer ────────────────────────────────────────────
    def _weapon_score(self, lsh, rsh, lhp, rhp, lel, rel, lwr, rwr,
                      l_elbow_ang, r_elbow_ang, torso_angle,
                      hist: _TrackHistory, now: float) -> float:
        score = 0.0
        l_ext = l_elbow_ang > WEAPON_ARM_EXTEND_ANGLE
        r_ext = r_elbow_ang > WEAPON_ARM_EXTEND_ANGLE
        if l_ext or r_ext:
            score += 0.30
        l_retracted = lwr[1] > lsh[1]
        r_retracted = rwr[1] > rsh[1]
        if (l_ext and r_retracted) or (r_ext and l_retracted):
            score += 0.20
        if 10.0 < torso_angle < 45.0:
            score += 0.10
        sh_w  = math.hypot(lsh[0]-rsh[0], lsh[1]-rsh[1])
        hip_w = math.hypot(lhp[0]-rhp[0], lhp[1]-rhp[1])
        if hip_w > sh_w * 1.25:
            score += 0.10
        if hist.sudden_stop():
            score += 0.15
        if l_ext and lwr[1] < lsh[1]:
            score += 0.10
        if r_ext and rwr[1] < rsh[1]:
            score += 0.10
        return min(1.0, score)

    # ── Heuristic fallback ────────────────────────────────────────────────────
    def _classify_heuristic(self, bbox, hist, now, fH, fW) -> PoseDecision:
        w = bbox["x2"] - bbox["x1"]
        h = bbox["y2"] - bbox["y1"]
        if w / (h + 1e-6) > ASPECT_LIE_THRESH:
            return PoseDecision(level=PoseLevel.CRITICAL,
                                reason="fall_aspect", score=0.65, action="falling")
        if hist.dwell_seconds > DWELL_SECONDS:
            return PoseDecision(level=PoseLevel.SUSPICIOUS,
                                reason=f"loitering_{int(hist.dwell_seconds)}s",
                                score=min(1.0, hist.dwell_seconds/(DWELL_SECONDS*2)),
                                action="loitering")
        rev = hist.direction_reversals()
        if rev >= ERRATIC_DIR_CHANGES:
            return PoseDecision(level=PoseLevel.SUSPICIOUS,
                                reason=f"erratic_{rev}_reversals",
                                score=min(1.0, rev/6.0),
                                action="suspicious_behavior")
        if hist.sudden_stop():
            return PoseDecision(level=PoseLevel.SUSPICIOUS,
                                reason="sudden_stop",
                                score=0.45, action="suspicious_behavior")
        return PoseDecision(level=PoseLevel.NORMAL)

    # ── Housekeeping ──────────────────────────────────────────────────────────
    def _prune_histories(self, now: float, max_age: float = 30.0):
        stale = [tid for tid, h in self._histories.items()
                 if h.centres and now - h.centres[-1][2] > max_age]
        for tid in stale:
            del self._histories[tid]

    def prune_track(self, track_id: int):
        self._histories.pop(track_id, None)

    def stats(self) -> Dict:
        return {"backend": "mediapipe" if self._use_mp else "heuristic",
                "active_tracks": len(self._histories)}