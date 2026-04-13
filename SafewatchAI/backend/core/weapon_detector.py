"""
weapon_detector.py — SafeWatch AI

Object-in-hand detection using a second-pass YOLOv8 model trained on weapons.

DESIGN:
  The main YOLO model detects persons. For each person crop, this module
  runs a second YOLO pass specifically tuned to detect hand-held objects.

  Detection pipeline:
    1. camera_manager._process_frame detects persons (YOLOv8n, person class)
    2. For each person crop, ai_engine.submit_weapon(crop) is called
    3. weapon_detector.detect() runs YOLOv8 on the crop with weapon classes
    4. Returns weapon_class, confidence, bbox within the crop

MODEL PRIORITY:
  1. YOLOv8n trained on COCO (has knife=43, gun/pistol via 'pistol' class if
     fine-tuned) — place yolov8n_weapons.pt in data/models/
  2. Fallback: run standard YOLOv8n on the person crop and look for
     COCO object classes that overlap with dangerous items:
       scissors(76), knife(43), baseball bat(38), bottle(39),
       handbag(26), cell phone(67), remote(65)
  3. Heuristic: bounding box aspect ratio + position relative to hand region

HOW TO USE WITH BETTER ACCURACY:
  Download a fine-tuned weapons model:
    https://github.com/ultralytics/assets/releases (YOLOv8n trained on
    weapons dataset) and place as data/models/yolov8n_weapons.pt

  Or train your own with:
    yolo train model=yolov8n.pt data=weapons.yaml epochs=100

COCO CLASSES USED FOR WEAPON HEURISTICS (when no weapons model available):
  These are objects that in a security context may indicate a threat:
    38: baseball bat
    43: knife
    76: scissors
    39: bottle       (could be weapon)
    40: wine glass
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from core.config import settings

logger = logging.getLogger(__name__)

# ── COCO classes that are potentially dangerous in security context ────────────
# Maps COCO class_id → (display_name, threat_level)
# threat_level: 'critical' | 'high' | 'medium'
_COCO_WEAPON_CLASSES: Dict[int, Tuple[str, str]] = {
    38:  ("baseball_bat",   "high"),
    43:  ("knife",          "critical"),
    76:  ("scissors",       "medium"),
    39:  ("bottle",         "medium"),
    40:  ("wine_glass",     "medium"),
}

# ── Weapon-specific YOLO classes (when using a weapons fine-tuned model) ──────
_WEAPON_MODEL_CLASSES: Dict[int, Tuple[str, str]] = {
    0:  ("pistol",          "critical"),
    1:  ("rifle",           "critical"),
    2:  ("knife",           "critical"),
    3:  ("gun",             "critical"),
    4:  ("axe",             "critical"),
    5:  ("machete",         "critical"),
    6:  ("baseball_bat",    "high"),
    7:  ("crowbar",         "high"),
    8:  ("explosive",       "critical"),
    9:  ("grenade",         "critical"),
}


class WeaponDetector:
    """
    Detects weapons / dangerous objects in a person crop.

    Usage:
        wd = WeaponDetector()
        results = wd.detect(person_crop_bgr, device)
        # returns list of {'weapon_class': str, 'confidence': float,
        #                  'bbox': {x1,y1,x2,y2}, 'threat_level': str}
    """

    _WEAPONS_MODEL_NAME = "yolov8n_weapons.pt"
    _CONF_THRESHOLD     = 0.45   # minimum confidence to report
    _MIN_CROP_W         = 60     # skip tiny crops (unlikely to contain useful info)
    _MIN_CROP_H         = 60

    def __init__(self):
        self._model       = None    # fine-tuned weapons model (optional)
        self._yolo        = None    # standard YOLO for COCO fallback
        self._mode        = "none"
        self._loaded      = False
        self._load_called = False

    def _lazy_load(self):
        """Load model on first use — avoids startup delay."""
        if self._load_called:
            return
        self._load_called = True
        self._try_weapons_model()
        if self._mode == "none":
            self._try_coco_fallback()

    def _try_weapons_model(self) -> bool:
        """Try to load a fine-tuned weapons YOLO model."""
        models_dir  = Path(settings.DATA_DIR) / "models"
        model_path  = models_dir / self._WEAPONS_MODEL_NAME
        if not model_path.exists():
            logger.info(
                f"[WeaponDetector] Weapons model not found at {model_path}\n"
                "  → Using COCO fallback (knife/bat/scissors detection)\n"
                "  → For better accuracy, download a weapons-trained YOLOv8 model\n"
                "    and place it at: " + str(model_path)
            )
            return False
        try:
            import torch, functools
            from ultralytics import YOLO
            _orig = torch.load
            if hasattr(torch.serialization, "add_safe_globals"):
                @functools.wraps(_orig)
                def _p(*a, **kw): kw["weights_only"] = False; return _orig(*a, **kw)
                torch.load = _p
            try:
                self._model = YOLO(str(model_path))
            finally:
                try: torch.load = _orig
                except Exception: pass
            self._mode   = "weapons_model"
            self._loaded = True
            logger.info(f"[WeaponDetector] Weapons model loaded: {model_path.name}")
            return True
        except Exception as e:
            logger.warning(f"[WeaponDetector] Weapons model load failed ({e})")
            return False

    def _try_coco_fallback(self) -> bool:
        """Use the main YOLO model with COCO dangerous-item classes."""
        try:
            import torch, functools
            from ultralytics import YOLO
            model_name = Path(settings.YOLO_MODEL).name
            models_dir = Path(settings.DATA_DIR) / "models"
            model_path = models_dir / model_name
            if not model_path.exists():
                model_path = Path(settings.YOLO_MODEL)
            if not model_path.exists():
                return False
            _orig = torch.load
            if hasattr(torch.serialization, "add_safe_globals"):
                @functools.wraps(_orig)
                def _p(*a, **kw): kw["weights_only"] = False; return _orig(*a, **kw)
                torch.load = _p
            try:
                self._yolo = YOLO(str(model_path))
            finally:
                try: torch.load = _orig
                except Exception: pass
            self._mode   = "coco_fallback"
            self._loaded = True
            logger.info(f"[WeaponDetector] COCO fallback loaded (knife/bat/scissors classes)")
            return True
        except Exception as e:
            logger.warning(f"[WeaponDetector] COCO fallback failed ({e})")
            return False

    def detect(self, person_crop_bgr: np.ndarray,
               device: str = "cpu") -> List[Dict]:
        """
        Run weapon detection on a person body crop.

        Args:
            person_crop_bgr: BGR image of one person (full body or upper body)
            device:          'cpu' or 'cuda'

        Returns:
            List of dicts, each with:
              weapon_class: str   — e.g. "knife", "pistol", "baseball_bat"
              confidence:   float — 0-1
              bbox:         dict  — {x1, y1, x2, y2} in crop pixel coordinates
              threat_level: str   — "critical" | "high" | "medium"
        """
        self._lazy_load()

        if not self._loaded:
            return []

        h, w = person_crop_bgr.shape[:2]
        if w < self._MIN_CROP_W or h < self._MIN_CROP_H:
            return []

        try:
            if self._mode == "weapons_model":
                return self._detect_weapons_model(person_crop_bgr, device)
            if self._mode == "coco_fallback":
                return self._detect_coco(person_crop_bgr, device)
        except Exception as e:
            logger.debug(f"[WeaponDetector] detect error: {e}")
        return []

    def _detect_weapons_model(self, crop: np.ndarray, device: str) -> List[Dict]:
        """Run fine-tuned weapons model."""
        results = self._model.predict(
            crop, conf=self._CONF_THRESHOLD, iou=0.45,
            device=device, verbose=False, max_det=10,
        )
        detections = []
        h, w = crop.shape[:2]
        for res in results:
            for i in range(len(res.boxes)):
                cls_id = int(res.boxes.cls[i].cpu().item())
                conf   = float(res.boxes.conf[i].cpu().item())
                box    = res.boxes.xyxy[i].cpu().numpy()
                info   = _WEAPON_MODEL_CLASSES.get(cls_id, (f"object_{cls_id}", "medium"))
                detections.append({
                    "weapon_class": info[0],
                    "confidence":   round(conf, 3),
                    "bbox": {
                        "x1": max(0, int(box[0])), "y1": max(0, int(box[1])),
                        "x2": min(w, int(box[2])), "y2": min(h, int(box[3])),
                    },
                    "threat_level": info[1],
                })
        return sorted(detections, key=lambda x: x["confidence"], reverse=True)

    def _detect_coco(self, crop: np.ndarray, device: str) -> List[Dict]:
        """
        Run standard YOLO on person crop, extract dangerous COCO classes.
        Classes: knife(43), baseball_bat(38), scissors(76), bottle(39).
        """
        target_classes = list(_COCO_WEAPON_CLASSES.keys())
        results = self._yolo.predict(
            crop, conf=self._CONF_THRESHOLD, iou=0.45,
            classes=target_classes, device=device,
            verbose=False, max_det=10,
        )
        detections = []
        h, w = crop.shape[:2]
        for res in results:
            for i in range(len(res.boxes)):
                cls_id = int(res.boxes.cls[i].cpu().item())
                conf   = float(res.boxes.conf[i].cpu().item())
                box    = res.boxes.xyxy[i].cpu().numpy()
                info   = _COCO_WEAPON_CLASSES.get(cls_id, (f"object_{cls_id}", "medium"))
                detections.append({
                    "weapon_class": info[0],
                    "confidence":   round(conf, 3),
                    "bbox": {
                        "x1": max(0, int(box[0])), "y1": max(0, int(box[1])),
                        "x2": min(w, int(box[2])), "y2": min(h, int(box[3])),
                    },
                    "threat_level": info[1],
                })
        return sorted(detections, key=lambda x: x["confidence"], reverse=True)

    def get_highest_threat(self, detections: List[Dict]) -> Optional[Dict]:
        """Return the single highest-threat detection from a list."""
        if not detections:
            return None
        priority = {"critical": 3, "high": 2, "medium": 1}
        return max(detections,
                   key=lambda d: (priority.get(d["threat_level"], 0), d["confidence"]))

    def format_alert_description(self, detections: List[Dict],
                                  person_name: str = "Unknown person") -> str:
        """
        Format a human-readable alert description for the frontend.

        Example:
          "Knife detected in person's hand (confidence: 92%). Person: John Smith"
        """
        if not detections:
            return ""
        top = self.get_highest_threat(detections)
        cls  = top["weapon_class"].replace("_", " ").title()
        conf = int(top["confidence"] * 100)
        desc = f"{cls} detected (confidence: {conf}%)"
        if len(detections) > 1:
            others = [d["weapon_class"].replace("_", " ") for d in detections[1:3]]
            desc  += f" + {', '.join(others)}"
        desc += f". Person: {person_name}"
        return desc


# ── Singleton ─────────────────────────────────────────────────────────────────
weapon_detector = WeaponDetector()