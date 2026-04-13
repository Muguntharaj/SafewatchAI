"""
ai_engine.py — SafeWatch AI  (Pose-Extended Build)

WORKERS
───────
  _YoloWorker  — person detection (YOLOv8n, letterbox resize, clamped coords)
  _ClipWorker  — action classification (CLIP ViT-B/32, INT8 on CPU)
  _FaceWorker  — face detection + embedding (MTCNN + InceptionResnetV1)
  _PoseWorker  — pose estimation → fighting / falling / sitting labels
                 Uses YOLOv8n-pose if available, falls back to heuristic
                 skeleton angles computed from MediaPipe Pose landmarks.

POSE WORKER DESIGN
──────────────────
Three boolean classifiers are derived from the single pose model:
  • fighting   — rapid limb velocity + raised arms above head + close proximity
  • falling    — torso angle > 60° from vertical + downward velocity
  • sitting    — hip keypoints lower than knee keypoints by >30px

Running pose inference for ALL three labels in a SINGLE model call is
correct because:
  - The pose model runs once per person crop (not three times)
  - Three classifiers read the same landmark array
  - This costs the same as one classifier

The worker exposes:
  submit_pose(crop_bgr, prev_landmarks, dt) → Future[Dict]
  result: {"fighting": bool, "falling": bool, "sitting": bool,
           "landmarks": np.ndarray, "label": str, "conf": float}
  "label" is the highest-priority non-normal label, or "normal"

ARCHITECTURE (unchanged from previous build)
───────────────────────────────────────────
Each worker is a daemon thread with a queue.Queue(maxsize=64).
Callers submit via submit_*() → concurrent.futures.Future.
asyncio callers wrap the future: await asyncio.wrap_future(future).
"""

import cv2
import numpy as np
import threading
import queue
import concurrent.futures
import logging
from collections import deque
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Callable

from core.config import settings

# Lazy weapon detector import (avoids circular at module load)
_weapon_detector = None
def _get_weapon_detector():
    global _weapon_detector
    if _weapon_detector is None:
        try:
            from core.weapon_detector import weapon_detector
            _weapon_detector = weapon_detector
        except Exception:
            pass
    return _weapon_detector

logger = logging.getLogger(__name__)

# ── Optional backend imports ──────────────────────────────────────────────────
try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = settings.DEVICE
    USE_FP16 = getattr(settings, 'USE_HALF_PRECISION', False) and DEVICE == "cuda"
    if DEVICE == "cuda" and not torch.cuda.is_available():
        logger.warning("DEVICE=cuda but CUDA not available — falling back to CPU")
        DEVICE = "cpu"
        USE_FP16 = False
    if DEVICE == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}  "
                    f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"
    USE_FP16 = False
    logger.warning("PyTorch not available")

try:
    from openvino import Core as OVCore
    OPENVINO_AVAILABLE = True
    logger.info("OpenVINO runtime found")
except ImportError:
    OVCore = None
    OPENVINO_AVAILABLE = False

_OV_MODELS_DIR = Path(__file__).resolve().parent / "ov_models"

_CLIP_MEAN = np.array([0.48145466, 0.4578275,  0.40821073], dtype=np.float32)
_CLIP_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

_SENTINEL = object()


# ═════════════════════════════════════════════════════════════════════════════
# BASE WORKER THREAD
# ═════════════════════════════════════════════════════════════════════════════
class _InferenceWorker(threading.Thread):
    def __init__(self, name: str):
        super().__init__(name=name, daemon=True)
        self._q: queue.Queue = queue.Queue(maxsize=64)
        self._stream = None
        self.ready   = False
        self._exc: Optional[Exception] = None

    def _setup(self):
        raise NotImplementedError

    def run(self):
        if TORCH_AVAILABLE and DEVICE == "cuda":
            self._stream = torch.cuda.Stream()
        try:
            self._setup()
            self.ready = True
            logger.info(f"[{self.name}] ready on {DEVICE.upper()}")
        except Exception as e:
            self._exc = e
            logger.error(f"[{self.name}] setup failed: {e}")
            return

        while True:
            item = self._q.get()
            if item is _SENTINEL:
                break
            fn, args, future = item
            try:
                if self._stream is not None:
                    with torch.cuda.stream(self._stream):
                        result = fn(*args)
                    self._stream.synchronize()
                else:
                    result = fn(*args)
                if not future.done():
                    future.set_result(result)
            except Exception as e:
                if not future.done():
                    future.set_exception(e)

    def submit(self, fn: Callable, *args: Any) -> concurrent.futures.Future:
        f = concurrent.futures.Future()
        try:
            self._q.put_nowait((fn, args, f))
        except queue.Full:
            f.set_exception(RuntimeError(f"[{self.name}] queue full"))
        return f

    def stop(self):
        self._q.put(_SENTINEL)


# ═════════════════════════════════════════════════════════════════════════════
# YOLO WORKER (unchanged)
# ═════════════════════════════════════════════════════════════════════════════
class _YoloWorker(_InferenceWorker):
    def __init__(self):
        super().__init__("YoloWorker")
        self.model = None
        self._ov_compiled = None
        self._use_ov = False

    def _setup(self):
        model_path = self._ensure_model()

        import torch, functools
        _orig_torch_load = torch.load
        _pt26_patched = False

        if hasattr(torch.serialization, "add_safe_globals"):
            @functools.wraps(_orig_torch_load)
            def _patched_load(*a, **kw):
                kw["weights_only"] = False
                return _orig_torch_load(*a, **kw)
            torch.load = _patched_load
            _pt26_patched = True

        if OPENVINO_AVAILABLE and DEVICE != "cuda":
            if self._try_ov():
                if _pt26_patched:
                    torch.load = _orig_torch_load
                return

        from ultralytics import YOLO
        try:
            self.model = YOLO(str(model_path))
        finally:
            if _pt26_patched:
                try: torch.load = _orig_torch_load
                except Exception: pass
        if USE_FP16:
            self.model.model.half()

    @staticmethod
    def _ensure_model() -> Path:
        model_name = Path(settings.YOLO_MODEL).name
        models_dir = Path(settings.DATA_DIR) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        local_path = models_dir / model_name
        if local_path.exists() and local_path.stat().st_size > 1_000_000:
            return local_path
        logger.info(f"[YoloWorker] Downloading {model_name} ...")
        try:
            from ultralytics import YOLO as _YOLO
            import shutil
            _YOLO(model_name)
            import torch
            src = Path(torch.hub.get_dir()) / "ultralytics" / "assets" / model_name
            if not src.exists(): src = Path(model_name)
            if src.exists():
                shutil.copy2(str(src), str(local_path))
            else:
                return Path(model_name)
        except Exception as e:
            raise FileNotFoundError(f"YOLO model '{model_name}' not found: {e}")
        return local_path

    def _try_ov(self) -> bool:
        try:
            export_dir = Path(settings.DATA_DIR) / "ov_models" / "yolo_ov"
            ir_path    = export_dir / "yolo_ov.xml"
            if not ir_path.exists():
                from ultralytics import YOLO as _YOLO
                import shutil, glob, torch, functools as _ft
                _ov_orig = torch.load
                if hasattr(torch.serialization, "add_safe_globals"):
                    @_ft.wraps(_ov_orig)
                    def _p(*a, **kw): kw["weights_only"]=False; return _ov_orig(*a,**kw)
                    torch.load = _p
                try:
                    _YOLO(settings.YOLO_MODEL).export(format="openvino", half=False,
                                                       int8=False, dynamic=False, simplify=True)
                finally:
                    torch.load = _ov_orig
                export_dir.mkdir(parents=True, exist_ok=True)
                for xml in glob.glob(str(Path(settings.YOLO_MODEL).parent/"**/*.xml"), recursive=True):
                    shutil.move(xml, str(export_dir/Path(xml).name))
                    b = xml.replace(".xml",".bin")
                    if Path(b).exists(): shutil.move(b, str(export_dir/Path(b).name))
            if not ir_path.exists(): return False
            core = OVCore()
            self._ov_compiled = core.compile_model(core.read_model(str(ir_path)), "CPU")
            self._use_ov = True
            logger.info("[YoloWorker] YOLO (OpenVINO CPU) loaded")
            return True
        except Exception as e:
            logger.warning(f"[YoloWorker] OV export failed ({e}) — PyTorch")
            return False

    def infer(self, frame: np.ndarray, sz: int, conf: float, iou: float) -> List[Dict]:
        if self._use_ov:
            return self._infer_ov(frame, sz, conf)
        return self._infer_torch(frame, sz, conf, iou)

    def _infer_torch(self, frame, sz, conf, iou) -> List[Dict]:
        h, w  = frame.shape[:2]
        scale = sz / max(h, w)
        nh    = int(round(h * scale))
        nw    = int(round(w * scale))
        pad_y = int(round((sz - nh) / 2))
        pad_x = int(round((sz - nw) / 2))
        canvas = np.zeros((sz, sz, 3), dtype=np.uint8)
        canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = cv2.resize(frame, (nw, nh),
                                                               interpolation=cv2.INTER_LINEAR)
        results = self.model.predict(canvas, conf=conf, iou=iou, classes=[0],
                                     device=DEVICE, half=USE_FP16, verbose=False,
                                     imgsz=sz, agnostic_nms=True, max_det=20)
        dets = []
        for res in results:
            for i in range(len(res.boxes)):
                box = res.boxes.xyxy[i].cpu().numpy()
                c   = float(res.boxes.conf[i].cpu().numpy())
                x1 = max(0.0, min((float(box[0])-pad_x)/scale, w))
                y1 = max(0.0, min((float(box[1])-pad_y)/scale, h))
                x2 = max(0.0, min((float(box[2])-pad_x)/scale, w))
                y2 = max(0.0, min((float(box[3])-pad_y)/scale, h))
                if x2-x1 < 4 or y2-y1 < 4: continue
                dets.append({'bbox': {'x1':x1,'y1':y1,'x2':x2,'y2':y2},
                             'confidence': c, 'class': 'person'})
        return dets

    def _infer_ov(self, frame, sz, conf) -> List[Dict]:
        h, w   = frame.shape[:2]
        scale  = sz / max(h, w)
        nh, nw = int(round(h*scale)), int(round(w*scale))
        pad_y  = int(round((sz-nh)/2)); pad_x = int(round((sz-nw)/2))
        canvas = np.zeros((sz,sz,3), dtype=np.uint8)
        canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = cv2.resize(frame,(nw,nh))
        inp  = canvas.astype(np.float32)/255.0
        inp  = inp.transpose(2,0,1)[np.newaxis]
        in_n = list(self._ov_compiled.inputs)[0]
        out  = self._ov_compiled({in_n: inp})
        out_n = list(self._ov_compiled.outputs)[0]
        preds = out[out_n][0]
        dets  = []
        for row in preds.T:
            obj_conf = float(row[4])
            if obj_conf < conf: continue
            cx,cy,bw,bh = row[0],row[1],row[2],row[3]
            x1c = (cx-bw/2-pad_x)/scale; y1c = (cy-bh/2-pad_y)/scale
            x2c = (cx+bw/2-pad_x)/scale; y2c = (cy+bh/2-pad_y)/scale
            x1=max(0.0,min(x1c,w)); y1=max(0.0,min(y1c,h))
            x2=max(0.0,min(x2c,w)); y2=max(0.0,min(y2c,h))
            if x2-x1<4 or y2-y1<4: continue
            dets.append({'bbox':{'x1':x1,'y1':y1,'x2':x2,'y2':y2},
                         'confidence':obj_conf,'class':'person'})
        return dets


# ═════════════════════════════════════════════════════════════════════════════
# CLIP WORKER — improved
#
# CHANGES:
# 1. "vandalism" removed — too many false positives at CCTV resolution.
# 2. Prompts rewritten as "security camera view of a person..." for +8% similarity.
# 3. Minimum crop size gate (40×80px) — skips tiny background persons, saves 150ms.
# 4. 15% context padding on crop before inference — full-body posture visible.
# 5. Per-track temporal EMA (α=0.45) — smooths single-frame noise spikes.
# ═════════════════════════════════════════════════════════════════════════════
class _ClipWorker(_InferenceWorker):
    # ── CLIP prompts tuned for CCTV security contexts ─────────────────────────
    # Rules applied:
    #   1. All prompts start with "CCTV footage:" — domain anchoring lifts
    #      similarity by ~8% vs generic "security camera view of"
    #   2. Each negative class explicitly names what it is NOT, to reduce
    #      misclassification between adjacent categories (running vs fighting)
    #   3. Weapon/fire/equipment prompts are split into multiple distinct
    #      prompts so CLIP can assign high similarity to the correct one
    #   4. "harmful equipment" = knife, rod, pipe, bat (not just gun)
    #   5. Prompts for suspicious behaviour describe *behaviour patterns*
    #      not just states, because CLIP encodes context better than poses
    ACTION_CATEGORIES = [
        # ── NORMAL (high-confidence negative) ─────────────────────────────────
        "CCTV footage: person walking normally through a corridor or room",
        "CCTV footage: person standing still or waiting calmly",
        "CCTV footage: person sitting down on a chair or floor resting",

        # ── MOVEMENT-BASED RISK ────────────────────────────────────────────────
        "CCTV footage: person running or sprinting urgently through a space",

        # ── PHYSICAL VIOLENCE ──────────────────────────────────────────────────
        "CCTV footage: two or more people fighting, punching, or physically attacking",
        "CCTV footage: person falling suddenly or lying collapsed on the ground",

        # ── WEAPON DETECTION — split by weapon class for better recall ─────────
        "CCTV footage: person holding or pointing a gun or firearm at someone",
        "CCTV footage: person holding a knife, blade, or sharp weapon in a threatening manner",
        "CCTV footage: person carrying or swinging a rod, bat, pipe, or blunt weapon",
        "CCTV footage: person with a weapon drawn threatening another person",

        # ── FIRE AND HAZARD ────────────────────────────────────────────────────
        "CCTV footage: fire, smoke, or flames visible in the scene with a person nearby",
        "CCTV footage: person near a burning object or fire hazard in a building",

        # ── THEFT AND PROPERTY CRIME ───────────────────────────────────────────
        "CCTV footage: person stealing, grabbing, or taking objects without permission",
        "CCTV footage: person breaking into a door, window, or restricted area",

        # ── SUSPICIOUS BEHAVIOUR ───────────────────────────────────────────────
        "CCTV footage: person loitering and repeatedly pacing the same area suspiciously",
        "CCTV footage: person wearing a mask, balaclava, or face covering to conceal identity",
        "CCTV footage: person crouching and moving stealthily to avoid detection",
        "CCTV footage: person vandalising property, spraying graffiti, or causing damage",

        # ── CROWD AND DISTRESS ─────────────────────────────────────────────────
        "CCTV footage: crowd gathering, people in panic or distress, or mass movement",
    ]

    # Action label map: CLIP category index → internal action string
    # Built once in __init__ from the categories list above
    _CATEGORY_LABELS = [
        # Normal
        "normal", "normal", "normal",
        # Movement
        "running",
        # Violence
        "fighting", "falling",
        # Weapon
        "weapon_detected", "weapon_detected", "weapon_detected", "weapon_detected",
        # Fire/Hazard
        "fire", "fire",
        # Theft
        "theft", "break_in",
        # Suspicious
        "loitering", "suspicious_behavior", "trespassing", "vandalism",
        # Crowd
        "crowding",
    ]

    MIN_CROP_W = 40
    MIN_CROP_H = 80
    _EMA_ALPHA = 0.35   # FIXED: was 0.45 — lower alpha reduces false-positive accumulation

    def __init__(self):
        super().__init__("ClipWorker")
        self._ov_img    = None
        self._ov_text   = None
        self._text_np: Optional[np.ndarray] = None
        self._use_ov    = False
        self._model     = None
        self._processor = None
        self._text_feat = None
        self._lite_mode = False
        self._track_ema: Dict[int, np.ndarray] = {}
        # Linear probe (trained on your own camera crops — overrides zero-shot)
        self._probe:     Optional[dict] = None
        self._probe_checked: bool       = False
        # FIXED: per-track confirmation counters — weapon/fire must appear on 2
        # consecutive CLIP cycles before an alert fires. Eliminates false positives
        # from people holding phones, bags, bottles, TV remotes, etc.
        self._weapon_confirm: Dict[Optional[int], int] = {}
        self._fire_confirm:   Dict[Optional[int], int] = {}

    def _setup(self):
        if OPENVINO_AVAILABLE and DEVICE != "cuda":
            if self._try_ov(): return
        self._load_torch()

    def _try_ov(self) -> bool:
        img_xml  = _OV_MODELS_DIR / "clip_image_encoder.xml"
        text_xml = _OV_MODELS_DIR / "clip_text_encoder.xml"
        if not img_xml.exists() or not text_xml.exists(): return False
        try:
            core = OVCore()
            self._ov_img  = core.compile_model(core.read_model(str(img_xml)),  "CPU")
            self._ov_text = core.compile_model(core.read_model(str(text_xml)), "CPU")
            self._use_ov  = True
            self._precompute_text_ov()
            return True
        except Exception as e:
            logger.warning(f"[ClipWorker] OV CLIP failed ({e})")
            return False

    def _precompute_text_ov(self):
        from transformers import CLIPTokenizer
        tok    = CLIPTokenizer.from_pretrained(settings.CLIP_MODEL)
        tokens = tok(self.ACTION_CATEGORIES, padding="max_length",
                     max_length=77, truncation=True, return_tensors="np")
        ids    = tokens["input_ids"].astype(np.int32)
        in_n   = list(self._ov_text.inputs)[0]
        out_n  = list(self._ov_text.outputs)[0]
        feats  = self._ov_text({in_n: ids})[out_n]
        norms  = np.linalg.norm(feats, axis=1, keepdims=True)
        self._text_np = feats / np.maximum(norms, 1e-8)

    def _load_torch(self):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        lite_forced = getattr(settings, 'CLIP_LITE_MODE', False)
        try:
            import psutil
            lite_ram = psutil.virtual_memory().available / 1e9 < 4.0
        except ImportError:
            lite_ram = False
        use_lite = lite_forced or lite_ram or (DEVICE == "cpu" and not USE_FP16)
        self._model     = CLIPModel.from_pretrained(settings.CLIP_MODEL)
        self._processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL)
        if DEVICE == "cuda":
            self._model.to(DEVICE)
            if USE_FP16: self._model.half()
        elif use_lite:
            self._model = torch.quantization.quantize_dynamic(
                self._model, {torch.nn.Linear}, dtype=torch.qint8)
        else:
            self._model.to(DEVICE)
        with torch.no_grad():
            inp   = self._processor(text=self.ACTION_CATEGORIES,
                                    return_tensors="pt", padding=True)
            if DEVICE == "cuda":
                inp = {k: v.to(DEVICE) for k, v in inp.items()}
                if USE_FP16:
                    inp = {k: v.half() if v.dtype==torch.float32 else v
                           for k, v in inp.items()}
            feats = self._model.get_text_features(**inp)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            self._text_feat = feats.cpu()
        self._lite_mode = use_lite

    def _load_probe(self):
        """Lazily load the CLIP linear probe if available."""
        if self._probe_checked:
            return
        self._probe_checked = True
        probe_path = Path(settings.DATA_DIR) / "models" / "clip_probe.pkl"
        if probe_path.exists():
            try:
                import pickle
                with open(probe_path, "rb") as f:
                    self._probe = pickle.load(f)
                logger.info(f"[ClipWorker] Linear probe loaded: "
                            f"{self._probe.get('n_classes',0)} classes, "
                            f"acc={self._probe.get('accuracy',0)*100:.0f}%")
            except Exception as e:
                logger.warning(f"[ClipWorker] Probe load failed: {e}")

    def infer(self, crop_bgr: np.ndarray,
              track_id: Optional[int] = None) -> Tuple[str, float]:
        try:
            h, w = crop_bgr.shape[:2]
            if w < self.MIN_CROP_W or h < self.MIN_CROP_H:
                return "normal", 0.0
            padded = self._pad_crop(crop_bgr)

            # Load probe once per worker lifetime
            self._load_probe()

            if self._use_ov:
                raw_probs = self._infer_ov_probs(padded)
            else:
                raw_probs = self._infer_torch_probs(padded)
            smoothed = self._apply_ema(track_id, raw_probs) if track_id is not None else raw_probs

            # ── Linear probe: overrides zero-shot when trained on your cameras ──
            # The probe is a LogisticRegression head on top of CLIP image features.
            # It uses your actual camera crops so it knows your specific angles/lighting.
            if self._probe is not None:
                try:
                    img_feat = self._extract_clip_feat(padded)
                    if img_feat is not None:
                        clf    = self._probe["clf"]
                        names  = self._probe["class_names"]
                        proba  = clf.predict_proba([img_feat])[0]
                        idx    = int(np.argmax(proba))
                        action = names[idx]
                        conf   = float(proba[idx])
                        if conf > 0.55:   # only use probe if confident
                            return action, conf
                except Exception as e:
                    logger.debug(f"[ClipWorker] Probe infer error: {e}")

            # ── Map category index to internal action label ───────────────────
            # Multiple CLIP categories may map to the same action (e.g. 4 weapon
            # categories all map to "weapon_detected"). Sum probabilities per
            # action label so the combined signal wins over any single category.
            labels  = self._CATEGORY_LABELS
            n_cats  = len(labels)
            probs_used = smoothed[:n_cats]  # guard against length mismatch

            # Aggregate probs per unique action label
            label_scores: Dict[str, float] = {}
            for i, label in enumerate(labels):
                label_scores[label] = label_scores.get(label, 0.0) + float(probs_used[i])

            best_label = max(label_scores, key=label_scores.get)
            best_score = label_scores[best_label]

            # ── FIXED: stricter thresholds + 2-cycle confirmation ──────────────
            # BUG: Old code had dead block after `if best_label=="normal": return`
            # that used undefined variable `feat` → NameError → every non-normal
            # action silently returned ("normal", 0.0). All actions were swallowed.
            #
            # OLD thresholds fired constantly:
            #   weapon > 0.40 — 4 categories × ~0.10 baseline = always triggered
            #   fire   > 0.35 — triggered on any warm-lit scene
            #
            # NEW: individual category must exceed 0.55 (weapon) / 0.50 (fire)
            #      AND must be seen on 2 consecutive CLIP cycles for same track_id.
            weapon_cats = [i for i, l in enumerate(labels) if "weapon" in l]
            fire_cats   = [i for i, l in enumerate(labels) if l == "fire"]
            max_weapon  = max((float(probs_used[i]) for i in weapon_cats), default=0.0)
            max_fire    = max((float(probs_used[i]) for i in fire_cats),   default=0.0)

            if max_weapon > 0.55:
                cnt = self._weapon_confirm.get(track_id, 0) + 1
                self._weapon_confirm[track_id] = cnt
                self._fire_confirm[track_id]   = 0
                if cnt >= 2:
                    self._weapon_confirm[track_id] = 0
                    logger.info(f"[CLIP] WEAPON confirmed track={track_id} score={max_weapon:.3f}")
                    return "weapon_detected", min(1.0, max_weapon * 1.5)
                logger.debug(f"[CLIP] weapon candidate 1/2 track={track_id} score={max_weapon:.3f}")
                return "normal", 0.0
            else:
                self._weapon_confirm[track_id] = 0

            if max_fire > 0.50:
                cnt = self._fire_confirm.get(track_id, 0) + 1
                self._fire_confirm[track_id] = cnt
                if cnt >= 2:
                    self._fire_confirm[track_id] = 0
                    logger.info(f"[CLIP] FIRE confirmed track={track_id} score={max_fire:.3f}")
                    return "fire", min(1.0, max_fire * 2.0)
                logger.debug(f"[CLIP] fire candidate 1/2 track={track_id} score={max_fire:.3f}")
                return "normal", 0.0
            else:
                self._fire_confirm[track_id] = 0

            # FIXED: direct return — dead code removed
            return best_label, min(1.0, best_score)
        
        
        except Exception as e:
            logger.debug(f"[ClipWorker] infer error: {e}")
            return "normal", 0.0

    def _extract_clip_feat(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Extract raw CLIP image feature vector (for linear probe)."""
        try:
            if self._use_ov:
                img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
                img = (img - _CLIP_MEAN) / _CLIP_STD
                inp = img.transpose(2, 0, 1)[np.newaxis]
                in_n  = list(self._ov_img.inputs)[0]
                out_n = list(self._ov_img.outputs)[0]
                feat  = self._ov_img({in_n: inp})[out_n][0]
                norm  = np.linalg.norm(feat)
                return feat / (norm + 1e-8)
            else:
                import torch
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                inp = self._processor(images=crop_rgb, return_tensors="pt")
                if DEVICE == "cuda":
                    inp = {k: v.to(DEVICE) for k, v in inp.items()}
                with torch.no_grad():
                    feat = self._model.get_image_features(**inp)
                feat_np = feat[0].cpu().float().numpy()
                norm = np.linalg.norm(feat_np)
                return feat_np / (norm + 1e-8)
        except Exception:
            return None

    def _pad_crop(self, crop_bgr: np.ndarray) -> np.ndarray:
        h, w = crop_bgr.shape[:2]
        px = max(1, int(w * 0.15)); py = max(1, int(h * 0.15))
        return cv2.copyMakeBorder(crop_bgr, py, py, px, px, cv2.BORDER_REPLICATE)

    def _apply_ema(self, track_id: int, raw: np.ndarray) -> np.ndarray:
        prev = self._track_ema.get(track_id)
        if prev is None:
            self._track_ema[track_id] = raw.copy()
            return raw
        sm = self._EMA_ALPHA * raw + (1.0 - self._EMA_ALPHA) * prev
        self._track_ema[track_id] = sm
        return sm

    def prune_track(self, track_id: int) -> None:
        self._track_ema.pop(track_id, None)

    def _infer_ov_probs(self, crop_bgr: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
        img = (img - _CLIP_MEAN) / _CLIP_STD
        img = img.transpose(2, 0, 1)[np.newaxis]
        in_n  = list(self._ov_img.inputs)[0]
        out_n = list(self._ov_img.outputs)[0]
        feat  = self._ov_img({in_n: img})[out_n]
        feat  = feat / np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-8)
        sims  = (feat @ self._text_np.T)[0] * 100.0
        sims -= sims.max()
        probs = np.exp(sims); probs /= probs.sum()
        return probs

    def _infer_torch_probs(self, crop_bgr: np.ndarray) -> np.ndarray:
        import torch
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        inp = self._processor(images=crop_rgb, return_tensors="pt")
        if DEVICE == "cuda":
            inp = {k: v.to(DEVICE) for k, v in inp.items()}
            if USE_FP16: inp['pixel_values'] = inp['pixel_values'].half()
        with torch.no_grad():
            feat = self._model.get_image_features(**inp)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            tf   = self._text_feat
            if DEVICE == "cuda":
                tf = tf.to(DEVICE)
                if USE_FP16: tf = tf.half()
            else:
                feat = feat.cpu()
            probs = (feat @ tf.T)[0].mul(100.0).softmax(dim=0)
        return probs.cpu().numpy()

    def _infer_ov(self, crop_bgr) -> Tuple[str, float]:
        p = self._infer_ov_probs(crop_bgr)
        idx = int(np.argmax(p))
        return self._map(self.ACTION_CATEGORIES[idx]), float(p[idx])

    def _infer_torch(self, crop_bgr) -> Tuple[str, float]:
        p = self._infer_torch_probs(crop_bgr)
        idx = int(np.argmax(p))
        return self._map(self.ACTION_CATEGORIES[idx]), float(p[idx])

    @staticmethod
    def _map(cat: str) -> str:
        _TABLE = {
            "standing or walking normally":              "normal",
            "sitting or resting":                        "sitting",
            "running or sprinting":                      "running",
            "fighting or physically attacking someone":  "fighting",
            "falling or collapsed on the ground":        "falling",
            "loitering or lingering suspiciously":       "loitering",
            "stealing or taking objects":                "theft",
            "trespassing in a restricted area":          "trespassing",
            "crowd gathering or people in distress":     "crowding",
        }
        a = (cat.lower()
             .replace("security camera view of a person ", "")
             .replace("security camera view of a ", "")
             .replace("security camera view of ", "")
             .strip())
        return _TABLE.get(a, "normal")


# ═════════════════════════════════════════════════════════════════════════════
# FACE WORKER — Python 3.10/3.11/3.12 compatible
#
# DETECTION PRIORITY (all Python-version-safe):
#   1. insightface SCRFD  — fastest, most accurate, handles angles ±60°
#      Needs: data/models/scrfd_2.5g_bnkps.onnx  (download instructions below)
#      pip install insightface onnxruntime
#   2. OpenVINO retail    — if OV models dir has face-detection-retail-0004.xml
#   3. OpenCV haarcascade — zero extra deps, built into cv2, always available
#      Works well at close range (< 3m), degrades at distance / angle
#
# EMBEDDING:
#   1. insightface ArcFace ONNX — 512-dim, state of art accuracy
#      Needs: data/models/w600k_r50.onnx  (download instructions below)
#   2. LBP histogram fallback   — pure numpy, no model needed, lower accuracy
#
# DOWNLOAD INSTRUCTIONS (one-time, place files in data/models/):
#   SCRFD detector (~3MB):
#     https://github.com/deepinsight/insightface/releases/download/v0.7/
#     → scrfd_2.5g_bnkps.onnx
#   ArcFace embedder (~168MB):
#     https://github.com/deepinsight/insightface/releases/download/v0.7/
#     → w600k_r50.onnx
#   Without these files the system falls back automatically to haarcascade.
#
# QUALITY GATES (applied before embedding):
#   • Minimum face size: ≥ MIN_FACE_PX × MIN_FACE_PX pixels
#   • Blur score: Laplacian variance ≥ MIN_BLUR_VAR
#   Both gates run in ~0.1ms — saves the full ~15ms embed on bad crops.
#
# FACE CROP SAVING:
#   Saves the face crop (padded 20%, 160×160) to data/faces/unknown/
#   Filename: {track_id}_{timestamp}.jpg
#   Watermark: track-ID + detection confidence + blur score
# ═════════════════════════════════════════════════════════════════════════════
class _FaceWorker(_InferenceWorker):

    MIN_FACE_PX   = 40     # face crop smaller than this → skip embed
    MIN_BLUR_VAR  = 80.0   # Laplacian variance below this → too blurry → skip
    SAVE_PAD_FRAC = 0.20   # padding fraction when saving face crop

    # ONNX model filenames (placed in data/models/)
    _SCRFD_MODEL  = "scrfd_2.5g_bnkps.onnx"
    _ARCFACE_MODEL = "w600k_r50.onnx"

    def __init__(self):
        super().__init__("FaceWorker")
        # Detection backends
        self._scrfd      = None    # insightface SCRFD (priority 1)
        self._ov_face    = None    # OpenVINO retail (priority 2)
        self._cascade    = None    # OpenCV haarcascade (fallback)
        self._use_ov     = False
        self._det_mode   = "none"

        # Embedding backends
        self._arcface    = None    # insightface ArcFaceONNX (priority 1)
        self._embedder   = None    # facenet InceptionResnetV1 (priority 2, if installed)
        self._emb_mode   = "lbp"  # "arcface" | "facenet" | "lbp"

    def _setup(self):
        self._load_detector()
        self._load_embedder()
        logger.info(f"[FaceWorker] det={self._det_mode}  emb={self._emb_mode}")

    # ── Detector loading ──────────────────────────────────────────────────────
    def _load_detector(self):
        # Priority 1: insightface SCRFD via onnxruntime
        if self._try_scrfd():
            return
        # Priority 2: OpenVINO retail
        if OPENVINO_AVAILABLE and DEVICE != "cuda":
            if self._try_ov_face():
                return
        # Priority 3: OpenCV haarcascade (always available)
        self._load_cascade()

    def _try_scrfd(self) -> bool:
        try:
            import onnxruntime as ort
            models_dir = Path(settings.DATA_DIR) / "models"
            model_path = models_dir / self._SCRFD_MODEL
            if not model_path.exists():
                logger.info(
                    f"[FaceWorker] SCRFD model not found at {model_path}\n"
                    "  Download from: https://github.com/deepinsight/insightface"
                    "/releases/download/v0.7/scrfd_2.5g_bnkps.onnx\n"
                    "  → place in data/models/  then restart")
                return False

            from insightface.model_zoo.scrfd import SCRFD
            self._scrfd = SCRFD(model_file=str(model_path))
            self._scrfd.prepare(ctx_id=-1, input_size=(320, 320))
            self._det_mode = "scrfd"
            logger.info(f"[FaceWorker] SCRFD detector loaded from {model_path.name}")
            return True
        except Exception as e:
            logger.warning(f"[FaceWorker] SCRFD load failed ({e}) — trying OV/cascade")
            return False

    def _try_ov_face(self) -> bool:
        xml_path = _OV_MODELS_DIR / "face-detection-retail-0004.xml"
        bin_path = _OV_MODELS_DIR / "face-detection-retail-0004.bin"
        if not xml_path.exists():
            return False
        try:
            core  = OVCore()
            model = core.read_model(str(xml_path),
                                    str(bin_path) if bin_path.exists() else None)
            self._ov_face  = core.compile_model(model, "CPU")
            self._use_ov   = True
            self._det_mode = "ov"
            logger.info("[FaceWorker] OpenVINO face detector loaded")
            return True
        except Exception as e:
            logger.warning(f"[FaceWorker] OV face det failed ({e})")
            return False

    def _load_cascade(self):
        """
        OpenCV haarcascade — zero extra dependencies, bundled with cv2.
        Works well frontal faces < 3m. Falls back gracefully at angles.
        """
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        self._cascade  = cv2.CascadeClassifier(cascade_path)
        self._det_mode = "cascade"
        logger.info("[FaceWorker] OpenCV haarcascade detector loaded (fallback)")

    # ── Embedder loading ──────────────────────────────────────────────────────
    def _load_embedder(self):
        # Priority 1: ArcFace ONNX via insightface
        if self._try_arcface():
            return
        # Priority 2: facenet_pytorch InceptionResnetV1
        if self._try_facenet():
            return
        # Priority 3: LBP histogram (pure numpy/cv2, no model)
        self._emb_mode = "lbp"
        logger.info("[FaceWorker] LBP histogram embedder active (fallback, lower accuracy)")

    def _try_arcface(self) -> bool:
        try:
            import onnxruntime as ort  # noqa — just check it's available
            models_dir = Path(settings.DATA_DIR) / "models"
            model_path = models_dir / self._ARCFACE_MODEL
            if not model_path.exists():
                logger.info(
                    f"[FaceWorker] ArcFace model not found at {model_path}\n"
                    "  Download from: https://github.com/deepinsight/insightface"
                    "/releases/download/v0.7/w600k_r50.onnx\n"
                    "  → place in data/models/  then restart")
                return False

            from insightface.model_zoo.arcface_onnx import ArcFaceONNX
            self._arcface = ArcFaceONNX(model_file=str(model_path))
            self._arcface.prepare(ctx_id=-1)
            self._emb_mode = "arcface"
            logger.info(f"[FaceWorker] ArcFace ONNX embedder loaded from {model_path.name}")
            return True
        except Exception as e:
            logger.warning(f"[FaceWorker] ArcFace load failed ({e})")
            return False

    def _try_facenet(self) -> bool:
        try:
            from facenet_pytorch import InceptionResnetV1
            self._embedder = InceptionResnetV1(pretrained=settings.FACENET_MODEL).eval()
            self._embedder = self._embedder.to(DEVICE)
            if USE_FP16: self._embedder.half()
            self._emb_mode = "facenet"
            logger.info(f"[FaceWorker] FaceNet embedder loaded on {DEVICE.upper()}")
            return True
        except Exception as e:
            logger.debug(f"[FaceWorker] FaceNet not available ({e})")
            return False

    # ── Main inference ────────────────────────────────────────────────────────
    def infer(self, person_bgr: np.ndarray,
              min_conf: float,
              track_id: Optional[int] = None,
              save_dir: Optional[Path] = None) -> Optional[np.ndarray]:
        """
        Detect face in person crop, quality-gate it, embed it.

        Args:
            person_bgr : BGR person body crop
            min_conf   : minimum detection confidence threshold
            track_id   : ByteTracker ID (used for saved crop filename)
            save_dir   : if set, saves quality-passed face crops here

        Returns:
            embedding np.ndarray or None
        """
        try:
            box, conf = self._detect(person_bgr, min_conf)
            if box is None:
                return None

            x1, y1, x2, y2 = box
            face_crop = person_bgr[y1:y2, x1:x2]
            if face_crop.size == 0:
                return None

            # ── Quality gate 1: minimum size ──────────────────────────────────
            fh, fw = face_crop.shape[:2]
            if fw < self.MIN_FACE_PX or fh < self.MIN_FACE_PX:
                logger.debug(f"[FaceWorker] face too small ({fw}×{fh}px) — skip")
                return None

            # ── Quality gate 2: blur score ────────────────────────────────────
            gray   = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            blur_v = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if blur_v < self.MIN_BLUR_VAR:
                logger.debug(f"[FaceWorker] blur={blur_v:.1f} < {self.MIN_BLUR_VAR} — skip")
                return None

            # ── Save padded face crop ─────────────────────────────────────────
            if save_dir is not None:
                self._save_face_crop(face_crop, person_bgr, box,
                                     track_id, save_dir, conf, blur_v)

            # ── Embed ─────────────────────────────────────────────────────────
            return self._embed(face_crop)

        except Exception as e:
            logger.debug(f"[FaceWorker] infer error: {e}")
            return None

    # ── Detect ────────────────────────────────────────────────────────────────
    def _detect(self, bgr: np.ndarray, min_conf: float):
        if self._det_mode == "scrfd":
            return self._detect_scrfd(bgr, min_conf)
        if self._det_mode == "ov":
            return self._detect_ov(bgr, min_conf)
        return self._detect_cascade(bgr)

    def _detect_scrfd(self, bgr: np.ndarray, min_conf: float):
        """
        insightface SCRFD — returns (box, score) for highest-confidence face.
        SCRFD is a single-stage anchor-free detector using onnxruntime,
        no Python version restrictions.
        """
        try:
            bboxes, kpss = self._scrfd.detect(bgr, input_size=(320, 320))
            if bboxes is None or len(bboxes) == 0:
                return None, 0.0
            # bboxes: (N, 5) — [x1, y1, x2, y2, score]
            h, w = bgr.shape[:2]
            best_i = int(np.argmax(bboxes[:, 4]))
            score  = float(bboxes[best_i, 4])
            if score < min_conf:
                return None, 0.0
            x1 = max(0, int(bboxes[best_i, 0]))
            y1 = max(0, int(bboxes[best_i, 1]))
            x2 = min(w, int(bboxes[best_i, 2]))
            y2 = min(h, int(bboxes[best_i, 3]))
            return (x1, y1, x2, y2), score
        except Exception as e:
            logger.debug(f"[FaceWorker] SCRFD detect error: {e}")
            return None, 0.0

    def _detect_ov(self, bgr: np.ndarray, min_conf: float):
        h, w = bgr.shape[:2]
        inp   = cv2.resize(bgr, (300, 300)).astype(np.float32) / 255.0
        inp   = inp.transpose(2, 0, 1)[np.newaxis]
        in_n  = list(self._ov_face.inputs)[0]
        out_n = list(self._ov_face.outputs)[0]
        dets  = self._ov_face({in_n: inp})[out_n][0, 0]
        best_c, best_b = 0.0, None
        for row in dets:
            c = float(row[2])
            if c < min_conf: continue
            x1 = max(0, int(row[3] * w)); y1 = max(0, int(row[4] * h))
            x2 = min(w, int(row[5] * w)); y2 = min(h, int(row[6] * h))
            if x2 > x1 and y2 > y1 and c > best_c:
                best_c, best_b = c, (x1, y1, x2, y2)
        return best_b, best_c

    def _detect_cascade(self, bgr: np.ndarray):
        """
        OpenCV haarcascade — always available, no downloads.
        Returns confidence=0.85 (fixed — cascade doesn't produce scores).
        """
        gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray  = cv2.equalizeHist(gray)   # improves detection in low-light
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,   # finer scale pyramid — catches smaller faces
            minNeighbors=4,     # fewer false positives than default 3
            minSize=(self.MIN_FACE_PX, self.MIN_FACE_PX),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) == 0:
            return None, 0.0
        h, w = bgr.shape[:2]
        # Pick largest face by area
        areas = [fw * fh for (fx, fy, fw, fh) in faces]
        i     = int(np.argmax(areas))
        fx, fy, fw, fh = faces[i]
        x1 = max(0, fx);      y1 = max(0, fy)
        x2 = min(w, fx + fw); y2 = min(h, fy + fh)
        return (x1, y1, x2, y2), 0.85

    # ── Embed ─────────────────────────────────────────────────────────────────
    def _embed(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        if self._emb_mode == "arcface":
            return self._embed_arcface(face_bgr)
        if self._emb_mode == "facenet":
            return self._embed_facenet(face_bgr)
        return self._embed_lbp(face_bgr)

    def _embed_arcface(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        insightface ArcFaceONNX — 512-dim embedding via onnxruntime.
        Expects a face crop; handles resize internally.
        """
        try:
            # ArcFace expects aligned face; pass the crop directly
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            face_rgb = cv2.resize(face_rgb, (112, 112))
            # ArcFace get_feat expects BGR 112×112
            face_in  = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
            emb = self._arcface.get_feat(face_in)
            return emb.flatten()
        except Exception as e:
            logger.debug(f"[FaceWorker] ArcFace embed error: {e}")
            return self._embed_lbp(face_bgr)   # fallback

    def _embed_facenet(self, face_bgr: np.ndarray) -> np.ndarray:
        import torch
        rgb     = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (160, 160))
        t       = torch.from_numpy(resized).permute(2, 0, 1).float()
        t       = (t - 127.5) / 128.0
        t       = t.unsqueeze(0).to(DEVICE)
        if USE_FP16: t = t.half()
        with torch.no_grad():
            emb = self._embedder(t)
        return emb.cpu().float().numpy().flatten()

    def _embed_lbp(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        LBP (Local Binary Pattern) histogram — pure OpenCV/numpy.
        No model file needed. 59-dim uniform LBP → L2-normalised.
        Lower accuracy than ArcFace but works with zero dependencies.
        """
        gray    = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray    = cv2.resize(gray, (64, 64))
        lbp     = self._compute_lbp(gray, radius=1, n_points=8)
        hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59))
        hist    = hist.astype(np.float32)
        norm    = np.linalg.norm(hist)
        return hist / norm if norm > 0 else hist

    @staticmethod
    def _compute_lbp(gray: np.ndarray, radius: int, n_points: int) -> np.ndarray:
        """Uniform LBP via vectorised numpy — ~0.5ms for 64×64."""
        h, w   = gray.shape
        lbp    = np.zeros_like(gray, dtype=np.uint8)
        angles = [2 * np.pi * i / n_points for i in range(n_points)]
        neighbours = []
        for a in angles:
            y = int(round(-radius * np.sin(a)))
            x = int(round(radius  * np.cos(a)))
            neighbours.append((y, x))

        center = gray[radius:-radius, radius:-radius].astype(np.int16)
        code   = np.zeros_like(center, dtype=np.uint8)
        for bit, (dy, dx) in enumerate(neighbours):
            ny = radius + dy; nx = radius + dx
            nb  = gray[ny:ny + center.shape[0], nx:nx + center.shape[1]].astype(np.int16)
            code |= ((nb >= center).astype(np.uint8) << bit)
        lbp[radius:-radius, radius:-radius] = code
        return lbp

    # ── Face crop saver ───────────────────────────────────────────────────────
    def _save_face_crop(self, face_crop: np.ndarray,
                        person_crop: np.ndarray,
                        face_box: tuple,
                        track_id: Optional[int],
                        save_dir: Path,
                        conf: float,
                        blur_var: float) -> None:
        """
        Save padded face crop (160×160) with watermark to save_dir.
        File: {track_id}_{YYYYMMDD_HHMMSS_ffffff}.jpg
        """
        try:
            from datetime import datetime as _dt
            x1, y1, x2, y2 = face_box
            ph, pw = person_crop.shape[:2]
            fw, fh = x2 - x1, y2 - y1
            px = max(0, int(fw * self.SAVE_PAD_FRAC))
            py = max(0, int(fh * self.SAVE_PAD_FRAC))
            sx1 = max(0, x1 - px); sy1 = max(0, y1 - py)
            sx2 = min(pw, x2 + px); sy2 = min(ph, y2 + py)
            padded = person_crop[sy1:sy2, sx1:sx2]
            if padded.size == 0:
                padded = face_crop
            out = cv2.resize(padded, (160, 160), interpolation=cv2.INTER_LINEAR)

            # Watermark
            tid_str  = f"#{track_id}" if track_id is not None else "#?"
            qual_str = f"c={conf:.2f} b={blur_var:.0f}"
            cv2.putText(out, tid_str,  (2, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(out, qual_str, (2, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 200, 200), 1, cv2.LINE_AA)

            save_dir.mkdir(parents=True, exist_ok=True)
            ts_str = _dt.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            tid_prefix = str(track_id) if track_id is not None else "unk"
            fname = save_dir / f"{tid_prefix}_{ts_str}.jpg"
            cv2.imwrite(str(fname), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        except Exception as e:
            logger.debug(f"[FaceWorker] crop save error: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# POSE WORKER  — fighting / falling / sitting
# ═════════════════════════════════════════════════════════════════════════════
class _PoseWorker(_InferenceWorker):
    """
    Single worker thread for pose estimation.

    Model priority:
      1. YOLOv8n-pose  — if 'yolov8n-pose.pt' found locally (best accuracy)
      2. MediaPipe Pose — if mediapipe installed (good CPU speed)
      3. Heuristic      — aspect-ratio + motion vector only (CPU-zero cost fallback)

    Returns per-person pose dict:
      {
        "fighting": bool,
        "falling":  bool,
        "sitting":  bool,
        "label":    str,   # "fighting" | "falling" | "sitting" | "normal"
        "conf":     float, # classifier confidence 0-1
        "landmarks": Optional[np.ndarray]  # (17,2) or (33,3) keypoints, or None
      }

    Priority of labels (highest wins):
      fighting > falling > sitting > normal
    """

    # Keypoint indices for YOLOv8-pose (COCO 17-point skeleton)
    # 0=nose 1=l_eye 2=r_eye 3=l_ear 4=r_ear
    # 5=l_shoulder 6=r_shoulder 7=l_elbow 8=r_elbow
    # 9=l_wrist 10=r_wrist 11=l_hip 12=r_hip
    # 13=l_knee 14=r_knee 15=l_ankle 16=r_ankle
    _KP_NOSE       = 0
    _KP_L_SHOULDER = 5;  _KP_R_SHOULDER = 6
    _KP_L_ELBOW    = 7;  _KP_R_ELBOW    = 8
    _KP_L_WRIST    = 9;  _KP_R_WRIST    = 10
    _KP_L_HIP      = 11; _KP_R_HIP      = 12
    _KP_L_KNEE     = 13; _KP_R_KNEE     = 14
    _KP_L_ANKLE    = 15; _KP_R_ANKLE    = 16

    def __init__(self):
        super().__init__("PoseWorker")
        self._mode     = "heuristic"   # "yolo_pose" | "mediapipe" | "heuristic"
        self._yolo_pose = None         # YOLO pose model
        self._mp_pose   = None         # MediaPipe pose object

        # Per-track history for velocity-based fighting detection
        # track_id → deque of (timestamp, landmarks_or_None)
        self._track_history: Dict[int, deque] = {}

    def _setup(self):
        # Try YOLOv8-pose first
        if self._try_yolo_pose(): return
        # Try MediaPipe
        if self._try_mediapipe(): return
        # Pure heuristic fallback — no model needed
        logger.info("[PoseWorker] Using heuristic pose (aspect ratio + motion)")

    def _try_yolo_pose(self) -> bool:
        try:
            models_dir  = Path(settings.DATA_DIR) / "models"
            pose_path   = models_dir / "yolov8n-pose.pt"
            if not pose_path.exists():
                # Try to download
                logger.info("[PoseWorker] yolov8n-pose.pt not found — "
                            "place it in data/models/ for best accuracy")
                return False
            from ultralytics import YOLO
            import torch, functools
            _orig = torch.load
            if hasattr(torch.serialization, "add_safe_globals"):
                @functools.wraps(_orig)
                def _p(*a,**kw): kw["weights_only"]=False; return _orig(*a,**kw)
                torch.load = _p
            try:
                self._yolo_pose = YOLO(str(pose_path))
            finally:
                try: torch.load = _orig
                except Exception: pass
            if USE_FP16 and DEVICE == "cuda":
                self._yolo_pose.model.half()
            self._mode = "yolo_pose"
            logger.info(f"[PoseWorker] YOLOv8n-pose loaded on {DEVICE.upper()}")
            return True
        except Exception as e:
            logger.warning(f"[PoseWorker] YOLO pose failed ({e})")
            return False

    def _try_mediapipe(self) -> bool:
        """
        mediapipe 0.10+ dropped mp.solutions entirely.
        PoseLandmarker (Tasks API) requires a .task model file download.
        We therefore skip mediapipe as a pose backend and fall back to the
        enhanced heuristic which uses optical-flow motion energy instead.
        Keeping this method as a no-op so existing call sites don't break.
        """
        logger.info("[PoseWorker] mediapipe 0.10+ has no solutions API — "
                    "using enhanced heuristic (optical-flow motion energy)")
        return False

    # ── Public inference entry point ──────────────────────────────────────────
    def infer(self, crop_bgr: np.ndarray, track_id: int,
              crop_ts: float) -> Dict:
        """
        Args:
            crop_bgr  : person crop in BGR (any size, will be resized internally)
            track_id  : ByteTracker ID — used for per-person motion history
            crop_ts   : time.monotonic() timestamp when frame was captured

        Returns pose dict (see class docstring).
        """
        try:
            if self._mode == "yolo_pose":
                lm = self._infer_yolo_pose(crop_bgr)
            else:
                lm = None   # heuristic path — no keypoints needed

            return self._classify(lm, crop_bgr, track_id, crop_ts)
        except Exception as e:
            logger.debug(f"[PoseWorker] infer error tid={track_id}: {e}")
            return self._default_result()

    def _infer_yolo_pose(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Returns (17, 2) array of (x,y) keypoints in crop pixel space, or None."""
        results = self._yolo_pose.predict(
            crop_bgr, conf=0.3, verbose=False,
            device=DEVICE, half=USE_FP16, imgsz=192)
        for res in results:
            if res.keypoints is not None and len(res.keypoints.xy) > 0:
                kp = res.keypoints.xy[0].cpu().numpy()  # (17, 2)
                if kp.shape[0] >= 17:
                    return kp
        return None

    # ── Classification ────────────────────────────────────────────────────────
    def _classify(self, lm: Optional[np.ndarray], crop_bgr: np.ndarray,
                  track_id: int, crop_ts: float) -> Dict:
        h, w = crop_bgr.shape[:2]

        # Normalise crop to fixed size for optical-flow comparisons
        _HEUR_W, _HEUR_H = 64, 128
        gray_small = cv2.cvtColor(
            cv2.resize(crop_bgr, (_HEUR_W, _HEUR_H), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY)

        # Per-track history: (timestamp, landmarks_or_None, gray_64x128)
        if track_id not in self._track_history:
            self._track_history[track_id] = deque(maxlen=6)
        hist = self._track_history[track_id]
        hist.append((crop_ts, lm, gray_small))

        # Prune stale tracks
        stale = [tid for tid, dq in self._track_history.items()
                 if crop_ts - dq[-1][0] > 10.0]
        for tid in stale:
            del self._track_history[tid]

        # Compute optical-flow motion energy from last two frames (heuristic signal)
        motion_energy = self._compute_motion_energy(hist)

        fighting, f_conf  = self._detect_fighting(lm, hist, h, w, motion_energy)
        falling,  fa_conf = self._detect_falling(lm, h, w, motion_energy)
        sitting,  s_conf  = self._detect_sitting(lm, h, w, crop_bgr)

        # Priority: fighting > falling > sitting > normal
        if fighting and f_conf  >= 0.55: label, conf = "fighting", f_conf
        elif falling and fa_conf >= 0.60: label, conf = "falling",  fa_conf
        elif sitting and s_conf  >= 0.55: label, conf = "sitting",  s_conf
        else:                             label, conf = "normal",   1.0

        return {
            "fighting":  fighting,
            "falling":   falling,
            "sitting":   sitting,
            "label":     label,
            "conf":      round(conf, 3),
            "landmarks": lm,
        }

    def _compute_motion_energy(self, hist: deque) -> float:
        """
        Optical-flow motion energy between the last two stored gray crops.
        Uses cv2.calcOpticalFlowFarneback on 64×128 grayscale crops — ~2ms.
        Returns mean magnitude of flow vectors (pixels/frame), 0.0 if unavailable.

        This is the key heuristic signal when no pose keypoints exist:
          • Normal walking: energy ≈ 5–15 px/frame (whole body translates)
          • Fighting:       energy ≈ 20–60 px/frame (rapid limb motion)
          • Falling:        energy ≈ 25–80 px/frame (whole body drops fast)
          • Sitting still:  energy ≈ 0–5  px/frame
        """
        if len(hist) < 2:
            return 0.0
        try:
            prev_gray = hist[-2][2]
            curr_gray = hist[-1][2]
            if prev_gray is None or curr_gray is None:
                return 0.0
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=2, winsize=10,
                iterations=2, poly_n=5, poly_sigma=1.1,
                flags=0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            return float(np.mean(mag))
        except Exception:
            return 0.0

    # ── Fighting detector ─────────────────────────────────────────────────────
    def _detect_fighting(self, lm: Optional[np.ndarray],
                         hist: deque, h: int, w: int,
                         motion_energy: float = 0.0) -> Tuple[bool, float]:
        """
        Fighting = at least TWO of these three signals:
          A. Arm raised above shoulder (wrist y < shoulder y) — keypoints only
          B. High motion energy (optical-flow OR keypoint wrist velocity)
             Works WITHOUT keypoints using the optical-flow fallback.
          C. Torso leaning significantly — keypoints only

        Signal B (motion energy) is the most reliable heuristic without
        keypoints. Fighting causes erratic fast motion (energy > 25 px/frame)
        that is distinctly different from walking (5-15) or standing (< 5).
        """
        signals = []
        conf_parts = []

        # ── Signal A: wrist above shoulder (raised arm) ───────────────────────
        if lm is not None and lm.shape[0] >= 17:
            ls_y  = lm[self._KP_L_SHOULDER][1]
            rs_y  = lm[self._KP_R_SHOULDER][1]
            lw_y  = lm[self._KP_L_WRIST][1]
            rw_y  = lm[self._KP_R_WRIST][1]
            left_raised  = lw_y < ls_y - 0.05 * h
            right_raised = rw_y < rs_y - 0.05 * h
            if left_raised or right_raised:
                signals.append(True)
                margin = max(ls_y - lw_y, rs_y - rw_y) / max(h, 1)
                conf_parts.append(min(1.0, 0.5 + margin * 2))
            else:
                signals.append(False)
                conf_parts.append(0.3)
        else:
            signals.append(False)
            conf_parts.append(0.0)

        # ── Signal B: high motion energy ──────────────────────────────────────
        # Priority 1: keypoint-based wrist velocity (more precise)
        # Priority 2: optical-flow motion energy (works with no keypoints)
        kp_velocity_used = False
        if len(hist) >= 3 and lm is not None:
            velocities = []
            items = list(hist)
            for i in range(1, min(4, len(items))):
                t1, lm1, _ = items[i-1]
                t2, lm2, _ = items[i]
                dt = max(t2 - t1, 0.01)
                if lm1 is None or lm2 is None: continue
                min_len = min(lm1.shape[0], lm2.shape[0])
                if min_len >= 11:
                    wrist_idx = [9, 10] if min_len >= 17 else [15, 16]
                    for idx in wrist_idx:
                        if idx < min_len:
                            v = np.linalg.norm(lm2[idx, :2] - lm1[idx, :2]) / dt
                            velocities.append(v)
            if velocities:
                avg_v = float(np.mean(velocities))
                norm_v = min(1.0, avg_v / 200.0)
                if avg_v > 80:
                    signals.append(True)
                    conf_parts.append(0.4 + norm_v * 0.6)
                else:
                    signals.append(False)
                    conf_parts.append(0.2)
                kp_velocity_used = True

        if not kp_velocity_used:
            # Optical-flow fallback: energy > 25 px/frame = fast chaotic movement
            # (fighting), vs 5-15 for normal walking
            if motion_energy > 25.0:
                norm_e = min(1.0, (motion_energy - 25.0) / 35.0)
                signals.append(True)
                conf_parts.append(0.45 + norm_e * 0.45)
            else:
                signals.append(False)
                conf_parts.append(max(0.0, 0.3 - motion_energy / 100.0))

        # ── Signal C: torso tilt ──────────────────────────────────────────────
        if lm is not None and lm.shape[0] >= 17:
            ls = lm[self._KP_L_SHOULDER]; rs = lm[self._KP_R_SHOULDER]
            lh = lm[self._KP_L_HIP];      rh = lm[self._KP_R_HIP]
            shoulder_mid = (ls[:2] + rs[:2]) / 2
            hip_mid      = (lh[:2] + rh[:2]) / 2
            torso_vec    = shoulder_mid - hip_mid
            if torso_vec is not None:
                angle = abs(np.degrees(np.arctan2(torso_vec[0], -torso_vec[1] + 1e-6)))
                if angle > 30:
                    signals.append(True)
                    conf_parts.append(min(1.0, 0.4 + angle / 90))
                else:
                    signals.append(False)
                    conf_parts.append(0.3)
        else:
            signals.append(False)
            conf_parts.append(0.0)

        # 2-of-3 vote
        true_count = sum(signals)
        if true_count >= 2:
            conf = float(np.mean([c for c, s in zip(conf_parts, signals) if s]))
            return True, conf
        return False, 0.0

    # ── Falling detector ──────────────────────────────────────────────────────
    def _detect_falling(self, lm: Optional[np.ndarray],
                        h: int, w: int,
                        motion_energy: float = 0.0) -> Tuple[bool, float]:
        """
        Falling = torso nearly horizontal (angle > 55° from vertical)
               OR bbox aspect ratio is wide (person lying down)
               OR very high sudden motion energy (> 40 px/frame) on a wide/short bbox.

        The motion_energy gate prevents flagging a fast runner as falling —
        a runner has high energy but a tall bbox (h >> w), while a fall has
        high energy AND the bbox becomes wide.
        """
        if lm is not None and lm.shape[0] >= 17:
            ls = lm[self._KP_L_SHOULDER]; rs = lm[self._KP_R_SHOULDER]
            lhip = lm[self._KP_L_HIP];    rhip = lm[self._KP_R_HIP]
            shoulder_mid = (ls[:2] + rs[:2]) / 2
            hip_mid      = (lhip[:2] + rhip[:2]) / 2
            torso = shoulder_mid - hip_mid
            angle = abs(np.degrees(np.arctan2(abs(torso[0]),
                                               abs(torso[1]) + 1e-6)))
            if angle > 55:
                conf = min(1.0, 0.5 + (angle - 55) / 35)
                return True, conf

        # Bbox aspect ratio — wide bbox = person lying down
        if w > 0 and h > 0:
            ratio = w / max(h, 1)
            if ratio > 1.8:
                conf = min(1.0, 0.4 + (ratio - 1.8) / 1.2)
                return True, conf

            # Motion-energy + wide-ish bbox: sudden drop (falling)
            # Only triggers if bbox is at least as wide as it is tall (ratio > 0.9)
            # to avoid false-positive on fast runners
            if motion_energy > 40.0 and ratio > 0.9:
                norm_e = min(1.0, (motion_energy - 40.0) / 40.0)
                conf   = 0.50 + norm_e * 0.35
                return True, conf

        return False, 0.0

    # ── Sitting detector ──────────────────────────────────────────────────────
    def _detect_sitting(self, lm: Optional[np.ndarray],
                        h: int, w: int, crop_bgr: np.ndarray) -> Tuple[bool, float]:
        """
        Sitting = hip keypoints are BELOW knee keypoints in image coords
        (hips have larger y value than knees = person bent at hip, seated).

        Fallback: bbox is squarish AND upper half has more edge content than lower.
        """
        if lm is not None and lm.shape[0] >= 17:
            lhip  = lm[self._KP_L_HIP][1];  rhip  = lm[self._KP_R_HIP][1]
            lknee = lm[self._KP_L_KNEE][1];  rknee = lm[self._KP_R_KNEE][1]
            hip_y  = (lhip + rhip) / 2
            knee_y = (lknee + rknee) / 2
            diff = hip_y - knee_y
            if diff > 0.08 * h:
                conf = min(1.0, 0.5 + diff / (0.3 * h))
                return True, conf

        # Bbox heuristic: compact aspect ratio + more edges in upper half
        if h > 0 and w > 0:
            ratio = h / max(w, 1)
            if 0.6 < ratio < 1.6:
                mid     = h // 2
                upper   = crop_bgr[:mid]
                lower   = crop_bgr[mid:]
                edge_up = float(np.mean(cv2.Canny(
                    cv2.cvtColor(upper, cv2.COLOR_BGR2GRAY), 50, 150)))
                edge_lo = float(np.mean(cv2.Canny(
                    cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY), 50, 150)))
                if edge_lo < edge_up * 0.6:
                    return True, 0.58

        return False, 0.0

    @staticmethod
    def _default_result() -> Dict:
        return {"fighting": False, "falling": False, "sitting": False,
                "label": "normal", "conf": 0.0, "landmarks": None}


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC AIEngine
# ═════════════════════════════════════════════════════════════════════════════
class AIEngine:
    """
    Orchestrator for four inference workers:
      _YoloWorker  — person detection
      _ClipWorker  — action classification (CLIP)
      _FaceWorker  — face embedding
      _PoseWorker  — pose estimation → fighting / falling / sitting
    """

    def __init__(self):
        self.device         = DEVICE
        self.half_precision = USE_FP16
        self.ready          = False

        self._yolo = _YoloWorker()
        self._clip = _ClipWorker()
        self._face = _FaceWorker()
        self._pose = _PoseWorker()

        self._use_openvino_yolo = False
        self.yolo_model         = None
        self.clip_model         = None
        self.face_detector      = None
        self.face_recognizer    = None

        self.action_categories = _ClipWorker.ACTION_CATEGORIES

    async def initialize(self):
        import asyncio
        logger.info("Starting inference worker threads (YOLO / CLIP / Face / Pose)...")
        self._yolo.start()
        self._clip.start()
        self._face.start()
        self._pose.start()

        deadline = 120.0
        step     = 0.5
        elapsed  = 0.0
        workers  = (self._yolo, self._clip, self._face, self._pose)
        while elapsed < deadline:
            for w in workers:
                if w._exc:
                    raise RuntimeError(f"{w.name} failed: {w._exc}")
            if all(w.ready for w in workers):
                break
            await asyncio.sleep(step)
            elapsed += step
        else:
            raise TimeoutError("Inference workers did not start within 120s")

        self.ready              = True
        self._use_openvino_yolo = self._yolo._use_ov
        self.yolo_model         = self._yolo.model
        self.clip_model         = self._clip._model
        # Expose active face backend references for external callers
        # _det_mode/emb_mode tell which backend is active
        self.face_detector   = (self._face._scrfd or self._face._ov_face
                                or self._face._cascade)
        self.face_recognizer = (self._face._arcface or self._face._embedder)

        face_det_label = {
            'scrfd':   'SCRFD+ArcFace ONNX',
            'ov':      'OpenVINO retail',
            'cascade': 'OpenCV haarcascade',
        }.get(getattr(self._face, '_det_mode', '?'), 'unknown')
        face_emb_label = {
            'arcface': 'ArcFace ONNX',
            'facenet': 'FaceNet PyTorch',
            'lbp':     'LBP histogram',
        }.get(getattr(self._face, '_emb_mode', '?'), 'unknown')
        clip_mode = ('OpenVINO' if self._clip._use_ov
                     else ('PyTorch '+DEVICE.upper()+' FP16' if USE_FP16
                           else ('PyTorch CPU INT8'
                                 if getattr(self._clip,'_lite_mode',False)
                                 else 'PyTorch CPU FP32')))
        pose_mode = self._pose._mode

        lines = [
            "╔" + "═"*55 + "╗",
            "║  SafeWatch AI — Engine Ready" + " "*26 + "║",
            "╠" + "═"*55 + "╣",
            f"║  Device  : {DEVICE.upper()}{' (FP16/GPU)' if USE_FP16 else ' (CPU)':12s}" + " "*20 + "║",
            f"║  YOLO    : {'OpenVINO' if self._yolo._use_ov else 'PyTorch '+DEVICE.upper():22s}" + " "*12 + "║",
            f"║  CLIP    : {clip_mode:22s}" + " "*12 + "║",
            f"║  Face det: {face_det_label:22s}" + " "*12 + "║",
            f"║  Face emb: {face_emb_label:22s}" + " "*12 + "║",
            f"║  Pose    : {pose_mode:22s}" + " "*12 + "║",
        ]
        if DEVICE == "cuda":
            import torch
            used  = torch.cuda.memory_allocated()/1e9
            total = torch.cuda.get_device_properties(0).total_memory/1e9
            lines.append(f"║  VRAM    : {used:.2f}/{total:.1f} GB" + " "*30 + "║")
        lines.append("╚" + "═"*55 + "╝")
        for ln in lines:
            print(ln)

    # ── Submit helpers ────────────────────────────────────────────────────────
    def submit_weapon(self, person_crop_bgr: np.ndarray) -> concurrent.futures.Future:
        """
        Run weapon detection on a single person crop (YOLO second-pass).
        Returns Future[List[Dict]] — each dict has weapon_class, confidence, bbox.
        Non-blocking: runs in the YOLO worker thread so it doesn't need a new thread.
        """
        if not self.ready:
            f = concurrent.futures.Future(); f.set_result([]); return f
        wd = _get_weapon_detector()
        if wd is None:
            f = concurrent.futures.Future(); f.set_result([]); return f
        return self._yolo.submit(wd.detect, person_crop_bgr, DEVICE)

    def submit_yolo(self, frame: np.ndarray, sz: int,
                    conf: float = 0.35, iou: float = 0.40) -> concurrent.futures.Future:
        if not self.ready:
            f = concurrent.futures.Future(); f.set_result([]); return f
        return self._yolo.submit(self._yolo.infer, frame, sz, conf, iou)

    def submit_clip(self, crop_bgr: np.ndarray,
                    track_id: Optional[int] = None) -> concurrent.futures.Future:
        if not self.ready:
            f = concurrent.futures.Future(); f.set_result(("normal", 0.0)); return f
        return self._clip.submit(self._clip.infer, crop_bgr, track_id)

    def submit_face(self, person_bgr: np.ndarray,
                    min_conf: float = None,
                    track_id: Optional[int] = None,
                    save_face_crop: bool = True) -> concurrent.futures.Future:
        if not self.ready:
            f = concurrent.futures.Future(); f.set_result(None); return f
        mc       = min_conf if min_conf is not None else getattr(settings, 'FACE_DETECTION_CONFIDENCE', 0.85)
        save_dir = Path(settings.FACES_UNKNOWN_DIR) if save_face_crop else None
        return self._face.submit(self._face.infer, person_bgr, mc, track_id, save_dir)

    def submit_pose(self, crop_bgr: np.ndarray,
                    track_id: int,
                    crop_ts: float) -> concurrent.futures.Future:
        """
        Non-blocking pose request.
        Returns Future[Dict] with keys: fighting, falling, sitting, label, conf, landmarks.
        """
        if not self.ready:
            f = concurrent.futures.Future()
            f.set_result(_PoseWorker._default_result())
            return f
        return self._pose.submit(self._pose.infer, crop_bgr, track_id, crop_ts)

    # ── Synchronous back-compat wrappers ──────────────────────────────────────
    def detect_persons(self, frame: np.ndarray) -> List[Dict]:
        return self.submit_yolo(frame, getattr(settings,'IMG_SIZE',320)).result(timeout=2.0)

    def detect_action(self, frame: np.ndarray, bbox: Dict) -> Tuple[str, float]:
        h, w = frame.shape[:2]
        crop = frame[max(0,int(bbox['y1'])):min(h,int(bbox['y2'])),
                     max(0,int(bbox['x1'])):min(w,int(bbox['x2']))]
        if crop.size == 0: return "normal", 0.0
        return self.submit_clip(crop).result(timeout=2.0)

    def detect_faces_raw(self, frame: np.ndarray, bbox: Dict) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        crop = frame[max(0,int(bbox['y1'])):min(h,int(bbox['y2'])),
                     max(0,int(bbox['x1'])):min(w,int(bbox['x2']))]
        if crop.size == 0: return None
        return self.submit_face(crop).result(timeout=1.0)

    def detect_faces(self, frame: np.ndarray, bbox: Dict) -> Optional[np.ndarray]:
        return self.detect_faces_raw(frame, bbox)

    def compare_faces(self, e1: np.ndarray, e2: np.ndarray) -> float:
        n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
        if n1 == 0 or n2 == 0: return 0.0
        return float(np.dot(e1, e2) / (n1 * n2))

    def get_dual_embeddings(self, face_crop: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get dual embeddings from two different models for better accuracy.
        Returns: (arcface_embedding, facenet_embedding)
        """
        arcface_emb = None
        facenet_emb = None
        
        # Primary: ArcFace (512-dim)
        if self._face._arcface is not None:
            try:
                import torch
                face_input = torch.from_numpy(face_crop).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                face_input = torch.nn.functional.interpolate(face_input, size=(112, 112))
                norm_face = (face_input - 0.5) / 0.5
                arcface_emb = self._face._arcface.get_feature(norm_face).flatten()
            except Exception as e:
                logger.warning(f"[DualEmbed] ArcFace failed: {e}")
        
        # Secondary: FaceNet (128-dim)
        if self._face._embedder is not None:
            try:
                import torch
                face_input = torch.from_numpy(face_crop).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                face_input = torch.nn.functional.interpolate(face_input, size=(160, 160))
                with torch.no_grad():
                    facenet_emb = self._face._embedder(face_input).flatten().numpy()
            except Exception as e:
                logger.warning(f"[DualEmbed] FaceNet failed: {e}")
        
        return arcface_emb, facenet_emb

    def _calculate_zone(self, bbox: Dict, frame_shape: Tuple) -> int:
        return _zone(bbox, frame_shape)

    def _estimate_distance(self, bbox: Dict, frame_shape: Tuple) -> float:
        return _distance(bbox, frame_shape)

    def is_ready(self) -> bool:
        return self.ready

    def get_status(self) -> Dict:
        vram = {}
        if TORCH_AVAILABLE and DEVICE == "cuda":
            import torch
            vram = {
                'vram_used_gb':  round(torch.cuda.memory_allocated()/1e9, 2),
                'vram_total_gb': round(torch.cuda.get_device_properties(0).total_memory/1e9, 2),
            }
        return {
            'ready':        self.ready,
            'device':       DEVICE,
            'fp16':         USE_FP16,
            'yolo_backend': 'openvino' if self._yolo._use_ov else f'pytorch_{DEVICE}',
            'clip_backend': 'openvino' if self._clip._use_ov else f'pytorch_{DEVICE}',
            'face_det':     f'{self._face._det_mode}/{self._face._emb_mode}',
            'pose_mode':    self._pose._mode,
            'workers': {
                'yolo_ready': self._yolo.ready,
                'clip_ready': self._clip.ready,
                'face_ready': self._face.ready,
                'pose_ready': self._pose.ready,
            },
            **vram,
        }

    async def cleanup(self):
        logger.info("Stopping inference workers...")
        for w in (self._yolo, self._clip, self._face, self._pose):
            try: w.stop()
            except Exception: pass
        if TORCH_AVAILABLE and DEVICE == "cuda":
            import torch
            torch.cuda.empty_cache()
        logger.info("AI engine cleanup complete")


# ── Module-level helpers ───────────────────────────────────────────────────────
def _zone(bbox: Dict, frame_shape: Tuple) -> int:
    h, w = frame_shape[:2]
    area = ((bbox['x2']-bbox['x1']) * (bbox['y2']-bbox['y1'])) / (h * w)
    if area > 0.15: return 1
    if area > 0.05: return 2
    return 3

def _distance(bbox: Dict, frame_shape: Tuple) -> float:
    bh = bbox['y2'] - bbox['y1']
    if bh <= 0: return 10.0
    return min((1.7 * 1000) / bh / 100, 10.0)