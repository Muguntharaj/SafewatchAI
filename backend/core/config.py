"""
Configuration Settings — SafeWatch AI
Tuned for i5-6500 (4C/4T, no GPU) with 2–6 cameras @ smooth 13 fps.

KEY CHANGES FROM OLD VERSION:
  ACTION_FRAME_SKIP  5  → 30   CLIP is 150ms, must run rarely
  DETECTION_FRAME_SKIP 2 → 3   YOLO every 3rd frame saves 33% inference
  AI_POOL_WORKERS    4  → 2    only 2 cores for AI, 2 free for capture+IO
  IMG_SIZE           416 → 320  18ms vs 28ms per YOLO call
  STREAM_FPS         10  → 13   matches the reference CCTV footage fps
  FACE_FRAME_SKIP    8   → 15   faces don't change fast, run less often
  JPEG_QUALITY       72  → 65   slightly lower = faster encode & send
  ENABLE_RECORDING   True→ False disabled by default — costs ~25% CPU
  CAMERA_FPS         12  → 13   matches actual footage rate
"""

from pydantic_settings import BaseSettings
from pydantic import Field
import torch
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application settings with CPU/GPU auto-detection"""

    # ── Server ────────────────────────────────────────────────────────────────
    HOST:  str  = "0.0.0.0"
    PORT:  int  = 8000
    DEBUG: bool = False           # was True — disables SQLAlchemy echo (saves ~5% CPU)

    # ── Device ────────────────────────────────────────────────────────────────
    #DEVICE: str = Field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    DEVICE: str = "cpu"
    USE_HALF_PRECISION: bool = Field(default_factory=lambda: torch.cuda.is_available())

    # ── AI Model Paths ────────────────────────────────────────────────────────
    YOLO_MODEL:     str = "yolov8n.pt"                    # Nano — fastest on CPU
    CLIP_MODEL:     str = "openai/clip-vit-base-patch32"
    FACENET_MODEL:  str = "vggface2"

    # ── Model thresholds ──────────────────────────────────────────────────────
    YOLO_CONFIDENCE:           float = 0.40   # SAFEWATCH_FIX_GUIDE §6: fewer ghost dets
    YOLO_IOU_THRESHOLD:        float = 0.45
    FACE_DETECTION_CONFIDENCE: float = 0.85   # was 0.9 — catches more faces at distance
    FACE_RECOGNITION_THRESHOLD: float = 0.60
    
    # Real-time tracking settings
    ENABLE_REALTIME_TRACKING: bool = True
    TRACKING_PREDICTION_FRAMES: int = 3  # Predict 3 frames ahead
    TRACKING_TIMEOUT_SECONDS: float = 1.0  # Remove objects after 1 second of no detection
    MIN_IOU_FOR_TRACKING: float = 0.3  # Minimum IoU to consider same object

    # ── Image Processing ──────────────────────────────────────────────────────
    # 320px: ~18ms YOLO on i5-6500.  416px: ~28ms.
    # 320 is sufficient for persons >40px tall (anyone within ~8m of a standard cam).
    # 416px catches seated/partial persons better.
    # On GPU (i7-4790 + GeForce) YOLO is ~2ms at any size — no penalty.
    IMG_SIZE:   int = 416   # SAFEWATCH_FIX_GUIDE §6: catches seated/partial better; square-squash distorted aspect ratio causing bbox misalignment
    BATCH_SIZE: int = Field(default_factory=lambda: 1) # Force 1 for CPU stability
    #BATCH_SIZE: int = Field(default_factory=lambda: 8 if torch.cuda.is_available() else 1)

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./safewatch.db"

    # ── Storage Paths ─────────────────────────────────────────────────────────
    BASE_DIR:          Path = Path(__file__).resolve().parent.parent
    DATA_DIR:          Path = BASE_DIR / "data"
    RECORDINGS_DIR:    Path = BASE_DIR / "data" / "recordings"
    ALERTS_DIR:        Path = BASE_DIR / "data" / "alerts"
    FACES_KNOWN_DIR:   Path = BASE_DIR / "data" / "faces" / "known"
    FACES_UNKNOWN_DIR: Path = BASE_DIR / "data" / "faces" / "unknown"
    LOGS_DIR:          Path = BASE_DIR / "data" / "logs"

    # ── Camera ────────────────────────────────────────────────────────────────
    CAMERA_FPS:                 int = 13   # matches reference CCTV footage
    RECORDING_SEGMENT_DURATION: int = 300
    ALERT_VIDEO_DURATION:       int = 30
    MAX_CAMERAS:                int = 16
    CAMERA_CONNECTION_TIMEOUT:  int = 20
    RTSP_TRANSPORT:             str = "tcp"

    # ── Zone Detection ────────────────────────────────────────────────────────
    ZONE_1_DISTANCE: float = 2.0
    ZONE_2_DISTANCE: float = 5.0
    ZONE_3_DISTANCE: float = 10.0

    # ── Alert Levels ──────────────────────────────────────────────────────────
    ALERT_LEVEL_1_ACTIONS: list = Field(default_factory=lambda: [
        "fighting", "weapon_detected", "weapon_grip", "weapon",
        "violence", "fire", "falling", "break_in"])
    ALERT_LEVEL_2_ACTIONS: list = Field(default_factory=lambda: [
        "running", "shouting", "vandalism", "theft", "trespassing",
        "stealing", "crowding", "distress"])
    ALERT_LEVEL_3_ACTIONS: list = Field(default_factory=lambda: [
        "loitering", "suspicious_behavior", "unauthorized_area",
        "possible_weapon", "sneaking"])

    # ── Messaging ─────────────────────────────────────────────────────────────
    SMTP_SERVER:   str = "smtp.gmail.com"
    SMTP_PORT:     int = 587
    SMTP_USERNAME: str = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    ALERT_EMAIL_RECIPIENTS: list = Field(default_factory=lambda: ["admin@neelaminds.com"])

    TWILIO_ACCOUNT_SID:  str = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN:   str = os.getenv("TWILIO_AUTH_TOKEN", "")
    TWILIO_PHONE_NUMBER: str = os.getenv("TWILIO_PHONE_NUMBER", "")
    ALERT_SMS_RECIPIENTS:   list = Field(default_factory=lambda: ["+916383345184"])
    WHATSAPP_RECIPIENTS:    list = Field(default_factory=lambda: ["+916383345184"])

    # ── Performance ───────────────────────────────────────────────────────────
    NUM_WORKERS:    int = Field(default_factory=lambda: 4 if torch.cuda.is_available() else 2)
    PREFETCH_FACTOR: int = 2

    # ── Frame Processing Control ──────────────────────────────────────────────
    # These three values are the most important tuning knobs on a CPU-only machine.
    #
    #   DETECTION_FRAME_SKIP = 3
    #     YOLO runs every 3rd capture frame.
    #     At 13fps capture → ~4.3 YOLO calls/sec per camera.
    #     With 2 cameras and AI_POOL_WORKERS=2 → exactly 2 concurrent = no thrash.
    #
    #   ACTION_FRAME_SKIP = 30
    #     CLIP (150ms) runs every 30th detection frame.
    #     At 4.3 detections/sec → CLIP fires ~0.14x/sec per camera = once every 7s.
    #     Enough to catch sustained fighting/running. Not enough to miss a 3s event.
    #
    #   FACE_FRAME_SKIP = 15
    #     FaceNet (25ms MTCNN + 15ms embed) runs every 15th detection frame.
    #     At 4.3 det/sec → ~0.29 face runs/sec per camera = once every 3.4s.
    #     Faces don't change — this is perfectly adequate.
    #
    # GPU: YOLO takes ~2ms. Run every 2nd frame = 6+ detections/sec.
    # CPU fallback: value of 3 keeps load under control.
    DETECTION_FRAME_SKIP: int = 0   # YOLO every AI cycle (0.5s) = 2 det/s — responsive
    # GPU: CLIP takes ~5ms (vs 150ms CPU). Run every 10 det-frames.
    ACTION_FRAME_SKIP:    int = 15  # SAFEWATCH_FIX_GUIDE §6: CLIP 2× more often = ~5s on CPU — catches events
    FACE_FRAME_SKIP:      int = 15  # face check every ~5s on CPU

    # ── Low-Spec Profile (i5-6500) ────────────────────────────────────────────
    #
    #   STREAM_FPS = 13
    #     Used for FPS reporting, queue size calculation, and AI loop timing.
    #     NOT a throttle — every camera frame is pushed to the browser.
    #     Set this to match your camera's actual capture rate.
    #     Reference footage: 13.09fps → set to 13.
    #
    #   AI_POOL_WORKERS = 2
    #     Max concurrent AI inference threads across ALL cameras.
    #     i5-6500 has 4 cores. AI gets 2, capture+network+DB get 2.
    #     Setting this to 4 causes CPU thrashing and visible stream stutter.
    #
    #   IO_POOL_WORKERS = 4
    #     Disk writes (face images, recordings). Separate pool so AI never
    #     competes with disk I/O.
    #
    #   JPEG_QUALITY = 65
    #     65 is virtually indistinguishable from 72 at CCTV resolution but
    #     encodes ~15% faster and sends ~10% fewer bytes.
    #
    #   ENABLE_RECORDING = False
    #     Continuous recording costs ~25% CPU (VideoWriter + disk writes).
    #     Disable on i5-6500 to keep streams smooth. Enable only on i7/GPU.
    #
    STREAM_FPS:       int  = 13   # Push encode thread to 13fps target
    # GPU handles AI concurrency internally; 4 workers keeps pipeline full.
    AI_POOL_WORKERS:  int  = 1   # FIX: was 4; 4 workers on 4-core CPU starves capture thread
    IO_POOL_WORKERS:  int  = 4
    JPEG_QUALITY:     int  = 70
    # ── CLIP mode ─────────────────────────────────────────────────────────────
    # False = auto-select based on GPU/RAM (recommended)
    # True  = force INT8 quantized CLIP regardless of hardware (low-spec override)
    CLIP_LITE_MODE: bool = False

    # ── Stream resolution cap ─────────────────────────────────────────────────
    # Downscales the MJPEG stream to browser only (AI and recording stay full-res)
    # 1280 = cap at 720p (recommended for 1920x1080 cameras: 3x encode speedup)
    # 0    = disabled (send full resolution to browser — heavy on CPU/network)
    STREAM_MAX_WIDTH: int = 1280

    # ── AI latency (seconds behind live feed) ───────────────────────────────
    # AI analyses frames from the ring buffer N seconds behind live.
    # 2.0s = default: gives CPU time to process without blocking stream.
    # 0.5s = near-realtime (needs GPU or fast CPU)
    AI_LATENCY_SECONDS: float = 0.3   # FIX: was 1.5 — caused bboxes to float 1.5s behind person

    ENABLE_RECORDING: bool = False

    # ── Pose-Gated Pipeline (NEW) ─────────────────────────────────────────────
    #
    # POSE_ENABLED = True
    #   Activates the MediaPipe pose decision layer. Pose runs on EVERY person
    #   every detection cycle (~8ms each on i5-6500) and gates CLIP/Face.
    #   Set False to revert to the old blind frame-skip behaviour.
    #
    # POSE_CLIP_ON_SUSPICIOUS = True
    #   Run CLIP only when pose says SUSPICIOUS. Saves ~150ms CLIP calls on
    #   normal scenes (standing/walking).
    #
    # POSE_ALERT_ON_CRITICAL = True
    #   Issue alert immediately when pose detects CRITICAL (fall/fighting/weapon)
    #   without waiting for CLIP confirmation. Reduces alert latency to ~1 YOLO cycle.
    #
    # FACE_GATE_BY_TRACK_ID = True
    #   Only run FaceNet on a track_id that hasn't been embedded yet, OR after
    #   FACE_REEMBED_SECONDS have elapsed since last embed for that track.
    #   Replaces FACE_FRAME_SKIP counter — far more efficient.
    #
    # FACE_REEMBED_SECONDS = 60
    #   Re-embed a track after this many seconds to catch face-angle changes.
    #
    # POSE_SUSPICIOUS_DWELL_SECONDS = 90
    #   Seconds a person must linger in same zone before flagged as loitering.
    #
    # POSE_FALL_ANGLE_THRESH = 50
    #   Torso degrees from vertical. 50° catches falls without false-positives
    #   on people tying shoes (which reach ~40°).
    #
    # ── Weapon Detection ─────────────────────────────────────────────────────
    WEAPON_MODEL:              str   = "data/models/weapon_nano.pt"
    WEAPON_DETECTION_ENABLED:  bool  = True   # uses COCO knife/scissors if no custom model
    WEAPON_CONF_THRESHOLD:     float = 0.45
    # ── Training data collection ──────────────────────────────────────────────
    COLLECT_TRAINING_DATA:     bool  = False  # set True to collect CLIP training crops
    TRAINING_CROPS_DIR:        str   = "data/training_crops"

    POSE_ENABLED:                bool  = True
    POSE_CLIP_ON_SUSPICIOUS:     bool  = True
    POSE_ALERT_ON_CRITICAL:      bool  = True
    FACE_GATE_BY_TRACK_ID:       bool  = True
    FACE_REEMBED_SECONDS:        float = 45.0   # SAFEWATCH_FIX_GUIDE §6
    POSE_SUSPICIOUS_DWELL_SECONDS: float = 120.0  # SAFEWATCH_FIX_GUIDE §6
    POSE_FALL_ANGLE_THRESH:      float = 55.0   # SAFEWATCH_FIX_GUIDE §6
    POSE_CHILD_HEIGHT_FRAC:      float = 0.25
    POSE_FACE_COVER_VIS_THRESH:  float = 0.60
    POSE_ERRATIC_DIR_CHANGES:    int   = 3

    # ── Resize mode (NEW) ─────────────────────────────────────────────────────
    # YOLO_RECT_RESIZE = True  →  resize frame to 640×360 (16:9 preserve aspect)
    #   before feeding YOLO.  Eliminates the bbox aspect-ratio distortion from
    #   square-squash (IMG_SIZE=320 letterbox).
    # When False, falls back to IMG_SIZE letterbox (previous behaviour).
    YOLO_RECT_WIDTH:  int  = 640
    YOLO_RECT_HEIGHT: int  = 360
    YOLO_RECT_RESIZE: bool = True

    # ── Face Clustering ───────────────────────────────────────────────────────
    MIN_CLUSTER_SIZE:         int   = 5
    CLUSTER_SELECTION_EPSILON: float = 0.5

    # ── Bbox overlay display ──────────────────────────────────────────────────
    # How long (seconds) to keep showing bbox overlays in the stream after the
    # last AI result.  At AI_INTERVAL=0.7s, 4.0s = ~5 missed cycles before fade.
    # Must be >= GHOST_BBOX_SECONDS so the two timers don't fight each other.
    BBOX_DISPLAY_SECONDS: float = 4.0

    # How long to keep showing the last known bbox when YOLO returns 0 detections
    # (person partially occluded / behind glass). ByteTracker handles occlusion
    # internally via track_buffer; this is the fallback when tracker is inactive.
    # Must be <= BBOX_DISPLAY_SECONDS.
    GHOST_BBOX_SECONDS: float = 4.0

    class Config:
        env_file      = ".env"
        case_sensitive = True
        extra          = "ignore"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.create_directories()
        self.log_device_info()

    def create_directories(self):
        for d in [self.DATA_DIR, self.RECORDINGS_DIR, self.ALERTS_DIR,
                  self.FACES_KNOWN_DIR, self.FACES_UNKNOWN_DIR, self.LOGS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    def log_device_info(self):
        print(f"\n{'='*60}")
        print(f"SafeWatch AI — System Configuration")
        print(f"{'='*60}")
        print(f"Device       : {self.DEVICE.upper()}")
        if self.DEVICE == "cuda":
            print(f"GPU          : {torch.cuda.get_device_name(0)}")
            print(f"VRAM         : {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
        else:
            print(f"Mode         : CPU-only (i5-6500 profile)")
        print(f"Stream FPS   : {self.STREAM_FPS}")
        print(f"YOLO img size: {self.IMG_SIZE}px")
        print(f"AI workers   : {self.AI_POOL_WORKERS}")
        print(f"YOLO skip    : every {self.DETECTION_FRAME_SKIP} frames")
        print(f"CLIP skip    : every {self.ACTION_FRAME_SKIP} det-frames")
        print(f"Face skip    : every {self.FACE_FRAME_SKIP} det-frames")
        print(f"Recording    : {'ON' if self.ENABLE_RECORDING else 'OFF'}")
        print(f"{'='*60}\n")


settings = Settings()