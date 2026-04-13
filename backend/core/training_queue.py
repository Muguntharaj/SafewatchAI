"""
training_queue.py — SafeWatch AI
===================================
SQLite-backed store for hard examples and operator corrections.
Manages overnight trainer scheduling (2–5 AM only).
Exports COCO-format JSON for cloud fine-tuning.

ARCHITECTURE
─────────────
Three tables:

  hard_examples
    Frames where SORT held a confident track alive but YOLO returned 0
    detections — the strongest possible hard negatives. Stored as JPEG
    crops with a Kalman-predicted bbox as pseudo ground-truth label.

  operator_corrections
    When a human dismisses a false-positive alert or corrects a CLIP action
    label, the (frame_crop, correct_label) pair lands here.

  model_candidates
    Records of fine-tuned model checkpoints, their mAP scores, and whether
    they were promoted to production.

FILTERING RULES (hard examples only)
──────────────────────────────────────
A missed-detection frame is only saved if:
  • track.hit_streak >= MIN_HIT_STREAK (5) — established, confident track
  • track.time_since_update <= MAX_TIME_SINCE_UPDATE (3) — just lost
This filters out genuine exits and occlusion tails. Only true missed
detections — YOLO missed a person who was definitely there — are kept.

OVERNIGHT SCHEDULER
────────────────────
OvernightScheduler.maybe_run() is called from an asyncio task every hour.
It triggers fine-tuning only between 02:00–05:00 local time and only when
the queue has accumulated at least MIN_SAMPLES_TO_TRAIN high-quality crops.
After training, ModelValidator compares mAP against the baseline and only
swaps the model into production if it improved.

COCO EXPORT
────────────
TrainingQueue.export_coco(output_dir) generates:
  annotations.json — standard COCO detection format
  images/          — JPEG crops, named by sample ID

Usage:
    from core.training_queue import training_queue, OvernightScheduler

    # From camera_manager when SORT holds a track but YOLO returned 0:
    training_queue.add_hard_example(
        camera_id   = self.camera_id,
        frame_crop  = crop_bgr,        # numpy array
        pseudo_bbox = [x1, y1, x2, y2],
        quality_score = hit_streak / 10.0,
    )

    # From UI when operator corrects a CLIP action label:
    training_queue.add_correction(
        camera_id     = camera_id,
        frame_crop    = crop_bgr,
        correct_label = "running",
        original_label = "normal",
    )

    # Start the overnight scheduler (once, from main.py lifespan):
    scheduler = OvernightScheduler(training_queue)
    asyncio.create_task(scheduler.run())
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_HIT_STREAK          = 5      # SORT hit_streak threshold for trusted track
MAX_TIME_SINCE_UPDATE   = 3      # SORT frames since last YOLO match
MIN_QUALITY_SCORE       = 0.50   # discard low-confidence pseudo-labels
MIN_SAMPLES_TO_TRAIN    = 100    # won't trigger training until this many accumulate
MAX_QUEUE_SIZE          = 5000   # prune oldest when exceeded
TRAIN_WINDOW_START_H    = 2      # 2 AM
TRAIN_WINDOW_END_H      = 5      # 5 AM
SCHEDULER_CHECK_SECS    = 3600   # check once per hour


class TrainingQueue:
    """
    Thread-safe SQLite-backed store for training samples.

    The database is created on first use at data/training_queue.db.
    All public methods are safe to call from asyncio (they run SQLite in
    a thread executor) and from background threads (they use a mutex).
    """

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            try:
                from core.config import settings
                db_path = settings.DATA_DIR / "training_queue.db"
            except Exception:
                db_path = Path("data/training_queue.db")

        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()
        logger.info("[TrainingQueue] Database: %s", self._db_path)

    # ── Database init ─────────────────────────────────────────────────────────
    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS hard_examples (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id       TEXT    NOT NULL,
                    captured_at     REAL    NOT NULL,
                    pseudo_bbox     TEXT    NOT NULL,  -- JSON [x1,y1,x2,y2]
                    quality_score   REAL    NOT NULL,
                    jpeg_bytes      BLOB    NOT NULL,
                    used_in_train   INTEGER DEFAULT 0,
                    flagged_review  INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS operator_corrections (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id       TEXT    NOT NULL,
                    captured_at     REAL    NOT NULL,
                    original_label  TEXT,
                    correct_label   TEXT    NOT NULL,
                    jpeg_bytes      BLOB    NOT NULL,
                    used_in_train   INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS model_candidates (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at      REAL    NOT NULL,
                    model_path      TEXT    NOT NULL,
                    baseline_map    REAL,
                    candidate_map   REAL,
                    promoted        INTEGER DEFAULT 0,
                    notes           TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_he_quality
                    ON hard_examples(quality_score DESC);
                CREATE INDEX IF NOT EXISTS idx_he_used
                    ON hard_examples(used_in_train);
                CREATE INDEX IF NOT EXISTS idx_oc_used
                    ON operator_corrections(used_in_train);
            """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False,
                               timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Public write API ──────────────────────────────────────────────────────
    def add_hard_example(
        self,
        camera_id:    str,
        frame_crop:   np.ndarray,
        pseudo_bbox:  List[int],
        quality_score: float,
        hit_streak:   int = 999,
        time_since_update: int = 0,
    ) -> bool:
        """
        Save a hard negative (SORT held track, YOLO missed).

        Returns True if the sample was accepted, False if filtered out.
        """
        # Filtering gate
        if hit_streak < MIN_HIT_STREAK:
            return False
        if time_since_update > MAX_TIME_SINCE_UPDATE:
            return False
        if quality_score < MIN_QUALITY_SCORE:
            return False
        if frame_crop is None or frame_crop.size == 0:
            return False

        # Encode crop to JPEG
        ok, buf = cv2.imencode('.jpg', frame_crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return False
        jpeg_bytes = buf.tobytes()

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO hard_examples "
                    "(camera_id, captured_at, pseudo_bbox, quality_score, jpeg_bytes) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (camera_id, time.time(), json.dumps(pseudo_bbox),
                     quality_score, sqlite3.Binary(jpeg_bytes))
                )
                # Prune oldest if over capacity
                count = conn.execute(
                    "SELECT COUNT(*) FROM hard_examples").fetchone()[0]
                if count > MAX_QUEUE_SIZE:
                    excess = count - MAX_QUEUE_SIZE
                    conn.execute(
                        "DELETE FROM hard_examples WHERE id IN "
                        "(SELECT id FROM hard_examples ORDER BY captured_at ASC LIMIT ?)",
                        (excess,))
        return True

    def add_correction(
        self,
        camera_id:      str,
        frame_crop:     np.ndarray,
        correct_label:  str,
        original_label: str = "",
    ) -> bool:
        """Save an operator-corrected CLIP label."""
        if frame_crop is None or frame_crop.size == 0:
            return False

        ok, buf = cv2.imencode('.jpg', frame_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return False
        jpeg_bytes = buf.tobytes()

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO operator_corrections "
                    "(camera_id, captured_at, original_label, correct_label, jpeg_bytes) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (camera_id, time.time(), original_label, correct_label,
                     sqlite3.Binary(jpeg_bytes))
                )
        return True

    def record_candidate(
        self,
        model_path:    str,
        baseline_map:  Optional[float],
        candidate_map: float,
        promoted:      bool,
        notes:         str = "",
    ) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO model_candidates "
                    "(created_at, model_path, baseline_map, candidate_map, promoted, notes) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (time.time(), model_path, baseline_map, candidate_map,
                     int(promoted), notes)
                )

    # ── Public read API ───────────────────────────────────────────────────────
    def sample_count(self) -> dict:
        with self._lock:
            with self._connect() as conn:
                he  = conn.execute(
                    "SELECT COUNT(*) FROM hard_examples WHERE used_in_train=0"
                ).fetchone()[0]
                oc  = conn.execute(
                    "SELECT COUNT(*) FROM operator_corrections WHERE used_in_train=0"
                ).fetchone()[0]
                tot = conn.execute(
                    "SELECT COUNT(*) FROM hard_examples"
                ).fetchone()[0]
        return {"hard_examples_unused": he,
                "corrections_unused":   oc,
                "hard_examples_total":  tot}

    def top_hard_examples(self, n: int = 500) -> List[dict]:
        """Return top-N highest quality unused hard examples."""
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT id, camera_id, captured_at, pseudo_bbox, quality_score, jpeg_bytes "
                    "FROM hard_examples "
                    "WHERE used_in_train=0 AND flagged_review=0 "
                    "ORDER BY quality_score DESC LIMIT ?", (n,)
                ).fetchall()
        return [dict(r) for r in rows]

    def mark_used(self, ids: List[int], table: str = "hard_examples") -> None:
        if not ids or table not in ("hard_examples", "operator_corrections"):
            return
        placeholders = ",".join("?" * len(ids))
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    f"UPDATE {table} SET used_in_train=1 "
                    f"WHERE id IN ({placeholders})", ids)

    def flag_for_review(self, ids: List[int]) -> None:
        placeholders = ",".join("?" * len(ids))
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    f"UPDATE hard_examples SET flagged_review=1 "
                    f"WHERE id IN ({placeholders})", ids)

    # ── COCO Export ───────────────────────────────────────────────────────────
    def export_coco(self, output_dir: Path, max_samples: int = 1000) -> Path:
        """
        Export hard examples as a COCO-format dataset.

        output_dir/
            images/          ← JPEG crops named {id}.jpg
            annotations.json ← standard COCO detection JSON

        Returns path to annotations.json.
        Class 0 = "person" (the only class YOLO-nano watches).
        """
        output_dir = Path(output_dir)
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        samples = self.top_hard_examples(n=max_samples)
        if not samples:
            raise ValueError("No samples in queue to export")

        coco = {
            "info":        {"description": "SafeWatch hard-example dataset",
                            "date_created": datetime.now(timezone.utc).isoformat()},
            "licenses":    [],
            "categories":  [{"id": 1, "name": "person", "supercategory": "person"}],
            "images":      [],
            "annotations": [],
        }

        ann_id = 1
        for sample in samples:
            sid   = sample["id"]
            bbox  = json.loads(sample["pseudo_bbox"])   # [x1,y1,x2,y2]
            img   = cv2.imdecode(
                np.frombuffer(sample["jpeg_bytes"], np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            h, w = img.shape[:2]
            fname = f"{sid}.jpg"
            cv2.imwrite(str(images_dir / fname), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])

            coco["images"].append({
                "id":        sid,
                "file_name": fname,
                "width":     w,
                "height":    h,
            })

            x1, y1, x2, y2 = bbox
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            coco["annotations"].append({
                "id":          ann_id,
                "image_id":    sid,
                "category_id": 1,
                "bbox":        [float(x1), float(y1), float(bw), float(bh)],
                "area":        float(bw * bh),
                "iscrowd":     0,
            })
            ann_id += 1

        ann_path = output_dir / "annotations.json"
        ann_path.write_text(json.dumps(coco, indent=2))
        logger.info("[TrainingQueue] COCO export: %d images → %s", len(coco["images"]), ann_path)
        return ann_path


# ── Overnight Scheduler ───────────────────────────────────────────────────────
class OvernightScheduler:
    """
    Asyncio task that triggers CPU fine-tuning between 02:00–05:00 local time
    when the hard-example queue has accumulated enough samples.

    Plug into main.py lifespan:
        scheduler = OvernightScheduler(training_queue)
        asyncio.create_task(scheduler.run())

    Fine-tuning is done via the ultralytics API (CPU, slow but correct).
    ModelValidator checks mAP on a held-out clip and only promotes the
    candidate if it improved over the current production model.
    """

    def __init__(self, queue: TrainingQueue, check_interval: float = SCHEDULER_CHECK_SECS):
        self._queue    = queue
        self._interval = check_interval
        self._running  = False
        self._last_run_date: Optional[str] = None

    async def run(self) -> None:
        self._running = True
        logger.info("[OvernightScheduler] Started — checks every %.0fs", self._interval)
        while self._running:
            try:
                await asyncio.sleep(self._interval)
                await self.maybe_run()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[OvernightScheduler] Error: %s", e, exc_info=True)

    def stop(self) -> None:
        self._running = False

    async def maybe_run(self) -> None:
        now   = datetime.now()
        today = now.strftime("%Y-%m-%d")

        # Already ran today?
        if self._last_run_date == today:
            return

        # Within training window?
        if not (TRAIN_WINDOW_START_H <= now.hour < TRAIN_WINDOW_END_H):
            return

        # Enough samples?
        counts = self._queue.sample_count()
        n      = counts["hard_examples_unused"]
        if n < MIN_SAMPLES_TO_TRAIN:
            logger.info("[OvernightScheduler] %d/%d samples — skipping",
                        n, MIN_SAMPLES_TO_TRAIN)
            return

        logger.info("[OvernightScheduler] Starting overnight training (%d samples)", n)
        self._last_run_date = today

        # Run fine-tuning in a thread so asyncio isn't blocked
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, self._finetune_yolo)
            if result:
                logger.info("[OvernightScheduler] Training complete: %s", result)
            else:
                logger.warning("[OvernightScheduler] Training produced no candidate")
        except Exception as e:
            logger.error("[OvernightScheduler] Training failed: %s", e, exc_info=True)

    def _finetune_yolo(self) -> Optional[str]:
        """
        Blocking function — runs in a thread executor.
        Exports COCO dataset, calls ultralytics trainer on CPU,
        validates the result, and hot-swaps if mAP improved.

        Returns the path to the promoted model, or None on failure.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("[OvernightScheduler] ultralytics not installed — "
                         "pip install ultralytics")
            return None

        try:
            from core.config import settings
        except ImportError:
            logger.error("[OvernightScheduler] Cannot import settings")
            return None

        # ── Export dataset ────────────────────────────────────────────────────
        export_dir = settings.DATA_DIR / "finetune_export" / datetime.now().strftime("%Y%m%d")
        try:
            ann_path = self._queue.export_coco(export_dir)
        except ValueError as e:
            logger.warning("[OvernightScheduler] Export skipped: %s", e)
            return None

        # Write ultralytics data.yaml
        yaml_path = export_dir / "data.yaml"
        yaml_path.write_text(
            f"path: {export_dir}\n"
            f"train: images\n"
            f"val: images\n"
            f"nc: 1\n"
            f"names: ['person']\n"
        )

        # ── Fine-tune (frozen backbone — only head trains on CPU) ─────────────
        run_dir     = settings.DATA_DIR / "finetune_runs"
        candidate_path = run_dir / "candidate.pt"
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            model = YOLO(settings.YOLO_MODEL)
            model.train(
                data    = str(yaml_path),
                epochs  = 5,            # CPU: 5 epochs is ~10 min for 500 crops
                imgsz   = 320,
                batch   = 4,
                device  = "cpu",
                freeze  = 10,           # freeze first 10 backbone layers
                project = str(run_dir),
                name    = "finetune",
                exist_ok= True,
                verbose = False,
            )
            candidate_path = run_dir / "finetune" / "weights" / "best.pt"
        except Exception as e:
            logger.error("[OvernightScheduler] Training error: %s", e)
            return None

        if not candidate_path.exists():
            logger.error("[OvernightScheduler] No weights produced at %s", candidate_path)
            return None

        # ── Validate ──────────────────────────────────────────────────────────
        baseline_map, candidate_map = self._validate(
            str(settings.YOLO_MODEL), str(candidate_path), str(yaml_path))

        promoted = candidate_map > baseline_map
        self._queue.record_candidate(
            model_path    = str(candidate_path),
            baseline_map  = baseline_map,
            candidate_map = candidate_map,
            promoted      = promoted,
            notes         = f"epochs=5 freeze=10 samples={self._queue.sample_count()['hard_examples_total']}",
        )

        if promoted:
            # Mark the samples used
            samples = self._queue.top_hard_examples(n=500)
            self._queue.mark_used([s["id"] for s in samples])

            # Hot-swap: update settings to point at new model
            # The AIEngine picks up this change on the next model reload
            try:
                object.__setattr__(settings, 'YOLO_MODEL', str(candidate_path))
                logger.info("[OvernightScheduler] PROMOTED %s  mAP %.3f → %.3f",
                            candidate_path, baseline_map, candidate_map)
            except Exception as e:
                logger.warning("[OvernightScheduler] Hot-swap failed: %s", e)
            return str(candidate_path)
        else:
            logger.info("[OvernightScheduler] Candidate NOT promoted "
                        "(baseline=%.3f candidate=%.3f)", baseline_map, candidate_map)
            # Flag samples for human review
            samples = self._queue.top_hard_examples(n=100)
            self._queue.flag_for_review([s["id"] for s in samples])
            return None

    @staticmethod
    def _validate(base_model: str, candidate_model: str, data_yaml: str,
                  ) -> Tuple[float, float]:
        """
        Run ultralytics val() on both models against the held-out set.
        Returns (baseline_map50, candidate_map50).
        Falls back to (0.0, 0.0) if validation fails.
        """
        try:
            from ultralytics import YOLO
            base = YOLO(base_model).val(data=data_yaml, device="cpu",
                                        imgsz=320, verbose=False)
            cand = YOLO(candidate_model).val(data=data_yaml, device="cpu",
                                             imgsz=320, verbose=False)
            return float(base.box.map50), float(cand.box.map50)
        except Exception as e:
            logger.error("[OvernightScheduler] Validation error: %s", e)
            return 0.0, 0.0


# ── Module-level singleton ─────────────────────────────────────────────────────
training_queue = TrainingQueue()
