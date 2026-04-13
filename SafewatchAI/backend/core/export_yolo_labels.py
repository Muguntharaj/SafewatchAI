# export_yolo_labels.py
# Run against your DB to export all Detection rows as YOLO label files
import asyncio, cv2
from pathlib import Path
from sqlalchemy import select
from core.database import AsyncSessionLocal, Detection

EXPORT_DIR = Path("data/training/yolo_dataset")
IMG_DIR    = EXPORT_DIR / "images/train"
LBL_DIR    = EXPORT_DIR / "labels/train"
IMG_DIR.mkdir(parents=True, exist_ok=True)
LBL_DIR.mkdir(parents=True, exist_ok=True)

async def main():
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(select(Detection))).scalars().all()
    for row in rows:
        # You need to have saved the frame; if ENABLE_RECORDING=True this works:
        # otherwise integrate frame saving into _process_frame when COLLECT_TRAINING_DATA=True
        frame_path = Path(f"data/frames/{row.camera_id}_{row.id}.jpg")
        if not frame_path.exists():
            continue
        img = cv2.imread(str(frame_path))
        H, W = img.shape[:2]
        cx = (row.bbox_x1 + row.bbox_x2) / 2 / W
        cy = (row.bbox_y1 + row.bbox_y2) / 2 / H
        bw = (row.bbox_x2 - row.bbox_x1) / W
        bh = (row.bbox_y2 - row.bbox_y1) / H
        stem = f"frame_{row.id}"
        cv2.imwrite(str(IMG_DIR / f"{stem}.jpg"), img)
        with open(LBL_DIR / f"{stem}.txt", "w") as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")  # class 0 = person

asyncio.run(main())