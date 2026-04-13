#!/bin/bash
# train_yolo_weapon.sh — Fine-tune YOLOv8n for weapon detection
#
# PREREQUISITES:
#   pip install ultralytics roboflow
#   Set ROBOFLOW_API_KEY in environment
#
# USAGE:
#   chmod +x train_yolo_weapon.sh
#   ROBOFLOW_API_KEY=your_key ./train_yolo_weapon.sh

set -e

echo "══════════════════════════════════════════════════"
echo "  SafeWatch AI — YOLOv8-nano Weapon Model Training"
echo "══════════════════════════════════════════════════"

# ── Step 1: Download weapon dataset from Roboflow ──────────────────────────
echo ""
echo "Step 1: Downloading weapon detection dataset..."
python3 - << 'PYEOF'
from roboflow import Roboflow
import os

api_key = os.environ.get("ROBOFLOW_API_KEY", "")
if not api_key:
    print("ERROR: Set ROBOFLOW_API_KEY environment variable")
    exit(1)

rf = Roboflow(api_key=api_key)
proj = rf.workspace("roboflow-universe-projects").project("weapon-detection-gnvem")
dataset = proj.version(8).download("yolov8", location="data/weapon_dataset")
print(f"Dataset downloaded: {dataset.location}")
PYEOF

# ── Step 2: Verify dataset structure ─────────────────────────────────────
echo ""
echo "Step 2: Verifying dataset..."
ls data/weapon_dataset/
cat data/weapon_dataset/data.yaml

# ── Step 3: Fine-tune YOLOv8n ─────────────────────────────────────────────
echo ""
echo "Step 3: Training (50 epochs, ~20-30 min on CPU, ~3-5 min on GPU)..."
yolo detect train \
  model=yolov8n.pt \
  data=data/weapon_dataset/data.yaml \
  epochs=50 \
  imgsz=416 \
  batch=16 \
  lr0=0.001 \
  freeze=10 \
  patience=15 \
  name=safewatch_weapon \
  project=runs/weapon \
  exist_ok=True \
  device=0

# ── Step 4: Copy best weights to models directory ─────────────────────────
echo ""
echo "Step 4: Installing model..."
mkdir -p data/models
cp runs/weapon/safewatch_weapon/weights/best.pt data/models/weapon_nano.pt
echo "✅ Weapon model installed: data/models/weapon_nano.pt"

# ── Step 5: Validate ──────────────────────────────────────────────────────
echo ""
echo "Step 5: Validating..."
yolo detect val \
  model=data/models/weapon_nano.pt \
  data=data/weapon_dataset/data.yaml \
  device=0

echo ""
echo "══════════════════════════════════════════════════"
echo "  Training complete!"
echo "  Model: data/models/weapon_nano.pt"
echo "  Add to config.py:"
echo "    WEAPON_MODEL: str = 'data/models/weapon_nano.pt'"
echo "    WEAPON_DETECTION_ENABLED: bool = True"
echo "══════════════════════════════════════════════════"
