# Manual Training Guide for SafeWatch Detection Models

## 1. CLIP Linear Probe (Fire/Weapon/Mask Detection)

### What it does:
Trains a logistic regression classifier on CLIP image features using your actual camera footage. This reduces false positives significantly compared to zero-shot CLIP.

### Training Steps:

```python
# Step 1: Collect training data
# Create folders: data/training/fire, data/training/weapon, data/training/mask, data/training/normal
# Add 50+ images per category from your cameras

# Step 2: Run training script
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

MODEL_NAME = "openai/clip-vit-base-patch32"
DATA_DIR   = Path("data/training")
OUTPUT     = Path("models/clip_probe.pkl")

def extract_features(image_paths, model, processor):
    features = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        inp = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            feat = model.get_image_features(**inp)[0].numpy()
        features.append(feat)
    return np.array(features)

# Load model
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

# Load data
categories = ["fire", "weapon", "mask", "normal"]
X, y = [], []
for i, cat in enumerate(categories):
    imgs = list((DATA_DIR / cat).glob("*.jpg")) + list((DATA_DIR / cat).glob("*.png"))
    feats = extract_features(imgs, model, processor)
    X.extend(feats)
    y.extend([i] * len(feats))

X, y = np.array(X), np.array(y)

# Train
clf = LogisticRegression(max_iter=1000, C=1.0)
clf.fit(X, y)

# Save
OUTPUT.parent.mkdir(exist_ok=True)
pickle.dump({
    "clf": clf,
    "class_names": categories,
    "accuracy": clf.score(X, y)
}, open(OUTPUT, "wb"))

print(f"Training complete. Accuracy: {clf.score(X, y)*100:.1f}%")
```

### To use the trained probe:
1. Place `clip_probe.pkl` in `data/models/`
2. Restart the backend - it auto-loads on startup

---

## 2. Reducing False Positives (Quick Fixes)

### Adjust confidence thresholds in `ai_engine.py`:

```python
# Line ~551: Increase weapon threshold
if max_weapon > 0.40:  # was 0.28
    return "weapon_detected", min(1.0, max_weapon * 2.0)

# Line ~553: Increase fire threshold  
if max_fire > 0.35:  # was 0.25
    return "fire", min(1.0, max_fire * 2.5)
```

### Add streak requirement for non-critical alerts:

In `pose_classifier.py`, the `mask_streak >= 3` already exists. Similar logic can be added for fire/weapon.

---

## 3. Face Recognition Issues (Known = Unknown)

### Root cause:
Person identified but not added to `known_cache` OR threshold too high.

### Quick fixes:

1. **Check embedding dimensions match** (`camera_manager_patches.py:45`):
```python
_THRESHOLDS = {512: 0.55, 128: 0.60, 59: 0.50}
# Lower threshold if needed:
_THRESHOLDS = {512: 0.50, 128: 0.55, 59: 0.45}
```

2. **Reload known cache manually** - call this endpoint:
```
POST /api/faces/invalidate-cache
```

3. **Verify person exists in database:**
```sql
SELECT id, name, classification FROM persons WHERE name = 'HR';
```

4. **Check embedding exists:**
```sql
SELECT person_id, embedding_dim FROM person_embeddings WHERE person_id = <id>;
```

---

## 4. Similarity Clustering for Notifications

The deduplication fix has been applied to `alert_manager.py`. For person-based grouping:

1. Alerts now cluster by (camera_id + action) within 60 seconds
2. To group by person instead, modify `_recent_alerts` key to include person_id

---

## 5. Verification Checklist

- [ ] CLIP probe trained on your camera angles
- [ ] Fire/Weapon thresholds raised if false positives occur
- [ ] Face threshold lowered if known people show as unknown
- [ ] Known cache invalidated and reloaded
- [ ] Person embeddings exist in database
