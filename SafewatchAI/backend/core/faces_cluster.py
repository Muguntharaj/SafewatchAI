"""
faces_cluster.py — Face Similarity Clustering  v2

CHANGES FROM v1
───────────────
  • Auto-cluster hook: _auto_cluster_new_face() runs in background every time
    a new unknown face is saved. It compares the new face against all existing
    Person records — if it matches a known person above threshold, the image
    is moved to that person's folder and the person's centroid is updated.
    This means clustering happens continuously, not just on manual trigger.

  • Centroid update: after clustering, the matched person's face_embedding is
    updated to np.mean(all_matched_embeddings) — progressive accuracy improvement.

  • Cross-model dim guard: embeddings with mismatched dimensions are skipped
    instead of producing garbage cosine scores.

  • Similarity threshold is now embedding-dimension aware:
    ArcFace 512 → 0.55, FaceNet 128 → 0.60, LBP 59 → 0.50

  • cluster_person_faces endpoint now returns centroid_updated bool.

ENDPOINT:
  POST /api/faces/cluster/{person_id}
  Body: { person_name, classification, similarity_threshold }

AUTO-CLUSTER TRIGGER:
  Called from camera_manager after _run_face_bg saves a new unknown face:
    from core.faces_cluster import auto_cluster_new_face
    asyncio.create_task(auto_cluster_new_face(new_image_path, embedding))
"""

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import select

from core.config import settings
from core.database import get_db, AsyncSession, AsyncSessionLocal, Person, Detection

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Dim-aware thresholds (must match faces.py) ────────────────────────────────
_THRESHOLDS    = {512: 0.55, 128: 0.60, 59: 0.50}
_DEFAULT_THRESHOLD = 0.55


def _dim_threshold(emb: np.ndarray, table: dict = _THRESHOLDS) -> float:
    return table.get(len(emb), _DEFAULT_THRESHOLD)


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or len(a) != len(b):
        return 0.0
    n1, n2 = np.linalg.norm(a), np.linalg.norm(b)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(a, b) / (n1 * n2))


def _load_emb(raw: Optional[bytes]) -> Optional[np.ndarray]:
    if not raw:
        return None
    try:
        return np.frombuffer(raw, dtype=np.float32).copy()
    except Exception:
        return None


# ── Embedding mtime-cache ─────────────────────────────────────────────────────
_embed_cache: dict = {}


def _load_image_embedding(image_path: Path) -> Optional[np.ndarray]:
    """CLIP or colour-histogram embedding for a face crop, mtime-cached."""
    key = str(image_path)
    try:
        mtime = image_path.stat().st_mtime
    except FileNotFoundError:
        return None

    if key in _embed_cache and _embed_cache[key][0] == mtime:
        return _embed_cache[key][1]

    try:
        from PIL import Image
        import torch
        try:
            from transformers import CLIPProcessor, CLIPModel
            cache = getattr(_load_image_embedding, '_clip', None)
            if cache is None:
                model = CLIPModel.from_pretrained(settings.CLIP_MODEL).eval()
                proc  = CLIPProcessor.from_pretrained(settings.CLIP_MODEL)
                _load_image_embedding._clip = (model, proc)
            model, proc = _load_image_embedding._clip
            img    = Image.open(image_path).convert("RGB")
            inputs = proc(images=img, return_tensors="pt")
            with torch.no_grad():
                feat = model.get_image_features(**inputs)
            embed = feat[0].cpu().numpy()
        except Exception:
            img = Image.open(image_path).convert("RGB").resize((64, 64))
            arr = np.array(img, dtype=np.float32).flatten()
            embed = arr
        norm = np.linalg.norm(embed)
        if norm > 0:
            embed = embed / norm
        _embed_cache[key] = (mtime, embed)
        return embed
    except Exception as e:
        logger.warning(f"[FaceCluster] Embedding failed for {image_path}: {e}")
        return None


# ── AUTO-CLUSTER: called after every new unknown face save ───────────────────
async def auto_cluster_new_face(
    new_image_path: Path,
    new_embedding:  np.ndarray,
) -> None:
    """
    Background task — runs after camera saves a new unknown face.

    Steps:
      1. Compare new_embedding against ALL Person records (known + unknown cluster)
      2. If match ≥ threshold → move image to person's folder, update centroid
      3. If no match → leave in FACES_UNKNOWN_DIR (shown in Unknown list)
    """
    if new_embedding is None or not new_image_path.exists():
        return

    threshold = _dim_threshold(new_embedding)

    try:
        async with AsyncSessionLocal() as db:
            persons = (await db.execute(select(Person))).scalars().all()
            best_sim = 0.0
            best_person = None

            for person in persons:
                p_emb = _load_emb(person.face_embedding)
                if p_emb is None:
                    continue
                sim = _safe_cosine(new_embedding, p_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_person = person

            if best_person and best_sim >= threshold:
                # Move to person's known folder
                folder = settings.FACES_KNOWN_DIR / best_person.person_id
                folder.mkdir(parents=True, exist_ok=True)
                dst = folder / new_image_path.name
                try:
                    shutil.move(str(new_image_path), str(dst))
                except Exception as e:
                    logger.debug(f"[AutoCluster] Move failed: {e}")
                    return

                # Update centroid embedding
                p_emb = _load_emb(best_person.face_embedding)
                if p_emb is not None and len(p_emb) == len(new_embedding):
                    centroid = np.mean([p_emb, new_embedding], axis=0).astype(np.float32)
                    best_person.face_embedding    = centroid.tobytes()
                    best_person.total_appearances = (best_person.total_appearances or 0) + 1
                    await db.commit()

                    # Refresh live camera cache with improved centroid
                    try:
                        from core.camera_manager import CameraStream
                        CameraStream.inject_known_person(
                            best_person.person_id,
                            best_person.name or "",
                            best_person.classification or "unknown",
                            centroid,
                        )
                    except Exception:
                        pass

                logger.info(
                    f"[AutoCluster] {new_image_path.name} → {best_person.person_id} "
                    f"({best_person.name}) sim={best_sim:.3f}"
                )
    except Exception as e:
        logger.error(f"[AutoCluster] auto_cluster_new_face error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
class ClusterRequest(BaseModel):
    person_name:          str
    classification:       str
    similarity_threshold: float = 0.0   # 0 = use dim-aware default


class ClusterResponse(BaseModel):
    person_id:        int
    matched:          int
    skipped:          int
    cluster_folder:   str
    centroid_updated: bool
    message:          str


def _cluster_person_sync(
    person_id:              int,
    person_name:            str,
    classification:         str,
    threshold_override:     float,
    classified_image_path:  Optional[Path],
) -> dict:
    """
    Synchronous clustering — run in executor so event loop stays free.

    Steps:
      1. Load reference embedding for the classified person
      2. Scan all images in FACES_UNKNOWN_DIR
      3. For each, compute similarity; move matches to cluster folder
      4. Compute centroid of all matched embeddings and return it
    """
    if not classified_image_path or not classified_image_path.exists():
        return {"matched": 0, "skipped": 0, "reason": "no source image",
                "centroid": None}

    ref_embed = _load_image_embedding(classified_image_path)
    if ref_embed is None:
        return {"matched": 0, "skipped": 0, "reason": "embedding failed",
                "centroid": None}

    # Use dim-aware threshold unless caller specified one
    threshold = (threshold_override
                 if threshold_override > 0.0
                 else _dim_threshold(ref_embed))

    folder_name = f"person_{person_id}_{person_name.replace(' ', '_').lower()}"
    if classification in ("employee", "owner"):
        cluster_dir = settings.FACES_KNOWN_DIR / folder_name
    else:
        cluster_dir = settings.FACES_UNKNOWN_DIR / folder_name
    cluster_dir.mkdir(parents=True, exist_ok=True)

    # Copy classified image into cluster folder
    try:
        dst = cluster_dir / classified_image_path.name
        if not dst.exists():
            shutil.copy2(classified_image_path, dst)
    except Exception as e:
        logger.warning(f"[FaceCluster] Could not copy classified image: {e}")

    matched_embs = [ref_embed]
    matched = 0
    skipped = 0

    unknown_dir = settings.FACES_UNKNOWN_DIR
    for img_path in unknown_dir.glob("*.jpg"):
        if img_path == classified_image_path:
            continue
        if img_path.stat().st_size < 1024:
            skipped += 1
            continue
        cand_emb = _load_image_embedding(img_path)
        if cand_emb is None or len(cand_emb) != len(ref_embed):
            skipped += 1
            continue
        sim = _safe_cosine(ref_embed, cand_emb)
        if sim >= threshold:
            try:
                dst = cluster_dir / img_path.name
                if not dst.exists():
                    shutil.move(str(img_path), str(dst))
                else:
                    img_path.unlink(missing_ok=True)
                matched_embs.append(cand_emb)
                matched += 1
                logger.info(f"[FaceCluster] {img_path.name} → {folder_name} sim={sim:.3f}")
            except Exception as e:
                logger.warning(f"[FaceCluster] Move failed {img_path}: {e}")
                skipped += 1
        else:
            skipped += 1

    # Compute centroid for caller to store in DB
    centroid = np.mean(matched_embs, axis=0).astype(np.float32) if len(matched_embs) > 1 else None

    return {
        "matched":       matched,
        "skipped":       skipped,
        "cluster_folder": str(cluster_dir),
        "centroid":      centroid,
    }


@router.post("/cluster/{person_id}", response_model=ClusterResponse)
async def cluster_person_faces(
    person_id: int,
    body:      ClusterRequest,
    db:        AsyncSession = Depends(get_db),
):
    """
    Cluster all unknown faces against a classified person.
    After clustering, updates the person's face_embedding to the centroid
    of all matched faces — improving future identification accuracy.
    """
    result = await db.execute(select(Person).where(Person.id == person_id))
    person = result.scalar_one_or_none()
    if not person:
        raise HTTPException(status_code=404, detail=f"Person {person_id} not found")

    face_image_path: Optional[Path] = None
    if person.face_image_path:
        p = Path(person.face_image_path)
        if p.exists():
            face_image_path = p
        else:
            fname = p.name
            for candidate in [settings.FACES_KNOWN_DIR / fname,
                               settings.FACES_UNKNOWN_DIR / fname]:
                if candidate.exists():
                    face_image_path = candidate
                    break

    if not face_image_path:
        raise HTTPException(status_code=422,
                            detail="Person has no face image — cannot cluster")

    loop = asyncio.get_event_loop()
    result_dict = await loop.run_in_executor(
        None, _cluster_person_sync,
        person_id, body.person_name, body.classification,
        body.similarity_threshold, face_image_path,
    )

    # ── Update centroid in DB ─────────────────────────────────────────────────
    centroid_updated = False
    centroid = result_dict.get("centroid")
    if centroid is not None:
        person.face_embedding     = centroid.tobytes()
        person.total_appearances  = (person.total_appearances or 0) + result_dict["matched"]
        await db.commit()
        centroid_updated = True
        # Refresh live camera cache
        try:
            from core.camera_manager import CameraStream
            CameraStream.inject_known_person(
                person.person_id, body.person_name,
                body.classification, centroid,
            )
        except Exception:
            pass

    logger.info(
        f"[FaceCluster] person_id={person_id} name={body.person_name} "
        f"matched={result_dict['matched']} skipped={result_dict['skipped']} "
        f"centroid_updated={centroid_updated}"
    )

    return ClusterResponse(
        person_id=person_id,
        matched=result_dict["matched"],
        skipped=result_dict["skipped"],
        cluster_folder=result_dict.get("cluster_folder", ""),
        centroid_updated=centroid_updated,
        message=(
            f"Clustered {result_dict['matched']} face(s) into {body.person_name}'s folder. "
            f"{'Centroid updated.' if centroid_updated else ''}"
        ),
    )


@router.get("/cluster/{person_id}/preview")
async def preview_cluster(person_id: int):
    matches = []
    for base_dir in [settings.FACES_KNOWN_DIR, settings.FACES_UNKNOWN_DIR]:
        for folder in base_dir.glob(f"person_{person_id}_*"):
            if folder.is_dir():
                subdir = "known" if base_dir == settings.FACES_KNOWN_DIR else "unknown"
                images = [
                    f"/media/faces/{subdir}/{folder.name}/{img.name}"
                    for img in folder.glob("*.jpg")
                ]
                matches.extend(images)
    return JSONResponse({"person_id": person_id, "images": matches, "count": len(matches)})