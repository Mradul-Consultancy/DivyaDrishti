"""
Extract face crops from Celeb-DF zip files to populate:
  data/processed/celebdf/train/fake/   (was empty)
  data/processed/celebdf/val/real/ and val/fake/   (did not exist)

Usage:
  python scripts/extract_celebdf_faces.py
"""
import zipfile
import tempfile
import shutil
import os
import sys
import random
import logging
from pathlib import Path

import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS = ROOT / "data" / "downloads"
OUT_BASE = ROOT / "data" / "processed" / "celebdf"

# How many face crops to extract per video (spaced evenly)
FRAMES_PER_VIDEO = 25

# Val split fraction
VAL_FRAC = 0.2

MTCNN_AVAILABLE = False
try:
    from facenet_pytorch import MTCNN
    import torch
    MTCNN_AVAILABLE = True
except ImportError:
    pass


def extract_video_from_zip(zip_path: Path, tmp_dir: Path) -> Path | None:
    """Unzip a single mp4 and return its path."""
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        mp4s = [m for m in members if m.endswith(".mp4")]
        if not mp4s:
            log.warning(f"No MP4 found in {zip_path.name}")
            return None
        zf.extract(mp4s[0], tmp_dir)
        return tmp_dir / mp4s[0]


def sample_frames(video_path: Path, n: int) -> list:
    """Return n evenly-spaced BGR frames from a video."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = sorted(set([int(i * total / n) for i in range(n)]))
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def crop_faces_mtcnn(frames: list, detector) -> list:
    """Return face-cropped RGB images using MTCNN."""
    import torch
    from PIL import Image
    crops = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        try:
            boxes, _ = detector.detect(pil)
            if boxes is None:
                continue
            for box in boxes[:1]:  # take largest face
                x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                crop = rgb[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_resized = cv2.resize(crop, (224, 224))
                    crops.append(crop_resized)
        except Exception:
            continue
    return crops


def crop_faces_simple(frames: list) -> list:
    """Fallback: use OpenCV Haar cascade."""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    crops = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(80, 80))
        if len(faces) == 0:
            # No face detected — use centre crop as fallback
            h, w = frame.shape[:2]
            m = min(h, w)
            top, left = (h - m) // 2, (w - m) // 2
            crop = frame[top : top + m, left : left + m]
            crops.append(cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (224, 224)))
        else:
            x, y, w, h = faces[0]
            pad = int(0.2 * max(w, h))
            x1 = max(0, x - pad); y1 = max(0, y - pad)
            x2 = x + w + pad;     y2 = y + h + pad
            crop = frame[y1:y2, x1:x2]
            crops.append(cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (224, 224)))
    return crops


def save_crops(crops: list, out_dir: Path, prefix: str, start_idx: int = 0):
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, crop_rgb in enumerate(crops):
        dest = out_dir / f"{prefix}_{start_idx + i:04d}.jpg"
        bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(dest), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved.append(dest)
    return saved


def process_zips(zip_dir: Path, label: str, detector) -> list:
    """Process all zips in a directory; return list of (crop_rgb, path) results."""
    zips = sorted(zip_dir.glob("*.zip"))
    all_crops = []
    for zp in zips:
        log.info(f"  Processing {zp.name} ...")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            vid = extract_video_from_zip(zp, tmp_path)
            if vid is None:
                continue
            frames = sample_frames(vid, FRAMES_PER_VIDEO)
            log.info(f"    Extracted {len(frames)} frames from {vid.name}")
            if MTCNN_AVAILABLE and detector is not None:
                crops = crop_faces_mtcnn(frames, detector)
            else:
                crops = crop_faces_simple(frames)
            log.info(f"    Got {len(crops)} face crops")
            all_crops.extend(crops)
    return all_crops


def main():
    log.info("=" * 60)
    log.info("Celeb-DF face extraction — filling missing train/val splits")
    log.info("=" * 60)

    detector = None
    if MTCNN_AVAILABLE:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        detector = MTCNN(image_size=224, margin=20, keep_all=False, device=device)
        log.info(f"MTCNN detector ready on {device}")
    else:
        log.info("MTCNN not available — using OpenCV Haar cascade fallback")

    # --- FAKE videos (Celeb-synthesis) ---
    log.info("\n[1/2] Processing FAKE videos (Celeb-synthesis) ...")
    fake_crops = process_zips(DOWNLOADS / "Celeb-synthesis", "fake", detector)
    log.info(f"Total fake crops: {len(fake_crops)}")

    # --- REAL videos (Celeb-real) ---
    log.info("\n[2/2] Processing REAL videos (Celeb-real) ...")
    real_crops = process_zips(DOWNLOADS / "Celeb-real", "real", detector)
    log.info(f"Total real crops: {len(real_crops)}")

    # --- Split into train / val ---
    random.seed(42)
    random.shuffle(fake_crops)
    random.shuffle(real_crops)

    n_fake_val = max(1, int(len(fake_crops) * VAL_FRAC))
    n_real_val = max(1, int(len(real_crops) * VAL_FRAC))

    fake_val   = fake_crops[:n_fake_val]
    fake_train = fake_crops[n_fake_val:]
    real_val   = real_crops[:n_real_val]
    real_train = real_crops[n_real_val:]  # already populated, skip writing to avoid duplicates

    log.info("\n--- Saving ---")

    # Train fake (was empty)
    train_fake_dir = OUT_BASE / "train" / "fake"
    saved = save_crops(fake_train, train_fake_dir, "celebdf_fake")
    log.info(f"  train/fake: wrote {len(saved)} images → {train_fake_dir}")

    # Val fake (didn't exist)
    val_fake_dir = OUT_BASE / "val" / "fake"
    saved = save_crops(fake_val, val_fake_dir, "celebdf_fake_val")
    log.info(f"  val/fake:   wrote {len(saved)} images → {val_fake_dir}")

    # Val real (didn't exist) — supplement only if val/real is empty
    val_real_dir = OUT_BASE / "val" / "real"
    existing_val_real = list(val_real_dir.glob("*.jpg")) if val_real_dir.exists() else []
    if len(existing_val_real) == 0:
        saved = save_crops(real_val, val_real_dir, "celebdf_real_val")
        log.info(f"  val/real:   wrote {len(saved)} images → {val_real_dir}")
    else:
        log.info(f"  val/real:   already has {len(existing_val_real)} images — skipping")

    # Summary
    log.info("\n========== SUMMARY ==========")
    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            d = OUT_BASE / split / cls
            n = len(list(d.glob("*.jpg"))) if d.exists() else 0
            log.info(f"  celebdf/{split}/{cls}: {n} images")
    log.info("=============================")


if __name__ == "__main__":
    main()

