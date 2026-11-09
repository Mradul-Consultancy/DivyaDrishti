"""
Curate genuine generator-based and authentic portrait face images into data/real_life.

Sources:
  1. AI-face-detection-Dataset (Shahzaib Shazoo / HuggingFace/Kaggle):
     - AI: 500 genuine generative AI diffusion portrait faces
     - real: 500 authentic high-resolution portraits
  2. DeepFake Facial Dataset (FaceForensics++ / DeepFakes):
     - Fake: 100 face-swap manipulated crops
     - Real: 100 authentic face crops

Total: 600 Real, 600 Fake = 1,200 images (target: 1,000 - 2,000 in-the-wild images).
All resized to 224x224 with high quality JPEG encoding.
"""

import sys
import shutil
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
REAL_LIFE_DIR = ROOT / "data" / "real_life"
AI_GEN_DIR = ROOT / "data" / "downloads" / "ai_generated" / "AI-face-detection-Dataset"
FF_DIR = ROOT / "data" / "downloads" / "generator_dataset" / "Deep_Fakes"

def process_and_copy(img_path: Path, out_path: Path, target_size=(224, 224)):
    try:
        with Image.open(img_path) as img:
            rgb = img.convert("RGB")
            # If large portrait, take center-square crop before resizing
            w, h = rgb.size
            if w != h:
                dim = min(w, h)
                left = (w - dim) // 2
                top = (h - dim) // 2
                rgb = rgb.crop((left, top, left + dim, top + dim))
            resized = rgb.resize(target_size, Image.Resampling.BILINEAR)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            resized.save(out_path, format="JPEG", quality=92)
            return True
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False

def main():
    print("=" * 60)
    print("Curating 1,200 Genuine In-The-Wild Face Images into data/real_life")
    print("=" * 60)

    fake_out = REAL_LIFE_DIR / "fake"
    real_out = REAL_LIFE_DIR / "real"

    # Reset existing files in real_life to ensure clean generator-based dataset
    shutil.rmtree(fake_out, ignore_errors=True)
    shutil.rmtree(real_out, ignore_errors=True)
    fake_out.mkdir(parents=True, exist_ok=True)
    real_out.mkdir(parents=True, exist_ok=True)

    # 1. Ingest Fake - Generative AI Diffusion (500)
    ai_files = sorted(list((AI_GEN_DIR / "AI").glob("*.jpg")))[:500]
    count_fake = 0
    for i, p in enumerate(ai_files):
        dest = fake_out / f"gen_diffusion_{i:04d}.jpg"
        if process_and_copy(p, dest):
            count_fake += 1

    # 2. Ingest Fake - FaceForensics++ Face Swaps (100)
    ff_fake_files = sorted(list((FF_DIR / "Fake").glob("*.jpg")))[:100]
    for i, p in enumerate(ff_fake_files):
        dest = fake_out / f"ff_swap_{i:04d}.jpg"
        if process_and_copy(p, dest):
            count_fake += 1

    print(f"Curated {count_fake} fake images into {fake_out}")

    # 3. Ingest Real - Authentic High-Resolution Portraits (500)
    real_files = sorted(list((AI_GEN_DIR / "real").glob("*.jpg")))[:500]
    count_real = 0
    for i, p in enumerate(real_files):
        dest = real_out / f"auth_portrait_{i:04d}.jpg"
        if process_and_copy(p, dest):
            count_real += 1

    # 4. Ingest Real - Authentic Video Face Crops (100)
    ff_real_files = sorted(list((FF_DIR / "Real").glob("*.jpg")))[:100]
    for i, p in enumerate(ff_real_files):
        dest = real_out / f"auth_face_{i:04d}.jpg"
        if process_and_copy(p, dest):
            count_real += 1

    print(f"Curated {count_real} real images into {real_out}")
    print(f"Total in {REAL_LIFE_DIR}: {count_real + count_fake} images (Balanced: {count_real} Real / {count_fake} Fake)")
    print("=" * 60)

if __name__ == "__main__":
    main()

