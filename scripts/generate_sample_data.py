"""
Generates synthetic sample portrait images with face-like patterns and boundary artifacts.
This enables immediate testing of the entire data -> preprocessing -> training -> evaluation pipeline
without needing to download gigabytes of raw video/image datasets first.
"""

import sys
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def create_synthetic_portrait(is_fake: bool, size=(300, 300)) -> Image.Image:
    """
    Synthesizes a simple stylized facial portrait.
    If is_fake is True, injects synthetic boundary artifacts, color blending seams,
    or high-frequency noise typical of deepfake generation artifacts.
    """
    img = Image.new("RGB", size, (random.randint(180, 240), random.randint(180, 240), random.randint(200, 255)))
    draw = ImageDraw.Draw(img)

    cx, cy = size[0] // 2, size[1] // 2

    # Draw head/face oval
    face_w, face_h = int(size[0] * 0.35), int(size[1] * 0.45)
    skin_color = (random.randint(210, 255), random.randint(170, 210), random.randint(140, 180))
    draw.ellipse([cx - face_w, cy - face_h, cx + face_w, cy + face_h], fill=skin_color)

    # Eyes
    eye_offset_x, eye_y = int(face_w * 0.45), cy - int(face_h * 0.15)
    eye_r = int(face_w * 0.12)
    draw.ellipse([cx - eye_offset_x - eye_r, eye_y - eye_r, cx - eye_offset_x + eye_r, eye_y + eye_r], fill=(30, 30, 30))
    draw.ellipse([cx + eye_offset_x - eye_r, eye_y - eye_r, cx + eye_offset_x + eye_r, eye_y + eye_r], fill=(30, 30, 30))

    # Mouth
    mouth_y = cy + int(face_h * 0.45)
    draw.arc([cx - int(face_w * 0.35), mouth_y - 10, cx + int(face_w * 0.35), mouth_y + 15], start=0, end=180, fill=(180, 40, 50), width=3)

    if is_fake:
        # Inject deepfake-like artifacts:
        # 1. Blurring or warping around facial perimeter
        # 2. Inconsistent color boundary or frequency noise
        artifact_type = random.choice(["boundary_seam", "frequency_noise", "local_blur"])
        
        if artifact_type == "boundary_seam":
            # Distinct circular seam around face
            seam_box = [cx - int(face_w * 0.9), cy - int(face_h * 0.9), cx + int(face_w * 0.9), cy + int(face_h * 0.9)]
            draw.arc(seam_box, start=0, end=360, fill=(random.randint(50, 150), random.randint(50, 150), random.randint(50, 150)), width=2)
            
        elif artifact_type == "frequency_noise":
            # High-frequency checkerboard / noise overlay in face region
            np_img = np.array(img).astype(np.int16)
            noise = np.random.normal(0, 25, np_img.shape).astype(np.int16)
            mask = np.zeros((size[1], size[0], 1), dtype=bool)
            mask[cy - face_h:cy + face_h, cx - face_w:cx + face_w] = True
            np_img = np.where(mask, np.clip(np_img + noise, 0, 255), np_img).astype(np.uint8)
            img = Image.fromarray(np_img)

        elif artifact_type == "local_blur":
            # Excessive smoothing inside face region
            crop_box = (cx - face_w // 2, cy - face_h // 2, cx + face_w // 2, cy + face_h // 2)
            face_sub = img.crop(crop_box).filter(ImageFilter.GaussianBlur(radius=4))
            img.paste(face_sub, crop_box)

    return img


def generate_samples(num_real: int = 30, num_fake: int = 30, target_dir: Path = config.RAW_DATA_DIR):
    """Generates synthetic sample images for real and fake classes."""
    target_dir = Path(target_dir)
    real_dir = target_dir / "real"
    fake_dir = target_dir / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_real} authentic sample images in {real_dir}...")
    for i in range(1, num_real + 1):
        img = create_synthetic_portrait(is_fake=False)
        img.save(real_dir / f"real_{i:03d}.jpg", quality=95)

    print(f"Generating {num_fake} synthetic deepfake sample images in {fake_dir}...")
    for i in range(1, num_fake + 1):
        img = create_synthetic_portrait(is_fake=True)
        img.save(fake_dir / f"fake_{i:03d}.jpg", quality=90)

    print("\nSample dataset generation completed!")
    print(f"Raw Real: {len(list(real_dir.glob('*.jpg')))} images")
    print(f"Raw Fake: {len(list(fake_dir.glob('*.jpg')))} images")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic sample dataset for testing")
    parser.add_argument("--real", type=int, default=30, help="Number of real sample images")
    parser.add_argument("--fake", type=int, default=30, help="Number of fake sample images")
    parser.add_argument("--dest", type=str, default=str(config.RAW_DATA_DIR), help="Destination directory")
    args = parser.parse_args()

    generate_samples(num_real=args.real, num_fake=args.fake, target_dir=Path(args.dest))

