"""
Generates short sample MP4 videos and metadata for testing the video extraction,
MTCNN face cropping, and identity-aware dataset splitting pipeline.
"""

import sys
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.generate_sample_data import create_synthetic_portrait


def generate_test_video(output_path: Path, is_fake: bool, num_frames: int = 30, fps: int = 24, size=(300, 300)):
    """Generates an MP4 video of facial portraits."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, size)

    for f_idx in range(num_frames):
        pil_frame = create_synthetic_portrait(is_fake=is_fake, size=size)
        rgb_frame = np.array(pil_frame)
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)

    out.release()
    print(f"Created sample video: {output_path} ({num_frames} frames)")


def generate_sample_videos(target_dir: Path = Path("data/raw_videos"), num_identities: int = 6):
    target_dir = Path(target_dir)
    real_dir = target_dir / "real"
    fake_dir = target_dir / "fake"
    ensure_dirs = [real_dir, fake_dir]
    for d in ensure_dirs:
        d.mkdir(parents=True, exist_ok=True)

    metadata = {}

    print(f"Generating {num_identities} synthetic identities with real and derivative fake videos...")
    for i in range(1, num_identities + 1):
        real_name = f"video_real_{i:02d}.mp4"
        fake_name = f"video_fake_{i:02d}.mp4"
        actor_id = f"actor_{i:02d}"

        real_path = real_dir / real_name
        fake_path = fake_dir / fake_name

        generate_test_video(real_path, is_fake=False, num_frames=30)
        generate_test_video(fake_path, is_fake=True, num_frames=30)

        metadata[real_name] = {
            "label": "real",
            "actor_id": actor_id,
            "original": None
        }
        metadata[fake_name] = {
            "label": "fake",
            "actor_id": actor_id,
            "original": real_name
        }

    # Write DFDC-style metadata.json
    metadata_path = target_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata file saved to: {metadata_path}")

    # Also create sample benchmark splits for testing benchmark parsers
    # 1. Celeb-DF style List_of_testing_videos.txt (actor_06 as test)
    test_list_path = target_dir / "List_of_testing_videos.txt"
    with open(test_list_path, "w", encoding="utf-8") as f:
        f.write(f"1 Celeb-synthesis/video_fake_{num_identities:02d}.mp4\n")
        f.write(f"0 Celeb-real/video_real_{num_identities:02d}.mp4\n")
    print(f"Celeb-DF test list saved to: {test_list_path}")

    # 2. FF++ style official JSON splits
    ffpp_dir = target_dir / "ffpp_splits"
    ffpp_dir.mkdir(parents=True, exist_ok=True)
    with open(ffpp_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump([["01", "02"], ["03", "04"]], f)
    with open(ffpp_dir / "val.json", "w", encoding="utf-8") as f:
        json.dump([["05"]], f)
    with open(ffpp_dir / "test.json", "w", encoding="utf-8") as f:
        json.dump([["06"]], f)
    print(f"FaceForensics++ sample JSON splits saved to: {ffpp_dir}")

    print("\nSample videos and benchmark metadata generated successfully!")


if __name__ == "__main__":
    generate_sample_videos()
