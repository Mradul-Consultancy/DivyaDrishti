"""
DivyaDrishti AI — Automated Checkpoint Downloader.
Downloads pre-trained model weights from GitHub Releases v1.0.0.
"""

import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
CELEBDF_DIR = CHECKPOINT_DIR / "celebdf"

CHECKPOINTS = {
    "convnext_tiny.fb_in22k_ft_in1k_best.pth": "https://github.com/Mradul-Consultancy/DivyaDrishti/releases/download/v1.0.0/convnext_tiny.fb_in22k_ft_in1k_best.pth",
    "efficientnet_b3.ra2_in1k_best.pth": "https://github.com/Mradul-Consultancy/DivyaDrishti/releases/download/v1.0.0/efficientnet_b3.ra2_in1k_best.pth",
    "resnet18_best.pth": "https://github.com/Mradul-Consultancy/DivyaDrishti/releases/download/v1.0.0/resnet18_best.pth",
}

def download_file(url: str, dest_path: Path):
    if dest_path.exists():
        print(f"  [OK] Already exists: {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.1f} MB)")
        return

    print(f"  [-->] Downloading {dest_path.name} from {url}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    def _progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        mb = (count * block_size) / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\r    Progress: {percent:3d}% ({mb:.1f} / {total_mb:.1f} MB)")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=_progress)
        print(f"\n  [SUCCESS] Saved to {dest_path}")
    except Exception as e:
        print(f"\n  [FAILED] Could not download {dest_path.name}: {e}")
        if dest_path.exists():
            dest_path.unlink()

def main():
    print("=" * 65)
    print(" DivyaDrishti AI — Model Checkpoints Downloader")
    print(f" Target Directory: {CHECKPOINT_DIR}")
    print("=" * 65)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    CELEBDF_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in CHECKPOINTS.items():
        # Save to fallback path
        target = CHECKPOINT_DIR / filename
        download_file(url, target)

        # For CelebDF-specific models, also ensure domain copy exists
        celebdf_target = CELEBDF_DIR / filename
        if not celebdf_target.exists() and target.exists():
            import shutil
            shutil.copy2(target, celebdf_target)
            print(f"  [COPY] Synchronized to {celebdf_target}")

    print("\nCheckpoints verification complete.")

if __name__ == "__main__":
    main()
