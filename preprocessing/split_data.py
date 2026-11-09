"""
Dataset splitter: partitions raw/face-cropped datasets into train, val, and test splits.
Maintains class balance across real and fake folders.
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from typing import Tuple
import config


def split_dataset(
    source_dir: Path | str = config.RAW_DATA_DIR,
    target_dir: Path | str = config.PROCESSED_DATA_DIR,
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    copy_files: bool = True
):
    """
    Splits images from source_dir into target_dir/train, target_dir/val, and target_dir/test.
    Expected structure in source_dir:
        source_dir/
            real/
                image1.jpg, ...
            fake/
                image2.jpg, ...
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    random.seed(seed)

    train_ratio, val_ratio, test_ratio = split_ratios
    assert abs(sum(split_ratios) - 1.0) < 1e-4, "Split ratios must sum to 1.0"

    splits = ["train", "val", "test"]
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    for class_name in config.CLASS_NAMES:
        class_src = source_dir / class_name
        if not class_src.exists():
            print(f"[Warning] Class directory '{class_src}' does not exist. Skipping.")
            continue

        images = [p for p in class_src.glob("*") if p.suffix.lower() in valid_exts]
        random.shuffle(images)

        total = len(images)
        if total == 0:
            print(f"[Warning] No images found in '{class_src}'.")
            continue

        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        partition = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }

        print(f"\nProcessing class '{class_name}' ({total} images):")
        for split_name in splits:
            split_files = partition[split_name]
            split_dest = target_dir / split_name / class_name
            split_dest.mkdir(parents=True, exist_ok=True)

            print(f"  -> {split_name.capitalize()}: {len(split_files)} images -> {split_dest}")
            for img_path in split_files:
                dest_file = split_dest / img_path.name
                if copy_files:
                    shutil.copy2(img_path, dest_file)
                else:
                    shutil.move(img_path, dest_file)

    print("\nDataset split completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train, val, and test subsets")
    parser.add_argument("--source", type=str, default=str(config.RAW_DATA_DIR), help="Source directory containing 'real' and 'fake' folders")
    parser.add_argument("--dest", type=str, default=str(config.PROCESSED_DATA_DIR), help="Destination directory for splits")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train ratio (default: 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Val ratio (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")

    args = parser.parse_args()
    split_dataset(
        source_dir=args.source,
        target_dir=args.dest,
        split_ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        seed=args.seed,
        copy_files=not args.move
    )

