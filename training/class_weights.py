"""
Shared utilities for computing class weights and common training helpers
across all model backends (Ultralytics, timm, custom PyTorch).
"""

import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def compute_class_weights(
    data_dir: Path | str,
    class_to_idx: Dict[str, int] = None,
    strategy: str = "inverse_freq"
) -> torch.Tensor:
    """
    Scans the training directory and computes per-class weights to handle imbalance.

    Args:
        data_dir: Path to the training split (e.g., data/processed/train).
        class_to_idx: Mapping of class name to label index.
        strategy: 'inverse_freq' (1/count, normalised), 'effective' (uses beta=0.9999),
                  or 'sqrt_inverse' (sqrt of inverse frequency).

    Returns:
        torch.Tensor of shape (num_classes,) with weight per class index.
    """
    data_dir = Path(data_dir)
    class_to_idx = class_to_idx or config.CLASS_TO_IDX.copy()
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    counts = {}
    for class_name, idx in class_to_idx.items():
        class_folder = data_dir / class_name
        if class_folder.exists():
            n = sum(1 for p in class_folder.glob("*") if p.suffix.lower() in valid_exts)
        else:
            n = 0
        counts[idx] = max(n, 1)  # Avoid division by zero

    num_classes = len(class_to_idx)
    total = sum(counts.values())

    weights = torch.zeros(num_classes)

    if strategy == "inverse_freq":
        # w_i = total / (num_classes * count_i)
        for idx, count in counts.items():
            weights[idx] = total / (num_classes * count)

    elif strategy == "effective":
        # "Class-Balanced Loss Based on Effective Number of Samples" (Cui et al., 2019)
        beta = 0.9999
        for idx, count in counts.items():
            effective_num = 1.0 - beta ** count
            weights[idx] = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * num_classes

    elif strategy == "sqrt_inverse":
        for idx, count in counts.items():
            weights[idx] = np.sqrt(total / count)
        weights = weights / weights.sum() * num_classes

    else:
        raise ValueError(f"Unknown weighting strategy: '{strategy}'. Use 'inverse_freq', 'effective', or 'sqrt_inverse'.")

    return weights


def get_class_distribution(data_dir: Path | str, class_to_idx: Dict[str, int] = None) -> Dict[str, int]:
    """Returns per-class sample counts for a directory."""
    data_dir = Path(data_dir)
    class_to_idx = class_to_idx or config.CLASS_TO_IDX.copy()
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    dist = {}
    for class_name in class_to_idx:
        folder = data_dir / class_name
        if folder.exists():
            dist[class_name] = sum(1 for p in folder.glob("*") if p.suffix.lower() in valid_exts)
        else:
            dist[class_name] = 0
    return dist


def log_class_info(logger, train_dir: Path, class_to_idx: Dict[str, int] = None):
    """Logs the class distribution and computed weights."""
    dist = get_class_distribution(train_dir, class_to_idx)
    weights = compute_class_weights(train_dir, class_to_idx)
    logger.info(f"Class distribution: {dist}")
    logger.info(f"Computed class weights: {weights.tolist()}")
    total = sum(dist.values())
    if total > 0:
        ratio = max(dist.values()) / max(min(dist.values()), 1)
        logger.info(f"Imbalance ratio: {ratio:.2f}x")
    return weights

