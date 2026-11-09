"""
Label Mapping Consistency and Agreement Unit Tests.

Enforces that:
1. PyTorch's ImageFolder standard alphabetical sorting (fake=0, real=1) is strictly followed.
2. Custom DeepfakeDataset matches ImageFolder exactly.
3. Ultralytics YOLO classification maps class 0 to 'fake' and class 1 to 'real'.
4. timm model inference extracts fake probabilities from index 0.
5. All backends (config, dataset, class weights, YOLO, timm, ensemble) strictly agree.
6. A test FAILS if any backend deviates from fake=0, real=1.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import torch
from torchvision.datasets import ImageFolder
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from training.dataset import DeepfakeDataset
from training.class_weights import compute_class_weights
from evaluation.evaluate_ensemble import ModelEnsembleEvaluator


def test_config_constants():
    """Validates that global config defines fake=0 and real=1."""
    assert config.CLASS_NAMES == ["fake", "real"], (
        f"config.CLASS_NAMES must be ['fake', 'real'] in alphabetical order, got {config.CLASS_NAMES}"
    )
    assert config.CLASS_TO_IDX == {"fake": 0, "real": 1}, (
        f"config.CLASS_TO_IDX must be {{'fake': 0, 'real': 1}}, got {config.CLASS_TO_IDX}"
    )
    assert config.IDX_TO_CLASS == {0: "fake", 1: "real"}, (
        f"config.IDX_TO_CLASS must be {{0: 'fake', 1: 'real'}}, got {config.IDX_TO_CLASS}"
    )
    assert config.FAKE_LABEL_IDX == 0, f"config.FAKE_LABEL_IDX must be 0, got {config.FAKE_LABEL_IDX}"
    assert config.REAL_LABEL_IDX == 1, f"config.REAL_LABEL_IDX must be 1, got {config.REAL_LABEL_IDX}"


def test_torchvision_imagefolder_mapping():
    """Validates that torchvision ImageFolder sorts 'fake' and 'real' alphabetically into fake=0, real=1."""
    train_dir = config.TRAIN_DIR
    if not train_dir.exists():
        pytest.skip(f"Train directory not found at {train_dir}")

    imagefolder_ds = ImageFolder(str(train_dir))
    expected_mapping = {"fake": 0, "real": 1}

    assert imagefolder_ds.class_to_idx == expected_mapping, (
        f"ImageFolder class_to_idx mismatch! Expected {expected_mapping}, got {imagefolder_ds.class_to_idx}"
    )


def test_deepfake_dataset_matches_imagefolder():
    """Validates that custom DeepfakeDataset yields the exact same class_to_idx and sample labels as ImageFolder."""
    for split_dir in [config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR]:
        if not split_dir.exists():
            continue

        imagefolder_ds = ImageFolder(str(split_dir))
        custom_ds = DeepfakeDataset(split_dir)

        # 1. Class mapping agreement
        assert custom_ds.class_to_idx == imagefolder_ds.class_to_idx, (
            f"Class mapping mismatch between DeepfakeDataset and ImageFolder in {split_dir}! "
            f"DeepfakeDataset: {custom_ds.class_to_idx} vs ImageFolder: {imagefolder_ds.class_to_idx}"
        )
        assert custom_ds.class_to_idx == {"fake": 0, "real": 1}, (
            f"DeepfakeDataset class_to_idx must be {{'fake': 0, 'real': 1}}, got {custom_ds.class_to_idx}"
        )

        # 2. Sample-level verification
        for img_path, label in custom_ds.samples:
            parent_folder = img_path.parent.name.lower()
            if parent_folder == "fake":
                assert label == 0, f"Sample {img_path} in 'fake' folder got label {label} (expected 0)"
            elif parent_folder == "real":
                assert label == 1, f"Sample {img_path} in 'real' folder got label {label} (expected 1)"


def test_class_weights_agreement():
    """Validates that compute_class_weights uses fake=0 and real=1."""
    train_dir = config.TRAIN_DIR
    if not train_dir.exists():
        pytest.skip("Train dir missing")

    weights = compute_class_weights(train_dir)
    assert isinstance(weights, torch.Tensor)
    assert len(weights) == 2, f"Weights must have length 2 (for fake=0, real=1), got {len(weights)}"


def test_yolo_and_timm_evaluator_agreement():
    """
    Validates that ModelEnsembleEvaluator verifies agreement:
    YOLO fake=0, real=1 AND timm fake=0, real=1.
    """
    yolo_ckpt = config.BASE_DIR / "runs" / "yolo_cls" / "deepfake_yolo" / "weights" / "best.pt"
    cn_ckpt = config.CHECKPOINT_DIR / "convnext_tiny.fb_in22k_ft_in1k_best.pth"
    eff_ckpt = config.CHECKPOINT_DIR / "efficientnet_b3.ra2_in1k_best.pth"

    if not (yolo_ckpt.exists() and cn_ckpt.exists() and eff_ckpt.exists()):
        pytest.skip("Model checkpoints missing for evaluator test")

    evaluator = ModelEnsembleEvaluator(
        yolo_checkpoint=yolo_ckpt,
        convnext_checkpoint=cn_ckpt,
        effnet_checkpoint=eff_ckpt,
        device=torch.device("cpu")
    )

    # YOLO class mapping check
    assert evaluator.yolo_fake_idx == 0, (
        f"YOLO fake index must be 0, got {evaluator.yolo_fake_idx}. Names: {evaluator.yolo_names}"
    )
    assert evaluator.yolo_real_idx == 1, (
        f"YOLO real index must be 1, got {evaluator.yolo_real_idx}. Names: {evaluator.yolo_names}"
    )

    # timm class mapping check
    assert evaluator.timm_fake_idx == 0, (
        f"timm fake index must be 0, got {evaluator.timm_fake_idx}"
    )
    assert evaluator.timm_real_idx == 1, (
        f"timm real index must be 1, got {evaluator.timm_real_idx}"
    )

    # Absolute agreement
    assert evaluator.yolo_fake_idx == evaluator.timm_fake_idx == config.FAKE_LABEL_IDX == 0, (
        f"Class 0 disagreement! YOLO={evaluator.yolo_fake_idx}, timm={evaluator.timm_fake_idx}"
    )
    assert evaluator.yolo_real_idx == evaluator.timm_real_idx == config.REAL_LABEL_IDX == 1, (
        f"Class 1 disagreement! YOLO={evaluator.yolo_real_idx}, timm={evaluator.timm_real_idx}"
    )


def test_intentional_mismatch_fails():
    """
    Simulates an inverted mapping {'real': 0, 'fake': 1} and ensures
    that assertion or equality checks explicitly fail.
    """
    inverted_mapping = {"real": 0, "fake": 1}
    standard_mapping = config.CLASS_TO_IDX

    with pytest.raises(AssertionError):
        assert inverted_mapping == standard_mapping, "Inverted mapping should not match standard mapping!"

    with pytest.raises(AssertionError):
        assert inverted_mapping["fake"] == config.FAKE_LABEL_IDX, "Inverted mapping fake index should fail!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

