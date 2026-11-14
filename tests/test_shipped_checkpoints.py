"""
Tests that the shipped (fallback) checkpoints are actually trained.

The canonical paths `models/checkpoints/<model>_best.pth` are what the README
tells users to download from the GitHub Release and where the evaluation
pipeline looks when no domain-specific checkpoint exists.

They originally held epoch-1 warm-up weights, which score at chance
(ROC-AUC 0.49 / 0.47 / 0.46 on the Celeb-DF v2 test split) while looking
like a normal trained model on disk. These tests fail if that regresses:
a checkpoint that does not beat chance is not shippable.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import config


def _celebdf_test_dir():
    d = config.PROCESSED_CELEBDF_DIR / "test"
    if not (d / "real").exists() or not (d / "fake").exists():
        pytest.skip("Celeb-DF test split not extracted")
    return d


def _auc_for_timm(model_name: str, ckpt: Path):
    import torch
    import timm
    from torch.utils.data import DataLoader
    from sklearn.metrics import roc_auc_score
    from training.dataset import DeepfakeDataset
    from training.train_timm import get_timm_transforms

    model = timm.create_model(model_name, pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"])
    model.eval()

    ds = DeepfakeDataset(_celebdf_test_dir(),
                          transform=get_timm_transforms(model, is_training=False))
    probs, labels = [], []
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=32):
            probs += torch.softmax(model(x), 1)[:, config.FAKE_LABEL_IDX].tolist()
            labels += y.tolist()
    # FAKE is the positive class.
    return roc_auc_score([1 - l for l in labels], probs)


class TestShippedCheckpointsAreTrained:
    @pytest.mark.parametrize("model_name", [
        "efficientnet_b3.ra2_in1k",
        "convnext_tiny.fb_in22k_ft_in1k",
    ])
    def test_fallback_checkpoint_beats_chance(self, model_name):
        ckpt = config.CHECKPOINT_DIR / f"{model_name}_best.pth"
        if not ckpt.exists():
            pytest.skip(f"{ckpt.name} not present")
        auc = _auc_for_timm(model_name, ckpt)
        assert auc > 0.60, (
            f"{ckpt.name} scores ROC-AUC {auc:.4f} on the Celeb-DF v2 test split - "
            "a checkpoint that does not clearly beat chance must not be shipped."
        )

    def test_fallback_matches_a_per_domain_checkpoint(self):
        """
        The fallback path must hold the same weights as one of the
        per-domain checkpoints, not a superseded warm-up copy.
        """
        for model_name in ("efficientnet_b3.ra2_in1k", "convnext_tiny.fb_in22k_ft_in1k"):
            fallback = config.CHECKPOINT_DIR / f"{model_name}_best.pth"
            per_domain = config.CHECKPOINT_DIR / "celebdf" / f"{model_name}_best.pth"
            if not (fallback.exists() and per_domain.exists()):
                pytest.skip(f"{model_name} checkpoints not present")
            import torch
            a = torch.load(fallback, map_location="cpu")["state_dict"]
            b = torch.load(per_domain, map_location="cpu")["state_dict"]
            same = a.keys() == b.keys() and all(
                torch.equal(a[k], b[k]) for k in a
            )
            assert same, (
                f"models/checkpoints/{model_name}_best.pth differs from the "
                "per-domain Celeb-DF checkpoint - the fallback copy is stale."
            )
