"""
Independently re-measures the SHIPPED fallback checkpoints and records the
result so the test suite can verify reproduction, not merely "beats chance".

Why this exists
---------------
The GitHub Release v1.0.0 ships ``models/checkpoints/<model>_best.pth``. The
regression tests used to assert only ``AUC > 0.60`` on those weights, which
proves a checkpoint is trained but does NOT prove it reproduces the numbers
the README publishes. An auditor must be able to tie a released weight file to
a measured score. This script produces that tie:

  reports/shipped_checkpoint_probe.json

It scores the three fallback checkpoints on the canonical Celeb-DF v2 test
split (``data/processed/celebdf/test``) using the exact evaluation transform
each backbone was trained with:

  * timm backbones (EfficientNet-B3, ConvNeXt-Tiny) -> timm native eval
    transform (``resolve_data_config`` + ``create_transform``), the same path
    the cross-dataset table uses via ``get_eval_transform``;
  * the hand-written ResNet-18 -> torchvision val transform.

The JSON artefact is committed. ``tests/test_shipped_checkpoints.py`` then
re-runs the identical scoring and fails if the measured AUC drifts from the
recorded one (tolerance 0.01 to absorb hardware nondeterminism). Regenerate
with ``python scripts/probe_shipped_checkpoints.py`` after intentionally
replacing weights; the tests will pick up the new record.

Run:  python scripts/probe_shipped_checkpoints.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import config

PROBE_PATH = ROOT / "reports" / "shipped_checkpoint_probe.json"
TOLERANCE = 0.01


def _celebdf_test_dir() -> Path:
    d = config.PROCESSED_CELEBDF_DIR / "test"
    if not (d / "real").exists() or not (d / "fake").exists():
        raise FileNotFoundError(
            f"Celeb-DF test split not extracted at {d} - run the preprocessing "
            "pipeline first; the probe needs the canonical evaluation set."
        )
    return d


def _score_timm(model_name: str, ckpt: Path, eval_path: Path):
    import torch
    import timm
    from torch.utils.data import DataLoader
    from sklearn.metrics import roc_auc_score
    from training.dataset import DeepfakeDataset
    from training.train_timm import get_timm_transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(model_name, pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True)["state_dict"])
    model.to(device)
    model.eval()
    transform = get_timm_transforms(model, is_training=False)
    ds = DeepfakeDataset(eval_path, transform=transform)
    probs, labels = [], []
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=64 if device.type == "cuda" else 32):
            probs += torch.softmax(model(x.to(device)), 1)[:, config.FAKE_LABEL_IDX].tolist()
            labels += y.tolist()
    return {
        "roc_auc": round(roc_auc_score([1 - l for l in labels], probs), 4),
        "n_images": len(labels),
        "transform": "timm-native-eval (get_timm_transforms, is_training=False)",
    }


def _score_resnet18(ckpt: Path, eval_path: Path):
    import torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import roc_auc_score
    from models.detector import get_model
    from preprocessing.transforms import get_val_transforms
    from training.dataset import DeepfakeDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model("resnet18", num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True)["state_dict"])
    model.to(device)
    model.eval()
    ds = DeepfakeDataset(eval_path, transform=get_val_transforms())
    probs, labels = [], []
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=64 if device.type == "cuda" else 32):
            probs += torch.softmax(model(x.to(device)), 1)[:, config.FAKE_LABEL_IDX].tolist()
            labels += y.tolist()
    return {
        "roc_auc": round(roc_auc_score([1 - l for l in labels], probs), 4),
        "n_images": len(labels),
        "transform": "torchvision val (get_val_transforms)",
    }


def build_probe() -> dict:
    import torch

    eval_path = _celebdf_test_dir()
    entries = {}
    for model_name in ("efficientnet_b3.ra2_in1k", "convnext_tiny.fb_in22k_ft_in1k"):
        ckpt = config.CHECKPOINT_DIR / f"{model_name}_best.pth"
        if not ckpt.exists():
            entries[model_name] = {"status": "missing", "checkpoint": str(ckpt)}
            continue
        entries[model_name] = {
            "status": "measured",
            "checkpoint": str(ckpt.relative_to(ROOT)),
            **_score_timm(model_name, ckpt, eval_path),
        }
    ckpt = config.CHECKPOINT_DIR / "resnet18_best.pth"
    if ckpt.exists():
        entries["resnet18"] = {
            "status": "measured",
            "checkpoint": str(ckpt.relative_to(ROOT)),
            **_score_resnet18(ckpt, eval_path),
        }
    else:
        entries["resnet18"] = {"status": "missing", "checkpoint": str(ckpt)}

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_split": str(eval_path.relative_to(ROOT)),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "auc_tolerance": TOLERANCE,
        "models": entries,
    }


def main() -> None:
    probe = build_probe()
    PROBE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROBE_PATH.write_text(json.dumps(probe, indent=2), encoding="utf-8")
    print(f"wrote {PROBE_PATH}")
    for name, entry in probe["models"].items():
        if entry["status"] == "measured":
            print(f"  {name}: ROC-AUC {entry['roc_auc']:.4f} on {entry['n_images']} images")
        else:
            print(f"  {name}: {entry['status']} ({entry['checkpoint']})")


if __name__ == "__main__":
    main()
