"""
Convenience wrappers to fine-tune ConvNeXt-Tiny and EfficientNet-B3
from timm for deepfake detection.

These are thin entry points that call train_timm.py with
pre-configured model names and sensible defaults.

JPEG Augmentation
-----------------
Both wrappers expose ``--no-jpeg-aug``, ``--jpeg-q-min``, ``--jpeg-q-max``,
and ``--jpeg-p`` flags that are forwarded to ``run_timm_training``.  JPEG
augmentation is **on by default** (quality range 50–95, p=0.5) to improve
robustness to social-media re-encoding artefacts at inference time.
"""

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from training.train_timm import run_timm_training


# ── ConvNeXt-Tiny ──────────────────────────────────────────────────

def train_convnext_tiny():
    """Fine-tune ConvNeXt-Tiny (ImageNet-22k → 1k) for real-vs-fake."""
    parser = argparse.ArgumentParser(description="Fine-tune ConvNeXt-Tiny for deepfake detection")
    parser.add_argument("--data-dir", type=str, default=str(config.PROCESSED_DATA_DIR))
    parser.add_argument("--checkpoint-dir", type=str, default=str(config.CHECKPOINT_DIR))
    parser.add_argument("--reports-dir", type=str, default=str(config.REPORTS_DIR))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="ConvNeXt benefits from a lower LR during fine-tuning (default: 5e-5)")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=str(config.DEVICE))
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    # JPEG augmentation
    parser.add_argument("--no-jpeg-aug", dest="jpeg_aug", action="store_false",
                        help="Disable RandomJPEGCompression augmentation")
    parser.set_defaults(jpeg_aug=True)
    parser.add_argument("--jpeg-q-min", type=int, default=50,
                        help="Minimum JPEG quality for augmentation (default: 50)")
    parser.add_argument("--jpeg-q-max", type=int, default=95,
                        help="Maximum JPEG quality for augmentation (default: 95)")
    parser.add_argument("--jpeg-p", type=float, default=0.5,
                        help="Per-sample JPEG augmentation probability (default: 0.5)")

    args = parser.parse_args()
    args.model_name = "convnext_tiny.fb_in22k_ft_in1k"
    run_timm_training(args)


# ── EfficientNet-B3 ───────────────────────────────────────────────

def train_efficientnet_b3():
    """Fine-tune EfficientNet-B3 (RA2 recipe, ImageNet-1k) for real-vs-fake."""
    parser = argparse.ArgumentParser(description="Fine-tune EfficientNet-B3 for deepfake detection")
    parser.add_argument("--data-dir", type=str, default=str(config.PROCESSED_DATA_DIR))
    parser.add_argument("--checkpoint-dir", type=str, default=str(config.CHECKPOINT_DIR))
    parser.add_argument("--reports-dir", type=str, default=str(config.REPORTS_DIR))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=24,
                        help="EfficientNet-B3 uses 300×300 input — lower batch size to fit in VRAM (default: 24)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=str(config.DEVICE))
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    # JPEG augmentation
    parser.add_argument("--no-jpeg-aug", dest="jpeg_aug", action="store_false",
                        help="Disable RandomJPEGCompression augmentation")
    parser.set_defaults(jpeg_aug=True)
    parser.add_argument("--jpeg-q-min", type=int, default=50,
                        help="Minimum JPEG quality for augmentation (default: 50)")
    parser.add_argument("--jpeg-q-max", type=int, default=95,
                        help="Maximum JPEG quality for augmentation (default: 95)")
    parser.add_argument("--jpeg-p", type=float, default=0.5,
                        help="Per-sample JPEG augmentation probability (default: 0.5)")

    args = parser.parse_args()
    args.model_name = "efficientnet_b3.ra2_in1k"
    run_timm_training(args)


if __name__ == "__main__":
    import sys as _sys

    USAGE = """
Usage:
    python training/train_convnext_effnet.py convnext   [OPTIONS]
    python training/train_convnext_effnet.py effnet     [OPTIONS]

Examples:
    python training/train_convnext_effnet.py convnext --epochs 20 --batch-size 32 --lr 5e-5
    python training/train_convnext_effnet.py effnet   --epochs 20 --batch-size 24 --lr 1e-4
"""

    if len(_sys.argv) < 2 or _sys.argv[1] in ("-h", "--help"):
        print(USAGE)
        _sys.exit(0)

    subcommand = _sys.argv.pop(1).lower()

    if subcommand in ("convnext", "convnext_tiny"):
        train_convnext_tiny()
    elif subcommand in ("effnet", "efficientnet", "efficientnet_b3", "effnet_b3"):
        train_efficientnet_b3()
    else:
        print(f"Unknown subcommand: '{subcommand}'")
        print(USAGE)
        _sys.exit(1)

