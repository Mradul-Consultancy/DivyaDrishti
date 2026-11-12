"""
Fine-tune timm models (ConvNeXt-Tiny, EfficientNet-B3, or any timm backbone)
for real-vs-fake deepfake detection with class-weighted loss.

Features:
- Loads any timm pretrained model and replaces its classification head.
- Computes class weights from the training directory to handle imbalance.
- Mixed-precision training (AMP) on CUDA.
- ReduceLROnPlateau scheduler, early stopping, best-model checkpointing.
- Generates training curves and JSON history to reports/.
- Optional RandomJPEGCompression augmentation inserted into the timm training
  pipeline before the ToTensor step (see --jpeg-aug / --jpeg-q-min / --jpeg-q-max).
"""

import sys
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
from timm.data import resolve_data_config, create_transform

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from utils.helpers import setup_logger, seed_everything, ensure_dirs
from training.dataset import DeepfakeDataset, create_dataloaders
from training.class_weights import compute_class_weights, log_class_info
from preprocessing.transforms import RandomJPEGCompression

logger = setup_logger("TrainTimm")

# ImageNet weights vendored for offline environments (no Hub access).
LOCAL_PRETRAINED_DIR = config.BASE_DIR / "pretrained"


def build_timm_model(model_name: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Creates a timm model with a fresh classification head for binary deepfake detection.

    When ``pretrained`` is True the ImageNet weights are fetched from the
    timm/HuggingFace Hub. In an offline environment (e.g. the Kaggle
    sandbox, which has no DNS) that download fails, so a locally vendored
    copy under ``pretrained/`` is used instead when present.

    If neither source is available the model is built without pretrained
    weights and a warning is logged - training still runs, just from a
    random initialisation.
    """
    if not pretrained:
        return timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    try:
        return timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    except Exception as exc:
        local = LOCAL_PRETRAINED_DIR / f"{model_name}.safetensors"
        if local.exists():
            logger.warning(
                f"Could not fetch '{model_name}' from the Hub ({exc}); "
                f"using the vendored weights at {local}."
            )
            return timm.create_model(
                model_name,
                pretrained=True,
                num_classes=num_classes,
                pretrained_cfg_overlay=dict(file=str(local)),
            )
        logger.warning(
            f"Could not fetch '{model_name}' from the Hub ({exc}) and no vendored "
            f"weights were found at {local}. Training from random initialisation."
        )
        return timm.create_model(model_name, pretrained=False, num_classes=num_classes)


def get_timm_transforms(model: nn.Module, is_training: bool = True,
                        jpeg_aug: bool = True,
                        jpeg_q_min: int = 50,
                        jpeg_q_max: int = 95,
                        jpeg_p: float = 0.5):
    """
    Resolves the correct transforms for a timm model (respecting its native
    input size, interpolation, mean/std).

    For training, optionally inserts ``RandomJPEGCompression`` immediately
    before the ``MaybeToTensor`` / ``ToTensor`` step so the model is exposed
    to JPEG artefacts while still operating on PIL Images.  This mirrors the
    JPEG degradation measured at test time by ``robustness_test.py`` and
    improves model robustness to social-media re-encoding.

    Args:
        model:       timm model (used to resolve its pretrained data config).
        is_training: If False, returns a deterministic val/test transform.
        jpeg_aug:    Insert RandomJPEGCompression in the train pipeline.
        jpeg_q_min:  Lower JPEG quality bound (default 50).
        jpeg_q_max:  Upper JPEG quality bound (default 95).
        jpeg_p:      Per-sample probability of applying JPEG aug (default 0.5).
    """
    data_config = resolve_data_config(model.pretrained_cfg)

    if is_training:
        transform = create_transform(
            **data_config,
            is_training=True,
            auto_augment="rand-m9-mstd0.5",
            re_prob=0.25,  # Random erasing
        )

        if jpeg_aug:
            # timm's train pipeline ends with:
            #   [..., RandAugment, MaybeToTensor, Normalize]
            # We insert RandomJPEGCompression before MaybeToTensor so it
            # operates on PIL Images (JPEG encoding requires a PIL Image, not
            # a tensor).  We detect the ToTensor step by class name to remain
            # robust to timm version differences.
            from torchvision.transforms import Compose
            from preprocessing.transforms import RandomJPEGCompression

            jpeg_transform = RandomJPEGCompression(
                q_min=jpeg_q_min, q_max=jpeg_q_max, p=jpeg_p
            )

            ops = list(transform.transforms)
            # Find the first ToTensor / MaybeToTensor step
            insert_idx = next(
                (i for i, t in enumerate(ops)
                 if "totensor" in type(t).__name__.lower()),
                len(ops) - 1,  # fallback: insert before last op (Normalize)
            )
            ops.insert(insert_idx, jpeg_transform)
            transform = Compose(ops)

    else:
        transform = create_transform(**data_config, is_training=False)

    return transform


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct / total * 100:.1f}%"})

    return running_loss / max(total, 1), correct / max(total, 1) * 100


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({"val_loss": f"{loss.item():.4f}", "val_acc": f"{correct / total * 100:.1f}%"})

    return running_loss / max(total, 1), correct / max(total, 1) * 100


def plot_curves(history: dict, save_path: Path):
    epochs_range = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_range, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    ax1.plot(epochs_range, history["val_loss"], "r--s", label="Val Loss", markersize=4)
    ax1.set_title("Loss", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.plot(epochs_range, history["train_acc"], "b-o", label="Train Acc", markersize=4)
    ax2.plot(epochs_range, history["val_acc"], "g--s", label="Val Acc", markersize=4)
    ax2.set_title("Accuracy (%)", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle(f"Training Curves", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def run_timm_training(args):
    seed_everything(args.seed)
    ensure_dirs(args.checkpoint_dir, args.reports_dir)

    log_path = Path(args.reports_dir) / f"{args.model_name}_training.log"
    logger = setup_logger(f"TimmTrainer_{args.model_name}", log_file=log_path)

    logger.info("=" * 65)
    logger.info(f"timm Fine-Tuning: {args.model_name}")
    logger.info("=" * 65)

    device = torch.device(args.device)

    # --- Build model ---
    logger.info(f"Loading timm model: {args.model_name} (pretrained={args.pretrained})")
    model = build_timm_model(args.model_name, num_classes=config.NUM_CLASSES, pretrained=args.pretrained)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # --- Transforms ---
    train_transform = get_timm_transforms(model, is_training=True,
                                          jpeg_aug=getattr(args, "jpeg_aug", True),
                                          jpeg_q_min=getattr(args, "jpeg_q_min", 50),
                                          jpeg_q_max=getattr(args, "jpeg_q_max", 95),
                                          jpeg_p=getattr(args, "jpeg_p", 0.5))
    val_transform = get_timm_transforms(model, is_training=False)

    # --- DataLoaders ---
    from torch.utils.data import DataLoader

    data_dir = Path(args.data_dir)
    train_dataset = DeepfakeDataset(data_dir / "train", transform=train_transform)
    val_dataset = DeepfakeDataset(data_dir / "val", transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda"
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda"
    )

    logger.info(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")
    if len(train_dataset) == 0:
        logger.error("No training samples found! Populate the dataset first.")
        return

    # --- Class Weights ---
    train_dir = data_dir / "train"
    class_weights = log_class_info(logger, train_dir)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # --- Optimizer & Scheduler ---
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    # --- Training Loop ---
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = 0.0
    patience_counter = 0
    best_ckpt = Path(args.checkpoint_dir) / f"{args.model_name}_best.pth"
    latest_ckpt = Path(args.checkpoint_dir) / f"{args.model_name}_latest.pth"

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n--- Epoch {epoch}/{args.epochs} ---")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        logger.info(
            f"Epoch {epoch:02d}: "
            f"Train Loss={train_loss:.4f} Acc={train_acc:.2f}% | "
            f"Val Loss={val_loss:.4f} Acc={val_acc:.2f}% | "
            f"LR={current_lr:.6f}"
        )

        # Save latest
        ckpt_data = {
            "epoch": epoch, "model_name": args.model_name,
            "state_dict": model.state_dict(),
            "val_acc": val_acc, "val_loss": val_loss,
            "optimizer": optimizer.state_dict(),
            "class_weights": class_weights.tolist()
        }
        torch.save(ckpt_data, latest_ckpt)

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(ckpt_data, best_ckpt)
            logger.info(f"[*] New best: {best_ckpt} (Val Acc: {best_val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"[!] Early stopping after {epoch} epochs.")
                break

    # --- Save artifacts ---
    model_tag = args.model_name.replace("/", "_")
    curves_path = Path(args.reports_dir) / f"{model_tag}_curves.png"
    plot_curves(history, curves_path)
    logger.info(f"Curves saved: {curves_path}")

    history_path = Path(args.reports_dir) / f"{model_tag}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    logger.info(f"History saved: {history_path}")

    logger.info("=" * 65)
    logger.info(f"Training complete! Best Val Accuracy: {best_val_acc:.2f}%")
    logger.info("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune a timm model for deepfake detection with class weights"
    )
    parser.add_argument("--model-name", type=str, required=True,
                        help="timm model name (e.g. convnext_tiny.fb_in22k_ft_in1k, efficientnet_b3.ra2_in1k)")
    parser.add_argument("--data-dir", type=str, default=str(config.PROCESSED_DATA_DIR),
                        help="Dataset root with train/ and val/ subfolders")
    parser.add_argument("--checkpoint-dir", type=str, default=str(config.CHECKPOINT_DIR))
    parser.add_argument("--reports-dir", type=str, default=str(config.REPORTS_DIR))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=str(config.DEVICE))
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")

    # JPEG augmentation
    parser.add_argument("--no-jpeg-aug", dest="jpeg_aug", action="store_false",
                        help="Disable RandomJPEGCompression augmentation during training")
    parser.set_defaults(jpeg_aug=True)
    parser.add_argument("--jpeg-q-min", type=int, default=50,
                        help="Minimum JPEG quality for augmentation (default: 50)")
    parser.add_argument("--jpeg-q-max", type=int, default=95,
                        help="Maximum JPEG quality for augmentation (default: 95)")
    parser.add_argument("--jpeg-p", type=float, default=0.5,
                        help="Per-sample probability of JPEG augmentation (default: 0.5)")

    args = parser.parse_args()
    run_timm_training(args)

