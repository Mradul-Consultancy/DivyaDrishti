"""
Training pipeline for Deepfake Detection.
Supports multiple backbones, early stopping, LR scheduling, checkpointing, and metrics reporting.
"""

import sys
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Ensure project root is in python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from utils.helpers import setup_logger, seed_everything, ensure_dirs
from models.detector import get_model
from preprocessing.transforms import get_train_transforms, get_val_transforms
from training.dataset import create_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Mixed precision training
        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct/total)*100:.1f}%"})

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = (correct / max(total, 1)) * 100
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

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

            pbar.set_postfix({"val_loss": f"{loss.item():.4f}", "val_acc": f"{(correct/total)*100:.1f}%"})

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = (correct / max(total, 1)) * 100
    return epoch_loss, epoch_acc


def plot_learning_curves(history: dict, save_path: Path):
    """Saves loss and accuracy curves across epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], "r--s", label="Val Loss")
    plt.title("Cross Entropy Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], "b-o", label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], "g--s", label="Val Accuracy")
    plt.title("Classification Accuracy (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def run_training(args):
    seed_everything(args.seed)
    ensure_dirs(args.checkpoint_dir, args.reports_dir)

    log_path = Path(args.reports_dir) / "training.log"
    logger = setup_logger("DeepfakeTrainer", log_file=log_path)
    logger.info("Initializing Deepfake Training Pipeline...")
    logger.info(f"Model Architecture: {args.model}")
    logger.info(f"Device: {args.device}")

    # Transforms & DataLoaders
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()

    train_loader, val_loader, _ = create_dataloaders(
        data_dir=args.data_dir,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    logger.info(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")
    if len(train_loader.dataset) == 0:
        logger.error(f"No training data found in {args.data_dir}/train! Please populate data or run preprocessing first.")
        return

    # Model & Loss & Optimizer
    device = torch.device(args.device)
    model = get_model(args.model, num_classes=config.NUM_CLASSES, pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }

    best_val_acc = 0.0
    patience_counter = 0
    best_checkpoint_path = Path(args.checkpoint_dir) / f"{args.model}_best.pth"
    latest_checkpoint_path = Path(args.checkpoint_dir) / f"{args.model}_latest.pth"

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n--- Epoch {epoch}/{args.epochs} ---")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        logger.info(
            f"Epoch {epoch:02d} Results: "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}"
        )

        # Save latest checkpoint
        torch.save({
            "epoch": epoch,
            "model_name": args.model,
            "state_dict": model.state_dict(),
            "val_acc": val_acc,
            "val_loss": val_loss,
            "optimizer": optimizer.state_dict()
        }, latest_checkpoint_path)

        # Check for best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_name": args.model,
                "state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "optimizer": optimizer.state_dict()
            }, best_checkpoint_path)
            logger.info(f"[*] New best model saved to {best_checkpoint_path} (Val Acc: {best_val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"[!] Early stopping triggered after {epoch} epochs.")
                break

    # Save training history JSON and plots
    curves_path = Path(args.reports_dir) / "training_curves.png"
    plot_learning_curves(history, curves_path)
    logger.info(f"Learning curves plot saved to {curves_path}")

    history_json_path = Path(args.reports_dir) / "training_history.json"
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
    logger.info(f"Training history metrics saved to {history_json_path}")
    logger.info(f"Training finished! Best Validation Accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Deepfake Image Detector")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet50", "mesonet", "efficientnet_b0"], help="Backbone model")
    parser.add_argument("--data-dir", type=str, default=str(config.PROCESSED_DATA_DIR), help="Path to processed dataset directory")
    parser.add_argument("--checkpoint-dir", type=str, default=str(config.CHECKPOINT_DIR), help="Directory to save checkpoints")
    parser.add_argument("--reports-dir", type=str, default=str(config.REPORTS_DIR), help="Directory to save training plots and logs")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY, help="Weight decay")
    parser.add_argument("--patience", type=int, default=config.EARLY_STOPPING_PATIENCE, help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=str(config.DEVICE), help="Device (cuda/cpu)")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", help="Do not load pre-trained ImageNet weights")

    args = parser.parse_args()
    run_training(args)

