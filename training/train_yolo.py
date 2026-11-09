"""
Fine-tune YOLOv11n-cls (Ultralytics classification head) for real-vs-fake deepfake detection.

Ultralytics classification models expect an ImageFolder layout:
    dataset_root/
        train/
            real/   ...
            fake/   ...
        val/
            real/   ...
            fake/   ...

Class weighting is applied via a custom loss callback that replaces the
default CrossEntropyLoss with a weighted variant computed from the training
distribution.

JPEG Augmentation
-----------------
Ultralytics does not expose a native JPEG quality augmentation parameter for
classification tasks.  We apply it via a dataset-level transform wrapper:
after Ultralytics builds its internal Dataset object, the ``on_train_start``
callback wraps each sample's tensor with a RandomJPEGCompression step applied
at the PIL stage.  Since Ultralytics classification loaders read images and
apply internal transforms, we instead monkey-patch the dataset's transform
pipeline to prepend JPEG augmentation after the existing PIL transforms and
before the ToTensor step — mirroring the approach used in ``train_timm.py``.
"""

import sys
import json
import shutil
import argparse
from pathlib import Path
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from utils.helpers import setup_logger, seed_everything, ensure_dirs
from training.class_weights import compute_class_weights, log_class_info
from preprocessing.transforms import RandomJPEGCompression


def patch_yolo_loss_with_weights(model, class_weights: torch.Tensor, device):
    """
    Replaces the classification loss inside an Ultralytics YOLO-cls model
    with a weighted CrossEntropyLoss.
    
    Ultralytics stores the loss function in model.model.loss.
    This patches it in-place before training begins.
    """
    weighted_loss = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Ultralytics stores loss as model.model.loss or via the trainer
    # We hook into the trainer's criterion after it's built
    return weighted_loss


def run_yolo_training(args):
    seed_everything(args.seed)
    ensure_dirs(args.project)

    log_path = Path(args.project) / "yolo_training.log"
    logger = setup_logger("YOLOTrainer", log_file=log_path)

    logger.info("=" * 65)
    logger.info("YOLO Classification Fine-Tuning for Deepfake Detection")
    logger.info("=" * 65)

    # Delayed import so the script can still parse args without ultralytics
    from ultralytics import YOLO

    # --- Class Weights ---
    train_dir = Path(args.data) / "train"
    class_weights = log_class_info(logger, train_dir)
    device_str = args.device

    # --- Load pretrained model ---
    model_name = args.model
    if not model_name.endswith(".pt") and not Path(model_name).exists():
        model_name = f"{model_name}.pt"
    logger.info(f"Loading pretrained model: {model_name}")
    model = YOLO(model_name)

    # --- Train ---
    logger.info(f"Starting training for {args.epochs} epochs...")
    logger.info(f"Data directory: {args.data}")
    logger.info(f"Image size: {args.imgsz}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")

    # Ultralytics YOLO train() handles everything internally.
    # We pass class weights via the 'cls_weights' mechanism or
    # by patching after the trainer is built.
    #
    # For classification tasks, Ultralytics uses CrossEntropyLoss.
    # We override it via a training callback.

    # Custom callback to inject weighted loss
    def on_train_start(trainer):
        """Callback: replace loss with weighted CrossEntropyLoss."""
        device = trainer.device
        w = class_weights.to(device)
        trainer.criterion = nn.CrossEntropyLoss(weight=w)
        logger.info(f"[Callback] Injected weighted CrossEntropyLoss: {w.tolist()} on {device}")

        # --- Optionally inject JPEG augmentation into the train dataset ---
        # Ultralytics classification datasets use torchvision Compose pipelines
        # internally.  We insert RandomJPEGCompression before the first ToTensor
        # step so PIL images are re-encoded before tensor conversion.
        if getattr(args, "jpeg_aug", True):
            from torchvision.transforms import Compose
            from preprocessing.transforms import RandomJPEGCompression

            jpeg_transform = RandomJPEGCompression(
                q_min=getattr(args, "jpeg_q_min", 50),
                q_max=getattr(args, "jpeg_q_max", 95),
                p=getattr(args, "jpeg_p", 0.5),
            )

            try:
                train_dataset = trainer.train_loader.dataset
                ops = list(train_dataset.transforms.transforms)
                insert_idx = next(
                    (i for i, t in enumerate(ops)
                     if "totensor" in type(t).__name__.lower()),
                    len(ops) - 1,
                )
                ops.insert(insert_idx, jpeg_transform)
                train_dataset.transforms = Compose(ops)
                logger.info(
                    f"[Callback] Injected RandomJPEGCompression "
                    f"(q={args.jpeg_q_min}-{args.jpeg_q_max}, p={args.jpeg_p}) "
                    f"at transform position {insert_idx}"
                )
            except AttributeError as exc:
                logger.warning(
                    f"[Callback] Could not inject JPEG aug into YOLO dataset "
                    f"(Ultralytics transform structure may differ): {exc}"
                )

    model.add_callback("on_train_start", on_train_start)

    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        lr0=args.lr,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        patience=args.patience,
        project=str(args.project),
        name=args.name,
        device=device_str,
        workers=args.workers,
        pretrained=True,
        optimizer=args.optimizer,
        seed=args.seed,
        exist_ok=True,
        verbose=True
    )

    # --- Export metrics summary ---
    best_model_path = Path(args.project) / args.name / "weights" / "best.pt"
    logger.info(f"Training complete. Best weights: {best_model_path}")

    summary = {
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "lr0": args.lr,
        "class_weights": class_weights.tolist(),
        "best_weights": str(best_model_path),
    }

    summary_path = Path(args.project) / args.name / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Training summary saved to {summary_path}")

    # --- Validate ---
    logger.info("Running validation on best weights...")
    best_model = YOLO(str(best_model_path))
    val_results = best_model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=device_str,
        split="val"
    )

    logger.info("=" * 65)
    logger.info("YOLO Classification Training Finished Successfully!")
    logger.info("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO-cls for deepfake detection with class weights"
    )
    parser.add_argument("--model", type=str, default="yolo26n-cls.pt",
                        help="Ultralytics YOLO classification model (default: yolo26n-cls.pt)")
    parser.add_argument("--data", type=str, default=str(config.PROCESSED_DATA_DIR),
                        help="Dataset root with train/ and val/ ImageFolder structure")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs (default: 30)")
    parser.add_argument("--imgsz", type=int, default=224,
                        help="Input image size (default: 224)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate (default: 1e-3)")
    parser.add_argument("--lrf", type=float, default=0.01,
                        help="Final LR fraction (default: 0.01)")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight decay (default: 5e-4)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (default: 10)")
    parser.add_argument("--optimizer", type=str, default="AdamW",
                        choices=["SGD", "Adam", "AdamW"],
                        help="Optimizer (default: AdamW)")
    parser.add_argument("--project", type=str,
                        default=str(config.BASE_DIR / "runs" / "yolo_cls"),
                        help="Output project directory")
    parser.add_argument("--name", type=str, default="deepfake_yolo",
                        help="Run name within project directory")
    parser.add_argument("--workers", type=int, default=config.NUM_WORKERS,
                        help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=str(config.DEVICE),
                        help="Device: cuda, cpu, or device id")

    # JPEG augmentation
    parser.add_argument("--no-jpeg-aug", dest="jpeg_aug", action="store_false",
                        help="Disable RandomJPEGCompression augmentation during training")
    parser.set_defaults(jpeg_aug=True)
    parser.add_argument("--jpeg-q-min", type=int, default=50,
                        help="Minimum JPEG quality for augmentation (default: 50)")
    parser.add_argument("--jpeg-q-max", type=int, default=95,
                        help="Maximum JPEG quality for augmentation (default: 95)")
    parser.add_argument("--jpeg-p", type=float, default=0.5,
                        help="Per-sample JPEG augmentation probability (default: 0.5)")

    args = parser.parse_args()
    run_yolo_training(args)

