"""
Orchestrate training of all three models on every available dataset.

Training grid (6 runs):
  Datasets : celebdf, dfdc
  Models   : efficientnet_b3.ra2_in1k (timm)
             convnext_tiny.fb_in22k_ft_in1k (timm)
             yolo26n-cls  (Ultralytics)

Checkpoint naming convention:
  models/checkpoints/<dataset>/<model_tag>_best.pth
  runs/yolo_cls/<dataset>/weights/best.pt

Usage
-----
  # Full 25-epoch run on both datasets
  python scripts/train_all_datasets.py

  # Quick 5-epoch smoke-test (pilot / debug)
  python scripts/train_all_datasets.py --epochs 5 --pilot

  # Only one dataset
  python scripts/train_all_datasets.py --datasets celebdf

  # Only timm models (skip YOLO)
  python scripts/train_all_datasets.py --skip-yolo
"""

import sys
import types
import argparse
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from utils.helpers import setup_logger, ensure_dirs

# ── Logging ────────────────────────────────────────────────────────────────
LOG_DIR = config.REPORTS_DIR / "training_grid"
ensure_dirs(LOG_DIR)
logger = setup_logger("TrainAllDatasets", log_file=LOG_DIR / "train_all.log")

# ── Dataset registry ───────────────────────────────────────────────────────
DATASET_DIRS = {
    "celebdf": config.PROCESSED_CELEBDF_DIR,
    "dfdc":    config.PROCESSED_DFDC_DIR,
}

# ── Model registry ─────────────────────────────────────────────────────────
TIMM_MODELS = [
    "efficientnet_b3.ra2_in1k",
    "convnext_tiny.fb_in22k_ft_in1k",
]
YOLO_MODEL = "yolo26n-cls.pt"


def checkpoint_path(dataset: str, model_tag: str) -> Path:
    """Returns the expected best-checkpoint path for a given (dataset, model) slot."""
    ckpt_dir = config.CHECKPOINT_DIR / dataset
    ensure_dirs(ckpt_dir)
    return ckpt_dir / f"{model_tag}_best.pth"


def yolo_best_path(dataset: str, run_name: str) -> Path:
    return config.BASE_DIR / "runs" / "yolo_cls" / dataset / run_name / "weights" / "best.pt"


def _dataset_has_splits(data_dir: Path) -> bool:
    """Returns True if data_dir contains non-empty train/real and train/fake subfolders."""
    train_real = data_dir / "train" / "real"
    train_fake = data_dir / "train" / "fake"
    return (
        train_real.exists() and any(train_real.iterdir())
        and train_fake.exists() and any(train_fake.iterdir())
    )


# ── Individual trainers ────────────────────────────────────────────────────

def train_timm_model(model_name: str, dataset: str, data_dir: Path, args):
    """Fine-tunes one timm model on one dataset."""
    from training.train_timm import run_timm_training

    model_tag = model_name.replace("/", "_").replace(".", "_")
    ckpt_dir  = config.CHECKPOINT_DIR / dataset
    rep_dir   = LOG_DIR / dataset
    ensure_dirs(ckpt_dir, rep_dir)

    best_ckpt = ckpt_dir / f"{model_name}_best.pth"
    if best_ckpt.exists() and not args.force:
        logger.info(f"[SKIP] {model_name} on {dataset}: checkpoint already exists at {best_ckpt}")
        return True

    logger.info(f"\n{'='*65}")
    logger.info(f"  TRAIN: {model_name}  |  DATASET: {dataset}")
    logger.info(f"{'='*65}")

    train_args = types.SimpleNamespace(
        model_name     = model_name,
        data_dir       = str(data_dir),
        checkpoint_dir = str(ckpt_dir),
        reports_dir    = str(rep_dir),
        epochs         = args.epochs,
        batch_size     = args.batch_size,
        lr             = 1e-4,
        weight_decay   = 1e-4,
        patience       = args.patience,
        num_workers    = getattr(args, "num_workers", config.NUM_WORKERS),
        seed           = 42,
        device         = str(config.DEVICE),
        pretrained     = True,
        jpeg_aug       = True,
        jpeg_q_min     = 50,
        jpeg_q_max     = 95,
        jpeg_p         = 0.5,
    )

    try:
        run_timm_training(train_args)
        logger.info(f"[OK] {model_name} on {dataset} — checkpoint: {best_ckpt}")
        return True
    except Exception as exc:
        logger.error(f"[FAIL] {model_name} on {dataset}: {exc}\n{traceback.format_exc()}")
        return False


def train_yolo_model(dataset: str, data_dir: Path, args):
    """Fine-tunes YOLO26n-cls on one dataset."""
    from training.train_yolo import run_yolo_training

    run_name  = f"deepfake_{dataset}"
    best_pt   = yolo_best_path(dataset, run_name)

    if best_pt.exists() and not args.force:
        logger.info(f"[SKIP] YOLO on {dataset}: weights already at {best_pt}")
        return True

    logger.info(f"\n{'='*65}")
    logger.info(f"  TRAIN: YOLO26n-cls  |  DATASET: {dataset}")
    logger.info(f"{'='*65}")

    yolo_args = types.SimpleNamespace(
        model        = str(config.BASE_DIR / YOLO_MODEL),
        data         = str(data_dir),
        epochs       = args.epochs,
        imgsz        = 224,
        batch_size   = args.batch_size,
        lr           = 1e-3,
        lrf          = 0.01,
        weight_decay = 5e-4,
        patience     = args.patience,
        optimizer    = "AdamW",
        project      = str(config.BASE_DIR / "runs" / "yolo_cls" / dataset),
        name         = run_name,
        workers      = getattr(args, "num_workers", config.NUM_WORKERS),
        seed         = 42,
        device       = str(config.DEVICE),
        jpeg_aug     = True,
        jpeg_q_min   = 50,
        jpeg_q_max   = 95,
        jpeg_p       = 0.5,
    )

    try:
        run_yolo_training(yolo_args)
        logger.info(f"[OK] YOLO26n-cls on {dataset} — weights: {best_pt}")
        return True
    except Exception as exc:
        logger.error(f"[FAIL] YOLO on {dataset}: {exc}\n{traceback.format_exc()}")
        return False


# ── Orchestrator ───────────────────────────────────────────────────────────

def run_all(args):
    datasets = args.datasets if args.datasets else list(DATASET_DIRS.keys())
    results  = {}   # (dataset, model) -> bool

    for dataset in datasets:
        data_dir = DATASET_DIRS.get(dataset)
        if data_dir is None:
            logger.warning(f"Unknown dataset '{dataset}' — skipping.")
            continue
        if not _dataset_has_splits(data_dir):
            logger.warning(
                f"Dataset '{dataset}' at {data_dir} has no populated train split. "
                "Run the extraction pipeline first (python main.py extract-videos). Skipping."
            )
            continue

        logger.info(f"\n{'#'*65}")
        logger.info(f"  DATASET: {dataset.upper()}  ({data_dir})")
        logger.info(f"{'#'*65}")

        # timm models
        for model_name in TIMM_MODELS:
            ok = train_timm_model(model_name, dataset, data_dir, args)
            results[(dataset, model_name)] = ok

        # YOLO
        if not args.skip_yolo:
            ok = train_yolo_model(dataset, data_dir, args)
            results[(dataset, "yolo26n-cls")] = ok

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  TRAINING GRID SUMMARY")
    print("=" * 65)
    print(f"{'Dataset':<12}  {'Model':<38}  {'Status'}")
    print("-" * 65)
    for (ds, mdl), ok in sorted(results.items()):
        status = "✅  OK" if ok else "❌  FAILED"
        print(f"{ds:<12}  {mdl:<38}  {status}")
    print("=" * 65)

    n_ok   = sum(results.values())
    n_fail = len(results) - n_ok
    logger.info(f"Grid complete: {n_ok}/{len(results)} runs succeeded, {n_fail} failed.")
    if n_fail:
        logger.warning("Some runs failed — check logs in " + str(LOG_DIR))


def main():
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-B3, ConvNeXt-Tiny, and YOLO26n-cls on each dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        choices=list(DATASET_DIRS.keys()),
        help="Which datasets to train on (default: all)",
    )
    parser.add_argument("--epochs",     type=int,   default=25,  help="Training epochs (default: 25)")
    parser.add_argument("--batch-size", type=int,   default=32,  help="Batch size (default: 32)")
    parser.add_argument("--patience",   type=int,   default=7,   help="Early-stopping patience (default: 7)")
    parser.add_argument("--pilot",      action="store_true",     help="Shortcut: set --epochs 5 --patience 3")
    parser.add_argument("--skip-yolo",  action="store_true",     help="Skip YOLO training (timm only)")
    parser.add_argument("--force",      action="store_true",     help="Re-train even if checkpoint exists")

    args = parser.parse_args()

    if args.pilot:
        args.epochs   = 5
        args.patience = 3
        logger.info("Pilot mode: 5 epochs, patience 3.")

    run_all(args)


if __name__ == "__main__":
    main()

