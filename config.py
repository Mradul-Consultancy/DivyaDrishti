"""
Central configuration for Deepfake Detection Pipeline.
"""

from pathlib import Path
import os
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent
WORK_DIR = BASE_DIR  # Alias for backward compatibility with Makefile
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = BASE_DIR / "models" / "checkpoints"
REPORTS_DIR = BASE_DIR / "reports"

# Structured Dataset Directories (for in-domain and cross-dataset testing)
PROCESSED_CELEBDF_DIR = PROCESSED_DATA_DIR / "celebdf"
PROCESSED_DFDC_DIR = PROCESSED_DATA_DIR / "dfdc"
REAL_LIFE_DIR = DATA_DIR / "real_life"

DATASETS = {
    "celebdf": PROCESSED_CELEBDF_DIR,
    "dfdc": PROCESSED_DFDC_DIR,
    "real_life": REAL_LIFE_DIR,
}

# Sub-splits (defaults to global processed, or can point to specific dataset)
TRAIN_DIR = PROCESSED_DATA_DIR / "train"
VAL_DIR = PROCESSED_DATA_DIR / "val"
TEST_DIR = PROCESSED_DATA_DIR / "test"

# Image and Model Settings
IMAGE_SIZE = (224, 224)  # (Height, Width) for standard CNN backbones
BATCH_SIZE = 32
# DataLoader workers. JPEG decode + augmentation is the bottleneck for
# GPU training, so this should be scaled with the host's core count rather
# than left at 2 (which leaves a GPU idle waiting for data). Capped at 4:
# each worker holds a decoded-image buffer, and too many of them exhaust
# the Windows page file on a modest RAM machine.
NUM_WORKERS = min(4, max(2, (os.cpu_count() or 4) - 1))
NUM_CLASSES = 2

# Standard PyTorch ImageFolder and Ultralytics YOLO sort directories alphabetically:
# 'fake' < 'real' -> fake=0, real=1
CLASS_NAMES = ["fake", "real"]
CLASS_TO_IDX = {"fake": 0, "real": 1}
IDX_TO_CLASS = {0: "fake", 1: "real"}
FAKE_LABEL_IDX = 0
REAL_LABEL_IDX = 1

# Training Hyperparameters
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5

# Normalization constants (ImageNet defaults)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Device configuration
def get_device() -> torch.device:
    """Returns CUDA if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()

