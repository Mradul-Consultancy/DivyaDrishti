"""
Utility helper functions for Deepfake Detection.
"""

import os
import random
import logging
from pathlib import Path
import numpy as np
import torch

def setup_logger(name: str = "deepfake_detector", log_file: Path | str | None = None) -> logging.Logger:
    """Configures a standardized console and optional file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
    return logger

def seed_everything(seed: int = 42):
    """Sets seeds across libraries for reproducible experiments."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def ensure_dirs(*dirs):
    """Ensures that one or multiple directories exist."""
    for d in dirs:
        if isinstance(d, (list, tuple, set)):
            for sub_d in d:
                Path(sub_d).mkdir(parents=True, exist_ok=True)
        else:
            Path(d).mkdir(parents=True, exist_ok=True)

