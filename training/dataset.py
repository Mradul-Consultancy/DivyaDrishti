"""
PyTorch Dataset and DataLoader loaders for Deepfake Detection.
"""

from pathlib import Path
from typing import Callable, Optional, List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import config


class DeepfakeDataset(Dataset):
    """
    Dataset loader for binary classification (real vs fake).
    Scans a directory containing 'real' and 'fake' subfolders.
    """

    def __init__(
        self,
        root_dir: Path | str,
        transform: Optional[Callable] = None,
        class_to_idx: Optional[dict] = None,
        return_path: bool = False
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.return_path = return_path

        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            # The label convention is fixed project-wide: fake=0, real=1
            # (config.CLASS_TO_IDX). The directory listing decides only WHICH
            # classes are present, never their indices - deriving the indices
            # from the listing would label every image of a single-class
            # split 'fake', silently inverting the labels.
            self.class_to_idx = config.CLASS_TO_IDX.copy()

        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self):
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        for class_name, label_idx in self.class_to_idx.items():
            class_folder = self.root_dir / class_name
            if not class_folder.exists():
                continue

            for img_path in class_folder.glob("*"):
                if img_path.suffix.lower() in valid_exts:
                    self.samples.append((img_path, label_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # A corrupted file must NOT keep its label: substituting a black
            # placeholder while retaining the label teaches the model that
            # "black frame = <label>", and silently mis-scores the sample at
            # evaluation time.
            raise RuntimeError(f"could not read image {img_path}: {e}") from e

        if self.transform:
            image = self.transform(image)

        if self.return_path:
            return image, label, str(img_path)
        return image, label

    def get_class_counts(self) -> dict:
        """Returns the distribution of samples per class."""
        counts = {cls: 0 for cls in self.class_to_idx}
        idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        for _, label in self.samples:
            counts[idx_to_class[label]] += 1
        return counts


def create_dataloaders(
    data_dir: Path | str = config.PROCESSED_DATA_DIR,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates train, val, and test DataLoaders.
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    train_dataset = DeepfakeDataset(train_dir, transform=train_transform)
    val_dataset = DeepfakeDataset(val_dir, transform=val_transform)
    test_dataset = DeepfakeDataset(test_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader

