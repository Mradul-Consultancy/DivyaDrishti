"""
Data augmentation and preprocessing transforms for training and evaluation.

JPEG augmentation rationale
----------------------------
Deepfake images are typically distributed via social media platforms (YouTube,
Instagram, WhatsApp, TikTok) which aggressively re-encode video frames as JPEG
at qualities between ~50–95.  The robustness tests (evaluation/robustness_test.py)
already measure performance degradation under JPEG compression at test time; this
module complements that by **exposing the model to simulated JPEG artefacts during
training** so it learns features that are invariant to blocking, ringing, and
chroma-subsampling artefacts rather than over-fitting to lossless training crops.

Implementation: a single PIL → BytesIO → JPEG re-encode → PIL round-trip is
sufficient and adds no extra dependencies (Pillow is already required).  The
quality is sampled uniformly from [q_min, q_max] on every call, introducing
stochastic variation that acts as a regulariser.
"""

import io
import random
from PIL import Image
from torchvision import transforms
from typing import Tuple
import config


# ---------------------------------------------------------------------------
# Custom transform
# ---------------------------------------------------------------------------

class RandomJPEGCompression:
    """
    Randomly re-encodes a PIL Image through JPEG at a quality sampled
    uniformly from [q_min, q_max].

    Args:
        q_min (int): Minimum JPEG quality (1–100). Lower → heavier artefacts.
        q_max (int): Maximum JPEG quality (1–100).
        p     (float): Probability of applying the transform (default 0.5).

    Notes:
        - Applied **before** ToTensor so the input and output are PIL Images.
        - Operates in-memory via ``io.BytesIO``; no disk I/O.
        - RGBA images are converted to RGB before encoding (JPEG has no alpha).
    """

    def __init__(self, q_min: int = 50, q_max: int = 95, p: float = 0.5):
        if not (1 <= q_min <= q_max <= 100):
            raise ValueError(f"Invalid quality range: q_min={q_min}, q_max={q_max}")
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.q_min = q_min
        self.q_max = q_max
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        quality = random.randint(self.q_min, self.q_max)
        # JPEG does not support alpha channel
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).copy()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"q_min={self.q_min}, q_max={self.q_max}, p={self.p})"
        )


# ---------------------------------------------------------------------------
# Public transform factories
# ---------------------------------------------------------------------------

def get_train_transforms(
    image_size: Tuple[int, int] = config.IMAGE_SIZE,
    jpeg_aug: bool = True,
    jpeg_q_min: int = 50,
    jpeg_q_max: int = 95,
    jpeg_p: float = 0.5,
) -> transforms.Compose:
    """
    Transforms for training.

    Pipeline (all PIL-stage ops before ToTensor):
      1. Resize to ``image_size``
      2. RandomHorizontalFlip
      3. RandomRotation ±10°
      4. ColorJitter (brightness/contrast/saturation ±10 %)
      5. [optional] RandomJPEGCompression — simulates social-media re-encoding
      6. ToTensor
      7. Normalize (ImageNet mean/std)

    Args:
        image_size:  (H, W) target size.
        jpeg_aug:    If True, insert RandomJPEGCompression before ToTensor.
        jpeg_q_min:  Lower bound of JPEG quality range (default 50).
        jpeg_q_max:  Upper bound of JPEG quality range (default 95).
        jpeg_p:      Probability of applying JPEG compression per sample (default 0.5).
    """
    pil_transforms = [
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    ]

    if jpeg_aug:
        pil_transforms.append(
            RandomJPEGCompression(q_min=jpeg_q_min, q_max=jpeg_q_max, p=jpeg_p)
        )

    pil_transforms += [
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD),
    ]

    return transforms.Compose(pil_transforms)

def get_val_transforms(image_size: Tuple[int, int] = config.IMAGE_SIZE) -> transforms.Compose:
    """
    Transforms for validation and testing:
    - Deterministic resize
    - Tensor conversion and normalization
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])

def get_inference_transforms(image_size: Tuple[int, int] = config.IMAGE_SIZE) -> transforms.Compose:
    """Transforms for standalone single-image inference."""
    return get_val_transforms(image_size)

