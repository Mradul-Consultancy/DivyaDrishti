"""
Tests for RandomJPEGCompression augmentation.

Verifies:
1. Output is a PIL Image of the same size as the input.
2. With p=1.0 the transform always applies; pixel values change (JPEG is lossy).
3. With p=0.0 the transform never applies; output is identical to input.
4. Quality bounds are enforced (invalid ranges raise ValueError).
5. The transform integrates into the torchvision Compose pipeline and produces
   a tensor of the expected shape.
6. RGBA images are silently converted to RGB (JPEG has no alpha channel).
7. The transform is wired into get_train_transforms() when jpeg_aug=True and
   absent when jpeg_aug=False.
8. The timm pipeline insert_idx correctly places the transform before ToTensor.
"""

import sys
import io
from pathlib import Path

import pytest
from PIL import Image
import numpy as np

# ---------------------------------------------------------------------------
# Local path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from preprocessing.transforms import RandomJPEGCompression, get_train_transforms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_pil(color=(128, 64, 32), size=(224, 224)) -> Image.Image:
    """Return a solid-colour RGB PIL Image."""
    img = Image.new("RGB", size, color=color)
    return img


# ---------------------------------------------------------------------------
# 1. Output shape and type
# ---------------------------------------------------------------------------

class TestOutputProperties:
    def test_output_is_pil_image(self):
        t = RandomJPEGCompression(q_min=50, q_max=95, p=1.0)
        img = _solid_pil()
        out = t(img)
        assert isinstance(out, Image.Image), "Output must be a PIL Image"

    def test_output_size_unchanged(self):
        t = RandomJPEGCompression(q_min=50, q_max=95, p=1.0)
        img = _solid_pil(size=(224, 224))
        out = t(img)
        assert out.size == img.size, "Spatial dimensions must be preserved"

    def test_output_mode_rgb(self):
        t = RandomJPEGCompression(q_min=50, q_max=95, p=1.0)
        img = _solid_pil()
        out = t(img)
        assert out.mode == "RGB", "Output mode must be RGB"


# ---------------------------------------------------------------------------
# 2. Stochastic behaviour
# ---------------------------------------------------------------------------

class TestProbability:
    def test_always_applies_when_p_is_one(self):
        """With p=1.0 and a solid colour, JPEG artefacts will modify pixels."""
        t = RandomJPEGCompression(q_min=10, q_max=10, p=1.0)  # very low Q → visible artefacts
        # Use a gradient image so non-trivial JPEG artefacts appear
        arr = np.tile(np.arange(224, dtype=np.uint8), (224, 1))
        img = Image.fromarray(np.stack([arr, arr, arr], axis=-1))
        out = t(img)
        # At Q=10 some pixels will differ
        diff = np.abs(np.array(out).astype(int) - np.array(img).astype(int))
        assert diff.max() > 0, "At low quality, JPEG artefacts must alter at least one pixel"

    def test_never_applies_when_p_is_zero(self):
        """With p=0.0 the transform must be a no-op."""
        t = RandomJPEGCompression(q_min=10, q_max=10, p=0.0)
        img = _solid_pil()
        for _ in range(20):   # run many times to be sure
            out = t(img)
            assert list(out.getdata()) == list(img.getdata()), \
                "With p=0.0 the image must be returned unchanged"


# ---------------------------------------------------------------------------
# 3. Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_invalid_quality_range_raises(self):
        with pytest.raises(ValueError):
            RandomJPEGCompression(q_min=90, q_max=50)   # min > max

    def test_quality_out_of_bounds_raises(self):
        with pytest.raises(ValueError):
            RandomJPEGCompression(q_min=0, q_max=100)   # q_min < 1

    def test_invalid_p_raises(self):
        with pytest.raises(ValueError):
            RandomJPEGCompression(q_min=50, q_max=95, p=1.5)


# ---------------------------------------------------------------------------
# 4. RGBA → RGB conversion
# ---------------------------------------------------------------------------

class TestRGBAConversion:
    def test_rgba_input_produces_rgb_output(self):
        t = RandomJPEGCompression(q_min=80, q_max=95, p=1.0)
        img = Image.new("RGBA", (224, 224), color=(100, 150, 200, 255))
        out = t(img)
        assert out.mode == "RGB", "RGBA input must be converted to RGB"
        assert out.size == (224, 224), "Size must be unchanged after RGBA→RGB conversion"


# ---------------------------------------------------------------------------
# 5. Integration with torchvision Compose pipeline
# ---------------------------------------------------------------------------

class TestComposeIntegration:
    def test_produces_correct_tensor_shape(self):
        import torch
        transform = get_train_transforms(image_size=(224, 224), jpeg_aug=True,
                                         jpeg_q_min=50, jpeg_q_max=95, jpeg_p=1.0)
        img = _solid_pil()
        tensor = transform(img)
        assert isinstance(tensor, torch.Tensor), "Output of full pipeline must be a tensor"
        assert tensor.shape == (3, 224, 224), f"Expected (3, 224, 224), got {tensor.shape}"

    def test_jpeg_aug_absent_when_disabled(self):
        transform = get_train_transforms(image_size=(224, 224), jpeg_aug=False)
        names = [type(t).__name__ for t in transform.transforms]
        assert "RandomJPEGCompression" not in names, \
            "RandomJPEGCompression must not appear when jpeg_aug=False"

    def test_jpeg_aug_present_when_enabled(self):
        transform = get_train_transforms(image_size=(224, 224), jpeg_aug=True)
        names = [type(t).__name__ for t in transform.transforms]
        assert "RandomJPEGCompression" in names, \
            "RandomJPEGCompression must appear in the train pipeline when jpeg_aug=True"

    def test_jpeg_aug_before_totensor(self):
        """JPEG aug must come before ToTensor so it operates on a PIL Image."""
        transform = get_train_transforms(image_size=(224, 224), jpeg_aug=True)
        names = [type(t).__name__ for t in transform.transforms]
        jpeg_idx = names.index("RandomJPEGCompression")
        totensor_idx = next(
            i for i, n in enumerate(names) if "totensor" in n.lower()
        )
        assert jpeg_idx < totensor_idx, \
            "RandomJPEGCompression must appear before ToTensor in the pipeline"


# ---------------------------------------------------------------------------
# 6. timm pipeline injection
# ---------------------------------------------------------------------------

class TestTimmPipelineInjection:
    def test_jpeg_inserted_before_totensor_in_timm_pipeline(self):
        """
        get_timm_transforms should insert RandomJPEGCompression before the
        MaybeToTensor / ToTensor step in the timm training Compose chain.
        """
        import timm
        from training.train_timm import get_timm_transforms

        model = timm.create_model(
            "convnext_tiny.fb_in22k_ft_in1k", pretrained=False, num_classes=2
        )
        transform = get_timm_transforms(
            model, is_training=True, jpeg_aug=True,
            jpeg_q_min=50, jpeg_q_max=95, jpeg_p=0.5
        )
        names = [type(t).__name__ for t in transform.transforms]
        assert "RandomJPEGCompression" in names, \
            "RandomJPEGCompression must be injected into the timm training pipeline"
        jpeg_idx = names.index("RandomJPEGCompression")
        totensor_idx = next(
            i for i, n in enumerate(names) if "totensor" in n.lower()
        )
        assert jpeg_idx < totensor_idx, \
            "RandomJPEGCompression must appear before MaybeToTensor in the timm pipeline"

    def test_jpeg_absent_in_timm_val_pipeline(self):
        """Validation/test pipeline must NEVER apply JPEG augmentation."""
        import timm
        from training.train_timm import get_timm_transforms

        model = timm.create_model(
            "convnext_tiny.fb_in22k_ft_in1k", pretrained=False, num_classes=2
        )
        transform = get_timm_transforms(model, is_training=False)
        names = [type(t).__name__ for t in transform.transforms]
        assert "RandomJPEGCompression" not in names, \
            "RandomJPEGCompression must NOT appear in the val/test pipeline"

