"""
Tests for offline pretrained-weight resolution.

timm fetches ImageNet weights from the HuggingFace Hub. In an offline
environment (the Kaggle sandbox has no DNS) that download fails, which
previously aborted every timm training run before the first epoch.

training/train_timm.py therefore falls back to a vendored copy under
pretrained/<model>.safetensors, and only then to random initialisation.
These tests pin that behaviour down.
"""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))


class TestLocalPretrainedWeights:
    def test_vendored_weights_exist(self):
        import config

        pretrained_dir = config.BASE_DIR / "pretrained"
        if not pretrained_dir.is_dir():
            pytest.skip("pretrained/ directory not populated in repository root")

        for name in ("efficientnet_b3.ra2_in1k", "convnext_tiny.fb_in22k_ft_in1k"):
            f = pretrained_dir / f"{name}.safetensors"
            if not f.exists() or f.stat().st_size < 1_000_000:
                pytest.skip(f"vendored weights {name} not downloaded locally (excluded from Git)")

    def test_build_timm_model_uses_vendored_weights(self, monkeypatch):
        """
        When the Hub lookup fails, the vendored file must be used and the
        model must still be constructed.
        """
        import timm
        import training.train_timm as tt

        # Capture the real factory first: tt.timm is the same module object,
        # so patching tt.timm.create_model would make the fake call itself.
        real_create_model = timm.create_model
        calls = {"n": 0}

        def fake_create_model(name, **kwargs):
            calls["n"] += 1
            if kwargs.get("pretrained") and "pretrained_cfg_overlay" not in kwargs:
                raise OSError("simulated: no Hub access")
            return real_create_model(name, pretrained=False,
                                      num_classes=kwargs.get("num_classes", 2))

        monkeypatch.setattr(tt.timm, "create_model", fake_create_model)

        model = tt.build_timm_model("efficientnet_b3.ra2_in1k", num_classes=2, pretrained=True)
        assert model is not None
        assert calls["n"] >= 2, "the vendored fallback path was not exercised"

    def test_build_timm_model_without_pretrained(self):
        import training.train_timm as tt

        model = tt.build_timm_model("efficientnet_b3.ra2_in1k", num_classes=2, pretrained=False)
        assert model is not None
        assert model.classifier.out_features == 2
