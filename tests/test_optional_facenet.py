"""
Tests that the preprocessing package imports cleanly WITHOUT facenet-pytorch.

facenet-pytorch is an optional dependency: the preprocessing package's
__init__ imports the video extractor, which used to import MTCNN at module
level. That made every module touching preprocessing.transforms (including
the training scripts) fail on machines where facenet-pytorch is absent -
for example a Kaggle sandbox with no internet, where pip install is
impossible.

These tests simulate the package being unavailable and assert that:
  - importing preprocessing still works,
  - face extraction degrades to a centre crop instead of raising.
"""

import builtins
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))


def _import_without_facenet():
    """
    Import preprocessing with facenet_pytorch made unimportable.

    The preprocessing modules are purged from sys.modules first so the
    guarded import actually takes effect even when another test has already
    imported the package. The original modules are restored afterwards so
    the rest of the suite is unaffected.
    """
    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.startswith("facenet_pytorch"):
            raise ImportError("simulated: facenet_pytorch unavailable")
        return real_import(name, *args, **kwargs)

    saved = {k: v for k, v in sys.modules.items()
             if k == "preprocessing" or k.startswith("preprocessing.")}

    for k in list(saved):
        del sys.modules[k]

    builtins.__import__ = guard
    try:
        import preprocessing
        from preprocessing.extract_faces_from_videos import MTCNN_AVAILABLE
        result = (preprocessing, MTCNN_AVAILABLE)
    finally:
        builtins.__import__ = real_import
        for k in [k for k in sys.modules
                  if k == "preprocessing" or k.startswith("preprocessing.")]:
            del sys.modules[k]
        sys.modules.update(saved)

    return result


class TestOptionalFacenet:
    def test_preprocessing_imports_without_facenet(self):
        preprocessing, available = _import_without_facenet()
        assert available is False, "MTCNN should be reported unavailable"
        assert preprocessing is not None

    def test_extract_faces_falls_back_to_centre_crop(self):
        import torch

        preprocessing, _ = _import_without_facenet()
        extractor = preprocessing.extract_faces_from_videos.VideoFaceExtractor(
            frames_per_video=1,
            image_size=(224, 224),
            margin_ratio=0.2,
            min_face_prob=0.9,
            fallback_center_crop=True,
            device=torch.device("cpu"),
        )
        img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        faces, detected = extractor.crop_faces_mtcnn(img)
        assert len(faces) == 1
        assert faces[0].size == (224, 224)

    def test_image_size_accepts_int_and_tuple(self):
        import torch

        preprocessing, _ = _import_without_facenet()
        for image_size in (224, (224, 224)):
            extractor = preprocessing.extract_faces_from_videos.VideoFaceExtractor(
                frames_per_video=1,
                image_size=image_size,
                margin_ratio=0.2,
                min_face_prob=0.9,
                fallback_center_crop=True,
                device=torch.device("cpu"),
            )
            assert extractor.image_size == (224, 224)
