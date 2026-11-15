"""
Regression tests for defects found in the enterprise-readiness review.

Each test corresponds to a specific bug that shipped:

* face_extractor unpacked MTCNN's (x1, y1, x2, y2) corners as
  (x, y, w, h), so every served crop was far larger than the tight face
  crops the models were trained on.
* evaluation/evaluate.py contained a duplicated evaluation block that
  re-ran the model and overwrote the artefacts, silently ignoring
  --test-dir.
* api/server.py used json.loads without importing json, so /v1/models
  raised NameError once the calibration files existed.
* api/server.py passed a BytesIO to cv2.VideoCapture, which only accepts
  a path, so /v1/detect/video always returned 500.
* main.py's predict subparser did not expose the arguments
  evaluation/predict.py reads.
* training/dataset.py derived the label indices from the directory
  listing, so a single-class split was labelled entirely 'fake'.
"""

import inspect
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))


class TestFaceCropGeometry:
    """The inference crop must match the training crop."""

    def test_mtcnn_box_is_treated_as_corners(self):
        from preprocessing.face_extractor import FaceExtractor

        ex = FaceExtractor(margin_ratio=0.2)
        # A box in corner form must produce a crop of the box's own size.
        crop = ex._crop.__wrapped__ if hasattr(ex._crop, "__wrapped__") else ex._crop
        img = _make_image(400, 300)
        det = crop(img, (100, 50, 180, 110), 0.99)
        # 80x60 box + 20% margin on each side -> 112 x 84
        assert det.box == (84, 38, 196, 122)
        assert det.image.size == (112, 84)

    def test_haar_style_xywh_is_converted_to_corners(self):
        from preprocessing.face_extractor import FaceExtractor

        ex = FaceExtractor(margin_ratio=0.2)
        img = _make_image(400, 300)
        # Haar yields (x, y, w, h) = (100, 50, 80, 60) -> corners (100,50,180,110)
        corners = [(int(x), int(y), int(x + w), int(y + h))
                   for (x, y, w, h) in [(100, 50, 80, 60)]]
        det = ex._crop(img, corners[0], 1.0)
        assert det.box == (84, 38, 196, 122)

    def test_inference_crop_matches_training_crop(self):
        """
        The inference extractor and the training extractor must agree on the
        crop rectangle for the same face box.
        """
        from preprocessing.face_extractor import FaceExtractor
        from preprocessing.extract_faces_from_videos import VideoFaceExtractor

        train_src = inspect.getsource(VideoFaceExtractor.crop_faces_mtcnn)
        # The training extractor unpacks corners.
        assert "x1, y1, x2, y2" in train_src, (
            "training extractor no longer unpacks MTCNN boxes as corners"
        )
        # ...and the inference extractor must do the same.
        infer_src = inspect.getsource(FaceExtractor._crop)
        assert "x1, y1, x2, y2" in infer_src


class TestEvaluateSinglePass:
    def test_evaluate_runs_the_model_once(self):
        src = (ROOT / "evaluation" / "evaluate.py").read_text(encoding="utf-8")
        assert src.count("evaluation_summary.json") == 1, (
            "evaluation/evaluate.py writes the summary more than once - a "
            "duplicated evaluation block is overwriting the artefacts"
        )
        assert src.count("def evaluate_model") == 1

    def test_evaluate_honours_test_dir(self):
        src = (ROOT / "evaluation" / "evaluate.py").read_text(encoding="utf-8")
        assert "args.test_dir" in src
        # The explicit --test-dir must be consulted BEFORE the default
        # data-dir/test resolution, otherwise the explicit flag is ignored.
        assert src.index("args.test_dir") < src.index('Path(args.data_dir) / "test"')


class TestApiSurface:
    def test_json_is_imported(self):
        src = (ROOT / "api" / "server.py").read_text(encoding="utf-8")
        assert "import json" in src, (
            "api/server.py calls json.loads but never imports json - "
            "/v1/models raises NameError"
        )

    def test_video_capture_gets_a_path_not_a_buffer(self):
        src = (ROOT / "api" / "server.py").read_text(encoding="utf-8")
        assert "VideoCapture(io.BytesIO" not in src, (
            "cv2.VideoCapture only accepts a path"
        )
        assert "VideoCapture(tmp_path)" in src

    def test_request_threshold_is_clamped(self):
        src = (ROOT / "api" / "server.py").read_text(encoding="utf-8")
        assert "MIN_THRESHOLD" in src and "MAX_THRESHOLD" in src
        assert "MIN_THRESHOLD <= thr <= MAX_THRESHOLD" in src

    def test_upload_size_is_capped(self):
        src = (ROOT / "api" / "server.py").read_text(encoding="utf-8")
        assert "MAX_UPLOAD_BYTES" in src
        assert "_read_capped" in src

    def test_audit_endpoints_are_key_protected(self):
        src = (ROOT / "api" / "server.py").read_text(encoding="utf-8")
        assert "require_api_key" in src
        # The purge must leave a trace in the audit log.
        assert "audit_purged" in src


class TestCliSurface:
    def test_predict_subparser_exposes_the_arguments_predict_reads(self):
        src = (ROOT / "main.py").read_text(encoding="utf-8")
        i = src.find('p_pred = subparsers.add_parser("predict"')
        blk = src[i:i + 2000]
        for flag in ("--threshold", "--margin", "--min-confidence"):
            assert flag in blk, f"main.py predict is missing {flag}"


class TestDatasetLabelSafety:
    def test_single_class_split_keeps_the_global_label_convention(self):
        """
        A split containing only 'real' must not label those images 'fake'.
        """
        import config
        from training.dataset import DeepfakeDataset

        with tempfile.TemporaryDirectory() as tmp:
            only_real = Path(tmp) / "real"
            only_real.mkdir()
            (only_real / "a.jpg").write_bytes(b"x")
            ds = DeepfakeDataset(Path(tmp))
            assert ds.class_to_idx["real"] == config.REAL_LABEL_IDX
            assert ds.class_to_idx["fake"] == config.FAKE_LABEL_IDX

    def test_corrupt_images_are_not_replaced_with_labelled_placeholders(self):
        src = (ROOT / "training" / "dataset.py").read_text(encoding="utf-8")
        assert "Image.new(\"RGB\", config.IMAGE_SIZE, (0, 0, 0))" not in src, (
            "a corrupt image must not become a black placeholder that keeps "
            "its label"
        )


def _make_image(w, h):
    from PIL import Image
    return Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
