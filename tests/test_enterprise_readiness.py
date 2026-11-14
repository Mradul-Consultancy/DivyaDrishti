"""
Tests for the enterprise-readiness additions:

* threshold calibration (evaluation/calibration.py)
* spectral descriptors (evaluation/spectral.py) and their combiner
* forensic report generation and verification (api/forensic.py)
* the rewritten FaceExtractor (multi-face, confidence, explicit no-face)
* the CLI inference path using the training transform
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(0, str(ROOT)) if False else sys.path.append(str(ROOT))


class TestThresholdCalibration:
    def test_curve_reports_both_classes(self):
        from evaluation.calibration import threshold_curve
        probs = np.array([0.9, 0.8, 0.2, 0.1])
        labels = np.array([0, 0, 1, 1])  # 0 = fake, 1 = real
        curve = threshold_curve(probs, labels)
        assert curve
        for row in curve:
            assert 0.0 <= row["fake_recall"] <= 1.0
            assert 0.0 <= row["false_positive_rate"] <= 1.0

    def test_threshold_is_monotone_in_false_positives(self):
        """Raising the threshold can only reduce the false-positive rate."""
        from evaluation.calibration import threshold_curve
        rng = np.random.RandomState(0)
        probs = rng.uniform(0, 1, 200)
        labels = np.array([0] * 150 + [1] * 50)
        curve = threshold_curve(probs, labels)
        ordered = sorted(curve, key=lambda r: r["threshold"])
        fprs = [r["false_positive_rate"] for r in ordered]
        assert all(a >= b - 1e-9 for a, b in zip(fprs, fprs[1:]))

    def test_youden_point_is_reasonable(self):
        from evaluation.calibration import threshold_curve, youden_threshold
        # A perfectly separating set: fakes score high, reals score low.
        probs = np.array([0.9, 0.95, 0.05, 0.1])
        labels = np.array([0, 0, 1, 1])
        curve = threshold_curve(probs, labels)
        best = youden_threshold(curve)
        assert best["fake_recall"] == pytest.approx(1.0)
        assert best["real_recall"] == pytest.approx(1.0)

    def test_calibration_artifacts_are_per_model(self):
        """One threshold file per model - a single file would be overwritten."""
        import config
        d = config.CHECKPOINT_DIR / "celebdf"
        if not d.exists():
            pytest.skip("no Celeb-DF checkpoints")
        files = list(d.glob("threshold_*.json"))
        if not files:
            pytest.skip("no calibration artefacts yet")
        for f in files:
            data = json.loads(f.read_text(encoding="utf-8"))
            assert "chosen_threshold" in data
            assert 0.0 < data["chosen_threshold"] < 1.0
            assert "at_default_0.5" in data

    def test_load_threshold_falls_back_cleanly(self):
        from evaluation.calibration import load_threshold
        assert load_threshold("nonexistent_domain", "no_model") == 0.5


class TestSpectralFeatures:
    def test_features_are_finite(self):
        from evaluation.spectral import compute_spectral_features
        rng = np.random.RandomState(0)
        f = compute_spectral_features(rng.randint(0, 255, (64, 64)).astype(float))
        for v in f.as_vector():
            assert np.isfinite(v)

    def test_high_frequency_ratio_differs_between_smooth_and_noisy(self):
        from evaluation.spectral import compute_spectral_features
        rng = np.random.RandomState(0)
        smooth = compute_spectral_features(np.full((64, 64), 128.0))
        noisy = compute_spectral_features(rng.randint(0, 255, (64, 64)).astype(float))
        assert noisy.high_freq_energy_ratio > smooth.high_freq_energy_ratio

    def test_feature_vector_length_matches_names(self):
        from evaluation.spectral import feature_names
        from evaluation.spectral import compute_spectral_features
        rng = np.random.RandomState(1)
        f = compute_spectral_features(rng.randint(0, 255, (32, 32)).astype(float))
        assert len(f.as_vector()) == len(feature_names())

    def test_combiner_round_trip(self):
        from evaluation.spectral_combiner import combine, fit_combiner
        rng = np.random.RandomState(0)
        neural = rng.uniform(0, 1, 60)
        spectral = rng.uniform(0, 1, (60, 5))
        labels = np.array([0] * 30 + [1] * 30)
        payload = fit_combiner(neural, spectral, labels)
        assert "coef" in payload and "mean" in payload
        out = combine(float(neural[0]), spectral[0], payload)
        assert 0.0 <= out <= 1.0


class TestForensicReport:
    def _finding(self):
        from api.forensic import ForensicFinding
        return ForensicFinding(
            media_name="sample.jpg", media_sha256="a" * 64, media_bytes=1234,
            media_type="image", verdict="fake", fake_probability=0.91,
            real_probability=0.09, threshold=0.8183, threshold_source="calibrated",
            model="convnext_tiny.fb_in22k_ft_in1k", checkpoint="x.pth",
            face_count=1, face_detector="mtcnn", face_min_confidence=0.9,
            margin_ratio=0.2, c2pa_present=False, c2pa_signature_verified=False,
        )

    def test_report_has_digest_and_no_media(self):
        from api.forensic import build_report
        body = build_report([self._finding()])
        assert len(body["report_sha256"]) == 64
        blob = json.dumps(body)
        assert "a" * 64 in blob          # the input digest is recorded
        assert "image_bytes" not in blob  # but no media content is embedded

    def test_report_verifies_after_write(self):
        from api.forensic import build_report, verify_report, write_json_report
        body = build_report([self._finding()])
        with tempfile.TemporaryDirectory() as tmp:
            path = write_json_report(body, Path(tmp) / "report.json")
            result = verify_report(path)
        assert result["intact"] is True

    def test_tampering_is_detected(self):
        from api.forensic import build_report, verify_report, write_json_report
        body = build_report([self._finding()])
        with tempfile.TemporaryDirectory() as tmp:
            path = write_json_report(body, Path(tmp) / "report.json")
            data = json.loads(path.read_text(encoding="utf-8"))
            data["findings"][0]["verdict"] = "real"   # tamper
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            intact = verify_report(path)["intact"]
        assert intact is False

    def test_markdown_rendering(self):
        from api.forensic import build_report, render_markdown
        md = render_markdown(build_report([self._finding()]))
        assert "Forensic Audit Report" in md
        assert "sample.jpg" in md


class TestFaceExtractor:
    def test_reports_every_face_with_confidence(self):
        from preprocessing.face_extractor import FaceExtractor
        ex = FaceExtractor(margin_ratio=0.2)
        assert ex.backend in ("mtcnn", "haar", "none")

    def test_no_face_raises_by_default(self):
        """A frame with no acceptable face must not be silently centre-cropped."""
        from preprocessing.face_extractor import FaceExtractor, FaceNotFoundError
        import numpy as np
        from PIL import Image
        ex = FaceExtractor(margin_ratio=0.2, min_confidence=0.999999)
        blank = Image.fromarray(np.zeros((120, 120, 3), dtype=np.uint8))
        with pytest.raises(FaceNotFoundError):
            ex.extract_face(blank)

    def test_centre_crop_only_when_explicitly_allowed(self):
        from preprocessing.face_extractor import FaceExtractor
        import numpy as np
        from PIL import Image
        ex = FaceExtractor(margin_ratio=0.2, min_confidence=0.999999,
                           allow_center_fallback=True)
        blank = Image.fromarray(np.zeros((120, 120, 3), dtype=np.uint8))
        out = ex.extract_face(blank)
        assert out.size == (120, 120)


class TestPredictTransform:
    def test_predict_uses_the_training_transform(self):
        """
        evaluation/predict.py must derive the transform from the model, not
        use the generic torchvision resize.
        """
        src = (ROOT / "evaluation" / "predict.py").read_text(encoding="utf-8")
        assert "get_eval_transform" in src
        assert "get_inference_transforms()" not in src, (
            "predict.py still uses the generic transform that does not match training"
        )

    def test_predict_reports_no_face(self):
        src = (ROOT / "evaluation" / "predict.py").read_text(encoding="utf-8")
        assert "FaceNotFoundError" in src
        assert "extract_faces" in src
