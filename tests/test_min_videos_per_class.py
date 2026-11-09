"""
Tests that generate_reports_from_predictions raises when any test class has < min_videos_per_class.
Covers: evaluation/generate_reports.py::generate_reports_from_predictions
"""
import pytest
import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from evaluation.generate_reports import generate_reports_from_predictions


def _write_predictions_csv(path: Path, n_fake: int, n_real: int):
    """Write a minimal saved_predictions.csv with n_fake fake rows and n_real real rows."""
    rows = (
        [
            {
                "video_id": f"fake_video_{i}",
                "filename": f"fake_video_{i}_f000.jpg",
                "ground_truth": config.FAKE_LABEL_IDX,
                "ensemble_pred": config.FAKE_LABEL_IDX,
                "ensemble_prob_fake": 0.95,
            }
            for i in range(n_fake)
        ]
        + [
            {
                "video_id": f"real_video_{i}",
                "filename": f"real_video_{i}_f000.jpg",
                "ground_truth": config.REAL_LABEL_IDX,
                "ensemble_pred": config.REAL_LABEL_IDX,
                "ensemble_prob_fake": 0.05,
            }
            for i in range(n_real)
        ]
    )
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video_id", "filename", "ground_truth", "ensemble_pred", "ensemble_prob_fake"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_guard_fires_below_min_videos():
    """generate_reports_from_predictions must raise ValueError when a class has < min_videos_per_class."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pred_csv = tmp_path / "saved_predictions.csv"
        # 5 fake, 20 real -> fake is below minimum of 10
        _write_predictions_csv(pred_csv, n_fake=5, n_real=20)
        with pytest.raises(ValueError, match="Safety Guard Activated"):
            generate_reports_from_predictions(
                predictions_file=pred_csv,
                output_dir=tmp_path / "reports",
                min_videos_per_class=10,
            )


def test_guard_passes_at_minimum():
    """generate_reports_from_predictions must succeed when both classes meet the minimum."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pred_csv = tmp_path / "saved_predictions.csv"
        # 10 fake, 10 real -> meets minimum of 10
        _write_predictions_csv(pred_csv, n_fake=10, n_real=10)
        try:
            generate_reports_from_predictions(
                predictions_file=pred_csv,
                output_dir=tmp_path / "reports",
                min_videos_per_class=10,
            )
        except ValueError as e:
            pytest.fail(f"Guard fired unexpectedly: {e}")
