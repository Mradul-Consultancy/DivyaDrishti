"""
Tests that generate_reports_from_predictions raises when any test class has < min_videos_per_class.
Covers: evaluation/generate_reports.py::generate_reports_from_predictions
"""
import pytest
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.generate_reports import generate_reports_from_predictions


def _write_predictions_csv(path: Path, n_fake: int, n_real: int):
    """Write a minimal saved_predictions.csv with n_fake fake rows and n_real real rows."""
    import config
    rows = (
        [{"video_id": f"fake_{i}", "label": config.FAKE_LABEL_IDX,
          "pred": config.FAKE_LABEL_IDX, "prob_fake": 0.9} for i in range(n_fake)]
        + [{"video_id": f"real_{i}", "label": config.REAL_LABEL_IDX,
            "pred": config.REAL_LABEL_IDX, "prob_fake": 0.1} for i in range(n_real)]
    )
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "label", "pred", "prob_fake"])
        writer.writeheader()
        writer.writerows(rows)


def test_guard_fires_below_min_videos(tmp_path):
    """generate_reports_from_predictions must raise RuntimeError when a class has < 10 videos."""
    pred_csv = tmp_path / "saved_predictions.csv"
    # 5 fake, 20 real → fake is below the default minimum of 10
    _write_predictions_csv(pred_csv, n_fake=5, n_real=20)
    with pytest.raises(RuntimeError, match="[Mm]inimum"):
        generate_reports_from_predictions(
            predictions_csv=pred_csv,
            reports_dir=tmp_path / "reports",
            min_videos_per_class=10,
        )


def test_guard_passes_at_minimum(tmp_path):
    """generate_reports_from_predictions must succeed when both classes meet the minimum."""
    pred_csv = tmp_path / "saved_predictions.csv"
    # 10 fake, 10 real → exactly at the minimum
    _write_predictions_csv(pred_csv, n_fake=10, n_real=10)
    # Should not raise (may warn about small sample, but must not raise RuntimeError)
    try:
        generate_reports_from_predictions(
            predictions_csv=pred_csv,
            reports_dir=tmp_path / "reports",
            min_videos_per_class=10,
        )
    except RuntimeError as e:
        pytest.fail(f"Guard fired unexpectedly: {e}")
