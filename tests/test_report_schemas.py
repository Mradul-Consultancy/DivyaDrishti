"""
Tests for the report generator's schema handling (evaluation/generate_reports.py).

``main.py generate-reports`` used to assume the multi-model ensemble schema
produced by ``evaluation/evaluate_ensemble.py`` (reports/saved_predictions.csv)
and crashed confusingly - or produced empty artefacts - when handed the
single-model schema written by ``evaluation/evaluate.py``
(reports/saved_predictions_single.csv). Both schemas are now supported and
anything else is rejected with actionable guidance.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.generate_reports import PredictionReportGenerator


def _ensemble_df(n_videos: int = 12) -> pd.DataFrame:
    """Ensemble schema, one row per frame, video_id groups frames."""
    rng = np.random.default_rng(7)
    rows = []
    for v in range(n_videos):
        label = 0 if v % 2 == 0 else 1  # 0 fake, 1 real
        for f in range(5):
            p = float(np.clip(0.9 - 0.6 * label + rng.normal(0, 0.1), 0, 1))
            rows.append({
                "image_path": f"img_{v}_{f}.jpg",
                "video_id": f"video_{v}",
                "ground_truth": label,
                "yolo_prob_fake": p,
                "convnext_prob_fake": p,
                "effnet_prob_fake": p,
                "ensemble_prob_fake": p,
            })
    return pd.DataFrame(rows)


def _single_model_df(n_videos: int = 12) -> pd.DataFrame:
    """Single-model schema: true_label is the class NAME, one prob column."""
    rng = np.random.default_rng(11)
    rows = []
    for v in range(n_videos):
        label_name = "fake" if v % 2 == 0 else "real"
        label = 0 if label_name == "fake" else 1
        for f in range(5):
            p = float(np.clip(0.88 - 0.6 * label + rng.normal(0, 0.1), 0, 1))
            rows.append({
                "image_path": f"img_{v}_{f}.jpg",
                "video_id": f"video_{v}",
                "true_label": label_name,
                "predicted_label": "fake" if p >= 0.5 else "real",
                "prob_fake": p,
                "prob_real": round(1.0 - p, 6),
            })
    return pd.DataFrame(rows)


class TestEnsembleSchema:
    def test_metrics_and_tables_are_produced(self, tmp_path):
        gen = PredictionReportGenerator(_ensemble_df(), output_dir=tmp_path)
        metrics = gen.compute_all_metrics(threshold=0.5)
        assert set(metrics) == {"yolo", "convnext", "effnet", "ensemble"}
        md, csv = gen.export_results_table(metrics)
        assert md.exists() and csv.exists()
        assert "ROC-AUC" in md.read_text(encoding="utf-8")


class TestSingleModelSchema:
    def test_true_label_is_mapped_to_ground_truth(self, tmp_path):
        gen = PredictionReportGenerator(_single_model_df(), output_dir=tmp_path)
        # fake=0, real=1 per config.CLASS_TO_IDX
        assert set(gen.df["ground_truth"]) == {0, 1}

    def test_metrics_and_tables_are_produced(self, tmp_path):
        gen = PredictionReportGenerator(_single_model_df(), output_dir=tmp_path)
        metrics = gen.compute_all_metrics(threshold=0.5)
        assert set(metrics) == {"model"}
        auc = metrics["model"]["roc_auc"]
        assert 0.0 <= auc <= 1.0 and auc > 0.8  # the synthetic signal is strong
        md, csv = gen.export_results_table(metrics)
        assert md.exists() and csv.exists()
        text = md.read_text(encoding="utf-8")
        assert "Model (single)" in text

    def test_plots_render_for_a_single_model(self, tmp_path):
        gen = PredictionReportGenerator(_single_model_df(), output_dir=tmp_path)
        metrics = gen.compute_all_metrics(threshold=0.5)
        roc = tmp_path / "roc_plots.png"
        cm = tmp_path / "confusion_matrix_grid.png"
        gen.plot_roc_curves(metrics, roc)
        gen.plot_confusion_matrices(metrics, cm)
        assert roc.exists() and roc.stat().st_size > 0
        assert cm.exists() and cm.stat().st_size > 0


class TestUnknownSchema:
    def test_csv_without_any_prob_column_is_rejected_with_guidance(self, tmp_path):
        df = pd.DataFrame({
            "image_path": ["a.jpg"],
            "video_id": ["v0"],
            "ground_truth": [0],
        })
        with pytest.raises(ValueError, match="evaluate_ensemble.py|saved_predictions_single"):
            PredictionReportGenerator(df, output_dir=tmp_path)

    def test_unknown_true_label_value_is_rejected(self, tmp_path):
        df = _single_model_df()
        df.loc[0, "true_label"] = "synthetic"
        with pytest.raises(ValueError, match="true_label"):
            PredictionReportGenerator(df, output_dir=tmp_path)

    def test_neither_ground_truth_nor_true_label(self, tmp_path):
        df = pd.DataFrame({"prob_fake": [0.9], "video_id": ["v0"]})
        with pytest.raises(KeyError, match="ground_truth"):
            PredictionReportGenerator(df, output_dir=tmp_path)
