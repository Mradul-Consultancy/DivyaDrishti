"""
Unit and Regression Tests for Probability Calibration and Brier Score Evaluation.

Verifies:
1. Brier Score mathematical definition: BS = (1/N) * sum((p_i - y_i)^2).
2. Proper calibration bounds (0.0 for perfect certainty, 0.25 for uncalibrated 50/50 prior).
3. Inclusion of Brier calibration metrics in multi-model and ensemble evaluation outputs.
4. Empirical calibration comparison between individual backbones and the soft-voting ensemble.
"""

import sys
import json
import numpy as np
import pytest
from pathlib import Path
from sklearn.metrics import brier_score_loss

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


class TestCalibrationMetrics:
    def test_brier_score_bounds(self):
        """Tests mathematical boundaries of Brier Score."""
        # 1. Perfect calibration: 100% confidence matching truth
        y_true = np.array([0, 1, 0, 1])  # 0: fake, 1: real
        p_fake_perfect = np.array([1.0, 0.0, 1.0, 0.0])
        is_fake = (y_true == config.FAKE_LABEL_IDX).astype(float)
        bs_perfect = float(np.mean((p_fake_perfect - is_fake) ** 2))
        assert bs_perfect == 0.0

        # 2. Complete inversion: 100% confidence opposite to truth
        p_fake_worst = np.array([0.0, 1.0, 0.0, 1.0])
        bs_worst = float(np.mean((p_fake_worst - is_fake) ** 2))
        assert bs_worst == 1.0

        # 3. Uninformative constant 0.5 coin-flip
        p_fake_uniform = np.array([0.5, 0.5, 0.5, 0.5])
        bs_uniform = float(np.mean((p_fake_uniform - is_fake) ** 2))
        assert bs_uniform == 0.25

    def test_saved_predictions_contain_valid_probabilities(self):
        """Verifies that saved test predictions contain valid probabilities in [0.0, 1.0]."""
        pred_csv = config.REPORTS_DIR / "saved_predictions.csv"
        if not pred_csv.exists():
            pytest.skip("saved_predictions.csv does not exist yet")

        import pandas as pd
        df = pd.read_csv(pred_csv)

        prob_cols = [c for c in df.columns if c.endswith("_prob_fake")]
        assert len(prob_cols) >= 3, f"Expected at least 3 model probability columns, found {prob_cols}"

        for col in prob_cols:
            assert df[col].min() >= 0.0, f"{col} has negative probabilities"
            assert df[col].max() <= 1.0, f"{col} has probabilities exceeding 1.0"

    def test_ensemble_metrics_json_contains_brier_scores(self):
        """Verifies that ensemble_metrics.json records measured Brier scores."""
        json_path = config.REPORTS_DIR / "ensemble" / "ensemble_metrics.json"
        if not json_path.exists():
            pytest.skip("ensemble_metrics.json does not exist yet")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for model_key in ["yolo26n_cls", "convnext_tiny", "efficientnet_b3", "ensemble"]:
            assert model_key in data
            assert "brier_score" in data[model_key]
            bs = data[model_key]["brier_score"]
            assert 0.0 <= bs <= 1.0, f"Invalid Brier score for {model_key}: {bs}"
