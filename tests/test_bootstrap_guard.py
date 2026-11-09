"""
Tests that the bootstrap CI safety guard actually fires when < 20 video clusters are present.
Covers: evaluation/video_metrics.py::bootstrap_confidence_intervals
"""
import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.video_metrics import bootstrap_confidence_intervals, MIN_CLUSTERS_FOR_BOOTSTRAP


def _make_video_df(n: int) -> pd.DataFrame:
    """Return a minimal video-level DataFrame with n rows."""
    import numpy as np
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        "video_id": [f"v{i}" for i in range(n)],
        "prob_fake": rng.uniform(0.1, 0.9, n),
        "ground_truth": ([0] * (n // 2)) + ([1] * (n - n // 2)),
    })


def test_guard_fires_below_minimum():
    """bootstrap_confidence_intervals must raise ValueError when n < MIN_CLUSTERS_FOR_BOOTSTRAP."""
    df = _make_video_df(MIN_CLUSTERS_FOR_BOOTSTRAP - 1)
    with pytest.raises(ValueError, match="Safety Guard Activated"):
        bootstrap_confidence_intervals(df, prob_col="prob_fake")


def test_guard_allows_at_minimum():
    """bootstrap_confidence_intervals must succeed when n >= MIN_CLUSTERS_FOR_BOOTSTRAP."""
    df = _make_video_df(MIN_CLUSTERS_FOR_BOOTSTRAP)
    # Should not raise
    result = bootstrap_confidence_intervals(df, prob_col="prob_fake", n_bootstraps=50)
    assert "roc_auc" in result


def test_guard_allows_above_minimum():
    """bootstrap_confidence_intervals must succeed when n is comfortably above minimum."""
    df = _make_video_df(MIN_CLUSTERS_FOR_BOOTSTRAP + 10)
    result = bootstrap_confidence_intervals(df, prob_col="prob_fake", n_bootstraps=50)
    assert "roc_auc" in result
    assert "mean" in result["roc_auc"]
