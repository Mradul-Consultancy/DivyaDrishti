"""
Video-level aggregation, bootstrap confidence intervals, and recall at fixed FPR analysis.

This module provides statistical assessment for deepfake detection models:
1. Video-level aggregation: Aggregates frame/crop predictions by video (mean, max, median, trimmed mean).
2. Recall at Fixed False Positive Rates (e.g. Recall @ 1% FPR, Recall @ 5% FPR, Recall @ 10% FPR).
3. Cluster/Video-level Bootstrapping: Computes 95% confidence intervals by resampling whole videos (with replacement)
   to respect temporal and identity correlation within videos.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve,
    auc,
    accuracy_score,
    precision_recall_fscore_support,
    brier_score_loss
)

import config


def extract_video_id(filename_or_path: str) -> str:
    """
    Infers the parent video ID from a frame/face filename or path.
    Examples:
      - 'video_fake_06_f00000_face0.jpg' -> 'video_fake_06'
      - '001_003_f00010_face0.jpg' -> '001_003'
      - 'id0_0000_f00005_face0.png' -> 'id0_0000'
      - 'fake_007.jpg' -> 'fake_007'
    """
    stem = Path(filename_or_path).stem
    # Match patterns like <video_stem>_f<frame_idx>_face<face_idx> or <video_stem>_frame<frame_idx>
    m = re.match(r"^(.*?)(?:_f\d{3,6}|_frame\d{3,6}|_face\d+)+$", stem, re.IGNORECASE)
    if m:
        return m.group(1)
    
    # Handle single images (e.g., fake_007.jpg -> fake_007)
    return stem


def compute_recall_at_fixed_fpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_fprs: List[float] = [0.01, 0.05, 0.10],
    pos_label: int = config.FAKE_LABEL_IDX
) -> Dict[str, float]:
    """
    Computes the True Positive Rate (Recall) at fixed False Positive Rates.
    
    Args:
        y_true: Ground truth labels (e.g. 0 for fake, 1 for real)
        y_score: Probability score of the positive class (fake)
        target_fprs: List of target FPRs (e.g. [0.01, 0.05, 0.10])
        pos_label: Label index considered as positive (default: 0 for fake)
        
    Returns:
        Dict mapping metric name (e.g. 'recall_at_1%_fpr') to float score.
    """
    # sklearn roc_curve returns fpr, tpr, thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=pos_label)
    
    recalls = {}
    for target in target_fprs:
        key = f"recall_at_{int(target * 100)}%_fpr"
        # Find index where fpr <= target (highest TPR within target FPR allowance)
        valid_indices = np.where(fpr <= target)[0]
        if len(valid_indices) == 0:
            recalls[key] = 0.0
            recalls[f"{key}_threshold"] = 1.0
        else:
            best_idx = valid_indices[-1]
            recalls[key] = float(tpr[best_idx])
            thresh_val = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
            recalls[f"{key}_threshold"] = thresh_val
            
    return recalls


def aggregate_predictions_by_video(
    df: pd.DataFrame,
    prob_cols: Optional[List[str]] = None,
    video_id_col: str = "video_id",
    ground_truth_col: str = "ground_truth",
    method: str = "mean"
) -> pd.DataFrame:
    """
    Aggregates frame-level predictions to video-level predictions.
    
    Args:
        df: DataFrame containing frame predictions
        prob_cols: List of column names containing probability of fake
        video_id_col: Column name containing video ID (if missing, will be extracted)
        ground_truth_col: Column with ground truth label
        method: Aggregation method: 'mean', 'max', 'median', or 'trimmed_mean'
        
    Returns:
        DataFrame with one row per video.
    """
    df = df.copy()
    if video_id_col not in df.columns:
        if "filename" in df.columns:
            df[video_id_col] = df["filename"].apply(extract_video_id)
        elif "filepath" in df.columns:
            df[video_id_col] = df["filepath"].apply(extract_video_id)
        elif "image_path" in df.columns:
            df[video_id_col] = df["image_path"].apply(extract_video_id)
        else:
            raise KeyError("Could not find video_id, filename, filepath, or image_path column.")

    if prob_cols is None:
        # Auto-detect probability columns
        prob_cols = [c for c in df.columns if "prob_fake" in c or c.endswith("_prob") or c == "prob_fake"]
        if not prob_cols and "fake_probability" in df.columns:
            prob_cols = ["fake_probability"]

    def _agg_func(series):
        if method == "mean":
            return series.mean()
        elif method == "max":
            return series.max()
        elif method == "median":
            return series.median()
        elif method == "trimmed_mean":
            # Trim top/bottom 10% if at least 5 frames
            vals = np.sort(series.values)
            if len(vals) >= 5:
                k = int(len(vals) * 0.1)
                return vals[k:-k].mean()
            return series.mean()
        else:
            return series.mean()

    # Majority vote / first for ground truth
    agg_dict = {ground_truth_col: lambda s: s.mode()[0] if not s.mode().empty else s.iloc[0]}
    for col in prob_cols:
        agg_dict[col] = _agg_func

    # Track frame count per video
    df["frame_count"] = 1
    agg_dict["frame_count"] = "count"

    video_df = df.groupby(video_id_col).agg(agg_dict).reset_index()
    
    # Re-infer predicted labels at video level (threshold = 0.5)
    for col in prob_cols:
        pred_col = col.replace("prob_fake", "pred").replace("_prob", "_pred")
        if pred_col == col:
            pred_col = f"{col}_pred"
        video_df[pred_col] = np.where(video_df[col] >= 0.5, config.FAKE_LABEL_IDX, config.REAL_LABEL_IDX)

    return video_df


MIN_CLUSTERS_FOR_BOOTSTRAP = 20


def bootstrap_confidence_intervals(
    video_df: pd.DataFrame,
    prob_col: str,
    ground_truth_col: str = "ground_truth",
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
    target_fprs: List[float] = [0.01, 0.05, 0.10],
    threshold: float = 0.5,
    min_clusters: int = MIN_CLUSTERS_FOR_BOOTSTRAP
) -> Dict[str, Dict[str, float]]:
    """
    Performs cluster bootstrap (resampling videos with replacement) to estimate
    unbiased confidence intervals for all primary metrics.
    
    Safety Guard:
        Refuses to compute bootstrap CIs if video_df contains fewer than min_clusters (default: 20)
        to prevent degenerate intervals and spurious 1.0000 AUC claims.

    Returns:
        Dict of metric_name -> {'mean': float, 'std': float, 'ci_lower': float, 'ci_upper': float}
    """
    n_videos = len(video_df)
    if n_videos < min_clusters:
        raise ValueError(
            f"Safety Guard Activated: Cluster bootstrap CI refuses to run with fewer than {min_clusters} "
            f"video clusters (received only {n_videos}). Resampling on small cohorts produces "
            f"degenerate intervals (e.g. [50%, 100%]) and uninformative findings."
        )

    rng = np.random.RandomState(seed)

    y_true_all = video_df[ground_truth_col].values.astype(int)
    y_probs_all = video_df[prob_col].values.astype(float)

    bootstrap_metrics = {
        "accuracy": [],
        "f1_score": [],
        "precision": [],
        "recall": [],
        "roc_auc": [],
        "brier_score": []
    }
    for fpr_val in target_fprs:
        bootstrap_metrics[f"recall_at_{int(fpr_val*100)}%_fpr"] = []

    for _ in range(n_bootstraps):
        boot_idx = rng.choice(n_videos, size=n_videos, replace=True)
        y_b_true = y_true_all[boot_idx]
        y_b_probs = y_probs_all[boot_idx]

        # Require both classes to exist in the bootstrap sample for valid ROC/Recall
        unique_classes = np.unique(y_b_true)
        if len(unique_classes) < 2:
            continue

        y_b_pred = np.where(y_b_probs >= threshold, config.FAKE_LABEL_IDX, config.REAL_LABEL_IDX)
        acc = accuracy_score(y_b_true, y_b_pred)
        p, r, f1, _ = precision_recall_fscore_support(
            y_b_true, y_b_pred, average="binary", pos_label=config.FAKE_LABEL_IDX, zero_division=0
        )
        
        try:
            fpr, tpr, _ = roc_curve(y_b_true, y_b_probs, pos_label=config.FAKE_LABEL_IDX)
            roc_auc_val = auc(fpr, tpr)
        except Exception:
            roc_auc_val = 0.5

        is_fake = (y_b_true == config.FAKE_LABEL_IDX).astype(float)
        brier = brier_score_loss(is_fake, y_b_probs)

        bootstrap_metrics["accuracy"].append(acc)
        bootstrap_metrics["precision"].append(p)
        bootstrap_metrics["recall"].append(r)
        bootstrap_metrics["f1_score"].append(f1)
        bootstrap_metrics["roc_auc"].append(roc_auc_val)
        bootstrap_metrics["brier_score"].append(brier)

        # Fixed FPR recalls
        rec_fixed = compute_recall_at_fixed_fpr(y_b_true, y_b_probs, target_fprs=target_fprs)
        for fpr_k, val in rec_fixed.items():
            if not fpr_k.endswith("_threshold"):
                bootstrap_metrics[fpr_k].append(val)

    alpha = 1.0 - ci_level
    lower_pct = 100 * (alpha / 2.0)
    upper_pct = 100 * (1.0 - alpha / 2.0)

    summary = {}
    for metric_name, values in bootstrap_metrics.items():
        if len(values) == 0:
            continue
        arr = np.array(values)
        summary[metric_name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_lower": float(np.percentile(arr, lower_pct)),
            "ci_upper": float(np.percentile(arr, upper_pct))
        }

    return summary
