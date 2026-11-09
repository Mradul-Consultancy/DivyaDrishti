"""
Evaluation and Soft-Voting Ensemble Pipeline for Deepfake Detection.

Evaluates:
1. YOLO26n-cls (Ultralytics)
2. ConvNeXt-Tiny (timm)
3. EfficientNet-B3 (timm)
4. Soft-Voting Ensemble (weighted probability average of the three models)

Metrics Computed:
- Precision, Recall, F1-Score (Fake class & Real class)
- Accuracy
- ROC-AUC
- Confusion Matrix

Visual & Structured Reports Saved:
- reports/ensemble/ensemble_roc_curves.png (Combined multi-model ROC plot)
- reports/ensemble/ensemble_confusion_matrices.png (2x2 confusion matrix grid)
- reports/ensemble/ensemble_metrics.json (Machine-readable metrics)
- reports/ensemble/ensemble_report.md (Markdown comparison table)
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report
)
import torch
import torch.nn as nn
import timm
from timm.data import resolve_data_config, create_transform

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from utils.helpers import setup_logger, ensure_dirs


class ModelEnsembleEvaluator:
    """
    Loads YOLO, ConvNeXt-Tiny, and EfficientNet-B3 models,
    runs inference, computes individual & soft-voting ensemble metrics,
    and exports comprehensive visual reports.
    """

    def __init__(
        self,
        yolo_checkpoint: Path | str,
        convnext_checkpoint: Path | str,
        effnet_checkpoint: Path | str,
        convnext_model_name: str = "convnext_tiny.fb_in22k_ft_in1k",
        effnet_model_name: str = "efficientnet_b3.ra2_in1k",
        ensemble_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        device: Optional[torch.device] = None,
        logger=None
    ):
        self.device = device or config.DEVICE
        self.logger = logger or setup_logger("EnsembleEvaluator")
        self.ensemble_weights = np.array(ensemble_weights, dtype=np.float32)
        self.ensemble_weights /= self.ensemble_weights.sum()

        self.logger.info("Initializing models for evaluation...")
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Ensemble weights [YOLO, ConvNeXt, EfficientNet]: {self.ensemble_weights.tolist()}")

        # 1. Load YOLO model
        self.logger.info(f"Loading YOLO checkpoint from {yolo_checkpoint}...")
        from ultralytics import YOLO
        self.yolo_model = YOLO(str(yolo_checkpoint))
        # Determine index of 'fake' and 'real' in YOLO names
        self.yolo_names = self.yolo_model.names
        self.yolo_fake_idx = [k for k, v in self.yolo_names.items() if v.lower() == "fake"][0]
        self.yolo_real_idx = [k for k, v in self.yolo_names.items() if v.lower() == "real"][0]

        # PyTorch ImageFolder and timm convention: alphabetical order ('fake' < 'real')
        self.timm_fake_idx = config.FAKE_LABEL_IDX  # 0
        self.timm_real_idx = config.REAL_LABEL_IDX  # 1

        # Strict agreement assertion
        if self.yolo_fake_idx != self.timm_fake_idx or self.yolo_real_idx != self.timm_real_idx:
            raise ValueError(
                f"Label mapping mismatch! YOLO mapping: fake={self.yolo_fake_idx}, real={self.yolo_real_idx} | "
                f"timm/ImageFolder mapping: fake={self.timm_fake_idx}, real={self.timm_real_idx}. "
                f"Both backends must strictly agree on fake=0, real=1."
            )
        self.logger.info(f"Class mapping verified: YOLO and timm both agree on fake={self.timm_fake_idx}, real={self.timm_real_idx}")

        # 2. Load ConvNeXt-Tiny
        self.logger.info(f"Loading ConvNeXt-Tiny from {convnext_checkpoint}...")
        self.convnext_model = timm.create_model(convnext_model_name, pretrained=False, num_classes=2)
        ckpt_cn = torch.load(convnext_checkpoint, map_location=self.device)
        self.convnext_model.load_state_dict(ckpt_cn.get("state_dict", ckpt_cn))
        self.convnext_model = self.convnext_model.to(self.device).eval()
        cn_cfg = resolve_data_config(self.convnext_model.pretrained_cfg)
        self.convnext_transform = create_transform(**cn_cfg, is_training=False)

        # 3. Load EfficientNet-B3
        self.logger.info(f"Loading EfficientNet-B3 from {effnet_checkpoint}...")
        self.effnet_model = timm.create_model(effnet_model_name, pretrained=False, num_classes=2)
        ckpt_eff = torch.load(effnet_checkpoint, map_location=self.device)
        self.effnet_model.load_state_dict(ckpt_eff.get("state_dict", ckpt_eff))
        self.effnet_model = self.effnet_model.to(self.device).eval()
        eff_cfg = resolve_data_config(self.effnet_model.pretrained_cfg)
        self.effnet_transform = create_transform(**eff_cfg, is_training=False)

        self.softmax = nn.Softmax(dim=1)

    def set_weights(self, weights: Tuple[float, float, float] | List[float] | np.ndarray):
        """Sets and normalizes ensemble weights."""
        w = np.array(weights, dtype=np.float32)
        if w.sum() == 0:
            raise ValueError("Ensemble weights cannot sum to zero.")
        self.ensemble_weights = w / w.sum()
        self.logger.info(f"Updated ensemble weights to: {self.ensemble_weights.tolist()}")

    def load_weights(self, weights_path: Path | str):
        """Loads ensemble weights from a JSON file."""
        weights_path = Path(weights_path)
        with open(weights_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        weights = data.get("weights") or data.get("ensemble_weights")
        if not weights or len(weights) != 3:
            raise ValueError(f"Invalid weights format in {weights_path}: {data}")
        self.set_weights(weights)
        self.logger.info(f"Loaded ensemble weights from {weights_path}: {self.ensemble_weights.tolist()}")

    @staticmethod
    def generate_simplex_grid(step: float = 0.02) -> np.ndarray:
        """
        Generates all 3D weight vectors on the unit simplex:
        w1 + w2 + w3 = 1.0, w_i >= 0.
        """
        steps = int(round(1.0 / step))
        weights_list = []
        for i in range(steps + 1):
            for j in range(steps + 1 - i):
                k = steps - i - j
                weights_list.append((i / steps, j / steps, k / steps))
        return np.array(weights_list, dtype=np.float32)

    def collect_dataset_probabilities(
        self,
        dataset_dir: Path | str
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Runs inference over all images in dataset_dir across real/ and fake/ subfolders.
        Returns:
            y_true: (N,) binary ground truth labels (0=real, 1=fake)
            probs_matrix: (N, 3) matrix of fake probabilities [YOLO, ConvNeXt, EfficientNet]
            samples_info: List of dicts with filepath, filename, label
        """
        dataset_dir = Path(dataset_dir)
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        samples: List[Tuple[Path, int]] = []
        for class_name, label in [("fake", config.FAKE_LABEL_IDX), ("real", config.REAL_LABEL_IDX)]:
            folder = dataset_dir / class_name
            if folder.exists():
                for p in folder.glob("*"):
                    if p.suffix.lower() in valid_exts:
                        samples.append((p, label))

        if len(samples) == 0:
            raise ValueError(f"No valid images found in {dataset_dir} (expected 'real' and 'fake' subfolders).")

        self.logger.info(f"Extracting model predictions for {len(samples)} samples from {dataset_dir}...")

        y_true = []
        raw_probs = []
        samples_info = []

        for idx, (img_path, label) in enumerate(samples, start=1):
            probs = self.predict_image(img_path)
            y_true.append(label)
            raw_probs.append([
                probs["yolo26n_cls"],
                probs["convnext_tiny"],
                probs["efficientnet_b3"]
            ])
            samples_info.append({
                "path": str(img_path),
                "filename": img_path.name,
                "label": int(label)
            })

            if idx % 10 == 0 or idx == len(samples):
                self.logger.info(f"Inference progress: [{idx}/{len(samples)}]")

        return np.array(y_true, dtype=int), np.array(raw_probs, dtype=np.float32), samples_info

    def tune_weights_on_validation(
        self,
        val_dir: Path | str,
        metric: str = "roc_auc",
        step: float = 0.02,
        threshold: float = 0.5,
        save_path: Optional[Path | str] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Tunes ensemble weights STRICTLY on the validation set to eliminate data leakage into the test set.

        Optimization Protocol:
        1. Forward-passes all models across the validation set only (test set is unobserved).
        2. Evaluates candidate weight vectors over the unit simplex (w1 + w2 + w3 = 1.0, w_i >= 0).
        3. Optimizes target metric: 'roc_auc', 'f1', 'accuracy', 'log_loss', or 'brier'.
        4. Regularizes metric ties by selecting the candidate closest to the uniform prior [1/3, 1/3, 1/3].
        5. Updates self.ensemble_weights and exports tuning audit artifact.
        """
        self.logger.info(f"Starting ensemble weight tuning STRICTLY on validation set: {val_dir}")
        self.logger.info(f"Target optimization metric: {metric} | Grid simplex step: {step}")

        y_val, p_val, _ = self.collect_dataset_probabilities(val_dir)
        grid = self.generate_simplex_grid(step=step)
        num_candidates = len(grid)

        # Ensemble probabilities for all candidates: shape (N_val, num_candidates)
        # p_val is (N_val, 3), grid is (num_candidates, 3)
        p_ens_all = np.matmul(p_val, grid.T)

        scores = np.zeros(num_candidates, dtype=np.float32)
        metric_lower = metric.lower()

        if metric_lower == "roc_auc":
            for j in range(num_candidates):
                try:
                    fpr, tpr, _ = roc_curve(y_val, p_ens_all[:, j], pos_label=config.FAKE_LABEL_IDX)
                    scores[j] = auc(fpr, tpr)
                except Exception:
                    scores[j] = 0.5
        elif metric_lower == "f1":
            preds_all = np.where(p_ens_all >= threshold, config.FAKE_LABEL_IDX, config.REAL_LABEL_IDX)
            for j in range(num_candidates):
                _, _, f1, _ = precision_recall_fscore_support(
                    y_val, preds_all[:, j], average="binary", pos_label=config.FAKE_LABEL_IDX, zero_division=0
                )
                scores[j] = f1
        elif metric_lower == "accuracy":
            preds_all = np.where(p_ens_all >= threshold, config.FAKE_LABEL_IDX, config.REAL_LABEL_IDX)
            y_col = y_val[:, None]
            scores = np.mean(preds_all == y_col, axis=0)
        elif metric_lower == "log_loss":
            eps = 1e-7
            p_clipped = np.clip(p_ens_all, eps, 1.0 - eps)
            is_fake_col = (y_val == config.FAKE_LABEL_IDX).astype(float)[:, None]
            sample_losses = -(is_fake_col * np.log(p_clipped) + (1.0 - is_fake_col) * np.log(1.0 - p_clipped))
            scores = -np.mean(sample_losses, axis=0)  # Negative loss for maximization
        elif metric_lower == "brier":
            is_fake_col = (y_val == config.FAKE_LABEL_IDX).astype(float)[:, None]
            sample_brier = (p_ens_all - is_fake_col) ** 2
            scores = -np.mean(sample_brier, axis=0)  # Negative brier for maximization
        else:
            raise ValueError(
                f"Unsupported tuning metric: '{metric}'. "
                f"Choose from ['roc_auc', 'f1', 'accuracy', 'log_loss', 'brier']"
            )

        # Uniform baseline score for comparison
        uniform_w = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)
        p_uniform = np.dot(p_val, uniform_w)
        if metric_lower == "roc_auc":
            fpr_u, tpr_u, _ = roc_curve(y_val, p_uniform, pos_label=config.FAKE_LABEL_IDX)
            uniform_score = float(auc(fpr_u, tpr_u))
        elif metric_lower == "f1":
            preds_u = np.where(p_uniform >= threshold, config.FAKE_LABEL_IDX, config.REAL_LABEL_IDX)
            _, _, f1_u, _ = precision_recall_fscore_support(
                y_val, preds_u, average="binary", pos_label=config.FAKE_LABEL_IDX, zero_division=0
            )
            uniform_score = float(f1_u)
        elif metric_lower == "accuracy":
            preds_u = np.where(p_uniform >= threshold, config.FAKE_LABEL_IDX, config.REAL_LABEL_IDX)
            uniform_score = float(np.mean(preds_u == y_val))
        elif metric_lower == "log_loss":
            eps = 1e-7
            p_u_c = np.clip(p_uniform, eps, 1.0 - eps)
            is_fake_u = (y_val == config.FAKE_LABEL_IDX).astype(float)
            uniform_score = float(np.mean(-(is_fake_u * np.log(p_u_c) + (1.0 - is_fake_u) * np.log(1.0 - p_u_c))))
        elif metric_lower == "brier":
            is_fake_u = (y_val == config.FAKE_LABEL_IDX).astype(float)
            uniform_score = float(np.mean((p_uniform - is_fake_u) ** 2))

        # Find best candidate with regularized tie-breaking towards uniform prior
        max_score = float(np.max(scores))
        candidate_indices = np.where(np.isclose(scores, max_score, atol=1e-5))[0]
        distances = np.linalg.norm(grid[candidate_indices] - uniform_w, axis=1)
        best_idx = candidate_indices[np.argmin(distances)]
        optimal_weights = grid[best_idx]

        # Update evaluator weights
        self.ensemble_weights = optimal_weights.astype(np.float32)

        summary = {
            "weights": [round(float(w), 4) for w in optimal_weights],
            "weights_dict": {
                "yolo26n_cls": round(float(optimal_weights[0]), 4),
                "convnext_tiny": round(float(optimal_weights[1]), 4),
                "efficientnet_b3": round(float(optimal_weights[2]), 4)
            },
            "metric_optimized": metric,
            "val_score_before_tuning_uniform": round(uniform_score, 4),
            "val_score_after_tuning": round(
                float(max_score) if metric_lower not in ["log_loss", "brier"] else -float(max_score), 4
            ),
            "tuning_dataset": str(val_dir),
            "num_validation_samples": int(len(y_val)),
            "grid_candidates_evaluated": int(num_candidates),
            "data_leakage_prevented": True,
            "protocol_note": "Tuned strictly on validation set. Test set remained completely unobserved during weight selection."
        }

        self.logger.info("Validation weight tuning complete:")
        self.logger.info(f"  Optimal Weights [YOLO, ConvNeXt, EfficientNet]: {summary['weights']}")
        self.logger.info(f"  Val {metric} Uniform: {summary['val_score_before_tuning_uniform']:.4f} -> Tuned: {summary['val_score_after_tuning']:.4f}")

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=4)
            self.logger.info(f"Saved tuned ensemble weights audit to: {save_path}")

        return optimal_weights, summary

    def predict_image(self, img_input: Path | str | Image.Image) -> Dict[str, float]:
        """
        Runs inference across all 3 models on a single image and computes
        individual and soft-voting ensemble fake probabilities.
        """
        if isinstance(img_input, (str, Path)):
            pil_img = Image.open(img_input).convert("RGB")
        elif isinstance(img_input, Image.Image):
            pil_img = img_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(img_input)}")

        # 1. YOLO inference
        yolo_res = self.yolo_model(pil_img, verbose=False)
        yolo_probs = yolo_res[0].probs.data
        p_fake_yolo = float(yolo_probs[self.yolo_fake_idx])

        # 2. ConvNeXt inference
        cn_tensor = self.convnext_transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            cn_out = self.convnext_model(cn_tensor)
            cn_probs = self.softmax(cn_out).squeeze(0)
            # PyTorch ImageFolder and timm convention: index 0 is fake, index 1 is real
            p_fake_convnext = float(cn_probs[self.timm_fake_idx].item())

        # 3. EfficientNet inference
        eff_tensor = self.effnet_transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            eff_out = self.effnet_model(eff_tensor)
            eff_probs = self.softmax(eff_out).squeeze(0)
            p_fake_effnet = float(eff_probs[self.timm_fake_idx].item())

        # 4. Soft-Voting Ensemble (weighted probability average)
        w_yolo, w_cn, w_eff = self.ensemble_weights
        p_fake_ensemble = (w_yolo * p_fake_yolo) + (w_cn * p_fake_convnext) + (w_eff * p_fake_effnet)

        return {
            "yolo26n_cls": p_fake_yolo,
            "convnext_tiny": p_fake_convnext,
            "efficientnet_b3": p_fake_effnet,
            "ensemble": p_fake_ensemble
        }

    def evaluate_test_set(
        self,
        test_dir: Path | str,
        threshold: float = 0.5
    ) -> Dict[str, Dict]:
        """
        Evaluates test dataset across fake/ and real/ subfolders.
        Computes precision, recall, f1, roc-auc, and confusion matrix for each model & ensemble.
        """
        y_true, raw_probs, samples_info = self.collect_dataset_probabilities(test_dir)
        self.logger.info(f"Computing evaluation metrics for {len(y_true)} test samples...")

        # Soft-voting ensemble with current weights
        w_yolo, w_cn, w_eff = self.ensemble_weights
        p_ensemble = (w_yolo * raw_probs[:, 0]) + (w_cn * raw_probs[:, 1]) + (w_eff * raw_probs[:, 2])

        model_probs = {
            "yolo26n_cls": raw_probs[:, 0],
            "convnext_tiny": raw_probs[:, 1],
            "efficientnet_b3": raw_probs[:, 2],
            "ensemble": p_ensemble
        }

        results = {}
        for m_key, probs_arr in model_probs.items():
            # If fake probability >= threshold -> FAKE (0), else REAL (1)
            preds_arr = np.where(probs_arr >= threshold, config.FAKE_LABEL_IDX, config.REAL_LABEL_IDX)

            acc = accuracy_score(y_true, preds_arr)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, preds_arr, average="binary", pos_label=config.FAKE_LABEL_IDX, zero_division=0
            )

            # ROC & AUC with pos_label=0 (fake class)
            fpr, tpr, _ = roc_curve(y_true, probs_arr, pos_label=config.FAKE_LABEL_IDX)
            roc_auc_val = auc(fpr, tpr)
            cm = confusion_matrix(y_true, preds_arr, labels=[config.FAKE_LABEL_IDX, config.REAL_LABEL_IDX])

            # Calibration: Brier Score (Mean Squared Error of predicted probability vs binary truth)
            # Lower is better (0.0 = perfect calibration, 0.25 = uninformative coin toss)
            is_fake_binary = (y_true == config.FAKE_LABEL_IDX).astype(float)
            brier_val = float(np.mean((probs_arr - is_fake_binary) ** 2))

            results[m_key] = {
                "accuracy": float(acc),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "roc_auc": float(roc_auc_val),
                "brier_score": float(brier_val),
                "confusion_matrix": cm.tolist(),
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "probabilities": probs_arr.tolist(),
                "predictions": preds_arr.tolist()
            }

        results["_ground_truth"] = y_true.tolist()
        results["_samples"] = samples_info
        results["_weights"] = [float(w) for w in self.ensemble_weights]
        return results


def plot_combined_roc_curves(results: Dict[str, Dict], save_path: Path):
    """
    Plots ROC curves for YOLO26n-cls, ConvNeXt-Tiny, EfficientNet-B3,
    and Soft-Voting Ensemble on a single high-resolution graph.
    """
    palette = {
        "yolo26n_cls": ("#1f77b4", "YOLO26n-cls", "-."),
        "convnext_tiny": ("#2ca02c", "ConvNeXt-Tiny", ":"),
        "efficientnet_b3": ("#9467bd", "EfficientNet-B3", "--"),
        "ensemble": ("#d62728", "Soft-Voting Ensemble", "-")
    }

    plt.figure(figsize=(8, 7))

    for m_key, (color, label, linestyle) in palette.items():
        if m_key not in results:
            continue
        fpr = np.array(results[m_key]["fpr"])
        tpr = np.array(results[m_key]["tpr"])
        auc_score = results[m_key]["roc_auc"]
        lw = 3.0 if m_key == "ensemble" else 1.8
        plt.plot(fpr, tpr, color=color, lw=lw, linestyle=linestyle,
                 label=f"{label} (AUC = {auc_score:.4f})")

    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)", fontweight="bold", fontsize=11)
    plt.ylabel("True Positive Rate (TPR / Recall)", fontweight="bold", fontsize=11)
    plt.title("Multi-Model ROC Curves & Soft-Voting Ensemble", fontweight="bold", fontsize=13, pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_combined_confusion_matrices(results: Dict[str, Dict], save_path: Path):
    """
    Plots a 2x2 grid displaying confusion matrices for all three individual
    models and the soft-voting ensemble.
    """
    models_order = [
        ("yolo26n_cls", "YOLO26n-cls"),
        ("convnext_tiny", "ConvNeXt-Tiny"),
        ("efficientnet_b3", "EfficientNet-B3"),
        ("ensemble", "Soft-Voting Ensemble (All 3)")
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.flatten()

    for idx, (m_key, title) in enumerate(models_order):
        ax = axes[idx]
        cm = np.array(results[m_key]["confusion_matrix"])
        cmap = plt.cm.Reds if m_key == "ensemble" else plt.cm.Blues

        cax = ax.matshow(cm, cmap=cmap, alpha=0.85)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                ax.text(
                    x=j, y=i, s=f"{val}",
                    va="center", ha="center",
                    size=14, weight="bold",
                    color="white" if val > cm.max() / 2 else "black"
                )

        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["FAKE", "REAL"], fontweight="bold")
        ax.set_yticklabels(["FAKE", "REAL"], fontweight="bold")
        ax.set_xlabel("Predicted Label", fontweight="bold", labelpad=8)
        ax.set_ylabel("True Label", fontweight="bold", labelpad=8)
        acc = results[m_key]["accuracy"] * 100
        f1 = results[m_key]["f1_score"] * 100
        ax.set_title(f"{title}\n(Acc: {acc:.1f}% | F1: {f1:.1f}%)", fontweight="bold", pad=12)

    plt.suptitle("Confusion Matrix Comparison: Individual Models vs Ensemble", fontweight="bold", fontsize=14, y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def generate_ensemble_markdown_report(
    results: Dict[str, Dict],
    total_samples: int,
    save_path: Path,
    weights_summary: Optional[Dict] = None
):
    """Generates an executive markdown comparison report."""
    display_names = {
        "yolo26n_cls": "YOLO26n-cls (Ultralytics)",
        "convnext_tiny": "ConvNeXt-Tiny (timm)",
        "efficientnet_b3": "EfficientNet-B3 (timm)",
        "ensemble": "Soft-Voting Ensemble (All 3)"
    }

    current_weights = results.get("_weights", [1/3, 1/3, 1/3])
    weights_str = f"[{current_weights[0]:.4f}, {current_weights[1]:.4f}, {current_weights[2]:.4f}]"

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("# Deepfake Detection Ensemble Evaluation Report\n\n")
        f.write(f"- **Total Test Samples**: `{total_samples}`\n")
        f.write(f"- **Ensemble Strategy**: Soft-Voting (Weighted Average of Class Probabilities)\n")
        f.write(f"- **Ensemble Weights [YOLO, ConvNeXt, EfficientNet]**: `{weights_str}`\n")

        if weights_summary:
            f.write(f"- **Weight Tuning Protocol**: STRICTLY on Validation Split (`{weights_summary.get('tuning_dataset')}`)\n")
            f.write(f"  - Target Metric: `{weights_summary.get('metric_optimized')}`\n")
            f.write(f"  - Val Score (Uniform Baseline): `{weights_summary.get('val_score_before_tuning_uniform')}`\n")
            f.write(f"  - Val Score (After Tuning): `{weights_summary.get('val_score_after_tuning')}`\n")
            f.write(f"  - Validation Samples: `{weights_summary.get('num_validation_samples')}`\n")
            f.write(f"  - Data Leakage Prevention: Guaranteed zero test set leakage (`test/` split remained completely unobserved)\n\n")
        else:
            f.write(f"- **Weight Tuning Protocol**: Equal Uniform Baseline `[1/3, 1/3, 1/3]` (No tuning applied; zero test leakage)\n\n")

        f.write(f"- **Evaluated Checkpoints**:\n")
        f.write(f"  - YOLO26n-cls: `runs/yolo_cls/deepfake_yolo/weights/best.pt`\n")
        f.write(f"  - ConvNeXt-Tiny: `models/checkpoints/convnext_tiny.fb_in22k_ft_in1k_best.pth`\n")
        f.write(f"  - EfficientNet-B3: `models/checkpoints/efficientnet_b3.ra2_in1k_best.pth`\n\n")

        f.write("## Performance & Calibration Comparison Table\n\n")
        f.write("| Model | Accuracy | Precision (Fake) | Recall (Fake) | F1-Score | ROC-AUC | Brier Score (Calib) |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|\n")

        for key in ["yolo26n_cls", "convnext_tiny", "efficientnet_b3", "ensemble"]:
            m = results[key]
            name = display_names[key]
            bold = "**" if key == "ensemble" else ""
            f.write(
                f"| {bold}{name}{bold} | "
                f"{bold}{m['accuracy']*100:.2f}%{bold} | "
                f"{bold}{m['precision']*100:.2f}%{bold} | "
                f"{bold}{m['recall']*100:.2f}%{bold} | "
                f"{bold}{m['f1_score']*100:.2f}%{bold} | "
                f"{bold}{m['roc_auc']:.4f}{bold} | "
                f"{bold}{m['brier_score']:.4f}{bold} |\n"
            )

        f.write("\n> [!NOTE]\n")
        f.write("> **Brier Score (Probability Calibration Loss)**: Measures mean squared probability error $BS = \\frac{1}{N} \\sum_{i=1}^N (p_i - y_i)^2$.\n")
        f.write("> Lower values denote superior calibration ($0.0$ is perfect calibration; $0.25$ is an uninformative coin flip on balanced data).\n\n")

        f.write("\n## Combined ROC Curves\n\n")
        f.write("![ROC Curves](ensemble_roc_curves.png)\n\n")

        f.write("## Confusion Matrices (Individual vs Ensemble)\n\n")
        f.write("![Confusion Matrices](ensemble_confusion_matrices.png)\n\n")

        f.write("## Individual Confusion Matrices Breakdown\n\n")
        for key in ["yolo26n_cls", "convnext_tiny", "efficientnet_b3", "ensemble"]:
            cm = results[key]["confusion_matrix"]
            name = display_names[key]
            f.write(f"### {name}\n\n")
            f.write(f"- **True Fake (TP)**: `{cm[0][0]}` | **False Real (FN)**: `{cm[0][1]}`\n")
            f.write(f"- **False Fake (FP)**: `{cm[1][0]}` | **True Real (TN)**: `{cm[1][1]}`\n\n")


def print_ascii_summary(results: Dict[str, Dict]):
    """Prints a formatted console table comparing all models."""
    display_names = {
        "yolo26n_cls": "YOLO26n-cls",
        "convnext_tiny": "ConvNeXt-Tiny",
        "efficientnet_b3": "EfficientNet-B3",
        "ensemble": "Soft-Voting Ensemble"
    }

    print("\n" + "=" * 95)
    print("           DEEPFAKE DETECTION: MULTI-MODEL & ENSEMBLE EVALUATION")
    print("=" * 95)
    print(f"{'Model':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<8} | {'Brier (Calib)':<12}")
    print("-" * 95)

    for key in ["yolo26n_cls", "convnext_tiny", "efficientnet_b3", "ensemble"]:
        m = results[key]
        tag = display_names[key]
        print(
            f"{tag:<25} | "
            f"{m['accuracy']*100:8.2f}% | "
            f"{m['precision']*100:8.2f}% | "
            f"{m['recall']*100:8.2f}% | "
            f"{m['f1_score']*100:8.2f}% | "
            f"{m['roc_auc']:8.4f} | "
            f"{m['brier_score']:12.4f}"
        )
    print("=" * 95 + "\n")


def run_ensemble_evaluation(args):
    ensure_dirs(args.reports_dir)
    log_path = Path(args.reports_dir) / "ensemble_eval.log"
    logger = setup_logger("EnsembleEvaluator", log_file=log_path)

    evaluator = ModelEnsembleEvaluator(
        yolo_checkpoint=args.yolo_checkpoint,
        convnext_checkpoint=args.convnext_checkpoint,
        effnet_checkpoint=args.effnet_checkpoint,
        convnext_model_name=getattr(args, "convnext_model", "convnext_tiny.fb_in22k_ft_in1k"),
        effnet_model_name=getattr(args, "effnet_model", "efficientnet_b3.ra2_in1k"),
        ensemble_weights=tuple(args.weights),
        device=torch.device(args.device),
        logger=logger
    )

    weights_summary = None

    # Step A: Load or Tune Ensemble Weights (strictly on validation set)
    if getattr(args, "weights_file", None):
        evaluator.load_weights(args.weights_file)
    elif getattr(args, "tune_on_val", False):
        val_dir = Path(args.val_dir)
        weights_save_path = Path(args.reports_dir) / "ensemble_weights.json"
        _, weights_summary = evaluator.tune_weights_on_validation(
            val_dir=val_dir,
            metric=getattr(args, "tune_metric", "roc_auc"),
            threshold=args.threshold,
            save_path=weights_save_path
        )
        # Also persist to checkpoints directory for inference loading
        ckpt_weights_path = Path(config.CHECKPOINT_DIR) / "ensemble_weights.json"
        ensure_dirs(ckpt_weights_path.parent)
        with open(ckpt_weights_path, "w", encoding="utf-8") as f:
            json.dump(weights_summary, f, indent=4)
        logger.info(f"Saved deployment ensemble weights to {ckpt_weights_path}")

    # Mode 1: Single image inference
    if args.image:
        logger.info(f"Running ensemble inference on single image: {args.image}")
        res = evaluator.predict_image(Path(args.image))
        print("\n" + "=" * 60)
        print(f"Prediction for: {Path(args.image).name}")
        print("=" * 60)
        print(f"  YOLO26n-cls:     Fake Prob: {res['yolo26n_cls']*100:5.1f}%")
        print(f"  ConvNeXt-Tiny:   Fake Prob: {res['convnext_tiny']*100:5.1f}%")
        print(f"  EfficientNet-B3: Fake Prob: {res['efficientnet_b3']*100:5.1f}%")
        ens_pred = "FAKE" if res["ensemble"] >= args.threshold else "REAL"
        confidence = res["ensemble"] if ens_pred == "FAKE" else 1.0 - res["ensemble"]
        print("-" * 60)
        print(f"  Ensemble Weights: [YOLO={evaluator.ensemble_weights[0]:.3f}, ConvNeXt={evaluator.ensemble_weights[1]:.3f}, EffNet={evaluator.ensemble_weights[2]:.3f}]")
        print(f"  --> ENSEMBLE DECISION: [{ens_pred}] (Confidence: {confidence*100:.1f}%)")
        print("=" * 60 + "\n")
        return

    # Mode 2: Full test set evaluation
    if getattr(args, "dataset", None) and args.dataset in config.DATASETS:
        args.data_dir = str(config.DATASETS[args.dataset])
    test_dir = Path(args.data_dir) / "test" if (Path(args.data_dir) / "test").exists() else Path(args.data_dir)
    results = evaluator.evaluate_test_set(test_dir, threshold=args.threshold)

    # Save visual outputs
    roc_plot_path = Path(args.reports_dir) / "ensemble_roc_curves.png"
    plot_combined_roc_curves(results, roc_plot_path)
    logger.info(f"Saved combined ROC plot: {roc_plot_path}")

    cm_plot_path = Path(args.reports_dir) / "ensemble_confusion_matrices.png"
    plot_combined_confusion_matrices(results, cm_plot_path)
    logger.info(f"Saved 2x2 confusion matrix plot: {cm_plot_path}")

    # Save metrics JSON (clean up fpr/tpr lists for readability)
    json_summary = {}
    for k in ["yolo26n_cls", "convnext_tiny", "efficientnet_b3", "ensemble"]:
        json_summary[k] = {
            "accuracy": results[k]["accuracy"],
            "precision": results[k]["precision"],
            "recall": results[k]["recall"],
            "f1_score": results[k]["f1_score"],
            "roc_auc": results[k]["roc_auc"],
            "brier_score": results[k]["brier_score"],
            "confusion_matrix": results[k]["confusion_matrix"]
        }
    if weights_summary:
        json_summary["_weights_audit"] = weights_summary

    json_path = Path(args.reports_dir) / "ensemble_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=4)
    logger.info(f"Saved metrics JSON: {json_path}")

    # Save Markdown report
    total_test = len(results["_ground_truth"])
    report_md_path = Path(args.reports_dir) / "ensemble_report.md"
    generate_ensemble_markdown_report(results, total_test, report_md_path, weights_summary=weights_summary)
    logger.info(f"Saved markdown report: {report_md_path}")

    # Save detailed per-sample predictions CSV
    import pandas as pd
    samples_info = results.get("_samples", [])
    if samples_info:
        df_rows = []
        for i, s in enumerate(samples_info):
            lbl = s["label"]
            y_prob = results["yolo26n_cls"]["probabilities"][i]
            y_pred = results["yolo26n_cls"]["predictions"][i]
            cn_prob = results["convnext_tiny"]["probabilities"][i]
            cn_pred = results["convnext_tiny"]["predictions"][i]
            eff_prob = results["efficientnet_b3"]["probabilities"][i]
            eff_pred = results["efficientnet_b3"]["predictions"][i]
            ens_prob = results["ensemble"]["probabilities"][i]
            ens_pred = results["ensemble"]["predictions"][i]

            df_rows.append({
                "filename": s["filename"],
                "filepath": s["path"],
                "ground_truth": lbl,
                "ground_truth_label": "fake" if lbl == config.FAKE_LABEL_IDX else "real",
                "yolo_prob_fake": y_prob,
                "yolo_pred": y_pred,
                "yolo_pred_label": "fake" if y_pred == config.FAKE_LABEL_IDX else "real",
                "convnext_prob_fake": cn_prob,
                "convnext_pred": cn_pred,
                "convnext_pred_label": "fake" if cn_pred == config.FAKE_LABEL_IDX else "real",
                "effnet_prob_fake": eff_prob,
                "effnet_pred": eff_pred,
                "effnet_pred_label": "fake" if eff_pred == config.FAKE_LABEL_IDX else "real",
                "ensemble_prob_fake": ens_prob,
                "ensemble_pred": ens_pred,
                "ensemble_pred_label": "fake" if ens_pred == config.FAKE_LABEL_IDX else "real",
                "ensemble_correct": bool(ens_pred == lbl)
            })

        df_preds = pd.DataFrame(df_rows)
        csv_path = Path(args.reports_dir) / "saved_predictions.csv"
        df_preds.to_csv(csv_path, index=False)
        logger.info(f"Saved per-sample predictions CSV: {csv_path}")

        # Also copy to root reports/ for top-level access
        root_csv_path = Path(config.REPORTS_DIR) / "saved_predictions.csv"
        df_preds.to_csv(root_csv_path, index=False)
        logger.info(f"Saved top-level predictions CSV: {root_csv_path}")

    # Print console summary
    print_ascii_summary(results)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO26n-cls, ConvNeXt-Tiny, EfficientNet-B3 and their Soft-Voting Ensemble."
    )
    parser.add_argument("--data-dir", type=str, default=str(config.PROCESSED_DATA_DIR),
                        help="Dataset root containing test/real and test/fake")
    parser.add_argument("--dataset", type=str, default=None, choices=["celebdf", "dfdc", "real_life"],
                        help="Pre-configured dataset to evaluate on (overrides --data-dir)")
    parser.add_argument("--yolo-checkpoint", type=str,
                        default=str(config.BASE_DIR / "runs" / "yolo_cls" / "deepfake_yolo" / "weights" / "best.pt"),
                        help="YOLO checkpoint (.pt)")
    parser.add_argument("--convnext-checkpoint", type=str,
                        default=str(config.CHECKPOINT_DIR / "convnext_tiny.fb_in22k_ft_in1k_best.pth"),
                        help="ConvNeXt checkpoint (.pth)")
    parser.add_argument("--effnet-checkpoint", type=str,
                        default=str(config.CHECKPOINT_DIR / "efficientnet_b3.ra2_in1k_best.pth"),
                        help="EfficientNet-B3 checkpoint (.pth)")
    parser.add_argument("--convnext-model", type=str, default="convnext_tiny.fb_in22k_ft_in1k")
    parser.add_argument("--effnet-model", type=str, default="efficientnet_b3.ra2_in1k")
    parser.add_argument("--weights", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help="Soft-voting weights [yolo, convnext, effnet] (default: 1.0 1.0 1.0)")
    parser.add_argument("--tune-on-val", action="store_true",
                        help="Tune ensemble weights strictly on the validation set before testing (zero test leakage)")
    parser.add_argument("--val-dir", type=str, default=str(config.PROCESSED_DATA_DIR / "val"),
                        help="Path to validation directory for tuning weights")
    parser.add_argument("--tune-metric", type=str, default="roc_auc",
                        choices=["roc_auc", "f1", "accuracy", "log_loss", "brier"],
                        help="Validation metric to optimize (default: roc_auc)")
    parser.add_argument("--weights-file", type=str, default=None,
                        help="Path to JSON file containing pre-tuned ensemble weights")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification probability threshold for fake class (default: 0.5)")
    parser.add_argument("--reports-dir", type=str,
                        default=str(config.REPORTS_DIR / "ensemble"),
                        help="Directory to save evaluation plots and reports")
    parser.add_argument("--device", type=str, default=str(config.DEVICE), help="Device (cuda/cpu)")
    parser.add_argument("--image", type=str, help="Run single-image ensemble prediction instead of test set")

    args = parser.parse_args()
    run_ensemble_evaluation(args)


if __name__ == "__main__":
    main()

