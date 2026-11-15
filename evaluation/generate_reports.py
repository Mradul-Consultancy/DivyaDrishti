"""
Results Table and ROC Plot Generator from Saved Predictions.

Reads per-sample model predictions (from saved_predictions.csv) and computes:
- Accuracy, Balanced Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score
- ROC-AUC and PR-AUC (Average Precision)
- Brier Score (Calibration Loss)
- Confusion Matrices

Generates and exports:
- reports/results_table.md (Markdown table for reports and documentation)
- reports/results_summary.csv (CSV spreadsheet)
- reports/roc_plots.png (High-resolution multi-model ROC curves with AUC annotations)
- reports/precision_recall_curves.png (Precision-Recall curves with AP annotations)
- reports/confusion_matrix_grid.png (Side-by-side confusion matrix heatmaps)
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    brier_score_loss
)

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from utils.helpers import setup_logger, ensure_dirs
from evaluation.video_metrics import (
    aggregate_predictions_by_video,
    compute_recall_at_fixed_fpr,
    bootstrap_confidence_intervals
)


class PredictionReportGenerator:
    """
    Parses a saved predictions DataFrame or CSV, computes comprehensive
    classification metrics, and renders publication-quality ROC and PR plots.
    """

    MODEL_CONFIGS = [
        {
            "key": "yolo",
            "name": "YOLO26n-cls (Ultralytics)",
            "prob_col": "yolo_prob_fake",
            "pred_col": "yolo_pred",
            "color": "#1f77b4",
            "linestyle": "-."
        },
        {
            "key": "convnext",
            "name": "ConvNeXt-Tiny (timm)",
            "prob_col": "convnext_prob_fake",
            "pred_col": "convnext_pred",
            "color": "#2ca02c",
            "linestyle": ":"
        },
        {
            "key": "effnet",
            "name": "EfficientNet-B3 (timm)",
            "prob_col": "effnet_prob_fake",
            "pred_col": "effnet_pred",
            "color": "#9467bd",
            "linestyle": "--"
        },
        {
            "key": "ensemble",
            "name": "Soft-Voting Ensemble",
            "prob_col": "ensemble_prob_fake",
            "pred_col": "ensemble_pred",
            "color": "#d62728",
            "linestyle": "-"
        }
    ]

    def __init__(self, predictions_df: pd.DataFrame, output_dir: Path | str = config.REPORTS_DIR, logger=None):
        self.df = predictions_df
        self.output_dir = Path(output_dir)
        self.logger = logger or setup_logger("ReportGenerator")
        ensure_dirs(self.output_dir)

        # ---- schema normalisation ---------------------------------------
        # Two producers write prediction CSVs with different schemas:
        #   evaluation/evaluate_ensemble.py -> reports/saved_predictions.csv
        #       (yolo_prob_fake, convnext_prob_fake, effnet_prob_fake,
        #        ensemble_prob_fake, ground_truth, ...)
        #   evaluation/evaluate.py          -> reports/saved_predictions_single.csv
        #       (image_path, video_id, true_label, predicted_label,
        #        prob_fake, prob_real)
        # Both are accepted; anything else is rejected with guidance instead
        # of a KeyError deep inside the metric code.
        if "ground_truth" not in self.df.columns:
            if "true_label" in self.df.columns:
                label_map = {str(k).lower(): v for k, v in config.CLASS_TO_IDX.items()}
                mapped = self.df["true_label"].astype(str).str.lower().map(label_map)
                if mapped.isna().any():
                    unknown = sorted(set(
                        self.df.loc[mapped.isna(), "true_label"].astype(str)
                    ))
                    raise ValueError(
                        "Predictions CSV contains unknown true_label value(s) "
                        f"{unknown}; expected one of {sorted(label_map)}"
                    )
                self.df = self.df.assign(ground_truth=mapped.astype(int))
            else:
                raise KeyError(
                    "Predictions dataframe must contain a 'ground_truth' column "
                    "(ensemble schema) or a 'true_label' column (single-model "
                    "schema)."
                )

        present = [cfg for cfg in self.MODEL_CONFIGS if cfg["prob_col"] in self.df.columns]
        if present:
            self.model_configs = present
        elif "prob_fake" in self.df.columns:
            # Single-model schema: one probability column, one row per frame.
            self.model_configs = [{
                "key": "model",
                "name": "Model (single)",
                "prob_col": "prob_fake",
                "pred_col": "predicted_label",
                "color": "#d62728",
                "linestyle": "-",
            }]
        else:
            raise ValueError(
                "No model probability columns found in the predictions CSV. "
                "Expected either the ensemble schema written by "
                "evaluation/evaluate_ensemble.py to reports/saved_predictions.csv "
                "(columns: yolo_prob_fake, convnext_prob_fake, effnet_prob_fake, "
                "ensemble_prob_fake) or the single-model schema written by "
                "evaluation/evaluate.py to reports/saved_predictions_single.csv "
                "(columns: image_path, video_id, true_label, predicted_label, "
                "prob_fake, prob_real)."
            )

        self.y_true = self.df["ground_truth"].values.astype(int)

    @classmethod
    def from_csv(cls, csv_path: Path | str, output_dir: Path | str = config.REPORTS_DIR, logger=None):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Predictions CSV not found at: {csv_path}")
        df = pd.read_csv(csv_path)
        return cls(df, output_dir=output_dir, logger=logger)

    def compute_all_metrics(self, threshold: float = 0.5) -> Dict[str, Dict]:
        """
        Computes detailed metrics for each model present in the DataFrame.
        """
        results = {}

        for cfg in self.model_configs:
            prob_col = cfg["prob_col"]
            if prob_col not in self.df.columns:
                continue

            y_probs = self.df[prob_col].values
            y_preds = np.where(y_probs >= threshold, config.FAKE_LABEL_IDX, config.REAL_LABEL_IDX)

            acc = accuracy_score(self.y_true, y_preds)
            bal_acc = balanced_accuracy_score(self.y_true, y_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.y_true, y_preds, average="binary", pos_label=config.FAKE_LABEL_IDX, zero_division=0
            )

            cm = confusion_matrix(self.y_true, y_preds, labels=[config.FAKE_LABEL_IDX, config.REAL_LABEL_IDX])
            # cm layout with [FAKE=0, REAL=1]:
            # cm[0,0] = TP (fake as fake), cm[0,1] = FN (fake as real)
            # cm[1,0] = FP (real as fake), cm[1,1] = TN (real as real)
            tp, fn = cm[0, 0], cm[0, 1]
            fp, tn = cm[1, 0], cm[1, 1]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            # ROC Curve & AUC (pos_label = 0 -> FAKE)
            fpr, tpr, roc_thresholds = roc_curve(self.y_true, y_probs, pos_label=config.FAKE_LABEL_IDX)
            roc_auc = auc(fpr, tpr)

            # Precision-Recall Curve & AP
            pr_prec, pr_rec, _ = precision_recall_curve(self.y_true, y_probs, pos_label=config.FAKE_LABEL_IDX)
            is_fake_binary = (self.y_true == config.FAKE_LABEL_IDX).astype(int)
            ap_score = average_precision_score(is_fake_binary, y_probs)

            # Calibration / Brier score
            brier = brier_score_loss(is_fake_binary, y_probs)

            # Recall at fixed False Positive Rates (e.g. 1%, 5%, 10% FPR)
            rec_fixed = compute_recall_at_fixed_fpr(
                self.y_true, y_probs, target_fprs=[0.01, 0.05, 0.10], pos_label=config.FAKE_LABEL_IDX
            )

            # Optimal Youden's Index cutoff
            youden_idx = np.argmax(tpr - fpr)
            best_cutoff = float(roc_thresholds[youden_idx]) if youden_idx < len(roc_thresholds) else 0.5

            results[cfg["key"]] = {
                "name": cfg["name"],
                "color": cfg["color"],
                "linestyle": cfg["linestyle"],
                "accuracy": float(acc),
                "balanced_accuracy": float(bal_acc),
                "precision": float(precision),
                "recall": float(recall),
                "specificity": float(specificity),
                "f1_score": float(f1),
                "roc_auc": float(roc_auc),
                "pr_auc": float(ap_score),
                "brier_score": float(brier),
                "recall_at_1%_fpr": rec_fixed.get("recall_at_1%_fpr", 0.0),
                "recall_at_5%_fpr": rec_fixed.get("recall_at_5%_fpr", 0.0),
                "recall_at_10%_fpr": rec_fixed.get("recall_at_10%_fpr", 0.0),
                "optimal_threshold": best_cutoff,
                "confusion_matrix": cm.tolist(),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "fpr": fpr,
                "tpr": tpr,
                "pr_precision": pr_prec,
                "pr_recall": pr_rec
            }

        return results

    def plot_roc_curves(self, metrics: Dict[str, Dict], save_path: Path):
        """Generates publication-quality ROC plots with AUC annotations."""
        plt.figure(figsize=(8.5, 7.5))

        for key, m in metrics.items():
            lw = 3.2 if key == "ensemble" else 1.8
            label = f"{m['name']} (AUC = {m['roc_auc']:.4f})"
            plt.plot(m["fpr"], m["tpr"], color=m["color"], lw=lw, linestyle=m["linestyle"], label=label)

        # Baseline diagonal
        plt.plot([0, 1], [0, 1], color="#555555", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")

        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.05])
        plt.xlabel("False Positive Rate (1 - Specificity)", fontweight="bold", fontsize=11, labelpad=8)
        plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontweight="bold", fontsize=11, labelpad=8)
        plt.title("Receiver Operating Characteristic (ROC) Comparison", fontweight="bold", fontsize=13, pad=14)
        plt.legend(loc="lower right", frameon=True, fontsize=10, shadow=True)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_pr_curves(self, metrics: Dict[str, Dict], save_path: Path):
        """Generates Precision-Recall curves with Average Precision (AP) scores."""
        plt.figure(figsize=(8.5, 7.5))

        for key, m in metrics.items():
            lw = 3.2 if key == "ensemble" else 1.8
            label = f"{m['name']} (AP = {m['pr_auc']:.4f})"
            plt.plot(m["pr_recall"], m["pr_precision"], color=m["color"], lw=lw, linestyle=m["linestyle"], label=label)

        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.05])
        plt.xlabel("Recall", fontweight="bold", fontsize=11, labelpad=8)
        plt.ylabel("Precision", fontweight="bold", fontsize=11, labelpad=8)
        plt.title("Precision-Recall (PR) Curves & Average Precision", fontweight="bold", fontsize=13, pad=14)
        plt.legend(loc="lower left", frameon=True, fontsize=10, shadow=True)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_confusion_matrices(self, metrics: Dict[str, Dict], save_path: Path):
        """Plots the confusion matrices in a grid sized to the model count."""
        keys = list(metrics.keys())
        n = max(len(keys), 1)
        ncols = 2 if n > 1 else 1
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows))
        axes = np.atleast_1d(axes).flatten()

        for idx, k in enumerate(keys):
            ax = axes[idx]
            m = metrics[k]
            cm = np.array(m["confusion_matrix"])
            cmap = plt.cm.Reds if k == "ensemble" else plt.cm.Blues

            cax = ax.matshow(cm, cmap=cmap, alpha=0.85)
            for r in range(cm.shape[0]):
                for c in range(cm.shape[1]):
                    val = cm[r, c]
                    ax.text(
                        x=c, y=r, s=f"{val}",
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
            ax.set_title(
                f"{m['name']}\nAcc: {m['accuracy']*100:.1f}% | F1: {m['f1_score']*100:.1f}%",
                fontweight="bold", pad=12
            )

        for ax in axes[len(keys):]:
            ax.axis("off")

        plt.suptitle("Confusion Matrix Grid (Test Predictions)", fontweight="bold", fontsize=14, y=0.99)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def export_results_table(self, metrics: Dict[str, Dict]) -> Tuple[Path, Path]:
        """
        Exports both Markdown results table and CSV results summary.
        """
        rows = []
        for key, m in metrics.items():
            rows.append({
                "Model": m["name"],
                "Accuracy (%)": round(m["accuracy"] * 100, 2),
                "Balanced Acc (%)": round(m["balanced_accuracy"] * 100, 2),
                "Precision (%)": round(m["precision"] * 100, 2),
                "Recall (%)": round(m["recall"] * 100, 2),
                "Specificity (%)": round(m["specificity"] * 100, 2),
                "F1-Score (%)": round(m["f1_score"] * 100, 2),
                "ROC-AUC": round(m["roc_auc"], 4),
                "PR-AUC (AP)": round(m["pr_auc"], 4),
                "Brier Score": round(m["brier_score"], 4),
                "True Real (TN)": m["tn"],
                "False Fake (FP)": m["fp"],
                "False Real (FN)": m["fn"],
                "True Fake (TP)": m["tp"]
            })

        df_results = pd.DataFrame(rows)

        # Compute video-level metrics & bootstrap confidence intervals if video info is available
        video_level_results = {}
        try:
            video_df = aggregate_predictions_by_video(self.df, method="mean")
            self.logger.info(f"Aggregated {len(self.df)} frame predictions into {len(video_df)} video predictions.")
            
            for cfg in self.model_configs:
                prob_col = cfg["prob_col"]
                if prob_col in video_df.columns:
                    ci = bootstrap_confidence_intervals(
                        video_df,
                        prob_col=prob_col,
                        ground_truth_col="ground_truth",
                        n_bootstraps=1000,
                        ci_level=0.95
                    )
                    video_level_results[cfg["key"]] = {
                        "name": cfg["name"],
                        "confidence_intervals": ci
                    }
        except Exception as e:
            self.logger.warning(f"Could not compute video-level bootstrap confidence intervals: {e}")

        # 1. Export CSV
        csv_path = self.output_dir / "results_summary.csv"
        df_results.to_csv(csv_path, index=False)

        # 2. Export Markdown Table
        md_path = self.output_dir / "results_table.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Model Performance Results Table\n\n")
            f.write(f"- **Evaluated Test Frames**: `{len(self.y_true)}`\n")
            if video_level_results:
                f.write(f"- **Evaluated Test Videos (Aggregated)**: `{len(video_df)}`\n")
            f.write(f"- **Positive Class**: `FAKE (1)` | **Negative Class**: `REAL (0)`\n\n")

            f.write("## Frame-Level Results & Recall @ Fixed FPR\n\n")
            f.write("| Model | Accuracy | F1-Score | ROC-AUC | Brier Score | Recall @ 1% FPR | Recall @ 5% FPR | Recall @ 10% FPR |\n")
            f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")

            for key, m in metrics.items():
                is_ens = key == "ensemble"
                b = "**" if is_ens else ""
                r1 = f"{m.get('recall_at_1%_fpr', 0.0)*100:.1f}%"
                r5 = f"{m.get('recall_at_5%_fpr', 0.0)*100:.1f}%"
                r10 = f"{m.get('recall_at_10%_fpr', 0.0)*100:.1f}%"
                f.write(
                    f"| {b}{m['name']}{b} | "
                    f"{b}{m['accuracy']*100:.2f}%{b} | "
                    f"{b}{m['f1_score']*100:.2f}%{b} | "
                    f"{b}{m['roc_auc']:.4f}{b} | "
                    f"{b}{m['brier_score']:.4f}{b} | "
                    f"{b}{r1}{b} | "
                    f"{b}{r5}{b} | "
                    f"{b}{r10}{b} |\n"
                )

            if video_level_results:
                f.write("\n## Video-Level Aggregated Performance (Bootstrap 95% CI)\n\n")
                f.write("> Evaluated by aggregating frame probabilities per video (mean), resampling videos with replacement ($B=1000$).\n\n")
                f.write("| Model | Metric | Mean | 95% CI (Lower - Upper) |\n")
                f.write("|:---|:---|:---:|:---:|\n")
                for key, v_info in video_level_results.items():
                    m_name = v_info["name"]
                    ci_dict = v_info.get("confidence_intervals", {})
                    for m_name_key in ["roc_auc", "f1_score", "accuracy", "brier_score", "recall_at_5%_fpr"]:
                        if m_name_key in ci_dict:
                            ci_m = ci_dict[m_name_key]
                            is_pct = m_name_key in ["f1_score", "accuracy", "recall_at_5%_fpr"]
                            mean_str = f"{ci_m['mean']*100:.2f}%" if is_pct else f"{ci_m['mean']:.4f}"
                            ci_str = (
                                f"[{ci_m['ci_lower']*100:.2f}%, {ci_m['ci_upper']*100:.2f}%]"
                                if is_pct else
                                f"[{ci_m['ci_lower']:.4f}, {ci_m['ci_upper']:.4f}]"
                            )
                            f.write(f"| {m_name} | `{m_name_key}` | {mean_str} | {ci_str} |\n")

            f.write("\n## Confusion Breakdown\n\n")
            f.write("| Model | True Real (TN) | False Fake (FP) | False Real (FN) | True Fake (TP) |\n")
            f.write("|:---|:---:|:---:|:---:|:---:|\n")
            for r in rows:
                f.write(f"| {r['Model']} | `{r['True Real (TN)']}` | `{r['False Fake (FP)']}` | `{r['False Real (FN)']}` | `{r['True Fake (TP)']}` |\n")

            f.write("\n## Evaluation Plots\n\n")
            f.write("- **Receiver Operating Characteristic Curves**: `roc_plots.png`\n")
            f.write("- **Precision-Recall Curves**: `precision_recall_curves.png`\n")
            f.write("- **Confusion Matrix Grid**: `confusion_matrix_grid.png`\n\n")
            f.write("![ROC Curves](roc_plots.png)\n\n")
            f.write("![PR Curves](precision_recall_curves.png)\n\n")
            f.write("![Confusion Matrices](confusion_matrix_grid.png)\n")

        return md_path, csv_path

    def print_ascii_table(self, metrics: Dict[str, Dict]):
        """Prints a clean ASCII table to terminal."""
        print("\n" + "=" * 112)
        print("                  DIVYADRISHTI AI: RESULTS TABLE FROM SAVED PREDICTIONS")
        print("=" * 112)
        header = f"{'Model':<26} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Brier':<8}"
        print(header)
        print("-" * 112)

        for key, m in metrics.items():
            name = m["name"]
            print(
                f"{name:<26} | "
                f"{m['accuracy']*100:7.2f}% | "
                f"{m['precision']*100:7.2f}% | "
                f"{m['recall']*100:7.2f}% | "
                f"{m['f1_score']*100:7.2f}% | "
                f"{m['roc_auc']:8.4f} | "
                f"{m['pr_auc']:8.4f} | "
                f"{m['brier_score']:8.4f}"
            )
        print("=" * 112 + "\n")


def generate_reports_from_predictions(
    predictions_file: Path | str,
    output_dir: Path | str = config.REPORTS_DIR,
    threshold: float = 0.5,
    min_videos_per_class: int = 10,
    allow_toy: bool = False
):
    logger = setup_logger("ReportGenerator")
    predictions_path = Path(predictions_file)

    # If predictions file does not exist, run ensemble evaluation to generate it
    if not predictions_path.exists():
        logger.warning(f"Predictions file '{predictions_path}' not found! Running evaluation to generate it...")
        from evaluation.evaluate_ensemble import run_ensemble_evaluation
        class EvalArgs:
            data_dir = str(config.PROCESSED_DATA_DIR)
            yolo_checkpoint = str(config.BASE_DIR / "runs" / "yolo_cls" / "deepfake_yolo" / "weights" / "best.pt")
            convnext_checkpoint = str(config.CHECKPOINT_DIR / "convnext_tiny.fb_in22k_ft_in1k_best.pth")
            effnet_checkpoint = str(config.CHECKPOINT_DIR / "efficientnet_b3.ra2_in1k_best.pth")
            convnext_model = "convnext_tiny.fb_in22k_ft_in1k"
            effnet_model = "efficientnet_b3.ra2_in1k"
            weights = [1.0, 1.0, 1.0]
            threshold = 0.5
            reports_dir = str(config.REPORTS_DIR / "ensemble")
            device = str(config.DEVICE)
            image = None

        run_ensemble_evaluation(EvalArgs())
        predictions_path = Path(config.REPORTS_DIR) / "saved_predictions.csv"

    logger.info(f"Loading predictions from: {predictions_path}")
    generator = PredictionReportGenerator.from_csv(predictions_path, output_dir=output_dir, logger=logger)

    # Safety Guard: Enforce minimum videos/clusters per class before generating client report
    if not allow_toy:
        try:
            v_df = aggregate_predictions_by_video(generator.df, method="mean")
            counts = v_df["ground_truth"].value_counts()
            fake_count = int(counts.get(config.FAKE_LABEL_IDX, 0))
            real_count = int(counts.get(config.REAL_LABEL_IDX, 0))
            if fake_count < min_videos_per_class or real_count < min_videos_per_class:
                raise ValueError(
                    f"Safety Guard Activated: Test evaluation set contains only {real_count} real and {fake_count} "
                    f"fake video clusters (minimum required: {min_videos_per_class} per class). "
                    f"Refusing to generate client deliverable reports from under-sized toy test slices."
                )
        except KeyError:
            pass

    metrics = generator.compute_all_metrics(threshold=threshold)

    # 1. Plot ROC Curves
    roc_plot_path = Path(output_dir) / "roc_plots.png"
    generator.plot_roc_curves(metrics, roc_plot_path)
    logger.info(f"Saved ROC curves plot to: {roc_plot_path}")

    # 2. Plot Precision-Recall Curves
    pr_plot_path = Path(output_dir) / "precision_recall_curves.png"
    generator.plot_pr_curves(metrics, pr_plot_path)
    logger.info(f"Saved Precision-Recall curves plot to: {pr_plot_path}")

    # 3. Plot Confusion Matrix Grid
    cm_plot_path = Path(output_dir) / "confusion_matrix_grid.png"
    generator.plot_confusion_matrices(metrics, cm_plot_path)
    logger.info(f"Saved confusion matrix grid to: {cm_plot_path}")

    # 4. Export Results Tables (Markdown & CSV)
    md_path, csv_path = generator.export_results_table(metrics)
    logger.info(f"Saved markdown results table to: {md_path}")
    logger.info(f"Saved CSV results table to: {csv_path}")

    # 5. Print ASCII Table
    generator.print_ascii_table(metrics)


def main():
    parser = argparse.ArgumentParser(
        description="Generate results table and ROC plots from saved model predictions."
    )
    parser.add_argument(
        "--predictions-file", type=str,
        default=str(config.REPORTS_DIR / "saved_predictions.csv"),
        help="Path to saved_predictions.csv"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(config.REPORTS_DIR),
        help="Directory where tables and ROC plots will be saved"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Classification decision threshold (default: 0.5)"
    )

    parser.add_argument(
        "--min-videos-per-class", type=int, default=10,
        help="Minimum required video clusters per class to generate client report (default: 10)"
    )
    parser.add_argument(
        "--allow-toy", action="store_true",
        help="Bypass video cluster count guard for smoke testing"
    )

    args = parser.parse_args()
    generate_reports_from_predictions(
        predictions_file=args.predictions_file,
        output_dir=args.output_dir,
        threshold=args.threshold,
        min_videos_per_class=args.min_videos_per_class,
        allow_toy=args.allow_toy
    )


if __name__ == "__main__":
    main()

