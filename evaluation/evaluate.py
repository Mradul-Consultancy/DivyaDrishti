"""
Evaluation script for Deepfake Detection models.
Computes Confusion Matrix, Precision, Recall, F1-score, and ROC-AUC curve.
Saves visual evaluation plots and markdown reports to reports/.
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_recall_fscore_support
)
import torch
from torch.utils.data import DataLoader

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from utils.helpers import setup_logger, ensure_dirs
from models.detector import get_model
from preprocessing.transforms import get_val_transforms


def get_eval_transform(model, model_name: str):
    """
    Returns the evaluation transform that matches how ``model`` was trained.

    timm backbones are trained with their own native pipeline
    (``resolve_data_config`` + ``create_transform``), which uses the model's
    own crop_pct and interpolation. The generic torchvision resize used for
    the hand-written backbones differs, and using it on a timm model changes
    the scores measurably. So timm models get their native transform and
    everything else keeps the torchvision one.
    """
    try:
        import timm
        from timm.data import resolve_data_config, create_transform

        # Only treat it as a timm model if timm can actually build it.
        timm.create_model(model_name, pretrained=False)
        cfg = resolve_data_config(model.pretrained_cfg)
        return create_transform(**cfg, is_training=False)
    except Exception:
        return get_val_transforms()
from training.dataset import DeepfakeDataset


def plot_confusion_matrix(cm, class_names, save_path: Path):
    """Generates and saves a clean confusion matrix heatmap plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                x=j, y=i,
                s=f"{cm[i, j]}",
                va="center", ha="center",
                size="large",
                weight="bold",
                color="white" if cm[i, j] > cm.max() / 2 else "black"
            )

    fig.colorbar(cax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels([c.upper() for c in class_names])
    ax.set_yticklabels([c.upper() for c in class_names])
    ax.set_xlabel("Predicted Label", fontweight="bold", labelpad=10)
    ax.set_ylabel("True Label", fontweight="bold", labelpad=10)
    ax.set_title("Confusion Matrix", fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curve(y_true, y_probs, save_path: Path):
    """Generates and saves ROC curve and AUC score."""
    fpr, tpr, _ = roc_curve(y_true, y_probs, pos_label=config.FAKE_LABEL_IDX)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)", fontweight="bold")
    plt.ylabel("True Positive Rate (TPR)", fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC)", fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return roc_auc


def evaluate_model(args):
    # Ensure reports directory exists
    ensure_dirs(args.reports_dir)
    # Determine test directory: explicit test-dir overrides data-dir logic
    if args.test_dir:
        test_dir = Path(args.test_dir)
    else:
        # Existing behavior: if data_dir has a 'test' subfolder, use it; otherwise use data_dir itself
        test_dir = Path(args.data_dir) / "test" if (Path(args.data_dir) / "test").exists() else Path(args.data_dir)
    
    # Load dataset with path information
    #
    # The transform must match the one the model was TRAINED with. timm
    # backbones use their native pipeline (resolve_data_config +
    # create_transform), which differs from the generic torchvision resize in
    # preprocessing.transforms.get_val_transforms - a different crop_pct and
    # interpolation. Using the wrong one measurably changes the scores, so
    # the model is built first and the transform derived from it.
    device = torch.device(args.device)

    # Prepare logger
    log_path = Path(args.reports_dir) / "evaluation.log"
    logger = setup_logger("DeepfakeEvaluator", log_file=log_path)
    logger.info("Initializing Evaluation Routine...")

    # Load Model
    logger.info(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_name = checkpoint.get("model_name", args.model)
    model = get_model(model_name, num_classes=config.NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    test_transform = get_eval_transform(model, model_name)
    logger.info(f"Evaluation transform: {type(test_transform).__name__} for '{model_name}'")
    test_dataset = DeepfakeDataset(test_dir, transform=test_transform, return_path=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    logger.info(f"Loaded {len(test_dataset)} test samples from {test_dir}")

    y_true = []
    y_pred = []
    y_probs = []
    prediction_records = []  # For CSV output

    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for batch in test_loader:
            # Batch may contain (images, labels) or (images, labels, paths) depending on dataset __getitem__
            if len(batch) == 3:
                images, labels, paths = batch
                img_paths = [str(p) for p in paths]
            else:
                images, labels = batch
                img_paths = ["<unknown>"] * images.size(0)
            images = images.to(device)
            outputs = model(images)
            probs = softmax(outputs)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_probs.extend(probs[:, config.FAKE_LABEL_IDX].cpu().numpy().tolist())

            # Record per-sample predictions for CSV
            for img_path, true_lbl, pred_lbl, prob_fake in zip(img_paths, labels.cpu().numpy(), preds.cpu().numpy(), probs[:, config.FAKE_LABEL_IDX].cpu().numpy()):
                video_id = Path(img_path).parent.name  # Assuming parent folder corresponds to video ID
                prediction_records.append({
                    "image_path": img_path,
                    "video_id": video_id,
                    "true_label": config.IDX_TO_CLASS.get(int(true_lbl), "unknown"),
                    "predicted_label": config.IDX_TO_CLASS.get(int(pred_lbl), "unknown"),
                    "prob_fake": float(prob_fake),
                    "prob_real": float(1.0 - prob_fake),
                })

    # Compute metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=config.FAKE_LABEL_IDX, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[config.FAKE_LABEL_IDX, config.REAL_LABEL_IDX])
    clf_report = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, output_dict=True, zero_division=0)

    # Save metrics and CSV
    cm_path = Path(args.reports_dir) / "confusion_matrix.png"
    plot_confusion_matrix(cm, config.CLASS_NAMES, cm_path)
    logger.info(f"Saved confusion matrix plot: {cm_path}")

    roc_path = Path(args.reports_dir) / "roc_curve.png"
    roc_auc = plot_roc_curve(y_true, y_probs, roc_path)
    logger.info(f"Saved ROC curve plot: {roc_path}")

    # Save CSV of predictions
    #
    # NOTE: this is the SINGLE-model schema (one prob_fake column). The
    # multi-model ensemble schema (yolo_prob_fake, convnext_prob_fake, ...)
    # is owned by evaluation/evaluate_ensemble.py and lives at
    # reports/saved_predictions.csv. Writing both to the same path made the
    # two scripts overwrite each other depending on run order, which broke
    # evaluation/generate_reports.py and tests/test_calibration.py.
    import csv
    csv_path = Path(args.reports_dir) / "saved_predictions_single.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["image_path", "video_id", "true_label", "predicted_label", "prob_fake", "prob_real"])
        writer.writeheader()
        for rec in prediction_records:
            writer.writerow(rec)
    logger.info(f"Saved per-sample predictions CSV: {csv_path}")

    # Assemble metrics dict
    metrics = {
        "model_name": model_name,
        "checkpoint": str(args.checkpoint),
        "total_samples": len(test_dataset),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "detailed_report": clf_report,
    }

    # Save metrics JSON
    metrics_json_path = Path(args.reports_dir) / "evaluation_summary.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved evaluation metrics JSON: {metrics_json_path}")

    # Generate Markdown report (unchanged content but added reference to CSV)
    report_md_path = Path(args.reports_dir) / "evaluation_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Deepfake Detection Evaluation Report\n\n")
        f.write(f"- **Model**: `{model_name}`\n")
        f.write(f"- **Evaluated Checkpoint**: `{args.checkpoint}`\n")
        f.write(f"- **Total Test Images**: `{len(test_dataset)}`\n\n")
        f.write("## Performance Summary\n\n")
        f.write("| Metric | Score |\n|---|---|\n")
        f.write(f"| **Accuracy** | {accuracy * 100:.2f}% |\n")
        f.write(f"| **Precision (Fake)** | {precision * 100:.2f}% |\n")
        f.write(f"| **Recall (Fake)** | {recall * 100:.2f}% |\n")
        f.write(f"| **F1-Score** | {f1 * 100:.2f}% |\n")
        f.write(f"| **ROC-AUC** | {roc_auc:.4f} |\n\n")
        f.write("## Confusion Matrix\n\n")
        f.write(f"![Confusion Matrix](confusion_matrix.png)\n\n")
        f.write("## ROC Curve\n\n")
        f.write(f"![ROC Curve](roc_curve.png)\n\n")
        f.write("## Per‑Sample Predictions\n\n")
        f.write(f"[saved_predictions.csv]({csv_path.name}) contains image paths, video IDs, true labels, predicted labels and class probabilities.\n")
    logger.info(f"Saved markdown report: {report_md_path}")

    # Print summary to console (unchanged)
    print("\n" + "=" * 50)
    print("           EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("=" * 50 + "\n")
    ensure_dirs(args.reports_dir)
    log_path = Path(args.reports_dir) / "evaluation.log"
    logger = setup_logger("DeepfakeEvaluator", log_file=log_path)
    logger.info("Initializing Evaluation Routine...")

    device = torch.device(args.device)

    # Load dataset
    test_dir = Path(args.data_dir) / "test" if (Path(args.data_dir) / "test").exists() else Path(args.data_dir)

    logger.info(f"Loaded {len(test_dataset)} test samples from {test_dir}")
    if len(test_dataset) == 0:
        logger.error(f"No test images found in {test_dir}! Please check test directory.")
        return

    # Load Model
    logger.info(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_name = checkpoint.get("model_name", args.model)
    model = get_model(model_name, num_classes=config.NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Rebuild the dataset with the transform that matches training (see
    # get_eval_transform).
    test_transform = get_eval_transform(model, model_name)
    test_dataset = DeepfakeDataset(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    y_true = []
    y_pred = []
    y_probs = []

    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = softmax(outputs)

            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            # Probability of fake class (index 0 per standard ImageFolder alphabetical sorting)
            y_probs.extend(probs[:, config.FAKE_LABEL_IDX].cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # Calculate metrics with pos_label=0 (fake class)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=config.FAKE_LABEL_IDX, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[config.FAKE_LABEL_IDX, config.REAL_LABEL_IDX])
    clf_report = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, output_dict=True, zero_division=0)

    # Plots
    cm_path = Path(args.reports_dir) / "confusion_matrix.png"
    plot_confusion_matrix(cm, config.CLASS_NAMES, cm_path)
    logger.info(f"Saved confusion matrix plot: {cm_path}")

    roc_path = Path(args.reports_dir) / "roc_curve.png"
    roc_auc = plot_roc_curve(y_true, y_probs, roc_path)
    logger.info(f"Saved ROC curve plot: {roc_path}")

    metrics = {
        "model_name": model_name,
        "checkpoint": str(args.checkpoint),
        "total_samples": len(test_dataset),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "detailed_report": clf_report
    }

    # Save metrics JSON
    metrics_json_path = Path(args.reports_dir) / "evaluation_summary.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved evaluation metrics JSON: {metrics_json_path}")

    # Generate Markdown Summary Report
    report_md_path = Path(args.reports_dir) / "evaluation_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Deepfake Detection Evaluation Report\n\n")
        f.write(f"- **Model**: `{model_name}`\n")
        f.write(f"- **Evaluated Checkpoint**: `{args.checkpoint}`\n")
        f.write(f"- **Total Test Images**: `{len(test_dataset)}`\n\n")
        f.write("## Performance Summary\n\n")
        f.write("| Metric | Score |\n")
        f.write("|---|---|\n")
        f.write(f"| **Accuracy** | {accuracy * 100:.2f}% |\n")
        f.write(f"| **Precision (Fake)** | {precision * 100:.2f}% |\n")
        f.write(f"| **Recall (Fake)** | {recall * 100:.2f}% |\n")
        f.write(f"| **F1-Score** | {f1 * 100:.2f}% |\n")
        f.write(f"| **ROC-AUC** | {roc_auc:.4f} |\n\n")
        f.write("## Confusion Matrix\n\n")
        f.write(f"![Confusion Matrix](confusion_matrix.png)\n\n")
        f.write("## ROC Curve\n\n")
        f.write(f"![ROC Curve](roc_curve.png)\n")
    logger.info(f"Saved markdown report: {report_md_path}")

    print("\n" + "=" * 50)
    print("           EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Deepfake Detector on Test Dataset")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model .pth checkpoint")
    parser.add_argument("--model", type=str, default="resnet18", help="Model architecture")
    parser.add_argument("--data-dir", type=str, default=str(config.PROCESSED_DATA_DIR), help="Dataset directory containing test split")
    parser.add_argument("--reports-dir", type=str, default=str(config.REPORTS_DIR), help="Output directory for reports and charts")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--device", type=str, default=str(config.DEVICE), help="Device (cuda/cpu)")

    args = parser.parse_args()
    evaluate_model(args)

