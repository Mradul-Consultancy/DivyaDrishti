"""
DivyaDrishti AI: Robustness Testing Pipeline for Deepfake Detection.

Tests resilience of individual models (YOLO26n-cls, ConvNeXt-Tiny, EfficientNet-B3)
and the Soft-Voting Ensemble under realistic image degradations:
1. JPEG Compression (at multiple quality levels: 90, 75, 50, 30, 15)
2. Gaussian Blur (at multiple kernel radii: 1.0, 2.0, 3.0, 5.0)
3. Downscaling & Resizing (at multiple scale factors: 0.75, 0.50, 0.35, 0.20)
4. Combined Degradation (Social Media simulation: resize + blur + JPEG)

Calculates and reports metric drops (Delta Accuracy, Delta F1, Delta ROC-AUC).
Generates degradation curve plots and markdown reports in reports/robustness/.
"""

import io
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    confusion_matrix
)
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from utils.helpers import setup_logger, ensure_dirs
from evaluation.evaluate_ensemble import ModelEnsembleEvaluator


# =====================================================================
# Image Perturbation Transforms
# =====================================================================

def apply_jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    """Simulates JPEG compression artifacts by re-saving to in-memory bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_gaussian_blur(image: Image.Image, radius: float) -> Image.Image:
    """Simulates lens defocus or camera motion blur using Gaussian filter."""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_resize_downscale(image: Image.Image, scale: float) -> Image.Image:
    """
    Simulates resolution loss / downscaling (e.g. social media previews)
    by downsampling and upsampling back to original dimension.
    """
    w, h = image.size
    new_w = max(16, int(w * scale))
    new_h = max(16, int(h * scale))
    downsampled = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    upsampled = downsampled.resize((w, h), Image.Resampling.BILINEAR)
    return upsampled


def apply_combined_distortion(image: Image.Image, scale: float = 0.5, blur_radius: float = 1.5, jpeg_quality: int = 30) -> Image.Image:
    """Simulates combined platform recompression (downscaled, blurred, and heavily compressed)."""
    img = apply_resize_downscale(image, scale=scale)
    img = apply_gaussian_blur(img, radius=blur_radius)
    img = apply_jpeg_compression(img, quality=jpeg_quality)
    return img


# =====================================================================
# Robustness Evaluator Runner
# =====================================================================

class RobustnessTester:
    def __init__(
        self,
        evaluator: ModelEnsembleEvaluator,
        reports_dir: Path | str = config.REPORTS_DIR / "robustness",
        save_sample_images: bool = True,
        logger=None
    ):
        self.evaluator = evaluator
        self.reports_dir = Path(reports_dir)
        self.save_sample_images = save_sample_images
        self.logger = logger or setup_logger("RobustnessTester")
        ensure_dirs(self.reports_dir)

    def load_test_samples(self, test_dir: Path) -> List[Tuple[Path, int]]:
        """Loads all test image paths and their labels (fake=0, real=1)."""
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        samples = []
        for class_name, label in [("fake", config.FAKE_LABEL_IDX), ("real", config.REAL_LABEL_IDX)]:
            folder = test_dir / class_name
            if folder.exists():
                for p in folder.glob("*"):
                    if p.suffix.lower() in valid_exts:
                        samples.append((p, label))
        return samples

    def evaluate_dataset_under_distortion(
        self,
        samples: List[Tuple[Path, int]],
        transform_fn: Optional[Callable[[Image.Image], Image.Image]] = None,
        threshold: float = 0.5
    ) -> Dict[str, Dict]:
        """
        Runs evaluation across all models on test set, applying optional transform_fn.
        """
        model_keys = ["yolo26n_cls", "convnext_tiny", "efficientnet_b3", "ensemble"]
        probs = {k: [] for k in model_keys}
        y_true = []

        for img_path, label in samples:
            pil_img = Image.open(img_path).convert("RGB")
            if transform_fn is not None:
                pil_img = transform_fn(pil_img)

            preds = self.evaluator.predict_image(pil_img)
            y_true.append(label)
            for k in model_keys:
                probs[k].append(preds[k])

        y_true = np.array(y_true)
        results = {}

        for k in model_keys:
            p_arr = np.array(probs[k])
            preds_bin = np.where(p_arr >= threshold, config.FAKE_LABEL_IDX, config.REAL_LABEL_IDX)

            acc = accuracy_score(y_true, preds_bin)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, preds_bin, average="binary", pos_label=config.FAKE_LABEL_IDX, zero_division=0
            )
            fpr, tpr, _ = roc_curve(y_true, p_arr, pos_label=config.FAKE_LABEL_IDX)
            roc_auc_val = auc(fpr, tpr)
            cm = confusion_matrix(y_true, preds_bin, labels=[config.FAKE_LABEL_IDX, config.REAL_LABEL_IDX])

            results[k] = {
                "accuracy": float(acc),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "roc_auc": float(roc_auc_val),
                "confusion_matrix": cm.tolist()
            }

        return results

    def run_full_suite(
        self,
        test_dir: Path | str,
        jpeg_qualities: List[int] = [90, 75, 50, 30, 15],
        blur_radii: List[float] = [1.0, 2.0, 3.0, 5.0],
        resize_scales: List[float] = [0.75, 0.50, 0.35, 0.20],
        threshold: float = 0.5
    ) -> Dict:
        """Runs the entire benchmark suite across clean and all distorted versions."""
        test_dir = Path(test_dir)
        samples = self.load_test_samples(test_dir)
        if len(samples) == 0:
            raise ValueError(f"No test images found in {test_dir} (expected real/ and fake/ subfolders).")

        self.logger.info("=" * 70)
        self.logger.info("Starting Deepfake Detector Robustness Benchmark")
        self.logger.info(f"Test samples: {len(samples)} images from {test_dir}")
        self.logger.info("=" * 70)

        # 1. Clean Baseline
        self.logger.info("\n[1/5] Evaluating Baseline (Clean unperturbed images)...")
        clean_results = self.evaluate_dataset_under_distortion(samples, transform_fn=None, threshold=threshold)

        full_benchmark = {
            "baseline": clean_results,
            "jpeg_compression": {},
            "gaussian_blur": {},
            "resize_scaling": {},
            "combined": {}
        }

        # 2. JPEG Compression Sweep
        self.logger.info("\n[2/5] Evaluating JPEG Compression Robustness (Qualities: %s)...", jpeg_qualities)
        for q in jpeg_qualities:
            self.logger.info(f"  -> Testing JPEG Quality: {q}")
            res = self.evaluate_dataset_under_distortion(
                samples,
                transform_fn=lambda img, qual=q: apply_jpeg_compression(img, quality=qual),
                threshold=threshold
            )
            full_benchmark["jpeg_compression"][str(q)] = self._compute_drops(clean_results, res)

        # 3. Gaussian Blur Sweep
        self.logger.info("\n[3/5] Evaluating Gaussian Blur Robustness (Radii: %s)...", blur_radii)
        for r in blur_radii:
            self.logger.info(f"  -> Testing Blur Radius: {r}px")
            res = self.evaluate_dataset_under_distortion(
                samples,
                transform_fn=lambda img, rad=r: apply_gaussian_blur(img, radius=rad),
                threshold=threshold
            )
            full_benchmark["gaussian_blur"][str(r)] = self._compute_drops(clean_results, res)

        # 4. Resize Scaling Sweep
        self.logger.info("\n[4/5] Evaluating Resize Downscale Robustness (Scales: %s)...", resize_scales)
        for s in resize_scales:
            self.logger.info(f"  -> Testing Resize Scale: {s*100:.0f}%")
            res = self.evaluate_dataset_under_distortion(
                samples,
                transform_fn=lambda img, sc=s: apply_resize_downscale(img, scale=sc),
                threshold=threshold
            )
            full_benchmark["resize_scaling"][str(s)] = self._compute_drops(clean_results, res)

        # 5. Combined Distortion (Social Media Simulation)
        self.logger.info("\n[5/5] Evaluating Combined Social Media Distortion (Resize 50% + Blur 1.5 + JPEG 30)...")
        combined_res = self.evaluate_dataset_under_distortion(
            samples,
            transform_fn=lambda img: apply_combined_distortion(img, scale=0.5, blur_radius=1.5, jpeg_quality=30),
            threshold=threshold
        )
        full_benchmark["combined"]["social_media_sim"] = self._compute_drops(clean_results, combined_res)

        # Save Visual Artifacts
        if self.save_sample_images and len(samples) > 0:
            preview_img = Image.open(samples[0][0]).convert("RGB")
            self._save_perturbation_preview(preview_img)

        # Plots and Reports
        curves_path = self.reports_dir / "robustness_curves.png"
        self._plot_robustness_curves(full_benchmark, curves_path)
        self.logger.info(f"Saved robustness curves plot: {curves_path}")

        json_path = self.reports_dir / "robustness_metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(full_benchmark, f, indent=4)
        self.logger.info(f"Saved robustness metrics JSON: {json_path}")

        report_md_path = self.reports_dir / "robustness_report.md"
        self._generate_markdown_report(full_benchmark, len(samples), report_md_path)
        self.logger.info(f"Saved robustness report markdown: {report_md_path}")

        # Print console table
        self._print_console_summary(full_benchmark)

        return full_benchmark

    def _compute_drops(self, baseline: Dict[str, Dict], perturbed: Dict[str, Dict]) -> Dict[str, Dict]:
        """Calculates raw metrics and metric drop (Delta = Baseline - Perturbed)."""
        out = {}
        for k in baseline:
            base_acc = baseline[k]["accuracy"]
            base_f1 = baseline[k]["f1_score"]
            base_auc = baseline[k]["roc_auc"]

            pert_acc = perturbed[k]["accuracy"]
            pert_f1 = perturbed[k]["f1_score"]
            pert_auc = perturbed[k]["roc_auc"]

            out[k] = {
                "accuracy": pert_acc,
                "f1_score": pert_f1,
                "roc_auc": pert_auc,
                "precision": perturbed[k]["precision"],
                "recall": perturbed[k]["recall"],
                "confusion_matrix": perturbed[k]["confusion_matrix"],
                "delta_accuracy": float(base_acc - pert_acc),
                "delta_f1": float(base_f1 - pert_f1),
                "delta_roc_auc": float(base_auc - pert_auc)
            }
        return out

    def _save_perturbation_preview(self, sample_img: Image.Image):
        """Saves a multi-panel visual grid showing how perturbations alter face images."""
        variants = [
            ("Clean / Original", sample_img),
            ("JPEG Quality 50", apply_jpeg_compression(sample_img, 50)),
            ("JPEG Quality 15", apply_jpeg_compression(sample_img, 15)),
            ("Blur Radius 2px", apply_gaussian_blur(sample_img, 2.0)),
            ("Blur Radius 5px", apply_gaussian_blur(sample_img, 5.0)),
            ("Downscale 50%", apply_resize_downscale(sample_img, 0.50)),
            ("Downscale 20%", apply_resize_downscale(sample_img, 0.20)),
            ("Combined Social", apply_combined_distortion(sample_img, scale=0.5, blur_radius=1.5, jpeg_quality=30))
        ]

        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        axes = axes.flatten()

        for idx, (title, img) in enumerate(variants):
            axes[idx].imshow(img)
            axes[idx].set_title(title, fontweight="bold", fontsize=10)
            axes[idx].axis("off")

        plt.suptitle("Sample Image Perturbations (Robustness Test Suite)", fontweight="bold", fontsize=13)
        plt.tight_layout()
        preview_path = self.reports_dir / "perturbations_preview.png"
        plt.savefig(preview_path, dpi=250)
        plt.close()

    def _plot_robustness_curves(self, benchmark: Dict, save_path: Path):
        """Plots degradation curves for Accuracy and ROC-AUC under JPEG, Blur, and Resize."""
        models = [
            ("yolo26n_cls", "YOLO26n-cls", "#1f77b4", "-o"),
            ("convnext_tiny", "ConvNeXt-Tiny", "#2ca02c", "-s"),
            ("efficientnet_b3", "EfficientNet-B3", "#9467bd", "-^"),
            ("ensemble", "Soft-Voting Ensemble", "#d62728", "-*")
        ]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # 1. JPEG Compression (x-axis: quality decreasing 100 -> 15)
        jpeg_keys = sorted([int(k) for k in benchmark["jpeg_compression"].keys()], reverse=True)
        for m_key, m_label, color, style in models:
            # Add clean baseline at 100 quality
            clean_val = benchmark["baseline"][m_key]["accuracy"] * 100
            y_vals = [clean_val] + [benchmark["jpeg_compression"][str(q)][m_key]["accuracy"] * 100 for q in jpeg_keys]
            x_vals = [100] + jpeg_keys
            lw = 2.5 if m_key == "ensemble" else 1.5
            ax1.plot(x_vals, y_vals, style, color=color, label=m_label, lw=lw, markersize=5)

        ax1.set_xlabel("JPEG Quality (100 = Clean)", fontweight="bold")
        ax1.set_ylabel("Accuracy (%)", fontweight="bold")
        ax1.set_title("Robustness vs JPEG Compression", fontweight="bold")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.invert_xaxis()  # 100 on left, 15 on right (increasing degradation)
        ax1.legend(loc="lower left", fontsize=9)

        # 2. Gaussian Blur (x-axis: radius increasing 0 -> 5)
        blur_keys = sorted([float(k) for k in benchmark["gaussian_blur"].keys()])
        for m_key, m_label, color, style in models:
            clean_val = benchmark["baseline"][m_key]["accuracy"] * 100
            y_vals = [clean_val] + [benchmark["gaussian_blur"][str(r)][m_key]["accuracy"] * 100 for r in blur_keys]
            x_vals = [0.0] + blur_keys
            lw = 2.5 if m_key == "ensemble" else 1.5
            ax2.plot(x_vals, y_vals, style, color=color, label=m_label, lw=lw, markersize=5)

        ax2.set_xlabel("Gaussian Blur Radius (px)", fontweight="bold")
        ax2.set_ylabel("Accuracy (%)", fontweight="bold")
        ax2.set_title("Robustness vs Gaussian Blur", fontweight="bold")
        ax2.grid(True, linestyle=":", alpha=0.6)

        # 3. Resize Downscale (x-axis: scale decreasing 1.0 -> 0.20)
        resize_keys = sorted([float(k) for k in benchmark["resize_scaling"].keys()], reverse=True)
        for m_key, m_label, color, style in models:
            clean_val = benchmark["baseline"][m_key]["accuracy"] * 100
            y_vals = [clean_val] + [benchmark["resize_scaling"][str(s)][m_key]["accuracy"] * 100 for s in resize_keys]
            x_vals = [1.0] + resize_keys
            lw = 2.5 if m_key == "ensemble" else 1.5
            ax3.plot(x_vals, y_vals, style, color=color, label=m_label, lw=lw, markersize=5)

        ax3.set_xlabel("Resolution Downscale Factor (1.0 = Full)", fontweight="bold")
        ax3.set_ylabel("Accuracy (%)", fontweight="bold")
        ax3.set_title("Robustness vs Resolution Downscaling", fontweight="bold")
        ax3.grid(True, linestyle=":", alpha=0.6)
        ax3.invert_xaxis()  # 1.0 on left, 0.20 on right (increasing degradation)

        plt.suptitle("Model Performance Degradation Curves Under Perturbations", fontweight="bold", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _generate_markdown_report(self, b: Dict, total_samples: int, save_path: Path):
        """Generates comprehensive markdown report detailing metric drops."""
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("# DivyaDrishti AI — Robustness Benchmark Report\n\n")
            f.write(f"- **Total Test Images**: `{total_samples}`\n")
            f.write("- **Evaluated Models**: `YOLO26n-cls`, `ConvNeXt-Tiny`, `EfficientNet-B3`, `Soft-Voting Ensemble`\n")
            f.write("- **Distortion Types Tested**: JPEG Recompression, Gaussian Blur, Resolution Downscale, Combined Social Media Sim\n\n")

            f.write("## 1. Clean Baseline Performance\n\n")
            f.write("| Model | Accuracy | F1-Score | ROC-AUC |\n")
            f.write("|---|:---:|:---:|:---:|\n")
            for m in ["yolo26n_cls", "convnext_tiny", "efficientnet_b3", "ensemble"]:
                name = m.replace("_", " ").title()
                f.write(f"| {name} | {b['baseline'][m]['accuracy']*100:.2f}% | {b['baseline'][m]['f1_score']*100:.2f}% | {b['baseline'][m]['roc_auc']:.4f} |\n")

            # JPEG Compression Section
            f.write("\n## 2. JPEG Compression Metric Drops\n\n")
            f.write("| JPEG Quality | Model | Accuracy | Acc Drop (Δ) | F1-Score | F1 Drop (Δ) | ROC-AUC | AUC Drop (Δ) |\n")
            f.write("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|\n")
            for q_str, q_data in b["jpeg_compression"].items():
                for m in ["yolo26n_cls", "convnext_tiny", "efficientnet_b3", "ensemble"]:
                    d = q_data[m]
                    m_name = m.replace("_", " ").title()
                    f.write(f"| Q={q_str} | {m_name} | {d['accuracy']*100:.1f}% | {d['delta_accuracy']*100:+.1f}% | {d['f1_score']*100:.1f}% | {d['delta_f1']*100:+.1f}% | {d['roc_auc']:.4f} | {d['delta_roc_auc']:+.4f} |\n")

            # Blur Section
            f.write("\n## 3. Gaussian Blur Metric Drops\n\n")
            f.write("| Blur Radius | Model | Accuracy | Acc Drop (Δ) | F1-Score | F1 Drop (Δ) | ROC-AUC | AUC Drop (Δ) |\n")
            f.write("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|\n")
            for r_str, r_data in b["gaussian_blur"].items():
                for m in ["yolo26n_cls", "convnext_tiny", "efficientnet_b3", "ensemble"]:
                    d = r_data[m]
                    m_name = m.replace("_", " ").title()
                    f.write(f"| {r_str}px | {m_name} | {d['accuracy']*100:.1f}% | {d['delta_accuracy']*100:+.1f}% | {d['f1_score']*100:.1f}% | {d['delta_f1']*100:+.1f}% | {d['roc_auc']:.4f} | {d['delta_roc_auc']:+.4f} |\n")

            # Resize Section
            f.write("\n## 4. Resolution Downscaling Metric Drops\n\n")
            f.write("| Scale | Model | Accuracy | Acc Drop (Δ) | F1-Score | F1 Drop (Δ) | ROC-AUC | AUC Drop (Δ) |\n")
            f.write("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|\n")
            for s_str, s_data in b["resize_scaling"].items():
                for m in ["yolo26n_cls", "convnext_tiny", "efficientnet_b3", "ensemble"]:
                    d = s_data[m]
                    m_name = m.replace("_", " ").title()
                    scale_pct = float(s_str) * 100
                    f.write(f"| {scale_pct:.0f}% | {m_name} | {d['accuracy']*100:.1f}% | {d['delta_accuracy']*100:+.1f}% | {d['f1_score']*100:.1f}% | {d['delta_f1']*100:+.1f}% | {d['roc_auc']:.4f} | {d['delta_roc_auc']:+.4f} |\n")

            # Combined Section
            f.write("\n## 5. Combined Distortion (Social Media Recompression)\n\n")
            f.write("Simulation: Scale 50% + Gaussian Blur 1.5px + JPEG Quality 30\n\n")
            f.write("| Model | Accuracy | Acc Drop (Δ) | F1-Score | F1 Drop (Δ) | ROC-AUC | AUC Drop (Δ) |\n")
            f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|\n")
            comb = b["combined"]["social_media_sim"]
            for m in ["yolo26n_cls", "convnext_tiny", "efficientnet_b3", "ensemble"]:
                d = comb[m]
                m_name = m.replace("_", " ").title()
                f.write(f"| {m_name} | {d['accuracy']*100:.1f}% | {d['delta_accuracy']*100:+.1f}% | {d['f1_score']*100:.1f}% | {d['delta_f1']*100:+.1f}% | {d['roc_auc']:.4f} | {d['delta_roc_auc']:+.4f} |\n")

            f.write("\n## 6. Visual Artifacts\n\n")
            f.write("- **Performance Degradation Curves**: `robustness_curves.png`\n")
            f.write("- **Visual Perturbation Sample Grid**: `perturbations_preview.png`\n\n")
            f.write("![Degradation Curves](robustness_curves.png)\n\n")
            f.write("![Sample Perturbations](perturbations_preview.png)\n")

    def _print_console_summary(self, b: Dict):
        """Prints a compact ASCII table of metric drops for the combined and extreme tests."""
        print("\n" + "=" * 88)
        print("                 ROBUSTNESS BENCHMARK SUMMARY (METRIC DROPS)")
        print("=" * 88)
        print(f"{'Condition':<22} | {'Model':<20} | {'Acc (Drop)':<14} | {'F1 (Drop)':<14} | {'AUC (Drop)':<12}")
        print("-" * 88)

        # Baseline
        for m in ["yolo26n_cls", "convnext_tiny", "efficientnet_b3", "ensemble"]:
            base = b["baseline"][m]
            print(f"{'Clean (Baseline)':<22} | {m:<20} | {base['accuracy']*100:5.1f}%        | {base['f1_score']*100:5.1f}%        | {base['roc_auc']:5.4f}")

        print("-" * 88)
        # Heavy JPEG Q=30
        if "30" in b["jpeg_compression"]:
            for m in ["yolo26n_cls", "convnext_tiny", "efficientnet_b3", "ensemble"]:
                d = b["jpeg_compression"]["30"][m]
                print(f"{'JPEG Quality 30':<22} | {m:<20} | {d['accuracy']*100:5.1f}% ({d['delta_accuracy']*100:+4.1f}%) | {d['f1_score']*100:5.1f}% ({d['delta_f1']*100:+4.1f}%) | {d['roc_auc']:5.4f} ({d['delta_roc_auc']:+5.4f})")

        print("-" * 88)
        # Blur 2px
        if "2.0" in b["gaussian_blur"]:
            for m in ["yolo26n_cls", "convnext_tiny", "efficientnet_b3", "ensemble"]:
                d = b["gaussian_blur"]["2.0"][m]
                print(f"{'Blur Radius 2px':<22} | {m:<20} | {d['accuracy']*100:5.1f}% ({d['delta_accuracy']*100:+4.1f}%) | {d['f1_score']*100:5.1f}% ({d['delta_f1']*100:+4.1f}%) | {d['roc_auc']:5.4f} ({d['delta_roc_auc']:+5.4f})")

        print("-" * 88)
        # Combined
        comb = b["combined"]["social_media_sim"]
        for m in ["yolo26n_cls", "convnext_tiny", "efficientnet_b3", "ensemble"]:
            d = comb[m]
            print(f"{'Combined Social':<22} | {m:<20} | {d['accuracy']*100:5.1f}% ({d['delta_accuracy']*100:+4.1f}%) | {d['f1_score']*100:5.1f}% ({d['delta_f1']*100:+4.1f}%) | {d['roc_auc']:5.4f} ({d['delta_roc_auc']:+5.4f})")
        print("=" * 88 + "\n")


def run_robustness_benchmark(args):
    logger = setup_logger("RobustnessBenchmark")

    evaluator = ModelEnsembleEvaluator(
        yolo_checkpoint=args.yolo_checkpoint,
        convnext_checkpoint=args.convnext_checkpoint,
        effnet_checkpoint=args.effnet_checkpoint,
        device=torch.device(args.device),
        logger=logger
    )

    # Check for tuned ensemble weights
    weights_path = getattr(args, "weights_file", None)
    if weights_path and Path(weights_path).exists():
        evaluator.load_weights(weights_path)
    elif (config.CHECKPOINT_DIR / "ensemble_weights.json").exists():
        evaluator.load_weights(config.CHECKPOINT_DIR / "ensemble_weights.json")

    tester = RobustnessTester(
        evaluator=evaluator,
        reports_dir=args.reports_dir,
        logger=logger
    )

    test_dir = Path(args.data_dir) / "test" if (Path(args.data_dir) / "test").exists() else Path(args.data_dir)
    tester.run_full_suite(
        test_dir=test_dir,
        jpeg_qualities=args.jpeg_qualities,
        blur_radii=args.blur_radii,
        resize_scales=args.resize_scales,
        threshold=args.threshold
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Deepfake Detector Robustness under JPEG, Blur, and Resize Perturbations"
    )
    parser.add_argument("--data-dir", type=str, default=str(config.PROCESSED_DATA_DIR), help="Dataset root containing test split")
    parser.add_argument("--yolo-checkpoint", type=str, default=str(config.BASE_DIR / "runs" / "yolo_cls" / "deepfake_yolo" / "weights" / "best.pt"))
    parser.add_argument("--convnext-checkpoint", type=str, default=str(config.CHECKPOINT_DIR / "convnext_tiny.fb_in22k_ft_in1k_best.pth"))
    parser.add_argument("--effnet-checkpoint", type=str, default=str(config.CHECKPOINT_DIR / "efficientnet_b3.ra2_in1k_best.pth"))
    parser.add_argument("--jpeg-qualities", type=int, nargs="+", default=[90, 75, 50, 30, 15], help="JPEG compression quality levels")
    parser.add_argument("--blur-radii", type=float, nargs="+", default=[1.0, 2.0, 3.0, 5.0], help="Gaussian blur radii in pixels")
    parser.add_argument("--resize-scales", type=float, nargs="+", default=[0.75, 0.50, 0.35, 0.20], help="Downscaling factors")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification probability threshold")
    parser.add_argument("--reports-dir", type=str, default=str(config.REPORTS_DIR / "robustness"), help="Output reports directory")
    parser.add_argument("--device", type=str, default=str(config.DEVICE), help="Device (cuda/cpu)")

    args = parser.parse_args()
    run_robustness_benchmark(args)


if __name__ == "__main__":
    main()

