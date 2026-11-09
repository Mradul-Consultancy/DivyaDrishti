"""
Build the full cross-dataset evaluation table.

Evaluates every (train-domain, test-domain) combination for all three models
individually AND their soft-voting ensemble.

Training-domain × Test-domain matrix:

                    Test: CelebDF  Test: DFDC  Test: Real-Life
  Train: CelebDF   [in-domain]    [cross]      [cross]
  Train: DFDC      [cross]        [in-domain]  [cross]
  Train: Both*     [pooled]       [pooled]     [pooled]

  (* "Both" row uses checkpoints from whichever dataset produced the
     higher val-accuracy, acting as a simple model-selection baseline.)

Outputs
-------
  reports/cross_dataset/
    cross_table.md              — human-readable Markdown table
    cross_table.json            — machine-readable metrics
    cross_table_heatmap.png     — AUC heatmap (train-domain vs test-domain)
    per_model_bars.png          — grouped bar chart per model
    cross_dataset_eval.log
"""

import sys
import json
import types
import argparse
import traceback
import itertools
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import config
from utils.helpers import setup_logger, ensure_dirs
from evaluation.evaluate_cross_dataset import resolve_test_split

# ── Logging ────────────────────────────────────────────────────────────────
OUT_DIR = config.REPORTS_DIR / "cross_dataset"
ensure_dirs(OUT_DIR)
logger = setup_logger("CrossTable", log_file=OUT_DIR / "cross_dataset_eval.log")

# ── Registry ───────────────────────────────────────────────────────────────
TRAIN_DATASETS = {
    "celebdf": config.PROCESSED_CELEBDF_DIR,
    "dfdc":    config.PROCESSED_DFDC_DIR,
}

TEST_DATASETS = {
    "CelebDF":    config.PROCESSED_CELEBDF_DIR,
    "DFDC":       config.PROCESSED_DFDC_DIR,
    "Real-Life":  config.REAL_LIFE_DIR,
}

TIMM_MODELS = {
    "EfficientNet-B3":  "efficientnet_b3.ra2_in1k",
    "ConvNeXt-Tiny":    "convnext_tiny.fb_in22k_ft_in1k",
}


def _timm_checkpoint(train_ds: str, model_name: str) -> Path:
    return config.CHECKPOINT_DIR / train_ds / f"{model_name}_best.pth"


def _yolo_checkpoint(train_ds: str) -> Path:
    run_name = f"deepfake_{train_ds}"
    return config.BASE_DIR / "runs" / "yolo_cls" / train_ds / run_name / "weights" / "best.pt"


# ── Single-model timm evaluator ─────────────────────────────────────────────

def evaluate_timm_on_dir(
    model_name: str,
    checkpoint: Path,
    test_dir: Path,
    batch_size: int = 32,
    device: torch.device = None,
) -> Optional[Dict[str, Any]]:
    """
    Runs a single timm model on ``test_dir`` (which must contain real/ and fake/ subfolders).
    Returns a metrics dict or None if evaluation cannot be run.
    """
    if not checkpoint.exists():
        logger.warning(f"  Checkpoint not found: {checkpoint}")
        return None

    eval_path = resolve_test_split(test_dir)
    if eval_path is None:
        logger.warning(f"  Test split not found in: {test_dir}")
        return None

    device = device or config.DEVICE

    # Count samples
    n_real = sum(1 for _ in (eval_path / "real").glob("*.*")) if (eval_path / "real").exists() else 0
    n_fake = sum(1 for _ in (eval_path / "fake").glob("*.*")) if (eval_path / "fake").exists() else 0
    if n_real + n_fake == 0:
        logger.warning(f"  No images found in {eval_path}")
        return None

    import timm as _timm
    from torch.utils.data import DataLoader
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        roc_auc_score, brier_score_loss, confusion_matrix,
    )
    from training.dataset import DeepfakeDataset
    from training.train_timm import get_timm_transforms

    # Load model
    model = _timm.create_model(model_name, pretrained=False, num_classes=2)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()

    transform = get_timm_transforms(model, is_training=False)
    dataset   = DeepfakeDataset(eval_path, transform=transform)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))

    y_true, y_probs_all = [], []
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for batch in loader:
            imgs, labels = batch[0], batch[1]   # handles 2- or 3-tuple
            imgs = imgs.to(device)
            probs = softmax(model(imgs))
            y_true.extend(labels.cpu().numpy().tolist())
            y_probs_all.extend(probs[:, config.FAKE_LABEL_IDX].cpu().numpy().tolist())

    y_true  = np.array(y_true)
    y_probs = np.array(y_probs_all)
    y_pred  = (y_probs >= 0.5).astype(int)

    acc    = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=config.FAKE_LABEL_IDX, zero_division=0
    )
    auc    = float(roc_auc_score(y_true, y_probs))
    brier  = float(brier_score_loss(y_true == config.FAKE_LABEL_IDX, y_probs))
    cm     = confusion_matrix(y_true, y_pred, labels=[config.FAKE_LABEL_IDX, config.REAL_LABEL_IDX]).tolist()

    return {
        "accuracy":   acc,
        "precision":  float(p),
        "recall":     float(r),
        "f1_score":   float(f1),
        "roc_auc":    auc,
        "brier_score": brier,
        "confusion_matrix": cm,
        "n_real":     n_real,
        "n_fake":     n_fake,
    }


def evaluate_yolo_on_dir(
    checkpoint: Path,
    test_dir: Path,
    batch_size: int = 32,
    device: torch.device = None,
) -> Optional[Dict[str, Any]]:
    """Runs YOLO26n-cls on ``test_dir`` and returns a metrics dict."""
    if not checkpoint.exists():
        logger.warning(f"  YOLO checkpoint not found: {checkpoint}")
        return None

    eval_path = resolve_test_split(test_dir)
    if eval_path is None:
        logger.warning(f"  Test split not found in: {test_dir}")
        return None

    device = device or config.DEVICE
    n_real = sum(1 for _ in (eval_path / "real").glob("*.*")) if (eval_path / "real").exists() else 0
    n_fake = sum(1 for _ in (eval_path / "fake").glob("*.*")) if (eval_path / "fake").exists() else 0
    if n_real + n_fake == 0:
        return None

    from ultralytics import YOLO
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        roc_auc_score, brier_score_loss, confusion_matrix,
    )
    from pathlib import Path as _Path

    yolo = YOLO(str(checkpoint))
    # Determine fake/real indices from the YOLO model's class names
    names = yolo.names
    fake_idx = next((k for k, v in names.items() if v.lower() == "fake"), 0)
    real_idx = next((k for k, v in names.items() if v.lower() == "real"), 1)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths: List[Tuple[Path, int]] = []
    for cls_name, label_idx in [("real", config.REAL_LABEL_IDX), ("fake", config.FAKE_LABEL_IDX)]:
        cls_dir = eval_path / cls_name
        if cls_dir.exists():
            for p in cls_dir.iterdir():
                if p.suffix.lower() in valid_exts:
                    image_paths.append((p, label_idx))

    if not image_paths:
        return None

    y_true, y_probs_all = [], []
    # YOLO inference in batches of image file paths
    all_paths = [str(p) for p, _ in image_paths]
    all_labels = [lbl for _, lbl in image_paths]

    device_str = "0" if (device.type == "cuda") else "cpu"
    results = yolo.predict(all_paths, imgsz=224, batch=batch_size, device=device_str, verbose=False)

    for res, true_lbl in zip(results, all_labels):
        probs = res.probs.data.cpu().numpy()
        y_probs_all.append(float(probs[fake_idx]))
        y_true.append(true_lbl)

    y_true  = np.array(y_true)
    y_probs = np.array(y_probs_all)
    y_pred  = (y_probs >= 0.5).astype(int)

    acc    = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=config.FAKE_LABEL_IDX, zero_division=0
    )
    auc    = float(roc_auc_score(y_true, y_probs))
    brier  = float(brier_score_loss(y_true == config.FAKE_LABEL_IDX, y_probs))
    cm     = confusion_matrix(y_true, y_pred, labels=[config.FAKE_LABEL_IDX, config.REAL_LABEL_IDX]).tolist()

    return {
        "accuracy":   acc,
        "precision":  float(p),
        "recall":     float(r),
        "f1_score":   float(f1),
        "roc_auc":    auc,
        "brier_score": brier,
        "confusion_matrix": cm,
        "n_real":     n_real,
        "n_fake":     n_fake,
    }


# ── Soft-voting ensemble evaluator ─────────────────────────────────────────

def evaluate_ensemble_on_dir(
    timm_checkpoints: Dict[str, Path],     # {display_name: ckpt_path}
    timm_model_names: Dict[str, str],      # {display_name: timm_model_name}
    yolo_checkpoint: Path,
    test_dir: Path,
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    batch_size: int = 32,
    device: torch.device = None,
) -> Optional[Dict[str, Any]]:
    """Soft-voting ensemble over (YOLO, ConvNeXt, EfficientNet)."""
    import timm as _timm
    from torch.utils.data import DataLoader
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        roc_auc_score, brier_score_loss, confusion_matrix,
    )
    from training.dataset import DeepfakeDataset
    from training.train_timm import get_timm_transforms

    device = device or config.DEVICE
    eval_path = resolve_test_split(test_dir)
    if eval_path is None:
        return None

    # Collect per-model prob arrays
    model_probs: Dict[str, np.ndarray] = {}
    ground_truth = None

    for disp_name, ckpt in timm_checkpoints.items():
        m_name = timm_model_names[disp_name]
        m = _timm.create_model(m_name, pretrained=False, num_classes=2)
        if not ckpt.exists():
            logger.warning(f"  Ensemble: missing {disp_name} checkpoint {ckpt}")
            continue
        sd = torch.load(ckpt, map_location=device)
        m.load_state_dict(sd["state_dict"])
        m = m.to(device)
        m.eval()
        tf = get_timm_transforms(m, is_training=False)
        ds = DeepfakeDataset(eval_path, transform=tf)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
        probs_list, labels_list = [], []
        softmax = torch.nn.Softmax(dim=1)
        with torch.no_grad():
            for batch in dl:
                imgs, lbs = batch[0], batch[1]
                p = softmax(m(imgs.to(device)))
                probs_list.extend(p[:, config.FAKE_LABEL_IDX].cpu().numpy().tolist())
                labels_list.extend(lbs.numpy().tolist())
        model_probs[disp_name] = np.array(probs_list)
        if ground_truth is None:
            ground_truth = np.array(labels_list)

    # YOLO probs
    if yolo_checkpoint.exists():
        from ultralytics import YOLO
        yolo = YOLO(str(yolo_checkpoint))
        names = yolo.names
        fake_idx = next((k for k, v in names.items() if v.lower() == "fake"), 0)
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = []
        for cls_name in ["real", "fake"]:
            cls_dir = eval_path / cls_name
            if cls_dir.exists():
                image_paths.extend(p for p in cls_dir.iterdir() if p.suffix.lower() in valid_exts)
        device_str = "0" if device.type == "cuda" else "cpu"
        results = yolo.predict([str(p) for p in image_paths], imgsz=224, batch=batch_size,
                               device=device_str, verbose=False)
        model_probs["YOLO26n-cls"] = np.array([float(r.probs.data.cpu().numpy()[fake_idx]) for r in results])

    if not model_probs or ground_truth is None:
        return None

    # Weighted average
    w = np.array(weights, dtype=np.float32) / sum(weights)
    prob_arrays = list(model_probs.values())
    if len(prob_arrays) < 3:
        w = np.ones(len(prob_arrays), dtype=np.float32) / len(prob_arrays)
    ensemble_probs = sum(wi * pa for wi, pa in zip(w, prob_arrays))

    y_pred = (ensemble_probs >= 0.5).astype(int)
    acc    = float(accuracy_score(ground_truth, y_pred))
    p_sc, r_sc, f1, _ = precision_recall_fscore_support(
        ground_truth, y_pred, average="binary", pos_label=config.FAKE_LABEL_IDX, zero_division=0
    )
    auc    = float(roc_auc_score(ground_truth, ensemble_probs))
    brier  = float(brier_score_loss(ground_truth == config.FAKE_LABEL_IDX, ensemble_probs))
    cm     = confusion_matrix(ground_truth, y_pred, labels=[config.FAKE_LABEL_IDX, config.REAL_LABEL_IDX]).tolist()

    return {
        "accuracy":   acc, "precision": float(p_sc), "recall": float(r_sc),
        "f1_score":   float(f1), "roc_auc": auc, "brier_score": brier,
        "confusion_matrix": cm,
    }


# ── Reporting ───────────────────────────────────────────────────────────────

def _fmt(val: Optional[float], pct: bool = False) -> str:
    if val is None:
        return "—"
    return f"{val*100:.1f}%" if pct else f"{val:.4f}"


def build_markdown_table(results: Dict) -> str:
    """
    results[train_ds][test_ds][model_key] = metrics_dict | None
    """
    MODEL_ORDER = ["EfficientNet-B3", "ConvNeXt-Tiny", "YOLO26n-cls", "Ensemble"]
    TRAIN_ORDER = ["celebdf", "dfdc"]
    TEST_ORDER  = list(TEST_DATASETS.keys())

    lines = [
        "# Cross-Dataset Evaluation Table",
        "",
        "> Metric shown: **ROC-AUC** (F1 | Brier) — in-domain cells are highlighted with ⬛.",
        "",
        "| Trained on | Test dataset | Model | Accuracy | F1 | ROC-AUC | Brier |",
        "|---|---|---|---|---|---|---|",
    ]

    for train_ds in TRAIN_ORDER:
        for test_name in TEST_ORDER:
            is_indomain = (
                (train_ds == "celebdf" and test_name == "CelebDF") or
                (train_ds == "dfdc"    and test_name == "DFDC")
            )
            tag = "⬛ in-domain" if is_indomain else "↗ cross"
            first = True
            for model_key in MODEL_ORDER:
                m = (results.get(train_ds) or {}).get(test_name) or {}
                mets = m.get(model_key)
                acc   = _fmt(mets.get("accuracy")   if mets else None, pct=True)
                f1    = _fmt(mets.get("f1_score")   if mets else None, pct=True)
                auc   = _fmt(mets.get("roc_auc")    if mets else None)
                brier = _fmt(mets.get("brier_score") if mets else None)
                train_cell = f"{train_ds} ({tag})" if first else ""
                test_cell  = test_name if first else ""
                first = False
                lines.append(f"| {train_cell} | {test_cell} | {model_key} | {acc} | {f1} | {auc} | {brier} |")
        lines.append("|---|---|---|---|---|---|---|")

    return "\n".join(lines)


def save_heatmap(results: Dict, save_path: Path):
    """AUC heatmap: rows = train datasets, cols = test datasets, one subplot per model."""
    MODEL_ORDER = ["EfficientNet-B3", "ConvNeXt-Tiny", "YOLO26n-cls", "Ensemble"]
    TRAIN_ORDER = ["celebdf", "dfdc"]
    TEST_ORDER  = list(TEST_DATASETS.keys())

    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(4 * len(MODEL_ORDER), 3.5))
    fig.suptitle("Cross-Dataset ROC-AUC Heatmap\n(rows = trained on, cols = tested on)",
                 fontweight="bold", fontsize=12)

    for ax, model_key in zip(axes, MODEL_ORDER):
        mat = np.full((len(TRAIN_ORDER), len(TEST_ORDER)), np.nan)
        for ri, train_ds in enumerate(TRAIN_ORDER):
            for ci, test_name in enumerate(TEST_ORDER):
                mets = ((results.get(train_ds) or {}).get(test_name) or {}).get(model_key)
                if mets and mets.get("roc_auc") is not None:
                    mat[ri, ci] = mets["roc_auc"]

        im = ax.imshow(mat, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(TEST_ORDER)))
        ax.set_yticks(range(len(TRAIN_ORDER)))
        ax.set_xticklabels(TEST_ORDER, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels([d.upper() for d in TRAIN_ORDER], fontsize=8)
        ax.set_title(model_key, fontsize=9, fontweight="bold")
        for ri in range(len(TRAIN_ORDER)):
            for ci in range(len(TEST_ORDER)):
                val = mat[ri, ci]
                if not np.isnan(val):
                    ax.text(ci, ri, f"{val:.3f}", ha="center", va="center",
                            fontsize=8, color="black" if val > 0.75 else "white")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Heatmap saved: {save_path}")


def save_bar_chart(results: Dict, save_path: Path):
    """Grouped bar chart: AUC per model, grouped by test dataset."""
    MODEL_ORDER = ["EfficientNet-B3", "ConvNeXt-Tiny", "YOLO26n-cls", "Ensemble"]
    TRAIN_ORDER = ["celebdf", "dfdc"]
    TEST_ORDER  = list(TEST_DATASETS.keys())
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    fig, axes = plt.subplots(1, len(TRAIN_ORDER), figsize=(14, 5), sharey=True)
    fig.suptitle("ROC-AUC per Test Dataset (grouped by model)", fontweight="bold")

    for ax, train_ds in zip(axes, TRAIN_ORDER):
        x = np.arange(len(TEST_ORDER))
        width = 0.18
        for mi, (model_key, color) in enumerate(zip(MODEL_ORDER, colors)):
            aucs = []
            for test_name in TEST_ORDER:
                mets = ((results.get(train_ds) or {}).get(test_name) or {}).get(model_key)
                aucs.append(mets["roc_auc"] if (mets and mets.get("roc_auc") is not None) else 0.0)
            offset = (mi - 1.5) * width
            bars = ax.bar(x + offset, aucs, width, label=model_key, color=color, alpha=0.85)
            for bar, v in zip(bars, aucs):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=6)
        ax.set_title(f"Trained on {train_ds.upper()}", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(TEST_ORDER, rotation=15, ha="right")
        ax.set_ylim(0.4, 1.05)
        ax.set_ylabel("ROC-AUC" if train_ds == TRAIN_ORDER[0] else "")
        ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.5, label="Chance")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Bar chart saved: {save_path}")


# ── Main evaluation loop ────────────────────────────────────────────────────

def run_cross_table(args):
    device = torch.device(args.device) if hasattr(args, "device") else config.DEVICE

    # results[train_ds][test_name][model_key] = metrics_dict | None
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    timm_meta = {
        "EfficientNet-B3": "efficientnet_b3.ra2_in1k",
        "ConvNeXt-Tiny":   "convnext_tiny.fb_in22k_ft_in1k",
    }

    for train_ds in TRAIN_DATASETS:
        results[train_ds] = {}
        logger.info(f"\n{'='*70}")
        logger.info(f"  TRAIN DOMAIN: {train_ds.upper()}")
        logger.info(f"{'='*70}")

        for test_name, test_base_dir in TEST_DATASETS.items():
            results[train_ds][test_name] = {}
            logger.info(f"\n  → Evaluating on [{test_name}]  ({test_base_dir})")

            # timm models
            for disp_name, model_name in timm_meta.items():
                ckpt = _timm_checkpoint(train_ds, model_name)
                logger.info(f"    {disp_name}: {ckpt}")
                try:
                    m = evaluate_timm_on_dir(model_name, ckpt, test_base_dir,
                                             batch_size=args.batch_size, device=device)
                    results[train_ds][test_name][disp_name] = m
                    if m:
                        logger.info(f"      AUC={m['roc_auc']:.4f}  F1={m['f1_score']:.4f}  Brier={m['brier_score']:.4f}")
                except Exception as exc:
                    logger.error(f"    {disp_name} failed: {exc}\n{traceback.format_exc()}")
                    results[train_ds][test_name][disp_name] = None

            # YOLO
            yolo_ckpt = _yolo_checkpoint(train_ds)
            logger.info(f"    YOLO26n-cls: {yolo_ckpt}")
            try:
                m = evaluate_yolo_on_dir(yolo_ckpt, test_base_dir,
                                         batch_size=args.batch_size, device=device)
                results[train_ds][test_name]["YOLO26n-cls"] = m
                if m:
                    logger.info(f"      AUC={m['roc_auc']:.4f}  F1={m['f1_score']:.4f}  Brier={m['brier_score']:.4f}")
            except Exception as exc:
                logger.error(f"    YOLO failed: {exc}\n{traceback.format_exc()}")
                results[train_ds][test_name]["YOLO26n-cls"] = None

            # Ensemble
            logger.info(f"    Soft-Voting Ensemble…")
            try:
                timm_ckpts = {dn: _timm_checkpoint(train_ds, mn) for dn, mn in timm_meta.items()}
                m = evaluate_ensemble_on_dir(
                    timm_ckpts, timm_meta, yolo_ckpt, test_base_dir,
                    weights=(1.0, 1.0, 1.0), batch_size=args.batch_size, device=device
                )
                results[train_ds][test_name]["Ensemble"] = m
                if m:
                    logger.info(f"      Ensemble AUC={m['roc_auc']:.4f}  F1={m['f1_score']:.4f}  Brier={m['brier_score']:.4f}")
            except Exception as exc:
                logger.error(f"    Ensemble failed: {exc}\n{traceback.format_exc()}")
                results[train_ds][test_name]["Ensemble"] = None

    # ── Save outputs ───────────────────────────────────────────────────────
    json_path = OUT_DIR / "cross_table.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved JSON: {json_path}")

    md_path = OUT_DIR / "cross_table.md"
    md = build_markdown_table(results)
    md_path.write_text(md, encoding="utf-8")
    logger.info(f"Saved Markdown table: {md_path}")

    save_heatmap(results, OUT_DIR / "cross_table_heatmap.png")
    save_bar_chart(results, OUT_DIR / "per_model_bars.png")

    # Print console summary
    print("\n" + md)
    print(f"\nAll outputs saved to: {OUT_DIR}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Build cross-dataset evaluation table (train-domain × test-domain).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device",     type=str, default=str(config.DEVICE))
    args = parser.parse_args()
    run_cross_table(args)


if __name__ == "__main__":
    main()

