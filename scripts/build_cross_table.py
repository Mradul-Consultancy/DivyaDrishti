"""
Build the full cross-dataset evaluation table.

Evaluates every (train-domain, test-domain) combination for all three models
individually AND their soft-voting ensemble.

Training-domain × Test-domain matrix:

                    Test: CelebDF  Test: DFDC  Test: Real-Life
  Train: CelebDF   [in-domain]    [cross]      [cross]
  Train: DFDC      [cross]        [in-domain]  [cross]

  A row is only meaningful if a *domain-specific* checkpoint exists for that
  training domain (``models/checkpoints/<domain>/<model>_best.pth``). When no
  such checkpoint exists the shared/global checkpoint is used instead and the
  row is explicitly labelled as such in the output, because evaluating the
  same weights under two different "Trained on" labels would misrepresent the
  experiment.

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
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# The cross-table loads three backbones in sequence on one card. Without
# expandable segments the allocator fragments and the third model fails with
# an out-of-memory error even though the total fits.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import config
from utils.helpers import setup_logger, ensure_dirs
from evaluation.evaluate_cross_dataset import resolve_test_split
from evaluation.calibration import load_threshold, yolo_fake_probs_chunked

# The console summary contains box-drawing/arrow characters. On Windows the
# default console encoding is cp1252, which cannot represent them and raises
# UnicodeEncodeError - so force UTF-8 output with a replacement fallback.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover - non-reconfigurable stream
    pass

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


def _timm_checkpoint(train_ds: str, model_name: str) -> Tuple[Path, bool]:
    """
    Resolves the checkpoint for a training domain.

    Returns (path, is_domain_specific). ``is_domain_specific`` is False when
    the domain-specific checkpoint is missing and the shared/global checkpoint
    is substituted - in that case the row is NOT a genuine cross-domain
    experiment and must be labelled as such in the report.
    """
    p = config.CHECKPOINT_DIR / train_ds / f"{model_name}_best.pth"
    if p.exists():
        return p, True
    fallback = config.CHECKPOINT_DIR / f"{model_name}_best.pth"
    if fallback.exists():
        return fallback, False
    return p, False


def _yolo_checkpoint(train_ds: str) -> Tuple[Path, bool]:
    """
    Resolves the YOLO checkpoint for a training domain. See _timm_checkpoint.

    Canonical layout is ``runs/yolo_cls/<dataset>/deepfake_<dataset>/weights/best.pt``.
    Older runs wrote to ``runs/yolo_cls/<run_name>/weights/best.pt`` without the
    dataset directory, so both are probed.
    """
    run_name = f"deepfake_{train_ds}"
    candidates = [
        config.BASE_DIR / "runs" / "yolo_cls" / train_ds / run_name / "weights" / "best.pt",
        config.BASE_DIR / "runs" / "yolo_cls" / run_name / "weights" / "best.pt",
        config.BASE_DIR / "runs" / "yolo_cls" / "deepfake_yolo" / "weights" / "best.pt",
    ]
    for i, c in enumerate(candidates):
        if c.exists():
            # Only the first (dataset-scoped) candidate counts as domain-specific.
            return c, (i == 0)
    return candidates[0], False


# ── Shared metric computation ────────────────────────────────────────────────
#
# Label convention (see config.py): FAKE = 0, REAL = 1.
#
# Every probability below is P(fake). Therefore the positive class for
# ROC-AUC is FAKE, and the decision rule is "prob_fake >= threshold -> FAKE".
#
# NOTE: sklearn's roc_auc_score() defaults to pos_label=1, which would treat
# REAL as the positive class and silently return 1 - AUC when fed P(fake).
# We therefore always use roc_curve(..., pos_label=FAKE_LABEL_IDX) + auc(),
# matching evaluation/evaluate_ensemble.py and evaluation/generate_reports.py.


def resolve_operating_points(model_names: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """
    The decision threshold that is actually shipped, per (domain, model).

    The published metrics must describe the decision rule the product uses.
    Accuracy, precision, recall, F1 and the confusion matrix all depend on
    the threshold; ROC-AUC and Brier do not. Using a hardcoded 0.5 while the
    service ships 0.21-0.96 would report a system nobody deploys.

    Each model has its own calibrated threshold, because each has its own
    score distribution. The ensemble has no calibration of its own, so it
    uses the median of its members' thresholds - recorded explicitly so the
    number is traceable.
    """
    names = model_names or list(TIMM_MODELS.values())
    points: Dict[str, Dict[str, float]] = {}
    for domain in TRAIN_DATASETS:
        per_model = {name: load_threshold(domain, name) for name in names}
        # YOLO has its own calibration artefact; applying the ensemble's
        # threshold to YOLO scores would report a decision rule nobody runs.
        per_model["yolo26n-cls"] = load_threshold(domain, "yolo26n-cls", default=0.5)
        # The ensemble has its own calibration: the mean of the members'
        # P(fake) has a different distribution, and a member's threshold
        # starved it (F1 = 0 at a member's 0.93 on Celeb-DF).
        per_model["ensemble"] = load_threshold(domain, "ensemble",
                                              default=sorted(per_model.values())[len(per_model) // 2])
        points[domain] = per_model
    return points


def resolve_domain_notes() -> Dict[str, bool]:
    """
    Determines, per training domain, whether a domain-specific checkpoint
    exists on disk.

    Used to label every cross-dataset row with the provenance of the weights
    that produced it, and to rebuild the CSV from a stored run without
    re-running inference.
    """
    notes: Dict[str, bool] = {}
    for train_ds in TRAIN_DATASETS:
        has_timm = any(
            (config.CHECKPOINT_DIR / train_ds / f"{m}_best.pth").exists()
            for m in TIMM_MODELS.values()
        )
        has_yolo = (
            config.BASE_DIR / "runs" / "yolo_cls" / train_ds /
            f"deepfake_{train_ds}" / "weights" / "best.pt"
        ).exists()
        notes[train_ds] = bool(has_timm and has_yolo)
    return notes


# Per-model P(fake) arrays, keyed by (model key, test directory). The ensemble
# reuses these instead of re-running inference: it is ~3x cheaper and
# guarantees the ensemble is built from exactly the same per-model scores that
# the individual rows report.
_PROB_CACHE: Dict[Tuple[str, str], np.ndarray] = {}


def _cache_key(model_key: str, test_dir: Path) -> Tuple[str, str]:
    return (model_key, str(test_dir))


def _labels_for(eval_path: Path) -> np.ndarray:
    """Ground-truth labels for a split directory, in dataset order."""
    from training.dataset import DeepfakeDataset
    ds = DeepfakeDataset(eval_path)
    return np.asarray([lbl for _, lbl in ds.samples], dtype=int)



def _compute_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Computes the standard metric set for a binary deepfake detector.

    y_true  : int array of labels, 0 = fake, 1 = real
    y_probs : float array of P(fake) scores
    """
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        roc_curve, auc, brier_score_loss, confusion_matrix,
    )

    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs, dtype=float)

    # Decision rule: high P(fake) -> predict FAKE (label 0).
    y_pred = np.where(y_probs >= threshold, config.FAKE_LABEL_IDX, config.REAL_LABEL_IDX)

    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=config.FAKE_LABEL_IDX, zero_division=0
    )
    fpr, tpr, _ = roc_curve(y_true, y_probs, pos_label=config.FAKE_LABEL_IDX)
    auc_val = float(auc(fpr, tpr))
    is_fake = (y_true == config.FAKE_LABEL_IDX).astype(float)
    brier = float(brier_score_loss(is_fake, y_probs))
    cm = confusion_matrix(y_true, y_pred, labels=[config.FAKE_LABEL_IDX, config.REAL_LABEL_IDX]).tolist()

    return {
        "accuracy":   acc,
        "precision":  float(p),
        "recall":     float(r),
        "f1_score":   float(f1),
        "roc_auc":    auc_val,
        "brier_score": brier,
        "confusion_matrix": cm,
    }


# ── Single-model timm evaluator ─────────────────────────────────────────────

def evaluate_timm_on_dir(
    model_name: str,
    checkpoint: Path,
    test_dir: Path,
    batch_size: int = 32,
    device: torch.device = None,
    threshold: float = 0.5,
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
    from training.dataset import DeepfakeDataset
    from training.train_timm import get_timm_transforms

    # Load model
    model = _timm.create_model(model_name, pretrained=False, num_classes=2)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()

    transform = get_timm_transforms(model, is_training=False)
    dataset   = DeepfakeDataset(eval_path, transform=transform)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    y_true, y_probs_all = [], []
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for batch in loader:
            imgs, labels = batch[0], batch[1]   # handles 2- or 3-tuple
            imgs = imgs.to(device)
            probs = softmax(model(imgs))
            y_true.extend(labels.cpu().numpy().tolist())
            y_probs_all.extend(probs[:, config.FAKE_LABEL_IDX].cpu().numpy().tolist())

    res = _compute_metrics(np.array(y_true), np.array(y_probs_all), threshold=threshold)
    res["threshold"] = threshold
    res["n_real"] = n_real
    res["n_fake"] = n_fake
    return res


def evaluate_yolo_on_dir(
    checkpoint: Path,
    test_dir: Path,
    batch_size: int = 32,
    device: torch.device = None,
    threshold: float = 0.5,
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
    from pathlib import Path as _Path

    yolo = YOLO(str(checkpoint))
    # Determine fake/real indices from the YOLO model's class names
    names = yolo.names
    fake_idx = next((k for k, v in names.items() if v.lower() == "fake"), 0)
    real_idx = next((k for k, v in names.items() if v.lower() == "real"), 1)

    # The probability array is cached and later averaged with the timm
    # models' arrays, so the order MUST be the canonical DeepfakeDataset
    # order (classes sorted alphabetically, i.e. fake before real). Listing
    # real first here reversed the class order relative to the timm arrays
    # and silently averaged probabilities belonging to different images -
    # which showed up as a below-chance in-domain ensemble AUC.
    image_paths = _collect_samples(eval_path)
    if not image_paths:
        return None
    y_true, y_probs_all = [], []
    all_paths = [str(p) for p, _ in image_paths]
    all_labels = [lbl for _, lbl in image_paths]

    device_str = "0" if (device.type == "cuda") else "cpu"
    # Chunked prediction: Ultralytics takes a LIST source as one batch, so a
    # single forward pass grows with the split and dies on a 4 GB card no
    # matter what batch= says. yolo_fake_probs_chunked bounds each pass.
    probs_all = yolo_fake_probs_chunked(
        yolo, all_paths, fake_idx, batch_size=batch_size, device=device_str
    )
    y_probs_all.extend(probs_all)
    y_true.extend(all_labels)

    res = _compute_metrics(np.array(y_true), np.array(y_probs_all), threshold=threshold)
    res["threshold"] = threshold
    res["n_real"] = n_real
    res["n_fake"] = n_fake
    _PROB_CACHE[_cache_key("yolo26n-cls", eval_path)] = np.array(y_probs_all)
    return res


# ── Soft-voting ensemble evaluator ─────────────────────────────────────────

def _collect_samples(eval_path: Path) -> List[Tuple[Path, int]]:
    """
    Builds the canonical (path, label) sample list for a test directory.

    Order MUST be identical for every model in the ensemble, otherwise the
    per-model probability arrays are misaligned and the soft-voting average
    mixes probabilities belonging to different images.

    The order matches training.dataset.DeepfakeDataset: classes sorted
    alphabetically ('fake' before 'real'), files in directory-glob order.
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    samples: List[Tuple[Path, int]] = []
    dir_classes = sorted([d.name for d in eval_path.iterdir() if d.is_dir()])
    for class_name in dir_classes:
        label_idx = config.CLASS_TO_IDX.get(class_name)
        if label_idx is None:
            continue
        class_folder = eval_path / class_name
        if not class_folder.exists():
            continue
        for img_path in class_folder.glob("*"):
            if img_path.suffix.lower() in valid_exts:
                samples.append((img_path, label_idx))
    return samples

def evaluate_ensemble_on_dir(
    timm_checkpoints: Dict[str, Path],     # {display_name: ckpt_path}
    timm_model_names: Dict[str, str],      # {display_name: timm_model_name}
    yolo_checkpoint: Path,
    test_dir: Path,
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    batch_size: int = 32,
    device: torch.device = None,
    threshold: float = 0.5,
) -> Optional[Dict[str, Any]]:
    """Soft-voting ensemble over (YOLO, ConvNeXt, EfficientNet)."""
    import timm as _timm
    from torch.utils.data import DataLoader
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
        if not ckpt.exists():
            logger.warning(f"  Ensemble: missing {disp_name} checkpoint {ckpt}")
            continue
        # Reuse the probabilities already computed for this model's own row
        # instead of re-running inference: it is ~3x cheaper and guarantees
        # the ensemble is built from exactly the scores the rows report.
        timm_key = timm_model_names.get(disp_name, disp_name)
        cached = _PROB_CACHE.get(_cache_key(timm_key, eval_path))
        if cached is not None:
            model_probs[disp_name] = cached
            if ground_truth is None:
                ground_truth = _labels_for(eval_path)
            continue
        m_name = timm_model_names[disp_name]
        m = _timm.create_model(m_name, pretrained=False, num_classes=2)
        sd = torch.load(ckpt, map_location=device, weights_only=True)
        m.load_state_dict(sd["state_dict"])
        m = m.to(device)
        m.eval()
        tf = get_timm_transforms(m, is_training=False)
        ds = DeepfakeDataset(eval_path, transform=tf)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        probs_list, labels_list = [], []
        softmax = torch.nn.Softmax(dim=1)
        with torch.no_grad():
            for batch in dl:
                imgs, lbs = batch[0], batch[1]
                p = softmax(m(imgs.to(device)))
                probs_list.extend(p[:, config.FAKE_LABEL_IDX].cpu().numpy().tolist())
                labels_list.extend(lbs.numpy().tolist())
        model_probs[disp_name] = np.array(probs_list)
        _PROB_CACHE[_cache_key(disp_name, eval_path)] = np.array(probs_list)
        if ground_truth is None:
            ground_truth = np.array(labels_list)

    # YOLO probs.
    #
    # The file list is built from _collect_samples() so it is in the SAME
    # order as the timm arrays above. Previously this used a separate
    # ["real", "fake"] listing, which reversed the class order relative to
    # DeepfakeDataset and silently paired each image's YOLO probability with
    # a different image's timm probabilities.
    cached_yolo = _PROB_CACHE.get(_cache_key("yolo26n-cls", eval_path))
    if cached_yolo is not None:
        model_probs["YOLO26n-cls"] = cached_yolo
        if ground_truth is None:
            ground_truth = _labels_for(eval_path)
    elif yolo_checkpoint.exists():
        from ultralytics import YOLO
        yolo = YOLO(str(yolo_checkpoint))
        names = yolo.names
        fake_idx = next((k for k, v in names.items() if v.lower() == "fake"), 0)
        samples = _collect_samples(eval_path)
        if samples:
            device_str = "0" if (device.type == "cuda") else "cpu"
            arr = np.array(yolo_fake_probs_chunked(
                yolo, [str(p) for p, _ in samples], fake_idx,
                batch_size=batch_size, device=device_str))
            model_probs["YOLO26n-cls"] = arr
            _PROB_CACHE[_cache_key("yolo26n-cls", eval_path)] = arr

    if not model_probs or ground_truth is None:
        return None

    # Weighted average. Weights are applied by MODEL KEY (not by dict
    # insertion order) so the mapping stays correct if the model set changes.
    weight_by_key = {
        "YOLO26n-cls":      weights[0],
        "ConvNeXt-Tiny":    weights[1],
        "EfficientNet-B3":  weights[2],
    }
    used = [(weight_by_key.get(k, 0.0), v) for k, v in model_probs.items()]
    total_w = sum(w for w, _ in used)
    if total_w <= 0:
        used = [(1.0, v) for _, v in used]
        total_w = float(len(used))
    ensemble_probs = sum((w / total_w) * pa for w, pa in used)

    res = _compute_metrics(ground_truth, ensemble_probs, threshold=threshold)
    res["threshold"] = threshold
    return res


# ── Reporting ───────────────────────────────────────────────────────────────

def _fmt(val: Optional[float], pct: bool = False) -> str:
    if val is None:
        return "—"
    return f"{val*100:.1f}%" if pct else f"{val:.4f}"


def build_markdown_table(results: Dict, domain_note: Optional[Dict[str, bool]] = None) -> str:
    """
    results[train_ds][test_ds][model_key] = metrics_dict | None

    domain_note maps train_ds -> True when that domain has its own checkpoint.
    Rows without a domain-specific checkpoint are labelled explicitly, because
    re-labelling the same shared weights under two "Trained on" headings would
    misrepresent the experiment.
    """
    MODEL_ORDER = ["EfficientNet-B3", "ConvNeXt-Tiny", "YOLO26n-cls", "Ensemble"]
    TRAIN_ORDER = ["celebdf", "dfdc"]
    TEST_ORDER  = list(TEST_DATASETS.keys())
    domain_note = domain_note or {}

    lines = [
        "# Cross-Dataset Evaluation Table",
        "",
        "> Metric shown: **ROC-AUC** (F1 | Brier) — in-domain cells are highlighted with ⬛.",
        "> ROC-AUC is computed with **FAKE as the positive class**; 0.5 = chance.",
    ]

    shared_domains = [d for d in TRAIN_ORDER if not domain_note.get(d, True)]
    if shared_domains:
        lines += [
            "",
            "> ⚠️ **Shared-checkpoint rows:** "
            + ", ".join(f"**{d.upper()}**" for d in shared_domains)
            + " has no domain-specific checkpoint (no `models/checkpoints/<domain>/` entry), "
              "so the shared/global checkpoint is used. These rows are **not** independent "
              "cross-domain results — train a per-domain model before claiming "
              "cross-dataset generalisation.",
        ]
    lines += [
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
            if not domain_note.get(train_ds, True):
                tag = "⚠ shared ckpt"
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


def export_csv_table(results: Dict, target_paths: List[Path],
                     domain_note: Optional[Dict[str, bool]] = None):
    """
    Exports structured metrics to CSV files.

    Every model is emitted, including those with no metrics (as empty cells),
    so the table never silently drops a backbone. Rows are annotated with
    whether a domain-specific checkpoint was used.
    """
    import csv
    MODEL_ORDER = ["EfficientNet-B3", "ConvNeXt-Tiny", "YOLO26n-cls", "Ensemble"]
    TRAIN_ORDER = ["celebdf", "dfdc"]
    TEST_ORDER  = list(TEST_DATASETS.keys())
    domain_note = domain_note or {}

    rows = []
    for train_ds in TRAIN_ORDER:
        for test_name in TEST_ORDER:
            for model_key in MODEL_ORDER:
                m = (results.get(train_ds) or {}).get(test_name) or {}
                mets = m.get(model_key)
                if not mets:
                    rows.append({
                        "Trained on": train_ds.upper(),
                        "Test dataset": test_name,
                        "Model": model_key,
                        "Checkpoint": "domain-specific" if domain_note.get(train_ds, True) else "shared (no per-domain ckpt)",
                        "Accuracy (%)": "",
                        "Balanced Accuracy (%)": "",
                        "Recall Fake (%)": "",
                        "Recall Real (%)": "",
                        "F1-Score (%)": "",
                        "ROC-AUC": "",
                        "Brier Score": "",
                        "Threshold": ""
                    })
                    continue
                # Confusion matrix layout is [FAKE, REAL] for both axes:
                #   cm[0][0] TP (fake as fake), cm[0][1] FN (fake as real)
                #   cm[1][0] FP (real as fake), cm[1][1] TN (real as real)
                cm = mets.get("confusion_matrix") or []
                rec_fake = rec_real = bal_acc = None
                if len(cm) == 2 and len(cm[0]) == 2 and len(cm[1]) == 2:
                    tp, fn = cm[0][0], cm[0][1]
                    fp, tn = cm[1][0], cm[1][1]
                    rec_fake = tp / (tp + fn) if (tp + fn) else None
                    rec_real = tn / (tn + fp) if (tn + fp) else None
                    if rec_fake is not None and rec_real is not None:
                        bal_acc = (rec_fake + rec_real) / 2.0

                def _pct(v):
                    return "" if v is None else f"{v*100:.2f}"

                rows.append({
                    "Trained on": train_ds.upper(),
                    "Test dataset": test_name,
                    "Model": model_key,
                    "Checkpoint": "domain-specific" if domain_note.get(train_ds, True) else "shared (no per-domain ckpt)",
                    "Accuracy (%)": f"{mets.get('accuracy', 0)*100:.2f}",
                    "Balanced Accuracy (%)": _pct(bal_acc),
                    "Recall Fake (%)": _pct(rec_fake),
                    "Recall Real (%)": _pct(rec_real),
                    "F1-Score (%)": f"{mets.get('f1_score', 0)*100:.2f}",
                    "ROC-AUC": f"{mets.get('roc_auc', 0):.4f}",
                    "Brier Score": f"{mets.get('brier_score', 0):.4f}",
                    "Threshold": f"{mets.get('threshold', 0.5):.4f}"
                })
    if not rows:
        return
    for path in target_paths:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Saved CSV table: {path}")


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


def _free_device_memory():
    """
    Releases cached GPU memory.

    The cross-table loads three backbones (up to 334 MB of weights each) and
    evaluates them one after another. Without an explicit cache flush the
    previous model's allocations stay reserved, and the next one can fail
    with "CUDA error: out of memory" on a small GPU - which then poisons
    the context for every model that follows.
    """
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass



def _is_cuda_oom(exc: BaseException) -> bool:
    """True when an exception is a CUDA out-of-memory failure."""
    name = type(exc).__name__.lower()
    if "outofmemory" in name:
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


def _with_cpu_fallback(fn, *args, device, **kwargs):
    """
    Runs an evaluation, retrying on the CPU if the GPU runs out of memory.

    A 4 GB card cannot hold every backbone at once, and a CUDA OOM poisons
    the context for every model that follows - which previously blanked the
    whole cross-dataset table after one failure. The CPU retry keeps the
    table complete.
    """
    try:
        return fn(*args, device=device, **kwargs)
    except Exception as exc:
        if not _is_cuda_oom(exc):
            raise
        logger.warning(
            "  CUDA out of memory - retrying this model on the CPU. "
            "The table stays complete; this model is simply slower."
        )
        _free_device_memory()
        import torch
        return fn(*args, device=torch.device("cpu"), **kwargs)


# ── Main evaluation loop ────────────────────────────────────────────────────

def run_cross_table(args):
    device = torch.device(args.device) if hasattr(args, "device") else config.DEVICE

    # results[train_ds][test_name][model_key] = metrics_dict | None
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # train_ds -> True when that domain has its own checkpoint (genuine
    # cross-domain row), False when a shared checkpoint was substituted.
    domain_note: Dict[str, bool] = {}

    timm_meta = {
        "EfficientNet-B3": "efficientnet_b3.ra2_in1k",
        "ConvNeXt-Tiny":   "convnext_tiny.fb_in22k_ft_in1k",
    }

    # Metrics are reported at the operating point the service ships, not at
    # an arbitrary 0.5.
    operating_points = resolve_operating_points(list(timm_meta.values()))

    # Results are checkpointed after every evaluation so an interrupted run
    # (a machine restart, a killed process) does not throw away hours of
    # work, and so a re-run can pick up where it stopped.
    checkpoint_path = OUT_DIR / "cross_table.partial.json"
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if checkpoint_path.exists() and not getattr(args, "force", False):
        try:
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "results" in payload:
                results = payload["results"]
                for key, arr in (payload.get("probs") or {}).items():
                    mk, td = key.split("::", 1)
                    _PROB_CACHE[(mk, td)] = np.asarray(arr, dtype=float)
                logger.info(
                    f"Resuming from {checkpoint_path}: "
                    f"{sum(1 for t in results.values() for m in t.values() if m)} cells, "
                    f"{len(_PROB_CACHE)} cached probability arrays"
                )
            else:
                results = payload
                logger.info(f"Resuming from {checkpoint_path}")
        except Exception as exc:
            logger.warning(f"Could not resume from {checkpoint_path}: {exc}")
            results = {}

    def _save_checkpoint():
        try:
            payload = {
                "results": results,
                # The per-model P(fake) arrays let a resumed run rebuild the
                # ensemble without re-running inference.
                "probs": {f"{k[0]}::{k[1]}": v.tolist()
                          for k, v in _PROB_CACHE.items()},
            }
            checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Could not write the partial checkpoint: {exc}")

    only = args.only
    only_pair = None
    if only:
        only_pair = tuple(part.strip() for part in only.split(":", 1))
        if len(only_pair) != 2:
            parser.error("--only must be of the form TRAIN:TEST")

    for train_ds in TRAIN_DATASETS:
        if only_pair and train_ds != only_pair[0]:
            continue
        results.setdefault(train_ds, {})
        logger.info(f"\n{'='*70}")
        logger.info(f"  TRAIN DOMAIN: {train_ds.upper()}")
        logger.info(f"{'='*70}")

        # Resolve checkpoints once per domain so we can report honestly
        # whether this row is a genuine cross-domain experiment.
        ckpts = {dn: _timm_checkpoint(train_ds, mn) for dn, mn in timm_meta.items()}
        yolo_ckpt, yolo_is_domain = _yolo_checkpoint(train_ds)
        all_domain_specific = all(flag for _, flag in ckpts.values()) and yolo_is_domain

        if not all_domain_specific:
            missing = [dn for dn, (_, flag) in ckpts.items() if not flag]
            if not yolo_is_domain:
                missing.append("YOLO26n-cls")
            logger.warning(
                f"  No domain-specific checkpoint for '{train_ds}' (missing: {', '.join(missing)}). "
                "Falling back to the shared/global checkpoint - this row is NOT a genuine "
                "cross-domain result and will be labelled accordingly."
            )
        domain_note[train_ds] = all_domain_specific

        # The operating point actually shipped for each model in this domain.
        ops = operating_points.get(train_ds, {})
        logger.info(
            "  Operating points (shipped thresholds): "
            + ", ".join(f"{k}={v:.4f}" for k, v in sorted(ops.items()))
        )

        for test_name, test_base_dir in TEST_DATASETS.items():
            if only_pair and test_name != only_pair[1]:
                continue
            done = results[train_ds].get(test_name) or {}
            expected = {"EfficientNet-B3", "ConvNeXt-Tiny", "YOLO26n-cls", "Ensemble"}
            if expected.issubset(set(done.keys())) and all(done.get(k) for k in expected) and not only_pair:
                logger.info(f"\n  → [{test_name}] already evaluated - skipping")
                continue

            results[train_ds].setdefault(test_name, {})
            logger.info(f"\n  → Evaluating on [{test_name}]  ({test_base_dir})")

            # timm models
            for disp_name, model_name in timm_meta.items():
                ckpt, _ = ckpts[disp_name]
                logger.info(f"    {disp_name}: {ckpt}")
                try:
                    m = _with_cpu_fallback(
                        evaluate_timm_on_dir, model_name, ckpt, test_base_dir,
                        batch_size=args.batch_size,
                        threshold=ops.get(model_name, 0.5),
                        device=device)
                    results[train_ds][test_name][disp_name] = m
                    if m:
                        logger.info(f"      AUC={m['roc_auc']:.4f}  F1={m['f1_score']:.4f}  Brier={m['brier_score']:.4f}")
                    _save_checkpoint()
                except Exception as exc:
                    logger.error(f"    {disp_name} failed: {exc}\n{traceback.format_exc()}")
                    results[train_ds][test_name][disp_name] = None
                finally:
                    _free_device_memory()

            # YOLO
            logger.info(f"    YOLO26n-cls: {yolo_ckpt}")
            try:
                m = _with_cpu_fallback(
                    evaluate_yolo_on_dir, yolo_ckpt, test_base_dir,
                    batch_size=args.batch_size,
                    threshold=ops.get("yolo26n-cls", 0.5),
                    device=device)
                results[train_ds][test_name]["YOLO26n-cls"] = m
                if m:
                    logger.info(f"      AUC={m['roc_auc']:.4f}  F1={m['f1_score']:.4f}  Brier={m['brier_score']:.4f}")
                else:
                    logger.warning("      YOLO26n-cls: no metrics produced (checkpoint missing or no samples)")
                _save_checkpoint()
            except Exception as exc:
                logger.error(f"    YOLO failed: {exc}\n{traceback.format_exc()}")
                results[train_ds][test_name]["YOLO26n-cls"] = None
            finally:
                _free_device_memory()

            # Ensemble
            logger.info(f"    Soft-Voting Ensemble…")
            try:
                timm_ckpts = {dn: path for dn, (path, _) in ckpts.items()}
                m = _with_cpu_fallback(
                    evaluate_ensemble_on_dir, timm_ckpts, timm_meta, yolo_ckpt, test_base_dir,
                    weights=(1.0, 1.0, 1.0), batch_size=args.batch_size,
                    threshold=ops.get("ensemble", 0.5),
                    device=device)
                results[train_ds][test_name]["Ensemble"] = m
                if m:
                    logger.info(f"      Ensemble AUC={m['roc_auc']:.4f}  F1={m['f1_score']:.4f}  Brier={m['brier_score']:.4f}")
                _save_checkpoint()
            except Exception as exc:
                logger.error(f"    Ensemble failed: {exc}\n{traceback.format_exc()}")
                results[train_ds][test_name]["Ensemble"] = None
            finally:
                _free_device_memory()

    # ── Save outputs ───────────────────────────────────────────────────────
    json_path = OUT_DIR / "cross_table.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved JSON: {json_path}")
    try:
        checkpoint_path.unlink()
    except OSError:
        pass

    md_path = OUT_DIR / "cross_table.md"
    md = build_markdown_table(results, domain_note)
    md_path.write_text(md, encoding="utf-8")
    logger.info(f"Saved Markdown table: {md_path}")

    csv_path = OUT_DIR / "cross_table.csv"
    root_csv_path = config.REPORTS_DIR / "cross_dataset_table.csv"
    export_csv_table(results, [csv_path, root_csv_path], domain_note)

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
    parser.add_argument("--force", action="store_true",
                        help="Ignore any partial checkpoint and start from scratch")
    parser.add_argument(
        "--only", type=str, default=None,
        help="Restrict the run to one cell, as TRAIN:TEST (e.g. celebdf:CelebDF). "
             "Repeated invocations accumulate in the checkpoint.",
    )
    parser.add_argument("--device",     type=str, default=str(config.DEVICE))
    args = parser.parse_args()
    run_cross_table(args)


if __name__ == "__main__":
    main()

