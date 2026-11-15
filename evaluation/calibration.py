"""
Operating-point calibration for deepfake detectors.

A deepfake detector's default threshold of 0.5 is only sensible for a
balanced test set. On the real Celeb-DF v2 evaluation the fake class is
~91% of the data, and at a 0.5 threshold a large share of *authentic*
faces are called fake - which in a moderation or KYC setting means
accusing real people.

This module measures the trade-off on the VALIDATION split (never the test
split) and stores the operating point that meets a target false-positive
rate, so inference can use a threshold that is defensible rather than
arbitrary.

Artefact: models/checkpoints/<domain>/threshold.json
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

import config

DEFAULT_TARGET_FPR = 0.01

# Per-domain processed split roots.
DOMAIN_DIRS = {
    "celebdf": config.PROCESSED_CELEBDF_DIR,
    "dfdc": config.PROCESSED_DFDC_DIR,
}


def _resolve_model_transform(model_name: str):
    """Builds the model and the evaluation transform that matches training."""
    from models.detector import get_model
    from evaluation.evaluate import get_eval_transform

    model = get_model(model_name, num_classes=config.NUM_CLASSES, pretrained=False)
    transform = get_eval_transform(model, model_name)
    return model, transform


@torch.no_grad()
def collect_probabilities(
    model_name: str,
    checkpoint: Path,
    data_dir: Path,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Runs the model over a labelled split and returns P(fake) per sample.
    """
    from torch.utils.data import DataLoader
    from training.dataset import DeepfakeDataset

    device = device or config.DEVICE
    model, transform = _resolve_model_transform(model_name)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model = model.to(device).eval()

    dataset = DeepfakeDataset(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    probs: List[float] = []
    labels: List[int] = []
    for images, y in loader:
        logits = model(images.to(device))
        p = torch.softmax(logits, dim=1)[:, config.FAKE_LABEL_IDX]
        probs.extend(p.cpu().numpy().tolist())
        labels.extend(y.numpy().tolist())

    return {
        "probs": np.asarray(probs, dtype=float),
        "labels": np.asarray(labels, dtype=int),   # 0 = fake, 1 = real
        "n": len(labels),
    }


def threshold_curve(probs: np.ndarray, labels: np.ndarray) -> List[Dict[str, float]]:
    """
    Sweeps the decision threshold and reports the operating characteristics.

    ``labels`` uses the project convention: 0 = fake, 1 = real.
    """
    is_fake = (labels == config.FAKE_LABEL_IDX)
    is_real = ~is_fake
    rows = []
    for t in np.unique(np.round(np.concatenate([probs, [0.0, 1.0]]), 6))[::-1]:
        pred_fake = probs >= t
        tp = int(np.sum(pred_fake & is_fake))
        fp = int(np.sum(pred_fake & is_real))
        fn = int(np.sum(~pred_fake & is_fake))
        tn = int(np.sum(~pred_fake & is_real))
        rows.append({
            "threshold": float(t),
            "fake_recall": tp / max(1, tp + fn),          # TPR on the fake class
            "real_recall": tn / max(1, tn + fp),          # TNR on the real class
            "false_positive_rate": fp / max(1, fp + tn),  # real faces called fake
            "precision": tp / max(1, tp + fp),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    return rows


def choose_threshold(
    curve: List[Dict[str, float]],
    target_fpr: float = DEFAULT_TARGET_FPR,
    min_fake_recall: float = 0.0,
) -> tuple:
    """
    Picks the operating point to ship.

    Two candidates are considered:

    * **Balanced (Youden's J)** - maximises TPR - FPR. This is the shipped
      default: it is the point where the detector is most useful, trading
      detection against false accusation symmetrically.
    * **FPR-constrained** - maximises detection rate subject to the
      false-positive rate staying within ``target_fpr``. This is the point
      to use when wrongly accusing an authentic person is far more costly
      than missing a forgery, e.g. a KYC identity check.

    The FPR-constrained point can end up detecting almost nothing (on the
    imbalanced Celeb-DF v2 split the detector's scores are saturated, so
    there is no threshold that separates the classes at a 1% false-positive
    rate). That is reported rather than hidden.

    Returns:
        (fpr_constrained_row, target_met)
    """
    feasible = [
        r for r in curve
        if r["false_positive_rate"] <= target_fpr and r["fake_recall"] >= min_fake_recall
    ]
    if feasible:
        return max(feasible, key=lambda r: (r["fake_recall"], r["threshold"])), True

    floor_ok = [r for r in curve if r["fake_recall"] >= min_fake_recall]
    if floor_ok:
        return min(floor_ok, key=lambda r: (r["false_positive_rate"], -r["fake_recall"])), False
    return min(curve, key=lambda r: r["false_positive_rate"]), False


def youden_threshold(curve: List[Dict[str, float]]) -> Dict[str, float]:
    """The threshold maximising TPR - FPR (Youden's J), the balanced point."""
    return max(curve, key=lambda r: r["fake_recall"] - (1.0 - r["real_recall"]))


def calibrate(
    model_name: str,
    checkpoint: Path,
    val_dir: Path,
    domain: str,
    target_fpr: float = DEFAULT_TARGET_FPR,
    min_fake_recall: float = 0.10,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Measures the operating point on the validation split and writes it to
    models/checkpoints/<domain>/threshold.json.
    """
    data = collect_probabilities(model_name, checkpoint, val_dir, device=device)
    curve = threshold_curve(data["probs"], data["labels"])
    constrained, target_met = choose_threshold(curve, target_fpr, min_fake_recall)
    default_050 = min(curve, key=lambda r: abs(r["threshold"] - 0.5))
    youden = youden_threshold(curve)

    def _pack(r):
        return {
            "threshold": round(r["threshold"], 6),
            "fake_recall": round(r["fake_recall"], 6),
            "real_recall": round(r["real_recall"], 6),
            "false_positive_rate": round(r["false_positive_rate"], 6),
            "precision": round(r["precision"], 6),
        }

    report = {
        "domain": domain,
        "model_name": model_name,
        "checkpoint": str(checkpoint),
        "validation_dir": str(val_dir),
        "validation_samples": data["n"],
        "target_false_positive_rate": target_fpr,
        "target_fpr_achievable": target_met,
        # Shipped default: the balanced operating point.
        "chosen_threshold": round(youden["threshold"], 6),
        "chosen_policy": "youden (maximises TPR - FPR)",
        "at_chosen_threshold": _pack(youden),
        # Alternative for use cases where a false accusation is far costlier.
        "fpr_constrained_threshold": round(constrained["threshold"], 6),
        "at_fpr_constrained_threshold": _pack(constrained),
        "at_default_0.5": _pack(default_050),
        "note": (
            "Measured on the validation split only; the test split is never "
            "used for threshold selection. 'chosen_threshold' is the balanced "
            "operating point. 'fpr_constrained_threshold' maximises detection "
            "rate subject to the target false-positive rate; 'target_fpr_"
            "achievable' is False when no point on this split meets the "
            "target, which is itself a finding about detector quality."
        ),
    }

    out_dir = config.CHECKPOINT_DIR / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    # One file per model: a single threshold.json per domain would be
    # overwritten by each model calibrated in turn.
    out_path = out_dir / f"threshold_{model_name}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["written_to"] = str(out_path)
    return report




def yolo_fake_probs_chunked(
    yolo,
    paths: List[str],
    fake_idx: int,
    batch_size: int = 32,
    device: str = "cpu",
    imgsz: int = 224,
) -> List[float]:
    """
    P(fake) for each path, predicted in bounded chunks.

    Ultralytics' predictor takes a LIST source as one batch: the forward pass
    grows with the size of the split and a 4 GB card dies at a few thousand
    images, no matter what ``batch=`` says. Chunking the paths keeps each
    forward pass at ``batch_size`` images, which bounds GPU memory
    independent of the split size.
    """
    probs: List[float] = []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start:start + batch_size]
        # stream=True keeps only one Result alive at a time; without it
        # Ultralytics retains every Result and its tensors.
        results = yolo.predict(chunk, imgsz=imgsz, batch=batch_size,
                               device=device, verbose=False, stream=True)
        probs.extend(float(r.probs.data.cpu().numpy()[fake_idx]) for r in results)
    return probs


def collect_yolo_probabilities(
    checkpoint: Path,
    data_dir: Path,
    device: Optional[str] = None,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Runs the YOLO classifier over a labelled split and returns P(fake) per
    sample, in exactly the same order as ``collect_probabilities``.

    The order comes from ``DeepfakeDataset``, which is the canonical order
    every per-model probability array follows; mixing orders would average
    probabilities belonging to different images.
    """
    from ultralytics import YOLO
    from training.dataset import DeepfakeDataset

    dataset = DeepfakeDataset(data_dir)
    paths = [str(p) for p, _ in dataset.samples]
    labels = [lbl for _, lbl in dataset.samples]
    if not paths:
        return {"probs": np.asarray([], dtype=float),
                "labels": np.asarray([], dtype=int), "n": 0}

    yolo = YOLO(str(checkpoint))
    names = yolo.names
    fake_idx = next((k for k, v in names.items() if str(v).lower() == "fake"), 0)
    probs = yolo_fake_probs_chunked(
        yolo, paths, fake_idx, batch_size=batch_size, device=device or "cpu"
    )
    return {
        "probs": np.asarray(probs, dtype=float),
        "labels": np.asarray(labels, dtype=int),   # 0 = fake, 1 = real
        "n": len(labels),
    }


def calibrate_ensemble(
    checkpoints: Dict[str, Path],
    yolo_checkpoint: Path,
    val_dir: Path,
    domain: str,
    target_fpr: float = DEFAULT_TARGET_FPR,
    min_fake_recall: float = 0.10,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Calibrates the soft-voting ensemble on the validation split.

    The ensemble score is the mean of its members' P(fake), and that mean has
    a distribution of its own: a member's calibrated threshold does not
    transfer to it. Reusing the median member threshold left the ensemble
    calling almost nothing fake (F1 = 0 at the shipped operating point), so
    the ensemble gets its own validation split calibration like every other
    model.
    """
    member_probs = []
    labels = None
    n = 0
    for model_name, ckpt in sorted(checkpoints.items()):
        data = collect_probabilities(model_name, ckpt, val_dir, device=device, batch_size=batch_size)
        member_probs.append(data["probs"])
        labels = data["labels"]
        n = data["n"]

    yolo_data = collect_yolo_probabilities(
        yolo_checkpoint, val_dir,
        device="cpu" if (device is None or device.type == "cpu") else "0",
        batch_size=batch_size,
    )
    if yolo_data["n"] != n:
        raise RuntimeError(
            f"YOLO returned {yolo_data['n']} probabilities for {n} samples; "
            "the arrays would be misaligned"
        )
    member_probs.append(yolo_data["probs"])

    if any(len(p) != n for p in member_probs):
        raise RuntimeError("per-model probability arrays are misaligned")

    # Soft voting: the unweighted mean of the members' P(fake).
    ensemble_probs = np.mean(np.vstack(member_probs), axis=0)

    curve = threshold_curve(ensemble_probs, labels)
    constrained, target_met = choose_threshold(curve, target_fpr, min_fake_recall)
    default_050 = min(curve, key=lambda r: abs(r["threshold"] - 0.5))
    youden = youden_threshold(curve)

    def _pack(r):
        return {
            "threshold": round(r["threshold"], 6),
            "fake_recall": round(r["fake_recall"], 6),
            "real_recall": round(r["real_recall"], 6),
            "false_positive_rate": round(r["false_positive_rate"], 6),
            "precision": round(r["precision"], 6),
        }

    report = {
        "domain": domain,
        "model_name": "ensemble (soft voting, mean of members)",
        "members": sorted(checkpoints.keys()) + ["yolo26n-cls"],
        "checkpoint": str(yolo_checkpoint.parent),
        "validation_dir": str(val_dir),
        "validation_samples": n,
        "target_false_positive_rate": target_fpr,
        "target_fpr_achievable": target_met,
        "chosen_threshold": round(youden["threshold"], 6),
        "chosen_policy": "youden (maximises TPR - FPR)",
        "at_chosen_threshold": _pack(youden),
        "fpr_constrained_threshold": round(constrained["threshold"], 6),
        "at_fpr_constrained_threshold": _pack(constrained),
        "at_default_0.5": _pack(default_050),
        "note": (
            "Ensemble score = unweighted mean of the members' P(fake), "
            "calibrated on the validation split like a single model. The test "
            "split is never used for threshold selection."
        ),
    }

    out_dir = config.CHECKPOINT_DIR / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "threshold_ensemble.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["written_to"] = str(out_path)
    return report


def load_threshold(domain: str, model_name: str = None, default: float = 0.5) -> float:
    """
    Loads the calibrated threshold for a domain/model, falling back to
    ``default`` when no calibration has been run.
    """
    out_dir = config.CHECKPOINT_DIR / domain
    candidates = []
    if model_name:
        candidates.append(out_dir / f"threshold_{model_name}.json")
    candidates.append(out_dir / "threshold.json")
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return float(data.get("chosen_threshold", default))
        except Exception:
            continue
    return default


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Calibrate the decision threshold on validation")
    ap.add_argument("--domain", required=True, choices=["celebdf", "dfdc"])
    ap.add_argument("--model", default="efficientnet_b3.ra2_in1k")
    ap.add_argument("--target-fpr", type=float, default=DEFAULT_TARGET_FPR)
    args = ap.parse_args()

    ck = config.CHECKPOINT_DIR / args.domain / f"{args.model}_best.pth"
    val = DOMAIN_DIRS[args.domain] / "val"
    rep = calibrate(args.model, ck, val, args.domain, target_fpr=args.target_fpr)
    print(json.dumps(rep, indent=2))
