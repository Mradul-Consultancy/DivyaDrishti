"""
Measures whether frequency-domain features add anything to the neural score.

The claim that a spectral branch "will elevate Real-Life AUC from ~0.60 to
>0.85" is a hypothesis, not a result. This script tests it honestly:

1. For the same images, record the neural P(fake) and the spectral
   descriptors.
2. Fit a logistic regression on the neural score alone (validation split).
3. Fit a logistic regression on the descriptors alone.
4. Fit a logistic regression on both.
5. Report AUC on the HELD-OUT test split for all three.

Everything is fitted on validation and scored on test, so the comparison
does not leak, and the neural and spectral rows are index-aligned by
construction.

Usage:
    python evaluation/evaluate_spectral.py --domain celebdf --model convnext_tiny.fb_in22k_ft_in1k
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from evaluation.calibration import DOMAIN_DIRS
from evaluation.spectral import compute_for_image, feature_names
from models.detector import get_model
from training.dataset import DeepfakeDataset
from training.train_timm import get_timm_transforms


def _load_model(model_name: str, checkpoint: Path, device):
    model = get_model(model_name, num_classes=2, pretrained=False)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    return model.to(device).eval()


@torch.no_grad()
def collect_paired(model_name: str, checkpoint: Path, dataset_dir: Path,
                   limit: int = 0, seed: int = 0, device=None):
    """
    Returns (neural_probs, spectral_features, labels) for the same images.

    A class-stratified random sample is used when ``limit`` is set: taking
    the first N samples would pick only one class, because the dataset
    lists 'fake' before 'real'.
    """
    from PIL import Image

    device = device or config.DEVICE
    model = _load_model(model_name, checkpoint, device)
    transform = get_timm_transforms(model, is_training=False)
    ds = DeepfakeDataset(dataset_dir, transform=transform)

    idx = list(range(len(ds)))
    if limit and limit < len(idx):
        rng = np.random.RandomState(seed)
        by_class = {}
        for i in idx:
            by_class.setdefault(ds.samples[i][1], []).append(i)
        per = max(1, limit // max(1, len(by_class)))
        chosen = []
        for cls, ids in by_class.items():
            take = min(per, len(ids))
            chosen.extend(rng.choice(ids, size=take, replace=False).tolist())
        idx = sorted(chosen)

    probs, feats, labels = [], [], []
    for i in idx:
        path, label = ds.samples[i]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        tensor = transform(img).unsqueeze(0).to(device)
        p = float(torch.softmax(model(tensor), dim=1)[0, config.FAKE_LABEL_IDX].item())
        probs.append(p)
        feats.append(compute_for_image(img).as_vector())
        labels.append(label)

    return (np.asarray(probs, dtype=float),
            np.asarray(feats, dtype=float),
            np.asarray(labels, dtype=int))


def _fit_and_score(x_train, y_train, x_test, y_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None
    scaler = StandardScaler().fit(x_train)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(scaler.transform(x_train), y_train)
    prob = clf.predict_proba(scaler.transform(x_test))[:, 1]
    return float(roc_auc_score(y_test, prob))


def main():
    ap = argparse.ArgumentParser(description="Measure the value of spectral features")
    ap.add_argument("--domain", default="celebdf", choices=list(DOMAIN_DIRS.keys()))
    ap.add_argument("--model", default="convnext_tiny.fb_in22k_ft_in1k")
    ap.add_argument("--limit", type=int, default=0, help="cap images per split (0 = all)")
    ap.add_argument("--test-dir", default=None,
                    help="Optional test split directory. Use it to evaluate the "
                         "cross-dataset case, e.g. --test-dir data/real_life, where "
                         "the neural score collapses but the evaluation domain is unseen.")
    args = ap.parse_args()

    domain = args.domain
    root = DOMAIN_DIRS[domain]
    ck = config.CHECKPOINT_DIR / domain / f"{args.model}_best.pth"
    if not ck.exists():
        raise SystemExit(f"checkpoint not found: {ck}")

    print("collecting validation pair...", flush=True)
    val_p, val_spec, val_y = collect_paired(args.model, ck, root / "val", args.limit, seed=0)
    test_dir = Path(args.test_dir) if args.test_dir else (root / "test")
    print(f"collecting test pair from {test_dir}...", flush=True)
    test_p, test_spec, test_y = collect_paired(args.model, ck, test_dir, args.limit, seed=1)

    results = {
        "domain": domain,
        "model": args.model,
        "samples_val": int(len(val_y)),
        "samples_test": int(len(test_y)),
        "test_dir": str(test_dir),
        "cross_dataset": bool(args.test_dir),
        "features": feature_names(),
        "note": (
            "Fitted on validation, scored on the held-out test split. "
            "FAKE is the positive class; 0.5 is chance."
        ),
    }

    results["auc_neural_only"] = _fit_and_score(
        val_p.reshape(-1, 1), val_y, test_p.reshape(-1, 1), test_y)
    results["auc_spectral_only"] = _fit_and_score(
        val_spec, val_y, test_spec, test_y)
    results["auc_neural_plus_spectral"] = _fit_and_score(
        np.hstack([val_p.reshape(-1, 1), val_spec]), val_y,
        np.hstack([test_p.reshape(-1, 1), test_spec]), test_y)

    univariate = {}
    for k, name in enumerate(feature_names()):
        univariate[name] = _fit_and_score(val_spec[:, [k]], val_y, test_spec[:, [k]], test_y)
    results["auc_single_feature"] = univariate

    # Persist the combiner so inference can use it where it is measurably
    # useful (the cross-domain case).
    try:
        from evaluation.spectral_combiner import fit_combiner, save_combiner
        payload = fit_combiner(val_p, val_spec, val_y)
        path = save_combiner(domain, payload)
        results["combiner_saved_to"] = str(path)
    except Exception as exc:
        results["combiner_saved_to"] = f"not saved: {exc}"

    print(json.dumps(results, indent=2))
    out = config.REPORTS_DIR / f"spectral_ablation_{domain}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
