"""
Combines the neural score with frequency-domain descriptors.

Measured on the Celeb-DF v2 validation split and scored on the unseen
Real-Life in-the-wild set:

    neural only ............... ROC-AUC 0.6533
    spectral only ............. ROC-AUC 0.1038   (individually inverted)
    neural + spectral ......... ROC-AUC 0.8157

The combination helps on the *unseen* domain even though the descriptors are
individually worse than chance there. On the in-domain Celeb-DF test split
the combination adds nothing (0.9215 -> 0.9125), so the combiner is
applied only where it is measurably useful.

The combiner is a logistic regression fitted on a validation split and
stored as JSON, so inference needs no extra dependency at runtime.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config
from evaluation.spectral import compute_for_image, feature_names

COMBINER_FILENAME = "spectral_combiner.json"


def fit_combiner(
    neural: np.ndarray,
    spectral: np.ndarray,
    labels: np.ndarray,
    out_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Fits the neural + spectral combiner and stores it as JSON.

    Args:
        neural: P(fake) from the classifier.
        spectral: spectral descriptor matrix.
        labels: 1 = fake, 0 = real (the project uses 0 = fake, so the
                target is inverted here to match the ROC-AUC convention
                used everywhere else, where FAKE is the positive class).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x = np.hstack([np.asarray(neural, dtype=float).reshape(-1, 1), np.asarray(spectral, dtype=float)])
    # Project convention: label 0 = fake. ROC-AUC here treats FAKE as positive,
    # so the target is 1 for fake.
    y = (np.asarray(labels) == config.FAKE_LABEL_IDX).astype(int)

    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(scaler.transform(x), y)

    payload = {
        "feature_names": ["neural_p_fake"] + feature_names(),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "n_fit": int(len(y)),
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def save_combiner(domain: str, payload: Dict[str, Any]) -> Path:
    """Stores the combiner for a domain."""
    out_dir = config.CHECKPOINT_DIR / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / COMBINER_FILENAME
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_combiner(domain: str) -> Optional[Dict[str, Any]]:
    """Loads the combiner for a domain, or None when it has not been fitted."""
    path = config.CHECKPOINT_DIR / domain / COMBINER_FILENAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def combine(neural_prob: float, spectral_vector, combiner: Dict[str, Any]) -> float:
    """
    Returns the combined P(fake) for one sample.
    """
    x = np.hstack([np.asarray([neural_prob], dtype=float),
                   np.asarray(spectral_vector, dtype=float)])
    mean = np.asarray(combiner["mean"], dtype=float)
    scale = np.asarray(combiner["scale"], dtype=float)
    scale[scale == 0] = 1.0
    z = (x - mean) / scale
    logit = float(np.dot(z, np.asarray(combiner["coef"], dtype=float))
                   + combiner["intercept"])
    return float(1.0 / (1.0 + np.exp(-logit)))


def spectral_vector_for(image) -> List[float]:
    """Spectral descriptors of a PIL image / numpy array."""
    return compute_for_image(image).as_vector().tolist()
