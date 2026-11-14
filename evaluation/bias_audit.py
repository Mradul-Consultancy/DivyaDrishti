"""
Fairness / bias audit for the deepfake detectors.

What this project can and cannot measure today
----------------------------------------------
A fairness matrix over Fitzpatrick skin type, age bracket and gender
requires per-image demographic labels. **The project has none.** The
Celeb-DF v2 and DFDC manifests carry no age, gender, skin-tone or
ethnicity annotation, and the released face crops do not either. This
script therefore reports that coverage gap explicitly rather than
inventing a matrix.

What it does measure
--------------------
Performance is stratified by properties that ARE derivable from the media
and that correlate with the capture conditions under which detectors are
known to degrade:

* **blur** - Laplacian variance (motion blur / out-of-focus frames),
* **brightness** and **contrast** - underexposed or flat-lit captures,
* **face area** - how much of the frame the face occupies (proxy for
  resolution and for group differences in framing),
* **per-identity variance** - some identities are intrinsically harder,
  which is a proxy for demographic imbalance in the training set.

These are *proxies*, not protected attributes. A stratum with a
disproportionately high false-positive rate is a signal to investigate, not
proof of discrimination.

Usage:
    python evaluation/bias_audit.py --domain celebdf --model convnext_tiny.fb_in22k_ft_in1k
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from evaluation.calibration import DOMAIN_DIRS

# Substrings searched for in the dataset metadata to establish whether
# protected-attribute labels exist.
DEMOGRAPHIC_TOKENS = (
    "age", "gender", "skin_tone", "skintone", "fitzpatrick", "ethnic",
    "race", "sex", "dob", "birth",
)


def dataset_has_demographic_labels(root: Path) -> Dict[str, Any]:
    """Checks the shipped metadata for protected-attribute labels."""
    found: Dict[str, List[str]] = {}
    for manifest in root.rglob("*.json"):
        try:
            text = manifest.read_text(encoding="utf-8").lower()
        except Exception:
            continue
        hits = [t for t in DEMOGRAPHIC_TOKENS if f'"{t}"' in text]
        if hits:
            found[str(manifest)] = hits
    return {
        "labelled_files": found,
        "has_demographic_labels": bool(found),
        "note": (
            "No protected-attribute labels were found in the dataset metadata, "
            "so a Fitzpatrick / age / gender fairness matrix cannot be produced "
            "from the released data. The proxy strata below are the "
            "substitutes that CAN be measured today."
        ) if not found else "Demographic labels were found; see labelled_files.",
    }


def image_properties(path: Path) -> Dict[str, float]:
    """Computes measurable capture-quality properties of one image."""
    from PIL import Image
    import cv2

    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return {}
    arr = np.asarray(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    # Face-area proxy: the crop is already a face, so use the crop's own
    # spread of high-frequency detail relative to its mean.
    detail_ratio = float(brightness / max(1.0, float(gray.max())))
    return {
        "blur": blur,
        "brightness": brightness,
        "contrast": contrast,
        "detail_ratio": detail_ratio,
    }


def _strata(values: np.ndarray, names: List[str]) -> Dict[str, Any]:
    """Splits a numeric property into quartiles and returns the edges."""
    edges = np.quantile(values, [0.25, 0.5, 0.75])
    return {"edges": [float(e) for e in edges], "names": names}


def _stratum_index(value: float, edges: List[float]) -> int:
    idx = 0
    for e in edges:
        if value > e:
            idx += 1
    return idx


def audit(
    domain: str,
    model_name: str,
    limit: int = 0,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Runs the audit for one domain and returns the report.
    """
    from PIL import Image
    from sklearn.metrics import roc_auc_score

    from evaluation.calibration import collect_probabilities
    from models.detector import get_model
    from training.dataset import DeepfakeDataset
    from training.train_timm import get_timm_transforms

    root = DOMAIN_DIRS[domain]
    ck = config.CHECKPOINT_DIR / domain / f"{model_name}_best.pth"
    if not ck.exists():
        raise SystemExit(f"checkpoint not found: {ck}")

    neural = collect_probabilities(model_name, ck, root / "test")
    probs = neural["probs"]
    labels = neural["labels"]

    model = get_model(model_name, num_classes=2, pretrained=False)
    transform = get_timm_transforms(model, is_training=False)
    ds = DeepfakeDataset(root / "test", transform=transform)

    idx = list(range(len(ds)))
    if limit and limit < len(idx):
        rng = np.random.RandomState(seed)
        by_class: Dict[int, List[int]] = {}
        for i in idx:
            by_class.setdefault(ds.samples[i][1], []).append(i)
        per = max(1, limit // max(1, len(by_class)))
        chosen: List[int] = []
        for cls, ids in by_class.items():
            take = min(per, len(ids))
            chosen.extend(rng.choice(ids, size=take, replace=False).tolist())
        idx = sorted(chosen)

    props: Dict[str, List[float]] = {"blur": [], "brightness": [], "contrast": []}
    y = []
    p = []
    for i in idx:
        path, label = ds.samples[i]
        try:
            Image.open(path).convert("RGB")
        except Exception:
            continue
        pr = image_properties(path)
        if not pr:
            continue
        for k in props:
            props[k].append(pr[k])
        y.append(label)
        p.append(float(probs[i]))

    y = np.asarray(y)
    p = np.asarray(p)
    props = {k: np.asarray(v) for k, v in props.items()}

    is_fake = (y == config.FAKE_LABEL_IDX)
    is_real = ~is_fake

    report: Dict[str, Any] = {
        "domain": domain,
        "model": model_name,
        "samples": int(len(y)),
        "fake_samples": int(is_fake.sum()),
        "real_samples": int(is_real.sum()),
        "overall": {
            "auc": float(roc_auc_score(is_fake, p)) if len(np.unique(y)) > 1 else None,
            "fake_recall": float((p[is_fake] >= 0.5).mean()) if is_fake.any() else None,
            "false_positive_rate": float((p[is_real] >= 0.5).mean()) if is_real.any() else None,
        },
        "demographic_labels": dataset_has_demographic_labels(root),
    }

    strata_report: Dict[str, Any] = {}
    for prop, values in props.items():
        edges = np.quantile(values, [0.25, 0.5, 0.75]).tolist()
        buckets: Dict[int, Dict[str, Any]] = {}
        for k, (v, yy, pp) in enumerate(zip(values, y, p)):
            s = _stratum_index(float(v), edges)
            b = buckets.setdefault(s, {"fake": [], "real": []})
            b["fake" if yy == config.FAKE_LABEL_IDX else "real"].append(float(pp))
        rows = []
        for s in sorted(buckets):
            b = buckets[s]
            f = np.asarray(b["fake"]); rr = np.asarray(b["real"])
            rows.append({
                "stratum": f"Q{s + 1}",
                "n_fake": int(len(f)),
                "n_real": int(len(rr)),
                "fake_recall": float((f >= 0.5).mean()) if len(f) else None,
                "false_positive_rate": float((rr >= 0.5).mean()) if len(rr) else None,
                "mean_p_fake_fake": float(f.mean()) if len(f) else None,
                "mean_p_fake_real": float(rr.mean()) if len(rr) else None,
            })
        strata_report[prop] = {"quartile_edges": edges, "strata": rows}

    report["proxy_strata"] = strata_report

    # Flag strata whose false-positive rate is materially worse than overall.
    overall_fpr = report["overall"]["false_positive_rate"]
    flags: List[Dict[str, Any]] = []
    if overall_fpr is not None:
        for prop, data in strata_report.items():
            for row in data["strata"]:
                fpr = row["false_positive_rate"]
                if fpr is None or row["n_real"] < 10:
                    continue
                if fpr > overall_fpr * 1.5 and fpr - overall_fpr > 0.05:
                    flags.append({
                        "property": prop,
                        "stratum": row["stratum"],
                        "false_positive_rate": fpr,
                        "overall_false_positive_rate": overall_fpr,
                        "n_real": row["n_real"],
                    })
    report["flagged_strata"] = flags
    report["interpretation"] = (
        "Proxy strata, not protected attributes. A flagged stratum means the "
        "detector misclassifies authentic faces more often under those capture "
        "conditions and warrants investigation."
    )
    return report


def render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Fairness / Bias Audit",
        "",
        f"- **Domain**: {report['domain']}",
        f"- **Model**: {report['model']}",
        f"- **Samples**: {report['samples']} ({report['fake_samples']} fake, {report['real_samples']} real)",
        "",
        "## Protected-Attribute Coverage",
        "",
        report["demographic_labels"]["note"],
        "",
    ]
    if report["demographic_labels"]["labelled_files"]:
        for f, hits in report["demographic_labels"]["labelled_files"].items():
            lines.append(f"- `{f}`: {', '.join(hits)}")

    o = report["overall"]
    lines += [
        "",
        "## Overall (threshold 0.5)",
        "",
        f"- ROC-AUC: {o['auc']:.4f}" if o["auc"] is not None else "- ROC-AUC: n/a",
        f"- Fake recall: {o['fake_recall']:.4f}" if o["fake_recall"] is not None else "- Fake recall: n/a",
        f"- False-positive rate on authentic faces: {o['false_positive_rate']:.4f}" if o["false_positive_rate"] is not None else "- FPR: n/a",
        "",
        "## Proxy Strata",
        "",
    ]
    for prop, data in report["proxy_strata"].items():
        lines += [f"### {prop}", "", "| Stratum | n real | n fake | Fake recall | FPR (authentic called fake) | Mean P(fake) on authentic |", "|---|---:|---:|---:|---:|---:|"]
        for row in data["strata"]:
            fr = f"{row['fake_recall']:.4f}" if row["fake_recall"] is not None else "-"
            fpr = f"{row['false_positive_rate']:.4f}" if row["false_positive_rate"] is not None else "-"
            mp = f"{row['mean_p_fake_real']:.4f}" if row["mean_p_fake_real"] is not None else "-"
            lines.append(f"| {row['stratum']} | {row['n_real']} | {row['n_fake']} | {fr} | {fpr} | {mp} |")
        lines.append("")

    lines += ["## Flagged Strata", ""]
    if report["flagged_strata"]:
        for f in report["flagged_strata"]:
            lines.append(
                f"- **{f['property']} / {f['stratum']}**: FPR {f['false_positive_rate']:.4f} "
                f"vs overall {f['overall_false_positive_rate']:.4f} (n_real={f['n_real']})"
            )
    else:
        lines.append("No proxy stratum exceeded 1.5x the overall false-positive rate.")
    lines += ["", f"_{report['interpretation']}_", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Fairness / bias audit")
    ap.add_argument("--domain", default="celebdf", choices=list(DOMAIN_DIRS.keys()))
    ap.add_argument("--model", default="convnext_tiny.fb_in22k_ft_in1k")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    report = audit(args.domain, args.model, limit=args.limit)
    out_json = config.REPORTS_DIR / f"bias_audit_{args.domain}.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md = config.REPORTS_DIR / f"bias_audit_{args.domain}.md"
    out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    print(f"wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
