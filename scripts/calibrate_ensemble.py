"""
Calibrates the soft-voting ensemble on each training domain's validation split.

The ensemble score is the mean of its members' P(fake). That mean has a
distribution of its own, so a member's calibrated threshold does not transfer
to it: reusing the median member threshold left the ensemble calling almost
nothing fake at the shipped operating point. The ensemble therefore gets its
own validation-split calibration, exactly like a single model, and the result
is written next to the per-model thresholds:

    models/checkpoints/<domain>/threshold_ensemble.json

Artefacts read:  per-domain checkpoints (timm + YOLO)
Artefact written: threshold_ensemble.json per domain
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import torch  # noqa: E402

import config  # noqa: E402
from evaluation.calibration import calibrate_ensemble, calibrate_yolo  # noqa: E402

TIMM_MODELS = {
    "efficientnet_b3.ra2_in1k": "efficientnet_b3.ra2_in1k",
    "convnext_tiny.fb_in22k_ft_in1k": "convnext_tiny.fb_in22k_ft_in1k",
}


def main() -> None:
    device = torch.device(config.DEVICE)
    val_dirs = {
        "celebdf": config.PROCESSED_CELEBDF_DIR / "val",
        "dfdc": config.PROCESSED_DFDC_DIR / "val",
    }

    for domain, val_dir in val_dirs.items():
        if not val_dir.exists():
            print(f"[skip] {domain}: no validation split at {val_dir}")
            continue
        print(f"\n=== {domain}: calibrating the ensemble on {val_dir} ===")

        checkpoints = {}
        for name in TIMM_MODELS:
            ckpt = config.CHECKPOINT_DIR / domain / f"{name}_best.pth"
            if not ckpt.exists():
                print(f"  [skip] missing checkpoint {ckpt}")
                continue
            checkpoints[name] = ckpt

        yolo_ckpt = ROOT / "runs" / "yolo_cls" / domain / f"deepfake_{domain}" / "weights" / "best.pt"
        if not yolo_ckpt.exists():
            print(f"  [skip] missing YOLO checkpoint {yolo_ckpt}")
            continue

        yolo_report = calibrate_yolo(
            yolo_ckpt, val_dir, domain,
            device="0" if device.type == "cuda" else "cpu", batch_size=16,
        )
        at = yolo_report["at_chosen_threshold"]
        print(f"  YOLO chosen threshold : {yolo_report['chosen_threshold']}")
        print(f"  YOLO fake/real recall : {at['fake_recall']:.4f} / {at['real_recall']:.4f}")

        report = calibrate_ensemble(
            checkpoints, yolo_ckpt, val_dir, domain,
            device=device, batch_size=16,
        )
        at = report["at_chosen_threshold"]
        print(f"  chosen threshold : {report['chosen_threshold']}")
        print(f"  fake recall      : {at['fake_recall']:.4f}")
        print(f"  real recall      : {at['real_recall']:.4f}")
        print(f"  false positive   : {at['false_positive_rate']:.4f}")
        print(f"  written to       : {report['written_to']}")




def rebuild_summary() -> None:
    """
    Rebuilds reports/calibration_summary.json from the threshold artefacts.

    The client deliverable's Operating Points section is generated from this
    summary, so it must cover every calibrated model: the per-model
    thresholds plus YOLO and the ensemble.
    """
    import json

    rows = []
    for domain in ("celebdf", "dfdc"):
        d = config.CHECKPOINT_DIR / domain
        for path in sorted(d.glob("threshold_*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.append({
                "domain": domain,
                "model": data.get("model_name", path.stem),
                "threshold": data["chosen_threshold"],
                "fpr_at_default_0.5": data["at_default_0.5"]["false_positive_rate"],
                "fpr_at_chosen": data["at_chosen_threshold"]["false_positive_rate"],
                "fake_recall_at_default_0.5": data["at_default_0.5"]["fake_recall"],
                "fake_recall_at_chosen": data["at_chosen_threshold"]["fake_recall"],
                "real_recall_at_chosen": data["at_chosen_threshold"]["real_recall"],
            })
    out = ROOT / "reports" / "calibration_summary.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\ncalibration_summary.json rebuilt with {len(rows)} entries -> {out}")


if __name__ == "__main__":
    main()
    rebuild_summary()
