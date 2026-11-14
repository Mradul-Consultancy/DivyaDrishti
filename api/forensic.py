"""
Forensic audit report generator.

Produces a signed-off record of a detection decision suitable for legal,
insurance or compliance use. The report contains:

* the SHA-256 digest of the exact media bytes that were analysed,
* the model, checkpoint, threshold and operating point used,
* the detector configuration (face detector, confidence, crop margin),
* the calibration evidence for that operating point,
* the verdict and its probabilities,
* the C2PA provenance status of the input,
* a timestamp and a digest of the report body itself.

The media itself is never embedded - only its digest - so the report can be
handed to a third party without disclosing biometric content.
"""

import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import config

REPORT_VERSION = "1.0"


@dataclass
class ForensicFinding:
    """One detection decision, ready for the record."""

    media_name: str
    media_sha256: str
    media_bytes: int
    media_type: str
    verdict: str
    fake_probability: float
    real_probability: float
    threshold: float
    threshold_source: str
    model: str
    checkpoint: str
    face_count: int
    face_detector: str
    face_min_confidence: float
    margin_ratio: float
    c2pa_present: Optional[bool] = None
    c2pa_signature_verified: Optional[bool] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "media_name": self.media_name,
            "media_sha256": self.media_sha256,
            "media_bytes": self.media_bytes,
            "media_type": self.media_type,
            "verdict": self.verdict,
            "fake_probability": round(self.fake_probability, 6),
            "real_probability": round(self.real_probability, 6),
            "threshold": self.threshold,
            "threshold_source": self.threshold_source,
            "model": self.model,
            "checkpoint": self.checkpoint,
            "face_count": self.face_count,
            "face_detector": self.face_detector,
            "face_min_confidence": self.face_min_confidence,
            "margin_ratio": self.margin_ratio,
            "c2pa_present": self.c2pa_present,
            "c2pa_signature_verified": self.c2pa_signature_verified,
            "notes": self.notes,
        }


def build_report(
    findings: List[ForensicFinding],
    operator: str = "api",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Builds the report body and its self-digest."""
    body = {
        "report_version": REPORT_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "operator": operator,
        "system": {
            "project": "DivyaDrishti AI",
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "findings": [f.to_dict() for f in findings],
        "limitations": [
            "The verdict reflects a statistical model, not a legal determination.",
            "Detection probabilities are model outputs under the stated operating point.",
            "Face crops are processed in memory only and are not retained by this report.",
        ],
    }
    if extra:
        body.update(extra)

    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    body["report_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return body


def write_json_report(body: Dict[str, Any], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return out_path


def render_markdown(body: Dict[str, Any]) -> str:
    """Human-readable rendering of the report."""
    lines = [
        "# Deepfake Forensic Audit Report",
        "",
        f"- **Report version**: {body['report_version']}",
        f"- **Generated at**: {body['generated_at']}",
        f"- **Operator**: {body['operator']}",
        f"- **Report SHA-256**: `{body.get('report_sha256', '')}`",
        "",
        "## Findings",
        "",
        "| Media | SHA-256 | Verdict | P(fake) | Threshold | Model | Faces | C2PA |",
        "|---|---|---|---:|---:|---|---:|---|",
    ]
    for f in body["findings"]:
        c2pa = "-" if f.get("c2pa_present") is None else ("present" if f["c2pa_present"] else "absent")
        lines.append(
            f"| `{f['media_name']}` | `{f['media_sha256'][:16]}...` | **{f['verdict'].upper()}** | "
            f"{f['fake_probability']:.4f} | {f['threshold']:.4f} ({f['threshold_source']}) | "
            f"{f['model']} | {f['face_count']} | {c2pa} |"
        )
    lines += ["", "## Limitations", ""]
    lines += [f"- {lim}" for lim in body["limitations"]]
    lines += [
        "",
        "> The media itself is not embedded in this report; only its SHA-256 digest "
        "is recorded, so the document can be shared without disclosing biometric content.",
        "",
    ]
    return "\n".join(lines)


def write_markdown_report(body: Dict[str, Any], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(body), encoding="utf-8")
    return out_path


def verify_report(path: Path) -> Dict[str, Any]:
    """
    Re-computes the report digest and reports whether it still matches.

    Gives an evidentiary chain a way to check that the report has not been
    altered since it was produced.
    """
    path = Path(path)
    body = json.loads(path.read_text(encoding="utf-8"))
    claimed = body.pop("report_sha256", None)
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "path": str(path),
        "claimed_sha256": claimed,
        "recomputed_sha256": actual,
        "intact": claimed == actual,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Verify a forensic audit report")
    ap.add_argument("--report", required=True, help="Path to a report JSON file")
    args = ap.parse_args()
    result = verify_report(Path(args.report))
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["intact"] else 1)
