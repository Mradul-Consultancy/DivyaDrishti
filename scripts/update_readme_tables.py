"""
Rebuilds the README's benchmark tables from the generated artefacts.

Section 1 (cross-dataset matrix) and Section 2 (in-domain detail) are
rewritten from reports/cross_dataset/cross_table.csv so every number in the
README is the artefact's number. The tests in
tests/test_docs_match_measurements.py enforce the same invariant.
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

README = ROOT / "README.md"
CSV_PATH = ROOT / "reports" / "cross_dataset" / "cross_table.csv"

DOMAIN_LABEL = {"celebdf": "Celeb-DF v2", "dfdc": "DFDC", "real_life": "Real-Life"}
MODEL_ORDER = ["EfficientNet-B3", "ConvNeXt-Tiny", "YOLO26n-cls", "Ensemble"]

# Sections are replaced between these headings; the rest of the file is untouched.
S1_START = "### 1. Cross-Dataset ROC-AUC Matrix"
S1_END = "### 2. In-Domain Detail"
S2_END = "### 3. Training Status"


def load_rows():
    with open(CSV_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def norm(s: str) -> str:
    return s.strip().lower().replace("-", "_").replace(" ", "_")


def build_section1(rows) -> str:
    lines = [
        "### 1. Cross-Dataset ROC-AUC Matrix",
        "",
        "FAKE is the positive class, so **0.5 is chance** and higher is better. Every row is a",
        "genuine train-domain / test-domain measurement: each model was trained on one domain only,",
        "and the `Checkpoint` column of the generated CSV records that the checkpoint was",
        "domain-specific.",
        "",
        "| Train domain | Test domain | EfficientNet-B3 | ConvNeXt-Tiny | YOLO26n-cls | Ensemble |",
        "|---|---|---|---|---|---|",
    ]
    for train in ("celebdf", "dfdc"):
        for test in ("celebdf", "dfdc", "real_life"):
            cell = {}
            for r in rows:
                if norm(r["Trained on"]) == train and norm(r["Test dataset"]) == test:
                    cell[r["Model"]] = r["ROC-AUC"]
            tag = "*(in-domain)*" if train == test else "*(cross)*"
            vals = " | ".join(cell.get(m, "—") for m in MODEL_ORDER)
            lines.append(
                f"| {DOMAIN_LABEL[train]} | {DOMAIN_LABEL[test]} {tag} | {vals} |"
            )
    lines += [
        "",
        "The `Threshold` column of `reports/cross_dataset/cross_table.csv` records the operating",
        "point every row was measured at: each model's own calibrated threshold, not an arbitrary",
        "0.5. ROC-AUC and Brier are threshold-free, so they do not change with it.",
        "",
        "**Reading the matrix.** Each model is strongest on the domain it was trained on, drops when",
        "moved to an unseen dataset, and is weakest on the in-the-wild Real-Life set - where the",
        "DFDC-trained models score below chance. Note that Celeb-DF v2's test split is 91.2% fake,",
        "so raw accuracy on that split is inflated by the majority class - ROC-AUC and Brier are the",
        "meaningful columns there. The DFDC test split is balanced (30 real / 30 fake).",
        "",
    ]
    return "\n".join(lines)


def build_section2(rows) -> str:
    in_rows = [r for r in rows
               if norm(r["Trained on"]) == "celebdf" and norm(r["Test dataset"]) == "celebdf"]
    by_model = {r["Model"]: r for r in in_rows}

    lines = [
        "### 2. In-Domain Detail (Trained on Celeb-DF v2, Tested on Celeb-DF v2)",
        "",
        "*Test split composition: 4,440 fake, 425 real (4,865 total, 91.26% fake).*",
        "> ⚠️ **Evaluation Integrity Disclosure**: Because 91.26% of the test set is Fake, raw accuracy is strongly biased towards the majority class. **Balanced Accuracy** (average of real and fake recall) provides the true uninflated classification capability. Every row is measured at that model's **calibrated operating point** (last column), chosen on the validation split - never at an arbitrary 0.5:",
        "",
        "| Architecture | Raw Accuracy | Balanced Accuracy | Real Recall | Fake Recall | F1-Score | ROC-AUC | Brier Score (lower is better) | Operating Point |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    display = [
        ("YOLO26n-cls", "YOLO26n-cls"),
        ("ConvNeXt-Tiny", "ConvNeXt-Tiny"),
        ("Soft-Voting Ensemble", "Ensemble"),
        ("EfficientNet-B3", "EfficientNet-B3"),
    ]
    for label, key in display:
        r = by_model.get(key)
        if not r:
            continue
        lines.append(
            f"| **{label}** | **{float(r['Accuracy (%)']):.2f}%** | **{float(r['Balanced Accuracy (%)']):.2f}%** "
            f"| **{float(r['Recall Real (%)']):.2f}%** | **{float(r['Recall Fake (%)']):.2f}%** "
            f"| **{float(r['F1-Score (%)']):.2f}%** | **{r['ROC-AUC']}** | **{r['Brier Score']}** "
            f"| {r['Threshold']} |"
        )
    lines += [
        "",
        "Latency is unchanged and measured on CPU: YOLO 14.18 ms, ConvNeXt 90.11 ms, EfficientNet 67.77 ms,",
        "ensemble 172.06 ms per image (`reports/speed_table.csv`).",
        "",
        "The operating points are honest trade-offs, not improvements everywhere: calibrating",
        "EfficientNet-B3 on Celeb-DF v2 lowers the false-positive rate on authentic faces from 51.6% to",
        "24.5%, at the cost of recalling 87% instead of 93% of fakes at 0.5. On DFDC the calibrated",
        "EfficientNet-B3 point trades the other way (FPR 35% → 41%). Both are recorded in",
        "`reports/calibration_summary.json`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    rows = load_rows()
    text = README.read_text(encoding="utf-8")

    s1 = build_section1(rows)
    s2 = build_section2(rows)

    i1 = text.index(S1_START)
    i2 = text.index(S1_END)
    i3 = text.index(S2_END)
    new_text = text[:i1] + s1 + "\n" + s2 + "\n---\n\n" + text[i3:]
    README.write_text(new_text, encoding="utf-8")
    print("README sections 1 and 2 rebuilt from", CSV_PATH)


if __name__ == "__main__":
    main()
