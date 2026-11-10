"""
Regenerate reports/FINAL_CLIENT_DELIVERABLE.md from JSON/CSV artefacts only.
No hand-written numbers are injected — all data comes from the generated files.
"""
import json
import csv
import sys
from datetime import date
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
TEMPLATE = ROOT / "templates" / "client_deliverable_template.md"
OUT = REPORTS / "FINAL_CLIENT_DELIVERABLE.md"


def load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_csv_as_md_table(path: Path) -> str:
    if not path.exists():
        return "_Not yet generated — run the full training grid first._"
    return _csv_to_md(load_csv_rows(path))


def load_csv_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _csv_to_md(rows: List[dict]) -> str:
    if not rows:
        return "_Empty._"
    headers = list(rows[0].keys())
    header_row = "| " + " | ".join(headers) + " |"
    sep_row   = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_rows = ["| " + " | ".join(str(r.get(h, "")) for h in headers) + " |" for r in rows]
    return "\n".join([header_row, sep_row] + data_rows)


def build_metrics_section(summary: dict) -> str:
    if not summary:
        return "_Evaluation summary not found — run `python main.py evaluate` first._"
    lines = []
    for k, v in summary.items():
        if isinstance(v, float):
            lines.append(f"- **{k}**: {v:.4f}")
        else:
            lines.append(f"- **{k}**: {v}")
    return "\n".join(lines)


# ── Derived findings ────────────────────────────────────────────────────────
#
# Every statement below is derived from the generated artefacts. Nothing is
# hand-written, so the document cannot drift away from the actual results.

def _cm_mode(cm) -> str:
    """
    Classifies a confusion matrix (labels [FAKE, REAL]) as a constant-prediction
    collapse or a genuine (non-degenerate) decision boundary.

    cm[0] = row for true FAKE, cm[1] = row for true REAL.
    cm[:,1] is the "predicted REAL" column, cm[:,0] the "predicted FAKE" column.
    """
    if not cm or len(cm) < 2 or len(cm[0]) < 2:
        return "unknown"
    pred_fake_col = cm[0][0] + cm[1][0]
    pred_real_col = cm[0][1] + cm[1][1]
    if pred_fake_col == 0:
        return "all-real"
    if pred_real_col == 0:
        return "all-fake"
    return "discriminative"


def build_training_status_section() -> str:
    """
    Reports convergence status for each backbone, derived from its training
    history. A model whose best validation accuracy sits at the chance level
    has not learned a discriminative boundary and must not be presented as a
    trained detector.
    """
    rows = []
    for hist_path in sorted(REPORTS.glob("*_history.json")):
        name = hist_path.name.replace("_history.json", "")
        # "training_history.json" is the generic ResNet-18 demo run; give it a
        # readable label instead of the bare word "training".
        display = "resnet18 (demo run)" if name == "training" else name
        try:
            hist = json.loads(hist_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        val_acc = hist.get("val_acc") or []
        train_acc = hist.get("train_acc") or []
        if not val_acc:
            continue
        best_val = max(val_acc)
        best_train = max(train_acc) if train_acc else float("nan")
        epochs = len(val_acc)
        converged = best_val > 55.0
        rows.append({
            "name": display,
            "epochs": epochs,
            "best_val_acc": best_val,
            "best_train_acc": best_train,
            "status": "converged" if converged else "NOT converged (chance-level)",
        })

    if not rows:
        return "_No training histories found — models have not been trained yet._"

    lines = [
        "| Model | Epochs | Best Train Acc | Best Val Acc | Status |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['epochs']} | {r['best_train_acc']:.2f}% | "
            f"{r['best_val_acc']:.2f}% | {'Converged' if r['status'] == 'converged' else '**Not converged (chance-level)**'} |"
        )
    return "\n".join(lines)


def build_findings_section(summary: dict, cross_rows: list) -> str:
    """
    Derives the findings narrative from the artefacts:
      * collapse detection (constant prediction) from the confusion matrices
      * best/worst ROC-AUC across the cross-dataset table
      * shared-checkpoint disclosure from the CSV's Checkpoint column
    """
    lines: List[str] = []

    # 1. Collapse detection from the in-domain evaluation summary.
    cm = summary.get("confusion_matrix")
    if cm:
        mode = _cm_mode(cm)
        acc = summary.get("accuracy")
        auc = summary.get("roc_auc")
        if mode == "all-fake":
            lines.append(
                f"- **Constant-prediction collapse.** The evaluated checkpoint predicts *fake* for "
                f"every image in the test set (accuracy {acc:.4f}, ROC-AUC {auc:.4f}). "
                "An accuracy of 50% with F1 of 66.67% is the arithmetic signature of a "
                "single-class predictor on a balanced set, **not** evidence of a working detector."
            )
        elif mode == "all-real":
            lines.append(
                f"- **Constant-prediction collapse.** The evaluated checkpoint predicts *real* for "
                f"every image in the test set (accuracy {acc:.4f}, ROC-AUC {auc:.4f}). "
                "This is a degenerate decision boundary, not a usable detector."
            )
        else:
            lines.append(
                f"- The evaluated checkpoint produces a non-degenerate decision boundary "
                f"(accuracy {acc:.4f}, ROC-AUC {auc:.4f})."
            )

    # 2. Best / worst AUC across the cross-dataset table.
    scored = []
    for r in cross_rows:
        try:
            scored.append((float(r.get("ROC-AUC", "")), r))
        except (TypeError, ValueError):
            continue
    scored = [(a, r) for a, r in scored if a != ""]
    if scored:
        best_auc, best_row = max(scored, key=lambda x: x[0])
        worst_auc, worst_row = min(scored, key=lambda x: x[0])
        lines.append(
            f"- **Across the cross-dataset matrix** (Table 2), the strongest ROC-AUC is "
            f"{best_auc:.4f} ({best_row.get('Model')} on {best_row.get('Test dataset')}), and the "
            f"weakest is {worst_auc:.4f} ({worst_row.get('Model')} on {worst_row.get('Test dataset')})."
        )
        at_chance = [r for a, r in scored if 0.45 <= a <= 0.55]
        if at_chance:
            lines.append(
                f"- {len(at_chance)} of {len(scored)} cells sit within ±0.05 of the 0.50 chance "
                "line, indicating that the underlying weights are not yet discriminating between "
                "authentic and manipulated faces."
            )

    # 3. Shared-checkpoint disclosure.
    shared = {r.get("Trained on") for r in cross_rows if "shared" in str(r.get("Checkpoint", "")).lower()}
    if shared:
        lines.append(
            "- ⚠️ **Train-domain caveat.** The rows labelled "
            + ", ".join(f"**{s}**" for s in sorted(shared))
            + " in Table 2 were produced with the shared/global checkpoint, because no "
              "domain-specific checkpoint exists for them. Those rows are **not** independent "
              "cross-domain measurements; a per-domain model must be trained before any "
              "cross-dataset generalisation claim is valid."
        )

    # 4. Missing backbones.
    models_present = {r.get("Model") for r in cross_rows}
    for expected in ("YOLO26n-cls", "Ensemble"):
        if expected not in models_present:
            lines.append(f"- **{expected}** produced no metrics in this cycle and is absent from Table 2.")

    if not lines:
        return "_No findings could be derived — run the evaluation pipeline first._"
    return "\n".join(lines)


def main():
    summary     = load_json(REPORTS / "evaluation_summary.json")
    cross_rows  = load_csv_rows(REPORTS / "cross_dataset_table.csv")
    cross_table = _csv_to_md(cross_rows) if cross_rows else load_csv_as_md_table(REPORTS / "cross_dataset_table.csv")
    speed_table = load_csv_as_md_table(REPORTS / "speed_table.csv")
    metrics_md  = build_metrics_section(summary)
    findings_md = build_findings_section(summary, cross_rows)
    training_md = build_training_status_section()

    # Read the template (which must NOT contain hard-coded numbers)
    if not TEMPLATE.exists():
        sys.exit(f"ERROR: Template not found at {TEMPLATE}")
    template = TEMPLATE.read_text(encoding="utf-8")

    # Inject real data
    report = (
        template
        .replace("{{METRICS}}", metrics_md)
        .replace("{{CROSS_DATASET_TABLE}}", cross_table)
        .replace("{{SPEED_TABLE}}", speed_table)
        .replace("{{FINDINGS}}", findings_md)
        .replace("{{TRAINING_STATUS}}", training_md)
        .replace("{{DATE}}", date.today().isoformat())
    )

    # Guard: the shipped document must not contain unsubstituted placeholders.
    leftover = [tok for tok in ("{{METRICS}}", "{{CROSS_DATASET_TABLE}}",
                                "{{SPEED_TABLE}}", "{{FINDINGS}}",
                                "{{TRAINING_STATUS}}", "{{DATE}}") if tok in report]
    if leftover:
        sys.exit(f"ERROR: unsubstituted placeholders remain in report: {leftover}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(report, encoding="utf-8")
    print(f"[OK] Client deliverable written to {OUT}")


if __name__ == "__main__":
    main()

