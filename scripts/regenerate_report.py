"""
Regenerate reports/FINAL_CLIENT_DELIVERABLE.md from JSON/CSV artefacts only.
No hand-written numbers are injected — all data comes from the generated files.
"""
import json
import csv
import sys
from pathlib import Path

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
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
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


def main():
    summary     = load_json(REPORTS / "evaluation_summary.json")
    cross_table = load_csv_as_md_table(REPORTS / "cross_dataset_table.csv")
    speed_table = load_csv_as_md_table(REPORTS / "speed_table.csv")
    metrics_md  = build_metrics_section(summary)

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
    )

    REPORTS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(report, encoding="utf-8")
    print(f"✅ Client deliverable written to {OUT}")


if __name__ == "__main__":
    main()

