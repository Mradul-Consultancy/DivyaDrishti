"""
Provenance tests for the client deliverable.

The deliverable must contain **measured** numbers only. These tests verify
that mechanically: every ROC-AUC / accuracy / F1 / Brier figure printed in
reports/FINAL_CLIENT_DELIVERABLE.md must appear verbatim in the generated
artefacts (reports/cross_dataset_table.csv, reports/evaluation_summary.json,
reports/speed_table.csv, reports/*_history.json).

If someone hand-edits the deliverable, or a generator starts emitting a
literal, these tests fail.
"""

import csv
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

REPORTS = ROOT / "reports"
DELIVERABLE = REPORTS / "FINAL_CLIENT_DELIVERABLE.md"
CROSS_CSV = REPORTS / "cross_dataset_table.csv"
SUMMARY_JSON = REPORTS / "evaluation_summary.json"
SPEED_CSV = REPORTS / "speed_table.csv"


def _read_deliverable() -> str:
    if not DELIVERABLE.exists():
        pytest.skip("client deliverable not generated yet")
    return DELIVERABLE.read_text(encoding="utf-8")


def _cross_rows():
    if not CROSS_CSV.exists():
        pytest.skip("cross-dataset table not generated yet")
    with open(CROSS_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestDeliverableProvenance:
    def test_no_unsubstituted_placeholders(self):
        txt = _read_deliverable()
        leftovers = re.findall(r"\{\{[A-Z_]+\}\}", txt)
        assert not leftovers, f"unsubstituted placeholders in deliverable: {leftovers}"

    def test_no_draft_markers(self):
        txt = _read_deliverable()
        for marker in ("To be completed", "_Not yet generated",
                       "_Evaluation summary not found", "_No training histories"):
            assert marker not in txt, f"draft marker {marker!r} still present"

    def test_every_cross_dataset_number_is_in_the_csv(self):
        """
        Each numeric cell in Table 2 of the deliverable must be present in
        reports/cross_dataset_table.csv.
        """
        txt = _read_deliverable()
        rows = _cross_rows()
        allowed = set()
        for r in rows:
            for key in ("Accuracy (%)", "F1-Score (%)", "ROC-AUC", "Brier Score"):
                v = (r.get(key) or "").strip()
                if v:
                    allowed.add(v)
                    # the deliverable prints percentages with two decimals
                    try:
                        allowed.add(f"{float(v):.2f}")
                        allowed.add(f"{float(v):.4f}")
                    except ValueError:
                        pass

        # Extract the Table 2 rows: "| TRAINED | TEST | MODEL | CKPT | a | f | auc | brier |"
        table_rows = [ln for ln in txt.splitlines()
                      if ln.startswith("|") and ln.count("|") == 9]
        assert table_rows, "Table 2 not found in the deliverable"

        checked = 0
        for ln in table_rows:
            cells = [c.strip() for c in ln.strip("|").split("|")]
            for value in cells[4:]:
                if not re.fullmatch(r"-?\d+(\.\d+)?", value):
                    continue
                assert value in allowed, (
                    f"number {value!r} in the deliverable is not present in "
                    f"reports/cross_dataset_table.csv - it must be measured, not typed"
                )
                checked += 1
        assert checked > 0, "no numeric cells were checked"

    def test_in_domain_metrics_match_evaluation_summary(self):
        txt = _read_deliverable()
        if not SUMMARY_JSON.exists():
            pytest.skip("evaluation summary not generated yet")
        summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))

        auc = summary.get("roc_auc")
        if auc is None:
            pytest.skip("no roc_auc in the evaluation summary")
        assert f"**roc_auc**: {auc:.4f}" in txt, (
            "the in-domain ROC-AUC in the deliverable does not match "
            "reports/evaluation_summary.json"
        )

    def test_speed_numbers_match_speed_table(self):
        txt = _read_deliverable()
        if not SPEED_CSV.exists():
            pytest.skip("speed table not generated yet")
        with open(SPEED_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                arch = row.get("Architecture", "")
                assert arch in txt, f"speed table row {arch!r} missing from the deliverable"
                fps = row.get("Throughput (FPS B=32)", "")
                if fps:
                    assert fps in txt, (
                        f"throughput {fps!r} for {arch!r} missing from the deliverable"
                    )

    def test_training_status_comes_from_history_files(self):
        """
        Every model row in the training-status table must correspond to a
        training history file, and the reported best validation accuracy must
        be the maximum in that file.
        """
        txt = _read_deliverable()
        section = txt.split("## 4. Model Training Status", 1)
        if len(section) < 2:
            pytest.skip("training status section not present")
        # Cut at the next "## " heading - the table's own |---| separator row
        # would otherwise end the block immediately.
        block = section[1].split("\n## ", 1)[0]

        rows = [ln for ln in block.splitlines() if ln.startswith("|")]
        assert len(rows) > 2, "training status table is empty"

        histories = {}
        for h in REPORTS.rglob("*_history.json"):
            try:
                data = json.loads(h.read_text(encoding="utf-8"))
            except Exception:
                continue
            val_acc = data.get("val_acc") or []
            if not val_acc:
                continue
            base = h.stem.replace("_history", "")
            domain = h.parent.name if h.parent.name in ("celebdf", "dfdc") else ""
            histories.setdefault((base, domain), []).append(max(val_acc))

        for ln in rows[2:]:
            cells = [c.strip() for c in ln.strip("|").split("|")]
            if len(cells) < 5:
                continue
            name = cells[0]
            best_val_txt = cells[3]
            base = name.split("[")[0].strip()
            domain = ""
            if "[" in name and name.endswith("]"):
                domain = name[name.index("[") + 1:-1].strip()
            # The deliverable renames training_history.json for readability.
            if base.startswith("resnet18 (demo run)"):
                base = "training"
            match = histories.get((base, domain))
            if match is None and domain:
                # fall back to the undomained history
                match = histories.get((base, ""))
            assert match, f"training status row {name!r} has no matching history file"
            expected = f"{max(match):.2f}%"
            assert best_val_txt == expected, (
                f"{name!r} reports best val acc {best_val_txt} but the history "
                f"file's maximum is {expected}"
            )


class TestArtefactsAreMachineGenerated:
    """The artefacts the deliverable reads must be machine-generated."""

    def test_cross_table_rows_are_all_populated(self):
        rows = _cross_rows()
        assert rows, "cross-dataset table is empty"
        empty = [
            (r.get("Trained on"), r.get("Test dataset"), r.get("Model"))
            for r in rows
            if not (r.get("ROC-AUC") or "").strip()
        ]
        assert not empty, f"cross-dataset rows without a measured ROC-AUC: {empty}"

    def test_cross_table_marks_checkpoint_provenance(self):
        rows = _cross_rows()
        for r in rows:
            ckpt = (r.get("Checkpoint") or "").strip()
            assert ckpt, "every cross-dataset row must record which checkpoint produced it"
            assert ckpt in ("domain-specific",) or ckpt.startswith("shared"), (
                f"unexpected checkpoint label {ckpt!r}"
            )
