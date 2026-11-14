"""
Tests that the documentation states the measured numbers.

The README and PROJECT_DOCUMENTATION quote figures (cross-dataset AUCs,
training accuracies, speed metrics, dataset sizes). Those quotes must match
the generated artefacts - otherwise the documentation drifts into describing
a system that no longer exists, or one that never did.

Every assertion here compares a documented value against the artefact it
came from.
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
README = ROOT / "README.md"
DOCS = ROOT / "PROJECT_DOCUMENTATION.md"
CROSS_CSV = REPORTS / "cross_dataset_table.csv"
SPEED_CSV = REPORTS / "speed_table.csv"
SUMMARY_JSON = REPORTS / "evaluation_summary.json"


def _read(path: Path) -> str:
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    return path.read_text(encoding="utf-8")


def _cross_rows():
    if not CROSS_CSV.exists():
        pytest.skip("cross-dataset table not generated yet")
    with open(CROSS_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestReadmeQuotesMeasuredNumbers:
    def test_readme_exists(self):
        assert README.exists()

    def test_cross_dataset_aucs_in_readme_are_measured(self):
        """
        Every ROC-AUC quoted in the README's results table must appear in
        reports/cross_dataset_table.csv.
        """
        txt = _read(README)
        rows = _cross_rows()
        measured = set()
        for r in rows:
            v = (r.get("ROC-AUC") or "").strip()
            if v:
                measured.add(v)
                measured.add(f"{float(v):.4f}")

        # The results table rows look like:
        # | Celeb-DF v2 | Celeb-DF v2 *(in-domain)* | 0.7684 | 0.9196 | ... |
        table = [ln for ln in txt.splitlines()
                 if ln.startswith("|") and "in-domain" in ln or
                    (ln.startswith("|") and "*(cross)*" in ln)]
        assert table, "README results table not found"

        checked = 0
        for ln in table:
            cells = [c.strip() for c in ln.strip("|").split("|")]
            for value in cells[2:]:
                if not re.fullmatch(r"\d\.\d{4}", value):
                    continue
                assert value in measured, (
                    f"README quotes ROC-AUC {value!r} which is not in "
                    f"reports/cross_dataset_table.csv"
                )
                checked += 1
        assert checked >= 24, f"expected 24 quoted AUCs, checked {checked}"

    def test_readme_speed_numbers_match_speed_table(self):
        txt = _read(README)
        if not SPEED_CSV.exists():
            pytest.skip("speed table not generated yet")
        with open(SPEED_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                arch = row.get("Architecture", "")
                assert arch in txt, f"speed row {arch!r} missing from README"
                gflops = row.get("Compute (GFLOPs)", "")
                if gflops:
                    assert gflops in txt, f"GFLOPs {gflops!r} for {arch!r} missing from README"

    def test_readme_does_not_claim_a_synthetic_dataset(self):
        txt = _read(README)
        assert "Local Synthetic Dataset (Active)" not in txt, (
            "the README still presents the synthetic demo set as the active dataset"
        )

    def test_readme_lists_the_real_benchmarks(self):
        txt = _read(README)
        for name in ("Celeb-DF v2", "DFDC", "Real-Life"):
            assert name in txt, f"README does not mention the {name} dataset"

    def test_readme_documents_the_generation_commands(self):
        """The README must say how to regenerate the numbers it quotes."""
        txt = _read(README)
        assert "scripts/build_cross_table.py" in txt
        assert "scripts/regenerate_report.py" in txt
        assert "test_deliverable_provenance.py" in txt


class TestProjectDocumentationQuotesMeasuredNumbers:
    def test_documentation_exists(self):
        assert DOCS.exists()

    def test_test_count_in_docs_matches_the_suite(self):
        """The documented test count must equal what pytest actually collects."""
        txt = _read(DOCS)
        m = re.search(r"\*\*(\d+)/(\d+) tests", txt)
        assert m, "no test-count claim found in PROJECT_DOCUMENTATION.md"
        documented = int(m.group(1))

        import subprocess
        out = subprocess.run(
            [sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q", "--collect-only"],
            cwd=str(ROOT), capture_output=True, text=True,
        )
        collected = len([l for l in out.stdout.splitlines() if "::" in l])
        assert documented == collected, (
            f"PROJECT_DOCUMENTATION.md claims {documented} tests but pytest "
            f"collects {collected}"
        )

    def test_verification_table_has_no_stale_environment_claim(self):
        txt = _read(DOCS)
        assert "torch 2.14" not in txt, (
            "the verification table still quotes the old PyTorch version"
        )

    def test_verification_table_lists_the_new_test_files(self):
        txt = _read(DOCS)
        for name in ("test_metric_orientation.py", "test_deliverable_provenance.py",
                     "test_eval_transform.py", "test_optional_facenet.py",
                     "test_local_pretrained.py"):
            assert name in txt, f"verification table omits {name}"

    def test_no_unverifiable_confidence_claims(self):
        """
        Percentage-confidence claims about face detection must not appear
        unless they can be traced to a manifest.
        """
        txt = _read(DOCS)
        assert ">99.99% confidence" not in txt, (
            "PROJECT_DOCUMENTATION.md claims a detection confidence figure that "
            "is not recorded in any manifest"
        )
