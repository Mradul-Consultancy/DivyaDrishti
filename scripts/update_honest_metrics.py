"""
Rebuild reports/cross_dataset_table.csv from an existing cross-table run.

This script previously wrote the CSV itself, with a different column set
from the canonical generator in scripts/build_cross_table.py - so running
it silently replaced the deliverable's source table (and dropped the
`Checkpoint` provenance column). It now delegates to the canonical
exporter, so there is exactly one writer and one schema.

The balanced-accuracy and per-class recall columns it introduced are now
produced by the canonical generator as well.

Usage:
    python scripts/update_honest_metrics.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config


def main():
    from scripts.build_cross_table import export_csv_table, resolve_domain_notes

    json_path = config.REPORTS_DIR / "cross_dataset" / "cross_table.json"
    if not json_path.exists():
        sys.exit(f"missing {json_path} - run `python scripts/build_cross_table.py` first")

    results = json.loads(json_path.read_text(encoding="utf-8"))
    domain_note = resolve_domain_notes()

    out_paths = [
        config.REPORTS_DIR / "cross_dataset" / "cross_table.csv",
        config.REPORTS_DIR / "cross_dataset_table.csv",
    ]
    export_csv_table(results, out_paths, domain_note)
    print("Rebuilt the cross-dataset CSV with balanced accuracy, per-class recall "
          "and checkpoint provenance (canonical schema).")


if __name__ == "__main__":
    main()
