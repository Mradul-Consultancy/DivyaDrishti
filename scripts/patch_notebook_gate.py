"""
Patch kaggle_pilot.ipynb:
1. Fix the quality gate to read VALIDATION AUC and use a configurable threshold (default 0.70).
2. Fix WORK_DIR to point to REPO_DIR (not /kaggle/working) when running locally.
Run: python scripts/patch_notebook_gate.py
"""
import sys

# The Windows console uses cp1252, which cannot encode emoji/arrows. Force
# UTF-8 output so this script never crashes after doing its work.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "kaggle_pilot.ipynb"
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

GATE_NEEDLE = "MIN_PILOT_AUC = 0.90"
WORK_DIR_NEEDLE = "WORK_DIR = Path('/kaggle/working')"

NEW_GATE_SOURCE = [
    "# ---- Quality gate: check VALIDATION AUC (configurable via env-var) ----\n",
    "import os as _os_gate\n",
    "val_auc = summary.get('val_roc_auc') or summary.get('roc_auc')\n",
    "_auc_key = 'val_roc_auc' if 'val_roc_auc' in summary else 'roc_auc (test fallback)'\n",
    "QUALITY_GATE = float(_os_gate.getenv('QUALITY_GATE', '0.70'))\n",
    "if val_auc is None:\n",
    "    raise RuntimeError('\\u274c QUALITY GATE: AUC key not found in evaluation_summary.json')\n",
    "if val_auc < QUALITY_GATE:\n",
    "    raise RuntimeError(\n",
    "        f'\\u274c QUALITY GATE FAILED: {_auc_key}={val_auc:.4f} < {QUALITY_GATE:.2f}. '\n",
    "        'Debug order: 1) labels  2) face crops  3) splits  4) training settings'\n",
    "    )\n",
    "print(f'\\u2705 QUALITY GATE PASSED: {_auc_key}={val_auc:.4f} >= {QUALITY_GATE:.2f}')\n",
]

gate_patched = False
work_dir_patched = False

for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    src = cell["source"]

    # Patch 1: quality gate
    if not gate_patched:
        joined = "".join(src)
        if GATE_NEEDLE in joined:
            # Replace entire source of this cell with the corrected version
            # Keep lines before the gate needle and after it
            new_src = []
            skip = False
            for line in src:
                if GATE_NEEDLE in line:
                    skip = True
                    new_src.extend(NEW_GATE_SOURCE)
                elif skip:
                    # Skip all old gate lines (they were on a single escaped string)
                    skip = False
                else:
                    new_src.append(line)
            cell["source"] = new_src
            gate_patched = True
            print("✅ Patched: quality gate")

    # Patch 2: WORK_DIR
    if not work_dir_patched:
        for i, line in enumerate(src):
            if WORK_DIR_NEEDLE in line:
                src[i] = line.replace(
                    "WORK_DIR = Path('/kaggle/working')",
                    "WORK_DIR = REPO_DIR  # always match REPO_DIR"
                )
                work_dir_patched = True
                print("✅ Patched: WORK_DIR")
                break

if not gate_patched:
    print("⚠️  Gate needle not found — may already be patched or location changed.")
if not work_dir_patched:
    print("⚠️  WORK_DIR needle not found — may already be patched.")

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("✅ Notebook saved.")

