"""
Re-package the codebase into a fresh deepfake_code.zip for Kaggle upload.
Excludes data/, __pycache__/, .git/, *.pyc, staging_dataset/, kaggle_upload/, kaggle_kernel/
"""
import zipfile
import os
import sys
from pathlib import Path

# The Windows console uses cp1252, which cannot encode arrows or emoji.
# Force UTF-8 output so this script never crashes after doing its work.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

REPO = Path(r'e:\deepfake')
OUT  = REPO / 'kaggle_upload' / 'deepfake_code.zip'
OUT.parent.mkdir(exist_ok=True)

EXCLUDES = {
    'data', '__pycache__', '.git', 'kaggle_upload', 'staging_dataset',
    'kaggle_kernel', 'runs', '.gitignore', 'node_modules', '.pytest_cache',
    'reports',  # don't ship toy reports; Kaggle will generate real ones
}
EXCLUDE_SUFFIXES = {'.pyc', '.pyo', '.log', '.tmp', '.pth', '.pt', '.tar', '.gz'}
EXCLUDE_PARTS = {'checkpoints', '__pycache__', '.pytest_cache', '.git'}
EXCLUDE_FILES = {'deepfake_code.zip'}

count = 0
with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(REPO.rglob('*')):
        # Skip directories themselves
        if path.is_dir():
            continue
        rel = path.relative_to(REPO)
        parts = rel.parts
        # Skip excluded top-level dirs
        if parts[0] in EXCLUDES:
            continue
        # Skip excluded directories anywhere in the path
        if any(p in EXCLUDE_PARTS for p in parts):
            continue
        if path.name in EXCLUDE_FILES:
            continue
        if path.suffix.lower() in EXCLUDE_SUFFIXES:
            continue
        # Skip hidden files and dirs
        if any(p.startswith('.') for p in parts):
            continue
        zf.write(path, rel)
        count += 1

size_kb = OUT.stat().st_size / 1024
print(f'Packaged {count} files → {OUT}  ({size_kb:.0f} KB)')

