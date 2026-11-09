# Makefile for Deepfake Detection pipeline
# Usage: make <target>
#   make pilot   – run the pilot (≈200 videos, 5‑epoch EfficientNet‑B3)
#   make full    – after pilot passes, run the full training grid (25 epochs for all models)
#   make reports – evaluate ensemble and generate JSON‑only reports
#   make real    – build the real‑life image set (1‑2k generated faces)
#   make deliver – rewrite the client deliverable using fresh artefacts

.PHONY: pilot full reports real deliver clean

# ----- configuration -----
MAX_VIDEOS ?= 200
FULL_EPOCHS ?= 25
FULL_BATCH ?= 32
FULL_PATIENCE ?= 7

# ----- helper commands -----
PY = python

# ----- targets -----
pilot:
	@echo "\n=== Pilot: downloading DFDC subset ($(MAX_VIDEOS) videos) ==="
	$(PY) -c "import config, utils.helpers as h;\
from scripts.download_dfdc_subset import download_dfdc_from_dataset;\
RAW = config.WORK_DIR / 'data' / 'raw_videos' / 'dfdc';\
DL = config.WORK_DIR / 'data' / 'downloads';\
h.ensure_dirs(RAW, DL);\
download_dfdc_from_dataset(RAW, DL, max_videos=int('$(MAX_VIDEOS)'))"

	@echo "\n=== Pilot: face extraction & split ==="
	papermill notebooks/kaggle_pilot.ipynb pilot_run.ipynb -p MAX_VIDEOS=$(MAX_VIDEOS) -p epochs=5 -p pilot=1

full:
	@echo "\n=== Full run: training all models (Manual step, no auto-cascade) ==="
	papermill notebooks/kaggle_pilot.ipynb full_run.ipynb -p MAX_VIDEOS=None -p epochs=$(FULL_EPOCHS) -p pilot=0

reports:
	@echo "\n=== Evaluating ensemble and generating reports ==="
	$(PY) main.py evaluate-ensemble
	$(PY) main.py generate-reports

real:
	@echo "\n=== Building real‑life image set ==="
	$(PY) scripts/build_real_life_dataset.py --output-dir data/real_life --num-images 2000

deliver:
	@echo "\n=== Regenerating FINAL_CLIENT_DELIVERABLE.md from JSON/CSV artefacts ==="
	$(PY) scripts/regenerate_report.py

clean:
	@echo "Cleaning intermediate artefacts…"
	$(PY) -c "import pathlib; [p.unlink() for p in pathlib.Path('.').glob('pilot_run.ipynb')]; [p.unlink() for p in pathlib.Path('.').glob('full_run.ipynb')]"
