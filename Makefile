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

full: pilot
	@echo "\n=== Full run: training all models ==="
	# Adjust notebook parameters for full training
	papermill pilot_run.ipynb full_run.ipynb -p MAX_VIDEOS=None -p epochs=$(FULL_EPOCHS) -p pilot=0

reports: full
	@echo "\n=== Evaluating ensemble and generating reports ==="
	$(PY) main.py evaluate-ensemble
	$(PY) main.py generate-reports

real:
	@echo "\n=== Building real‑life image set ==="
	$(PY) scripts/build_real_life_dataset.py --output-dir data/real_life --num-images 2000
	@echo "Updating data/README.md with provenance"
	$(PY) - <<'PY'
import pathlib, datetime
readme = pathlib.Path('data/README.md')
now = datetime.datetime.now().strftime('%Y-%m-%d')
readme.write_text(f"# Real‑life image set\nGenerated on {now}.\n\n| Source | # Images | Licence |\n|--------|----------|---------|\n| StyleGAN (MIT) | 1000 | MIT |\n| Stable Diffusion (CreativeML‑OpenRAIL‑M) | 1000 | CreativeML‑OpenRAIL‑M |\n\n", encoding='utf-8')
PY


deliver: reports real
	@echo "\n=== Re‑writing FINAL_CLIENT_DELIVERABLE.md ==="
	$(PY) - <<'PY'
import pathlib
final_md = pathlib.Path('reports/FINAL_CLIENT_DELIVERABLE.md')
template = pathlib.Path('templates/client_deliverable_template.md').read_text()
final_md.write_text(template)
print('Client deliverable regenerated.')
PY

clean:
	@echo "Cleaning intermediate artefacts…"
	rm -rf pilot_run.ipynb full_run.ipynb *.zip
