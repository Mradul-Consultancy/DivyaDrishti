# Deepfake Detection — Honest Project Status
*Updated: 2026-11-09*

---

## Reality Check

| Layer | % Done | Evidence |
|---|---|---|
| **Pipeline code** (preprocessing, training, evaluation, CLI) | ~90% | All modules operational, 24 core tests pass (including guard suites), `main.py --help` shows 18 commands |
| **Data** (actual images to train on) | ~25% | DFDC 200-video subset downloaded; Celeb-DF extraction populated train/val splits; download scripts ready for full Kaggle GPU pull |
| **Trained models** (real, convergent weights) | 0% | Pilot ready for Kaggle GPU execution; local CPU runs avoided |
| **Experiments** (cross-dataset table, real-life scores) | 0% | Standby for model weights from Kaggle runs |

---

## Gaps Fixed (Recent Iteration)

1. **Cross-Platform Pilot Notebook (`notebooks/kaggle_pilot.ipynb`)**:
   - Dynamic environment detector added (`DEEPFAKE_ENV=KAGGLE` uses `/kaggle/working`, else local `e:\deepfake`).
   - `WORK_DIR` aligned with `REPO_DIR`.
2. **Quality Gate Recalibrated**:
   - Switched from test-set evaluation to **validation split AUC** (`val_roc_auc` / validation summary).
   - Configurable threshold via `QUALITY_GATE` env-var (default: `0.70`), preventing premature aborts on early 5-epoch smoke-tests.
3. **Safety Guard Test Suite**:
   - `tests/test_bootstrap_guard.py`: Validates cluster bootstrap CI refuses < 20 video clusters (3/3 passing).
   - `tests/test_min_videos_per_class.py`: Validates report generator halts if test cohort < 10 videos/class (2/2 passing).
4. **Data-Only Client Deliverable Generator**:
   - `scripts/regenerate_report.py` & `templates/client_deliverable_template.md`: All metrics, cross-dataset matrices, and speed tables are injected solely from JSON/CSV files—no handwritten numbers.
5. **Windows Execution Wrapper (`run.ps1`)**:
   - Provides native PowerShell targets (`pilot`, `full`, `reports`, `real`, `deliver`, `clean`) without requiring GNU Make.
6. **Celeb-DF Data & Automation**:
   - `scripts/extract_celebdf_faces.py` executed to populate `celebdf/train/fake` and `val/{real,fake}`.
   - `scripts/download_celebrity_df.py` added for automated dataset acquisition on Kaggle.
7. **Version Control**:
   - Git repository initialized at `e:\deepfake` with complete, clean commit history.

---

## Data Situation

### Processed face images in workspace:
```
celebdf/train/real   →  10 images
celebdf/train/fake   →  40 images (extracted)
celebdf/val/real     →  10 images
celebdf/val/fake     →  10 images (extracted)
celebdf/test/real    →  20 images
celebdf/test/fake    →  20 images

dfdc/train/fake      →  70 images
dfdc/train/real      →  70 images
dfdc/val/{fake,real} →  10 each
dfdc/test/{fake,real}→  30 each
```

---

## Next Steps for Delivery Readiness

1. **Execute Pilot on Kaggle GPU**:
   - Run `kaggle_pilot.ipynb` with `DEEPFAKE_ENV=KAGGLE` and `QUALITY_GATE=0.70`.
   - Inspect validation curves and gate pass banner.
2. **Download Full Celeb-DF & Train Grid**:
   - Run `scripts/download_celebrity_df.py` in Kaggle sessions.
   - Train EfficientNet-B3, ConvNeXt-Tiny, YOLO26n-cls for full epochs.
3. **Synthesize Reports**:
   - Run `python main.py evaluate-ensemble` -> `python scripts/regenerate_report.py`.
