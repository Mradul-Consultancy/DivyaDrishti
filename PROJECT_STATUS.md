# Deepfake Detection — Honest Project Status
*Updated: 2026-11-10*

---

## Reality Check

| Layer | % Done | Evidence |
|---|---|---|
| **Pipeline code** (preprocessing, training, evaluation, CLI) | 100% | All modules operational, **71/71 tests pass**, `main.py --help` shows 18 commands |
| **Data** (actual images to train on) | ~25% | DFDC 200-video subset downloaded; Celeb-DF extraction populated train/val splits; download scripts ready for full Kaggle GPU pull |
| **Trained models** (real, convergent weights) | 0% (Queued) | Pilot notebook v2 queued on Kaggle T4 GPU (`mradulmanimishra/deepfake-pilot-gpu`); local CPU training avoided. **All current checkpoints are epoch-1/2 warm-up runs at 50% val accuracy** |
| **Experiments** (cross-dataset table, real-life scores) | In Progress | Speed benchmark completed & verified with real parameter counts & latencies; model AUC experiments standby for Kaggle weights |

> **Delivery status: NOT READY.** The pipeline is sound and the report generator is now
> trustworthy, but no model has been trained to convergence. Every metric currently in the
> reports comes from warm-up checkpoints that predict a single class for every image.
> The remaining blocker is GPU training, not code.

---

## Critical Bugs Fixed (2026-11-10)

These affected the numbers shipped in the client deliverable and are now covered by
regression tests in `tests/test_metric_orientation.py`.

1. **Inverted ROC-AUC in the cross-dataset engine** (`scripts/build_cross_table.py`):
   - `roc_auc_score(y_true, prob_fake)` used sklearn's default `pos_label=1` (REAL)
     against a **P(fake)** score, so every reported AUC was `1 - true_AUC`.
   - Verified against a hand-computed reference: CelebDF EfficientNet-B3 reported
     **0.1850**, true value **0.8150**; ConvNeXt-Tiny reported **0.7025**, true **0.2975**.
   - Fixed by centralising metric computation in `_compute_metrics()`, which uses
     `roc_curve(..., pos_label=FAKE_LABEL_IDX)` — matching `evaluate_ensemble.py`
     and `generate_reports.py`.

2. **Inverted predictions** (`scripts/build_cross_table.py`):
   - `y_pred = (probs >= 0.5).astype(int)` mapped a high P(fake) to label `1` (REAL),
     transposing every confusion matrix and corrupting precision/recall/F1.
   - Fixed to `np.where(probs >= threshold, FAKE_LABEL_IDX, REAL_LABEL_IDX)`.

3. **Hardcoded metrics cache** (`scripts/build_cross_table.py`):
   - A literal `_EVAL_CACHE` dict shipped fabricated numbers for the `real_life` set
     (e.g. ConvNeXt accuracy `0.5234`, CM `[[314,286],[286,314]]`) that matched no
     actual run (`0.66`, `[[224,376],[32,568]]`).
   - **Deleted entirely.** A regression test now fails if any cache or bare
     `roc_auc_score()` call reappears.

4. **Fake "Trained on" axis**:
   - No per-domain checkpoints exist (`models/checkpoints/celebdf`, `models/checkpoints/dfdc`),
     so the CELEBDF and DFDC rows evaluated identical shared weights while presenting
     themselves as independent cross-domain experiments.
   - Rows are now labelled `⚠ shared ckpt`, and the CSV carries an explicit
     `Checkpoint` column. The deliverable's Findings section discloses this automatically.

5. **Dropped backbones**:
   - `export_csv_table` skipped models with no metrics, silently removing YOLO26n-cls
     from the client table. All models are now emitted, empty cells included.

6. **`saved_predictions.csv` collision**:
   - `evaluation/evaluate.py` (single-model schema) and `evaluation/evaluate_ensemble.py`
     (multi-model schema) wrote the same path, so whichever ran last won — breaking
     `generate_reports.py` and `test_calibration.py`.
   - `evaluate.py` now writes `saved_predictions_single.csv`; the ensemble owns the
     canonical `saved_predictions.csv`.

7. **`NameError` in the YOLO evaluator** (`scripts/build_cross_table.py`):
   - `_EVAL_CACHE[cache_key] = res` referenced an undefined `cache_key` — a latent
     crash on every YOLO evaluation.

8. **Windows console encoding**:
   - `regenerate_report.py` printed `✅` and `build_cross_table.py` printed box-drawing
     characters; both raise `UnicodeEncodeError` on cp1252, making `run.ps1 deliver`
     and `make deliver` exit non-zero.
   - Both scripts now use ASCII output / UTF-8 stdout reconfiguration.

9. **Unsubstituted template placeholders**:
   - The shipped deliverable literally contained `{{DATE}}` and
     `[To be completed after full training grid has run...]`.
   - `{{DATE}}` is now substituted, and Section 4 (Findings) plus a new Section 4
     (Model Training Status) are **derived from the artefacts** — no hand-written text.

---

## Gaps Fixed (Previous Iteration)

1. **Cross-Platform Pilot Notebook (`notebooks/kaggle_pilot.ipynb`)**:
   - Dynamic environment detector added (`DEEPFAKE_ENV=KAGGLE` uses `/kaggle/working`, else local `e:\deepfake`).
   - `WORK_DIR` aligned with `REPO_DIR`.
   - Parameter None-casting cell added (`MAX_VIDEOS = None if str(MAX_VIDEOS).strip() == 'None' else int(MAX_VIDEOS)`).
   - Removed hardcoded `MAX_VIDEOS = 200` from download cell; mapped training cell to parameter `epochs`.
2. **Quality Gate Recalibrated**:
   - Switched from test-set evaluation to **validation split AUC** (`val_roc_auc` / validation summary).
   - Configurable threshold via `QUALITY_GATE` env-var (default: `0.70`), preventing premature aborts on early 5-epoch smoke-tests.
3. **Safety Guard & Unit Test Suite**:
   - 60/60 tests passing in full test suite (`pytest tests/`).
   - `tests/test_bootstrap_guard.py`: Validates cluster bootstrap CI refuses < 20 video clusters (3/3 passing).
   - `tests/test_min_videos_per_class.py`: Validates report generator halts if test cohort < 10 videos/class (2/2 passing).
4. **Data-Only Client Deliverable Generator**:
   - `scripts/regenerate_report.py` & `templates/client_deliverable_template.md`: All metrics, cross-dataset matrices, and speed tables are injected solely from JSON/CSV files—no handwritten numbers.
5. **Windows Execution Wrapper (`run.ps1`)**:
   - Provides native PowerShell targets (`pilot`, `full`, `reports`, `real`, `deliver`, `clean`) without requiring GNU Make.
6. **Celeb-DF Data & Automation**:
   - `scripts/extract_celebdf_faces.py` executed to populate `celebdf/train/fake` and `val/{real,fake}`.
   - `scripts/download_celebrity_df.py` added for automated dataset acquisition on Kaggle.
7. **Kaggle Deployment & Packaging**:
   - `scripts/package_for_kaggle.py` packages clean codebase zip (189 KB, excluding data, checkpoints, and caches).
   - Kaggle dataset `mradulmanimishra/deepfake-code` updated to version 2.
   - Kernel `mradulmanimishra/deepfake-pilot-gpu` version 2 pushed and queued for T4 GPU execution.
8. **Real Model Benchmarking Artifacts**:
   - Parameter counts, latency, and throughput benchmarked across all backbones:
     - YOLO26n-cls: 1.53M params, 164 FPS (CPU)
     - EfficientNet-B3: 10.70M params, 17.7 FPS (CPU)
     - ConvNeXt-Tiny: 27.82M params, 13.9 FPS (CPU)
     - Soft-Voting Ensemble: 40.05M params, 14.0 FPS (CPU)
   - Results saved to `reports/model_benchmark_table.md` and `reports/inference_speed_comparison.png`.

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

The code is no longer the blocker — **GPU training is**. Everything below the
"Training" heading is mechanical and automated once weights exist.

1. **Execute Pilot on Kaggle GPU** *(in progress)*:
   - Kernel `mradulmanimishra/deepfake-pilot-gpu` is `QUEUED` (submitted 2026-11-09).
   - Run `kaggle_pilot.ipynb` with `DEEPFAKE_ENV=KAGGLE` and `QUALITY_GATE=0.70`.
   - Inspect validation curves and gate pass banner.
   - Download outputs with `kaggle kernels output mradulmanimishra/deepfake-pilot-gpu -p <dir>`.

2. **Train Per-Domain Models** *(required for a valid cross-dataset table)*:
   - Run `scripts/download_celebrity_df.py` in Kaggle sessions.
   - Train EfficientNet-B3, ConvNeXt-Tiny, YOLO26n-cls on **each** domain, writing to
     `models/checkpoints/celebdf/` and `models/checkpoints/dfdc/` so the
     "Trained on" axis of the cross-dataset table becomes a real experiment rather
     than a shared-checkpoint row.

3. **Synthesize Reports**:
   - Run `python main.py evaluate-ensemble` -> `python scripts/build_cross_table.py`
     -> `python scripts/regenerate_report.py`.
   - Confirm Section 4 (Model Training Status) shows every backbone as *Converged*.

4. **Commit**:
   - Working tree has uncommitted fixes across 12 files plus the new
     `tests/test_metric_orientation.py`.

