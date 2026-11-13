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


### GPU Training (Local RTX 2050) — 2026-11-10

The Kaggle GPU kernels could not complete a training run:
- The sandbox had **no DNS/internet**, so `pip install` of `ultralytics` /
  `facenet-pytorch` failed and `git clone` could not reach GitHub.
- The `deepfake-code` dataset did not mount at `/kaggle/input/deepfake-code`.

A CUDA build of PyTorch was installed locally instead:

```
pip install "torch==2.9.1+cu128" "torchvision==0.24.1+cu128" \
    --index-url https://download.pytorch.org/whl/cu128 --no-deps
```

Verified: `torch.cuda.is_available() == True`,
`NVIDIA GeForce RTX 2050`, compute capability (8, 6), 4.29 GB VRAM.

`config.DEVICE` resolves to `cuda` automatically, and the training loop
enables mixed precision (`torch.amp.autocast` + `GradScaler`) when the
device is CUDA, so no code changes were needed to train on the GPU.

Per-domain checkpoints are written to `models/checkpoints/<dataset>/`,
which is exactly the layout `scripts/build_cross_table.py` looks for, so
the cross-dataset table gains genuine per-domain rows instead of
shared-checkpoint duplicates.

### Kaggle GPU Kernels — Not Usable (2026-11-10)

Both kernels (`deepfake-pilot-gpu`, `deepfake-celebdf-gpu-training`) were
pushed with `--accelerator NvidiaTeslaT4` and re-run, but both **failed**:

| Failure | Detail |
|---|---|
| No internet | DNS resolution fails for `pypi.org`, `api.kaggle.com` and `github.com`, so `pip install` of `ultralytics` / `facenet-pytorch` fails and `git clone` cannot reach GitHub. |
| Dataset not mounted | `/kaggle/input/deepfake-code` does not exist inside the kernel, so the codebase is unavailable. |

The notebooks were still fixed in response (mount now handles both the zipped
and the already-extracted dataset layout; dependency installation is
conditional and never aborts the run; the invalid `!git clone` inside an
`if` block was replaced). The notebooks remain the portable path for anyone
running where Kaggle's sandbox has working networking.

**Local GPU training is the path actually used** — see above.

### Full Celeb-DF v2 Dataset Acquired and Extracted (2026-11-10)

The complete Celeb-DF v2 dataset was downloaded from Kaggle
(`reubensuju/celeb-df-v2`, 9.29 GB, 6,229 videos: 590 real + 5,639 synthesis)
and extracted with the new `identity_primary` split mode.

| Split | Real | Fake | Total | Videos |
|---|---|---|---|---|
| train | 436 | 3,942 | 4,378 | 4,378 (1 frame each) |
| val | 380 | 4,275 | 4,655 | 947 |
| test | 425 | 4,440 | 4,865 | 969 |

All 31,141 extracted face crops are identity-disjoint across splits on the
primary identity. Training uses one frame per video so that every source
video and identity is represented, which matters more for generalisation
than additional frames of the same video.

DFDC was also re-extracted at 10 frames per video (previously only ~2), giving
1,807 train / 320 val / 60 test images across 212 videos, up from 292 training
images across 27 videos.

Note that Celeb-DF v2 is inherently imbalanced (590 real vs 5,639 synthesis
videos); the pipeline's inverse-frequency class weighting handles this, but
accuracy on the validation split is therefore not directly comparable to a
balanced set — AUC is the meaningful metric.

### Training Grid Must Run One Model per Process (2026-11-10)

Running the whole grid in a single process (`train-all-datasets --datasets
celebdf dfdc`) crashed with a native CUDA abort (0xC0000409, raised from
`c10/util/AbortHandler.h` inside `torch::autograd::THPCppFunction_requires_grad`)
at the *first batch* of the first model of the second dataset — i.e. immediately
after Ultralytics YOLO training had finished in the same process.

Celeb-DF completed all three models before the crash; DFDC's EfficientNet-B3
died on batch 0 and had to be run in a fresh process, where it trained normally.

**Practical rule: run one model per process.** The grid script should spawn a
subprocess per (dataset, model) rather than looping inside one interpreter,
so a CUDA-context problem in one backend cannot take down the rest.

### Kaggle Sandbox Profile (measured 2026-11-12)

Probed with a dedicated diagnostic kernel (`deepfake-env-check`). The
sandbox this account gets is **not** what a GPU-enabled Kaggle session
normally provides:

| Resource | Kaggle sandbox | Local workstation |
|---|---|---|
| GPU | **none** (no `/dev/nvidia*`, `nvidia-smi` absent) | RTX 2050, 4 GB, CUDA 13.0 |
| CPU | 4 cores | 12 logical cores |
| RAM | 31 GB (27 GB free) | 11.7 GB total, **0.7 GB free** (page file exhausted) |
| Disk | 20.9 GB free | local |
| Internet | **none** (DNS fails for pypi.org, api.kaggle.com, github.com) | available |
| Python | 3.13.18, torch 2.11.0+cpu | 3.12.10, torch 2.9.1+cu128 |
| Preinstalled | timm, torchvision, sklearn, cv2, pandas, numpy, PIL, matplotlib, tqdm | all + ultralytics, facenet-pytorch |

**GPU is never granted.** `enable_gpu: true` plus `--accelerator
NvidiaTeslaT4`, `NvidiaTeslaT4x2`, `NvidiaTeslaP100` and `gpu` were each
tried; every run produced the same 4-CPU, no-GPU sandbox. The account's
GPU quota reports 21600 s allowed and 0 s used, so the fallback is
server-side, not a quota exhaustion.

Two consequences drove real code changes:

1. **Datasets mount at `/kaggle/input/datasets/<owner>/<slug>/`**, not
   `/kaggle/input/<slug>/`. The notebooks looked in the old path and
   concluded the dataset was missing. The training kernel now probes both.
2. **No internet means no `pip install`**, so `ultralytics` and
   `facenet-pytorch` cannot be installed. Both are now vendored under
   `vendor/` and shipped in the `deepfake-code` dataset, together with the
   `yolo26n-cls.pt` pretrained weights.

The Kaggle sandbox is used for training because its 31 GB of RAM is the
only environment with enough headroom; the local workstation exhausted its
page file when two training processes ran concurrently.

### Kaggle Execution Path (final, 2026-11-13)

Training now runs on Kaggle. The working setup:

| Asset | Kaggle id | Contents |
|---|---|---|
| Code | `mradulmanimishra/deepfake-code` | repository + vendored `ultralytics`, `facenet_pytorch`, `yolo26n-cls.pt` |
| Face crops | `mradulmanimishra/deepfake-crops-v2` | 17,285 crops: Celeb-DF, DFDC and Real-Life splits |
| ImageNet weights | `mradulmanimishra/deepfake-pretrained` | `efficientnet_b3`, `convnext_tiny` safetensors |
| Celeb-DF checkpoints | `mradulmanimishra/deepfake-ckpt-celebdf` | per-domain timm + YOLO weights |

Three environment problems had to be solved in code rather than by
configuration:

1. **Datasets mount at `/kaggle/input/datasets/<owner>/<slug>/`.** The
   notebooks looked at `/kaggle/input/<slug>/` and concluded the dataset
   was missing. The kernel now probes both layouts.
2. **No internet.** `pip install` is impossible, so `ultralytics` and
   `facenet-pytorch` are vendored into the code dataset. `facenet-pytorch`
   was previously imported at module level by the preprocessing package,
   which broke *every* module that touched `preprocessing.transforms` -
   including all three trainers. The import is now optional and degrades
   to a Haar/centre-crop path.
3. **The HuggingFace Hub is unreachable**, so `timm.create_model(pretrained=True)`
   failed before the first epoch. The ImageNet weights are vendored, and a
   `sitecustomize.py` on `PYTHONPATH` redirects `timm.create_model` at them
   for every process, including the training subprocesses (a
   `sitecustomize` module is imported at interpreter startup, so it also
   covers workers that do not inherit `sys.path`).

### Trained Models (2026-11-13)

Both training domains now have real, converged, per-domain checkpoints.

| Domain | Model | Trained on | Best val acc |
|---|---|---|---|
| DFDC | EfficientNet-B3 | Kaggle CPU (4 cores) | 72.81% |
| DFDC | ConvNeXt-Tiny | Kaggle CPU (4 cores) | 92.19% |
| DFDC | YOLO26n-cls | Kaggle CPU (4 cores) | 87% (val) |
| Celeb-DF | EfficientNet-B3 | local RTX 2050 | 84.27% |
| Celeb-DF | ConvNeXt-Tiny | local RTX 2050 | 100% (val, imbalanced) |
| Celeb-DF | YOLO26n-cls | local RTX 2050 | - |

Kaggle's 4-CPU sandbox was used for DFDC because the local workstation had
only 0.7 GB of free RAM (the page file was exhausted by concurrent training
processes); the Kaggle sandbox has 31 GB. Celeb-DF was trained earlier on the
local RTX 2050, before the RAM pressure appeared.

Both domains use the same protocol: 4 epochs, batch 32, inverse-frequency
class weighting, 1 frame per source video in the training split so that every
video and identity is represented.

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

