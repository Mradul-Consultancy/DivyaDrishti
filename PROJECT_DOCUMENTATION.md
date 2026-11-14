# Deepfake Image Detection: Comprehensive Architecture & Project Audit

This document provides an exhaustive, 360-degree technical audit of the Deepfake Detection project. It breaks down **What** was built, **What** has been done, **Where** every component lives, **Why** each engineering and architectural decision was made, and **How** every pipeline stage operates.

---

## 1. What Has Been Built (System Architecture)

The project is an end-to-end, production-grade deepfake image and video detection system. It spans the entire machine learning lifecycle: from raw video frame extraction and facial alignment to multi-architecture training with class weighting, multi-model soft-voting ensembling, perturbation robustness testing, and automated publication-quality reporting.

```text
                          [ Raw Videos (.mp4, .avi) / Static Photos ]
                                              │
                                              ▼
                        ┌───────────────────────────────────────────┐
                        │      Pre-Extraction Video Partition       │
                        │       (Zero Data Leakage by Video)        │
                        └─────────────────────┬─────────────────────┘
                                              │
                                              ▼
                        ┌───────────────────────────────────────────┐
                        │        OpenCV Frame Sampling (N fps)      │
                        │                     +                     │
                        │         MTCNN Facial Crop (20% Pad)       │
                        └─────────────────────┬─────────────────────┘
                                              │
                                              ▼
                        ┌───────────────────────────────────────────┐
                        │   Organized Dataset (train / val / test)  │
                        └─────────────────────┬─────────────────────┘
                                              │
                      ┌───────────────────────┼───────────────────────┐
                      │                       │                       │
                      ▼                       ▼                       ▼
            ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
            │   YOLO26n-cls     │   │   ConvNeXt-Tiny   │   │  EfficientNet-B3  │
            │   (Ultralytics)   │   │      (timm)       │   │      (timm)       │
            └─────────┬─────────┘   └─────────┬─────────┘   └─────────┬─────────┘
                      │                       │                       │
                      │ (Weighted CE Loss)    │ (Weighted CE Loss)    │ (Weighted CE Loss)
                      ▼                       ▼                       ▼
            ┌───────────────────────────────────────────────────────────────────┐
            │                     Soft-Voting Ensemble                          │
            │           P(Fake) = w1*P_yolo + w2*P_convnext + w3*P_effnet       │
            └─────────────────────────────────┬─────────────────────────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
        ┌───────────────────────────┐                   ┌───────────────────────────┐
        │   Evaluation & Reporting  │                   │    Robustness Benchmark   │
        │ - Precision, Recall, F1   │                   │ - JPEG Compression Sweep  │
        │ - ROC-AUC & PR-AUC (AP)   │                   │ - Gaussian Blur Sweep     │
        │ - 2x2 Confusion Matrices  │                   │ - Resolution Downscaling  │
        │ - Per-sample CSV Audit    │                   │ - Social Media Distortion │
        └───────────────────────────┘                   └───────────────────────────┘
```

---

## 2. Where Everything is Located (Directory Tree & Responsibilities)

```text
e:\deepfake\
│
├── config.py                               <- Centralized configuration (paths, hyperparameters, device)
├── main.py                                 <- Unified master CLI entrypoint for all pipeline stages
├── requirements.txt                        <- Strict dependency specifications
├── .gitignore                              <- Ignore rules for datasets, checkpoints, caches
├── README.md                               <- User-facing step-by-step guide
├── PROJECT_DOCUMENTATION.md                <- Technical project audit documentation
│
├── data/                                   <- Dataset root directory
│   ├── README.md                           <- Dataset layout, benchmark links & leakage policies
│   ├── raw/                                <- Static uncropped image dataset (real/ & fake/)
│   ├── raw_videos/                         <- Source video files (real/ & fake/)
│   │   ├── metadata.json                   <- DFDC-style identity metadata (actor_id per video)
│   │   ├── List_of_testing_videos.txt      <- Celeb-DF-style official test list
│   │   └── ffpp_splits/                    <- FF++ JSON split files (train/val/test.json)
│   └── processed/                          <- Aligned & partitioned face crops
│       ├── video_split_manifest.json       <- Audit trail: video→split mapping + per-video drop stats
│       ├── train/ (real/ & fake/)          <- Training partition
│       ├── val/   (real/ & fake/)          <- Validation partition
│       └── test/  (real/ & fake/)          <- Unseen test partition
│
├── preprocessing/                          <- Data ingestion and computer vision transforms
│   ├── __init__.py                         <- Module exports
│   ├── extract_faces_from_videos.py        <- N-frame extraction + MTCNN crop + identity-aware splitting
│   ├── split_by_identity.py               <- DisjointSet Union-Find + IdentitySourceSplitter + leakage verifier
│   ├── face_extractor.py                   <- Static photo face detection (Haar cascade fallback)
│   ├── split_data.py                       <- Stratified train/val/test splitter for image folders
│   └── transforms.py                       <- Torchvision data augmentation pipelines
│
├── models/                                 <- Model architectures & checkpoint storage
│   ├── __init__.py                         <- Module exports
│   ├── detector.py                         <- MesoNet-4, ResNet-18/50, EfficientNet-B0 architectures
│   └── checkpoints/                        <- Serialized PyTorch model weights (.pth)
│       ├── convnext_tiny.*_best.pth        <- Fine-tuned ConvNeXt-Tiny weights
│       ├── efficientnet_b3.*_best.pth      <- Fine-tuned EfficientNet-B3 weights
│       ├── resnet18_best.pth               <- Fine-tuned ResNet-18 weights
│       └── ensemble_weights.json           <- Validation-tuned ensemble weights (w1, w2, w3)
│
├── training/                               <- Model training engines & loss optimization
│   ├── __init__.py                         <- Module exports
│   ├── class_weights.py                    <- Automated class weight calculations (inverse-freq/effective)
│   ├── dataset.py                          <- PyTorch DeepfakeDataset & DataLoader factory
│   ├── train.py                            <- Baseline PyTorch training loop (AMP, early stopping)
│   ├── train_yolo.py                       <- Fine-tunes YOLO26n-cls with callback-injected weighted loss
│   ├── train_timm.py                       <- General timm training loop (cosine LR, gradient clipping)
│   └── train_convnext_effnet.py            <- Pre-configured runners for ConvNeXt-Tiny & EfficientNet-B3
│
├── evaluation/                             <- Testing, ensembling & perturbation benchmarks
│   ├── __init__.py                         <- Module exports
│   ├── evaluate.py                         <- Standard single-model evaluation (ROC, confusion matrix)
│   ├── evaluate_ensemble.py                <- Multi-model evaluator & soft-voting ensemble engine
│   ├── robustness_test.py                  <- JPEG, Gaussian blur, and resize degradation suite
│   ├── generate_reports.py                 <- Computes results table and ROC/PR plots from saved predictions
│   └── predict.py                          <- Single-image and directory inference with visual banner
│
├── reports/                                <- Generated metrics, plots, and audit logs
│   ├── README.md                           <- Catalog of report artifacts
│   ├── saved_predictions.csv               <- Per-sample probabilities and predictions audit log
│   ├── results_table.md                    <- Markdown summary table of all models & ensemble
│   ├── results_summary.csv                 <- Machine-readable tabular summary
│   ├── roc_plots.png                       <- Multi-model ROC curves with AUC annotations
│   ├── precision_recall_curves.png         <- Precision-Recall curves with AP annotations
│   ├── confusion_matrix_grid.png           <- 2x2 confusion matrix heatmap grid
│   ├── training_curves.png                 <- Baseline loss & accuracy curves
│   ├── ensemble/                           <- Ensemble-specific evaluation outputs
│   │   ├── ensemble_metrics.json           <- Detailed numerical JSON summary
│   │   ├── ensemble_report.md              <- Markdown ensemble comparison report
│   │   ├── ensemble_roc_curves.png         <- Standalone ensemble ROC plot
│   │   └── ensemble_confusion_matrices.png <- Ensemble confusion heatmaps
│   └── robustness/                         <- Perturbation benchmark outputs
│       ├── robustness_metrics.json         <- JSON data of all perturbation sweeps and delta drops
│       ├── robustness_report.md            <- Markdown report detailing metric drops under degradation
│       ├── robustness_curves.png           <- Degradation curves (Accuracy vs Distortion Severity)
│       └── perturbations_preview.png       <- Visual grid showing sample faces under perturbations
│
├── runs/                                   <- Ultralytics YOLO run artifacts
│   └── yolo_cls/deepfake_yolo/weights/     <- Best and last weights for YOLO26n-cls (best.pt)
│
└── scripts/                                <- Utilities & data acquisition scripts
    ├── generate_sample_data.py             <- Creates synthetic portraits with manipulation artifacts
    ├── generate_sample_videos.py           <- Creates short MP4 videos with 6 identities + metadata stubs
    └── download_dfdc_subset.py             <- Downloads real DFDC videos & preserves source-fake pairs

tests/                                      <- Automated regression tests
    ├── __init__.py
    ├── test_label_mapping.py               <- 6 tests: fake=0/real=1 alignment across all models
    ├── test_identity_split.py              <- 10 tests: Union-Find, parsers, clustering, leakage detection
    ├── test_leakage.py                     <- 3 tests: on-disk manifest leakage assertion
    ├── test_drop_frames.py                 <- 3 tests: MTCNN drop logging, fallback center-crop
    ├── test_calibration.py                 <- 3 tests: Brier score bounds & ensemble_metrics.json
    ├── test_jpeg_augmentation.py           <- 15 tests: stochastic JPEG compression & pipeline order
    └── test_dfdc_downloader.py             <- 4 tests: Kaggle auth, zip/folder parsing, pair preservation
```

---

## 3. What Has Been Done (Chronological Evolution)

1. **Phase 1: Project Scaffolding & Base Pipeline**
   - Built modular folder hierarchy: `data/`, `preprocessing/`, `training/`, `evaluation/`, `reports/`.
   - Developed `config.py` with centralized hyperparameters and automated device detection (CUDA/CPU).
   - Created `models/detector.py` implementing classic MesoNet-4, ResNet-18/50, and EfficientNet-B0.
   - Built full baseline training loop (`training/train.py`) with mixed-precision training (`torch.amp`), early stopping, learning rate scheduler, checkpointing, and curve plotting.
   - Built evaluation routine (`evaluation/evaluate.py`) with ROC-AUC, Confusion Matrix, and single-image inference (`evaluation/predict.py`).
   - Created synthetic data generator (`scripts/generate_sample_data.py`) enabling instant out-of-the-box pipeline testing without needing large external downloads.
   - Executed and validated full demo pipeline.

2. **Phase 2: Video Extraction & MTCNN Cropping (By-Video Splitting)**
   - Installed `facenet-pytorch` without downgrading the environment's PyTorch.
   - Implemented `preprocessing/extract_faces_from_videos.py`:
     - Extracts $N$ evenly spaced frames across video duration with OpenCV.
     - Detects and crops faces using MTCNN with proportional margin padding (20%).
     - **Enforced strict By-Video partitioning**: splits video files before frame extraction, ensuring 0% data leakage across splits.
     - Exports `video_split_manifest.json` recording video-to-split mappings.
   - Created `scripts/generate_sample_videos.py` and validated on synthetic video files.

3. **Phase 3: Fine-Tuning YOLO26n-cls, ConvNeXt-Tiny, and EfficientNet-B3 with Class Weights**
   - Installed `ultralytics` and `timm`.
   - Built `training/class_weights.py` supporting `inverse_freq`, `effective` (Cui et al., 2019), and `sqrt_inverse` weighting strategies.
   - Built `training/train_yolo.py`: fine-tunes `yolo26n-cls.pt` (downloaded from official Ultralytics v8.4.0 release assets) with weighted `CrossEntropyLoss` injected via `on_train_start` callback.
   - Built `training/train_timm.py` and `training/train_convnext_effnet.py`: fine-tunes `convnext_tiny.fb_in22k_ft_in1k` and `efficientnet_b3.ra2_in1k` with cosine annealing LR, gradient clipping, AMP, and class weights.
   - Trained and saved checkpoints for all 3 architectures.

4. **Phase 4: Multi-Model Evaluation & Soft-Voting Ensemble**
   - Created `evaluation/evaluate_ensemble.py`: loads YOLO26n-cls, ConvNeXt-Tiny, and EfficientNet-B3 simultaneously.
   - Enforced standard alphabetical label mapping (`fake=0, real=1`) universally across PyTorch ImageFolder, custom DeepfakeDataset, timm backbones, and Ultralytics YOLO.
   - Implemented soft-voting ensemble: computes weighted average of predicted class probabilities.
   - Evaluated models on test set, outputting Precision, Recall, F1, ROC-AUC, 2x2 confusion matrix heatmaps, and Markdown reports.

5. **Phase 5: Perturbation Robustness Testing & Metric Drop Analysis**
   - Created `evaluation/robustness_test.py`: evaluates models and ensemble against real-world degradation.
   - Simulated:
     - JPEG Compression (qualities 90, 75, 50, 30, 15).
     - Gaussian Blur (radii 1.0, 2.0, 3.0, 5.0 px).
     - Resolution Downscaling (scales 75%, 50%, 35%, 20%).
     - Combined Social Media Simulation (50% scale + 1.5px blur + JPEG Q=30).
   - Computed and reported metric drops ($\Delta \text{Acc}$, $\Delta \text{F1}$, $\Delta \text{AUC}$) and rendered 3-panel degradation curves.

6. **Phase 6: Results Table & ROC Plotting from Saved Predictions**
   - Enhanced `evaluate_ensemble.py` to persist a full per-sample audit log `reports/saved_predictions.csv`.
   - Created `evaluation/generate_reports.py`:
     - Reads saved predictions and computes Accuracy, Balanced Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, PR-AUC, and Brier Score.
     - Exports publication-grade Markdown table (`reports/results_table.md`) and CSV summary (`reports/results_summary.csv`).
     - Plots high-resolution ROC curves (`reports/roc_plots.png`), Precision-Recall curves (`reports/precision_recall_curves.png`), and Confusion Matrix grid (`reports/confusion_matrix_grid.png`).

7. **Phase 7: Validation-Only Ensemble Weight Optimization (Zero-Leakage Protocol)**
   - Addressed ensemble weight provenance: verified that original baseline weights were strictly equal/uniform `[1/3, 1/3, 1/3]`.
   - Implemented `tune_weights_on_validation()` in [`evaluation/evaluate_ensemble.py`](file:///e:/deepfake/evaluation/evaluate_ensemble.py):
     - Optimizes $(w_1, w_2, w_3)$ on the unit simplex ($w_i \ge 0, \sum w_i = 1$) exclusively over the **validation set** (`data/processed/val`).
     - Supports optimization of non-differentiable and proper metrics: ROC-AUC, F1-Score, Balanced Accuracy, Cross-Entropy (Log-Loss), and Brier Score.
     - Uses dense simplex grid search ($\delta = 0.02$, evaluating 1,326 simplex vectors) with regularized tie-breaking towards the uniform prior $[1/3, 1/3, 1/3]$ to prevent overfitting on small validation splits.
     - Enforces strict data segregation: the test set (`test/`) remains 100% unobserved during weight selection, eliminating data snooping bias.
     - Exports weight audit metadata to `reports/ensemble/ensemble_weights.json` and deployment weights to `models/checkpoints/ensemble_weights.json`.

8. **Phase 8: Universal Label Mapping Fix (`fake=0, real=1`)**
   - **Problem**: PyTorch's `ImageFolder` sorts class subdirectories alphabetically, so `'fake' < 'real'` → `fake=0, real=1`. An earlier iteration of `evaluate_ensemble.py` had a comment stating `real=0, fake=1`, which was backwards. The custom dataset, YOLO evaluator, and timm evaluator could silently disagree, flipping all predictions.
   - **Fix**: Audited all files that reference class indices or class names. Standardized to `fake=0, real=1` everywhere: `evaluate_ensemble.py` (YOLO fake-index lookup via `self.yolo_fake_idx`), `generate_reports.py` (positive-class header), `training/dataset.py`, and `evaluation/evaluate.py`.
   - **Guard**: Added [`tests/test_label_mapping.py`](file:///e:/deepfake/tests/test_label_mapping.py) with 6 assertions:
     - `ImageFolder` on a synthetic directory gives `fake=0, real=1`.
     - YOLO's internal class dict reports `fake` before `real`.
     - Custom `DeepfakeDataset` agrees with `ImageFolder`.
     - Ensemble evaluator extracts YOLO fake-probability at the correct index.
     - `results_table.md` positive-class label is `FAKE (0)`.
     - All three model probability columns refer to `prob_fake` (not `prob_real`).
   - **Outcome**: 6 / 6 tests pass.

9. **Phase 9: Identity- & Source-Aware Splitting + Leakage Tests + Frame Drop Logging**

   *Sub-task A — Identity-aware splitting*
   - **Problem**: Splitting by video ID alone is insufficient for datasets where the same person appears in many videos (Celeb-DF) or where a fake video is derived from a specific real "source" video (FF++). If a source real is in `train` and the derived fake is in `test`, the model sees the subject's identity during training, inflating test scores.
   - **Implementation** — [`preprocessing/split_by_identity.py`](file:///e:/deepfake/preprocessing/split_by_identity.py):
     - `DisjointSet` (Union-Find with path compression + rank): merges video IDs that share an identity or source-video relationship into connected components.
     - `IdentitySourceSplitter.cluster_by_identity()`: shuffles components by their total-frame count and greedily assigns whole components to train/val/test to match target ratios.
     - Dataset parsers: `extract_dfdc_identities()` (DFDC `metadata.json` → actor ID mapping), `extract_celeb_df_splits()` (official `List_of_testing_videos.txt`), `extract_ffpp_splits()` (FF++ JSON split files).
     - `verify_zero_leakage()`: asserts no video appears in two splits and no identity cluster is split across partitions.
     - `split_by_official_benchmark()`: uses datasets' own published test lists and assigns remaining videos to train/val.
   - **Integration** — [`preprocessing/extract_faces_from_videos.py`](file:///e:/deepfake/preprocessing/extract_faces_from_videos.py): new `--split-mode {random,identity}` flag (default `identity`); `--metadata-file`, `--celeb-df-test-list`, `--ffpp-splits-dir`; `--clean` (removes stale crops before re-extraction).
   - **CLI** — [`main.py`](file:///e:/deepfake/main.py): `extract-videos` subcommand updated with all new flags; new `verify-leakage` subcommand.
   - **Sample data** — [`scripts/generate_sample_videos.py`](file:///e:/deepfake/scripts/generate_sample_videos.py): generates 6 synthetic identities, `metadata.json`, `List_of_testing_videos.txt`, and FF++ JSON split stubs for deterministic pipeline testing.
   - **Tests** — [`tests/test_identity_split.py`](file:///e:/deepfake/tests/test_identity_split.py): 10 tests covering Union-Find operations, key extraction, benchmark parsers, component clustering, and leakage detection; all pass.

   *Sub-task B — Leakage assertion script*
   - [`tests/test_leakage.py`](file:///e:/deepfake/tests/test_leakage.py): 3 on-disk integration tests:
     1. Reads `video_split_manifest.json` and asserts no video ID appears in two splits.
     2. Asserts no identity cluster is split across partitions.
     3. Intentional leakage injection test — asserts the leakage detector fails when it should.
   - All 3 pass; `python main.py verify-leakage` exits cleanly.

   *Sub-task C — Frame drop logging*
   - **Problem**: MTCNN sometimes returns `None` boxes (occluded, blurry, or non-face frames). Silently skipping them hides data quality issues.
   - **Fix**: `crop_faces_mtcnn()` now returns `(crops, face_detected: bool)`. The extraction loop increments per-video and per-class drop counters. At completion, `video_split_manifest.json` records `frames_attempted`, `frames_dropped`, `drop_rate`, and a `fallback_center_crop` flag per video. A summary table is printed to `stdout` grouped by class.
   - **CLI**: `--fallback-center-crop` flag activates a center-crop fallback when MTCNN finds no face, so no frame is silently lost.
   - **Tests** — [`tests/test_drop_frames.py`](file:///e:/deepfake/tests/test_drop_frames.py): 3 tests: (1) blank-frame detection causes `face_detected=False`, (2) fallback center-crop produces a valid PIL image, (3) manifest contains correct drop statistics. All pass.

10. **Phase 10: Documentation Accuracy & Brier Score Calibration**
    - **Unverified architecture claims removed**: The YOLO26n-cls checkpoint *does* contain `C3k2`, `C3k`, and `C2PSA` blocks (confirmed by introspecting `YOLO('yolo26n-cls.pt').model`), but the backbone is NOT "CSPDarkNet" — that term names YOLOv5/v8's backbone. The docs now say "Lightweight classification model loaded via Ultralytics" and avoid naming the backbone architecture. All references to "CSPDarkNet" have been removed from `PROJECT_DOCUMENTATION.md` and from the architecture audit artifact.
    - **Dataset name claims corrected**: Replaced "FaceForensics++ contains Deepfakes, Face2Face, FaceSwap, NeuralTextures" in `README.md` with a factual "Dataset Compatibility" section listing the benchmarks actually present locally (Celeb-DF v2, DFDC, and the Real-Life in-the-wild set) and the parsers shipped for FF++.
    - **Brier Score added as calibration metric**:
      - Formula: $BS = \frac{1}{N}\sum_{i=1}^{N}(p_i - y_i)^2$ where $p_i$ is predicted P(fake) and $y_i \in \{0,1\}$.
      - A perfect oracle scores 0.0; a uniform random classifier scores 0.25. Scores > 0.25 indicate worse-than-random calibration.
      - Added to `evaluate_ensemble.py` (computed per model and ensemble, stored in `ensemble_metrics.json`), `generate_reports.py` (Brier column in `results_table.md`), and printed in ASCII summary table.
      - Measured on the current DFDC test split (60 samples, EfficientNet-B3): Brier = 0.3234
        (see `reports/cross_dataset_table.csv`). Per-model Brier scores for the full matrix are
        in the `Brier Score` column of that file.
    - **Tests** — [`tests/test_calibration.py`](file:///e:/deepfake/tests/test_calibration.py): 3 tests: Brier score is in `[0,1]`, saved predictions have valid probability columns, `ensemble_metrics.json` contains `brier_score`. All pass.

11. **Phase 11: Real Data Acquisition & Automated DFDC Ingestion Pipeline**
    - **Authentication & Setup**: Integrated official Kaggle API with seamless support for legacy `kaggle.json` (`username` + `key`) and modern authorization tokens (`KGAT_...` format stored in `~/.kaggle/access_token`).
    - **Automated Ingestion Script** — [`scripts/download_dfdc_subset.py`](file:///e:/deepfake/scripts/download_dfdc_subset.py):
      - Connects via Kaggle API to public DFDC dataset mirrors (e.g. `pranay22077/dfdc-10`) and official competition archives.
      - Implements `select_paired_subset()`: when downloading a target number of videos (e.g. 10 or 50), it ensures that for every selected manipulated video (`label == 'FAKE'`), its pristine original source video (`'original'`) is also retrieved. This preserves identity and source pairing for zero-leakage clustering.
      - Automatically partitions acquired videos into `data/raw_videos/real/` and `data/raw_videos/fake/`.
      - Generates or updates `data/raw_videos/metadata.json` preserving original source video pointers.
    - **Master CLI Command**: Exposed via `python main.py download-dfdc` with options `--max-videos`, `--source {dataset, competition}`, and `--output-dir`.
    - **Real-World Video Verification**: Acquired real full-HD videos from DFDC and Celeb-DF v2. The measured face-detection yield is recorded per dataset in the extraction manifest: Celeb-DF v2 sampled 31,141 frames from 6,229 videos and saved 31,141 face crops (8 frames dropped, 0.03%); DFDC sampled 2,120 frames from 212 videos and saved 2,120 crops (162 dropped, 7.64%). Both figures come from `data/processed/<domain>/video_split_manifest.json`.
    - **Zero-Leakage Audit**: Verified with `python main.py verify-leakage --metadata-file data/raw_videos/metadata.json` (PASSED: 0 source or identity cross-contamination across splits).
    - **Tests** — [`tests/test_dfdc_downloader.py`](file:///e:/deepfake/tests/test_dfdc_downloader.py): 4 tests covering credential checking, zip extraction, folder parsing, and paired source preservation. All pass.

12. **Phase 12: Metric Correctness, Real Training & Verified Deliverable** *(2026-11-10 to 2026-11-14)*

    The client deliverable was rebuilt on measurements rather than on the warm-up checkpoints
    that were present when this document was first written. Everything below is reproduced by
    `pytest tests/` and by the artefacts under `reports/`.

    - **Inverted ROC-AUC corrected** (`scripts/build_cross_table.py`): the engine computed
      `roc_auc_score(y_true, prob_fake)` with sklearn's default `pos_label=1` (REAL) against a
      P(fake) score, so every reported AUC was `1 - true_AUC`. Celeb-DF EfficientNet-B3 was
      reported as 0.1850; the true value is 0.8150. Metric computation is now centralised in
      `_compute_metrics()`, which uses `roc_curve(..., pos_label=FAKE_LABEL_IDX)` and matches
      `evaluation/evaluate_ensemble.py`.
    - **Inverted predictions corrected**: `y_pred = (probs >= 0.5).astype(int)` mapped a high
      P(fake) to label 1 (REAL), transposing every confusion matrix and corrupting
      precision/recall/F1. The decision rule is now
      `np.where(probs >= threshold, FAKE_LABEL_IDX, REAL_LABEL_IDX)`.
    - **Hardcoded metric cache deleted**: a literal `_EVAL_CACHE` dict shipped fabricated
      numbers for the Real-Life set that matched no actual run. All numbers are now measured.
    - **Ensemble sample alignment fixed**: the YOLO branch built its file list in
      `["real", "fake"]` order while the timm branches used DeepfakeDataset's alphabetical
      order, so the soft-voting average paired each image's YOLO probability with a different
      image's timm probabilities. Both branches now share `_collect_samples()`.
    - **Evaluation transform fixed** (`evaluation/evaluate.py`): the generic torchvision
      resize was applied to timm backbones, which are trained with their native pipeline. The
      same checkpoint scored AUC 0.3956 with the timm transform and 0.5389 with the torchvision
      one; `get_eval_transform()` now derives the transform from the model, and the single-model
      evaluation and the cross-dataset table now agree exactly (0.6044 for the DFDC
      EfficientNet-B3 checkpoint).
    - **`saved_predictions.csv` collision fixed**: `evaluate.py` and `evaluate_ensemble.py` wrote
      the same path with different schemas, so whichever ran last won. `evaluate.py` now writes
      `saved_predictions_single.csv`.
    - **`identity_primary` split mode added** (`preprocessing/split_by_identity.py`): Celeb-DF
      v2's swap graph chains all 59 identities into 4 giant components, so the official test
      list swallowed the whole dataset and train/val came out empty. The new mode groups by
      primary identity, keeps the donor relationship recorded, and reports the residual
      donor-identity overlap as advisory. Primary-identity overlap remains fatal in every mode.
    - **Offline support**: `facenet-pytorch` was imported at module level and broke every
      trainer wherever it was absent; the import is now optional. `ultralytics`,
      `facenet-pytorch` and the ImageNet weights are vendored so the Kaggle sandbox (no
      internet, no Hub access) can train all three backbones.
    - **Real training completed**: Celeb-DF v2 (6,229 videos) and DFDC (212 videos) were
      extracted and trained per domain, giving 6 converged per-domain checkpoints and a
      24-cell cross-dataset matrix in which every row is `domain-specific`.
    - **Deliverable provenance enforced** (`tests/test_deliverable_provenance.py`): every
      number in `reports/FINAL_CLIENT_DELIVERABLE.md` must appear in the artefact it came
      from; editing the document by hand fails the suite.

13. **Phase 13: Enterprise Readiness — Serving, Calibration, Explainability & Forensics** *(2026-11-14)*

    A review against a Tier-1 deployment bar surfaced real defects in the inference
    path. All fixes below are covered by regression tests.

    **Correctness defects found and fixed**

    - `evaluation/predict.py` applied the generic torchvision resize to timm
      backbones, which are trained with their native bicubic pipeline. It now
      derives the transform from the model, matching `evaluation/evaluate.py`.
    - `preprocessing/face_extractor.py` silently centre-cropped every image:
      OpenCV 5 removed both `CascadeClassifier` and the bundled cascade XMLs, so
      the Haar path never initialised and the extractor always fell back. Face
      detection now uses MTCNN (the detector used in training), with the Haar
      cascade as a fallback when genuinely available.
    - The extractor assumed a single face and returned whichever face the
      detector happened to produce first. It now returns every face with its
      bounding box and confidence, and scores all of them.
    - A frame with no acceptable face was scored as if the background were a
      face crop. Detection now fails explicitly (`FaceNotFoundError`) and the
      API returns 422 instead of an arbitrary probability.

    **Operating-point calibration** (`evaluation/calibration.py`)

    At the default 0.5 threshold the detectors called a large share of
    authentic faces fake on the imbalanced Celeb-DF v2 split. Measured on each
    domain's validation split:

    | Domain | Model | FPR at 0.5 | FPR at calibrated threshold | Fake recall at threshold |
    |---|---|---:|---:|---:|
    | celebdf | efficientnet_b3.ra2_in1k | 51.6% | 24.5% | 65.8% |
    | celebdf | convnext_tiny.fb_in22k_ft_in1k | 75.5% | 28.4% | 87.1% |
    | celebdf | resnet18 | 46.8% | 10.0% | 86.2% |
    | dfdc | efficientnet_b3.ra2_in1k | 35.0% | 41.0% | 87.3% |
    | dfdc | convnext_tiny.fb_in22k_ft_in1k | 7.0% | 4.0% | 90.5% |

    Calibration is a real improvement but does not make the detector safe to
    deploy: a 10-28% false-positive rate on authentic faces remains far too
    high. The binding constraint is detector quality, not the threshold.

    **Frequency-domain branch** (`evaluation/spectral.py`) - measured, not assumed:

    | Evaluation | Neural only | Spectral only | Neural + spectral |
    |---|---:|---:|---:|
    | In-domain (Celeb-DF test) | 0.9215 | 0.6230 | 0.9125 |
    | Cross-domain (unseen Real-Life) | 0.6533 | 0.1038 | **0.8157** |

    The branch helps on the unseen domain (+0.16 AUC) and adds nothing
    in-domain. The combiner is a logistic regression fitted on validation and
    stored as JSON, so inference needs no extra dependency.

    **Fairness audit** (`evaluation/bias_audit.py`) - the project has **no
    protected-attribute labels** in any manifest, so a Fitzpatrick / age /
    gender matrix cannot be produced from the released data. The audit reports
    that coverage gap explicitly and instead measures performance stratified by
    capture-condition proxies that are derivable (blur, brightness, contrast)
    plus per-identity variance.

    **Serving and forensics**
    - `api/server.py`: FastAPI service with `/v1/detect/image`,
      `/v1/detect/video`, `/v1/models`, `/health` and audit endpoints.
      Video responses carry timeline hotspots and a temporal-consistency score.
    - `api/audit.py`: decision records keep a SHA-256 digest of the input and
      never the image bytes, so no biometric content is persisted.
    - `api/c2pa.py`: detects Content-Credentials manifests and explicitly
      reports that the signature was **not** cryptographically verified (the
      C2PA SDK is unavailable offline) rather than implying it was checked.
    - `api/forensic.py`: produces a report with the media digest, the model,
      threshold, detector configuration and C2PA status, plus a self-digest
      that detects tampering.
    - Grad-CAM is wired into the CLI, the Streamlit app and the API, and now
      uses the model's own transform so the heatmap matches the scored tensor.


---

## 4. Why Each Decision Was Made (Engineering & Scientific Rationale)

| Decision | Why It Was Made |
|---|---|
| **Split by Video, Never by Frame** | Deepfake video frames from the same video share identical identities, lighting, backgrounds, and compression noise. Splitting by frame causes severe data leakage where the model memorizes facial identity rather than learning forgery artifacts, yielding falsely inflated test scores. Splitting by video guarantees clean generalization testing. |
| **20% Contextual Margin Around Face** | Deepfake manipulation artifacts (boundary blending seams, color mismatches, warp distortions) predominantly occur at the jawline, chin, hairline, and forehead boundaries. Tightly cropped faces exclude these vital forensic cues. |
| **MTCNN Over Basic Haar Cascades** | Multi-Task Cascaded Convolutional Networks (MTCNN) use a 3-stage cascaded deep architecture (P-Net, R-Net, O-Net) that jointly performs face detection and 5-point facial landmark alignment, significantly reducing false positives on tilted, angled, or partially occluded faces. |
| **Class Weighting in Loss Functions** | Real-world deepfake datasets are inherently imbalanced (authentic videos often outnumber fakes, or vice versa depending on scraping). Standard Cross-Entropy loss leads to a model biased toward the majority class. Reweighting loss by inverse class frequency penalizes minority errors proportionately. |
| **Three Complementary Model Backbones** | **YOLO-cls (`yolo26n-cls.pt` checkpoint)**: Lightweight classification model loaded via Ultralytics for fast real-time inference.<br>**ConvNeXt-Tiny**: Modern pure convolutional architecture with $7 \times 7$ depthwise convolutions and inverted bottlenecks mimicking vision transformers.<br>**EfficientNet-B3**: Scalable compound-scaled CNN optimizing depth, width, and resolution with Squeeze-and-Excitation blocks. |
| **Soft-Voting Over Hard-Voting** | Hard-voting discards prediction confidence (a $51\%$ vote counts equally as a $99\%$ vote). Soft-voting computes the weighted average of continuous probabilities, penalizing uncertain predictions and rewarding high-confidence consensus across diverse feature extractors. |
| **Validation-Only Ensemble Weight Tuning** | Tuning ensemble weights $(w_1, w_2, w_3)$ on the test set is a form of data snooping (data leakage) that produces optimistically biased, over-fitted evaluation metrics. Optimizing weights on the validation split and freezing them prior to test set evaluation guarantees scientifically rigorous, leak-free performance estimates. |
| **Alphabetical Label Alignment (`fake=0, real=1`)** | PyTorch's `ImageFolder` discovers subdirectories by sorting folder names alphabetically (`'fake'` < `'real'`), giving `fake=0, real=1`. Ultralytics YOLO uses the exact same alphabetical sorting. Standardizing all datasets, loss weights, timm model heads, and evaluation metrics on `fake=0, real=1` eliminates silent prediction inversions and ensures multi-model cross-framework interoperability. |
| **Robustness Degradation Benchmarking** | Deepfake videos uploaded to platforms like YouTube, Instagram, WhatsApp, or TikTok undergo heavy aggressive transcoding, downsampling, and spatial smoothing. Evaluating on clean datasets alone gives a deceptive estimate of real-world efficacy. |
| **Identity-Aware Splitting (Union-Find)** | Splitting by video ID alone is insufficient when the same person appears in multiple videos (Celeb-DF) or when a fake video derives from a real source (FF++). If a source-real video is in `train` and its derived fake is in `test`, the model sees the subject's identity during training, inflating test-set metrics. Union-Find clusters all videos sharing an actor or source link into one indivisible component that is always assigned to the same split. |
| **Logging Dropped MTCNN Frames** | MTCNN returns `None` bounding boxes for blurry, occluded, or non-face frames. Silently skipping these hides data quality issues and can silently reduce the effective test set. Logging per-video and per-class drop rates in `video_split_manifest.json` makes the extraction reproducible and auditable, and the `--fallback-center-crop` option ensures no frame is lost without an explicit decision. |
| **Brier Score as Calibration Metric** | Accuracy and AUC measure discrimination but not probability calibration. A model that always outputs 0.6 for every sample can achieve high AUC while being completely miscalibrated. The Brier Score $BS = \frac{1}{N}\sum(p_i - y_i)^2$ penalizes confident wrong predictions quadratically and is a proper scoring rule. A perfect model scores 0.0; random scoring 0.25; > 0.25 means worse-than-random calibration. Only claiming "improved calibration" after measuring it avoids unverified assertions. |

---

## 5. How Each Aspect Works (Implementation Mechanics)

### A. Preprocessing & Video Ingestion (`preprocessing/`)
- [`extract_faces_from_videos.py`](file:///e:/deepfake/preprocessing/extract_faces_from_videos.py):
  1. Accepts `--split-mode {random,identity}` (default: `identity`).
  2. In **identity mode**: loads `IdentitySourceSplitter` from [`split_by_identity.py`](file:///e:/deepfake/preprocessing/split_by_identity.py), which reads `--metadata-file` / `--celeb-df-test-list` / `--ffpp-splits-dir` to build an identity graph. Videos sharing an actor or source-derivation link are merged into one cluster (Union-Find with path compression), and whole clusters are assigned to splits — preventing identity leakage even when the same person appears in many videos.
  3. In **random mode**: falls back to per-video shuffle + ratio split (original behavior).
  4. Uses `cv2.VideoCapture` to compute $N$ evenly spaced indices: `np.linspace(0, total_frames - 1, N)`.
  5. Feeds each frame to MTCNN (`mtcnn.detect(pil_img)`) to extract bounding boxes. Returns `(crops, face_detected: bool)`.
  6. If `face_detected=False` and `--fallback-center-crop` is set, a $(224 \times 224)$ center crop is used so no frame is silently discarded.
  7. Expands detected bounding box by `margin_ratio × width/height`, clamps to image boundaries, crops with PIL, and resamples to $(224, 224)$ with Lanczos interpolation.
  8. Per-video counters track `frames_attempted` and `frames_dropped`. On completion, drop statistics are printed per class and written to `video_split_manifest.json` (fields: `frames_attempted`, `frames_dropped`, `drop_rate`, `fallback_center_crop`).
  9. `--clean` flag: removes stale `video_*.jpg` crops from a previous run before re-extracting.
  10. Saves crops into `data/processed/{split}/{class_name}/` and writes `video_split_manifest.json`.

- [`split_by_identity.py`](file:///e:/deepfake/preprocessing/split_by_identity.py):
  - **`DisjointSet`**: Union-Find data structure with path compression and union-by-rank. Merges video IDs that share identity or source-derivation relationships in O(α(N)) amortized time.
  - **`IdentitySourceSplitter`**: orchestrates identity-aware splitting. `cluster_by_identity()` builds the graph by parsing metadata, then greedily assigns clusters (sorted by frame count) to train/val/test to match target ratios.
  - **Dataset parsers**: `extract_dfdc_identities()` (reads DFDC `metadata.json` → maps each video to its actor_id), `extract_celeb_df_splits()` (reads official `List_of_testing_videos.txt`), `extract_ffpp_splits()` (reads FF++ JSON split files).
  - **`verify_zero_leakage()`**: raises `AssertionError` if any video ID appears in two splits or if any identity cluster is split across partition boundaries.
  - **`split_by_official_benchmark()`**: directly uses a dataset's published test list; the remaining videos are distributed to train/val.

### B. Training Engines (`training/`)
- `class_weights.py`:
  - Scans training directory, counts valid image files per class.
  - Computes normalized inverse frequency: $w_i = \frac{N_{\text{total}}}{C \cdot N_i}$.
- `train_yolo.py`:
  - Initializes `YOLO("yolo26n-cls.pt")`.
  - Attaches `on_train_start` callback to inject `nn.CrossEntropyLoss(weight=w)`.
  - Runs Ultralytics native trainer with cosine LR decay, early stopping, and model checkpointing to `runs/yolo_cls/`.
- `train_timm.py` & `train_convnext_effnet.py`:
  - Creates model via `timm.create_model(...)` with replacement classification head (`num_classes=2`).
  - Resolves native pretrained configuration (`resolve_data_config`) for proper interpolation, normalization, and image dimensions.
  - Trains with AdamW optimizer, `CosineAnnealingLR`, mixed precision (`torch.amp.autocast`), gradient clipping (`max_norm=1.0`), and saving `*_best.pth`.

### C. Evaluation & Soft-Voting Ensemble (`evaluation/`)
- `evaluate_ensemble.py`:
  - Evaluates every test image through all 3 networks.
  - Dynamically extracts correct probability of `FAKE` from YOLO's internal class dictionary (`self.yolo_fake_idx`) and timm's index 1.
  - Computes soft-voting ensemble: $P_{\text{ensemble}} = \sum w_i P_i$.
  - Computes precision, recall, F1, ROC curve, and AUC using `sklearn.metrics`.
  - Saves all per-sample probabilities, labels, and decisions to `reports/saved_predictions.csv`.

### D. Robustness Benchmarking (`evaluation/robustness_test.py`)
- Employs PIL in-memory byte buffers for lossless-to-lossy JPEG compression simulation at specified quality factors ($Q \in [15, 90]$).
- Applies `ImageFilter.GaussianBlur(radius=r)` for optical smoothing simulation.
- Performs dual-stage bilinear resizing (`resize(scale) -> resize(original)`) for downsampling simulation.
- Evaluates the entire test suite under each distortion and computes $\Delta$ metric degradation against clean baselines.

### E. Reporting & Visualizations (`evaluation/generate_reports.py`)
- Ingests `saved_predictions.csv` and outputs:
  - Formatted terminal table.
  - `results_table.md` containing overall performance metrics and confusion breakdowns.
  - `roc_plots.png` (Multi-model ROC curves with AUC annotations).
  - `precision_recall_curves.png` (PR curves with Average Precision scores).
  - `confusion_matrix_grid.png` ($2 \times 2$ heatmap matrix).

### F. Cross-Dataset Generalization Engine (`evaluation/evaluate_cross_dataset.py`)
- Benchmarks model generalization transferability across multiple diverse evaluation domains:
  - **Celeb-DF (v2)** (`data/processed/celebdf/test`): Face-swap manipulation benchmark.
  - **DFDC** (`data/processed/dfdc/test`): Dual-encoder autoencoder deepfake challenge.
  - **Real-Life** (`data/real_life`): In-the-wild facial captures under varying resolutions and compressions.
- Provides dynamic directory resolution (`resolve_test_split()`) supporting both nested `test/` splits and direct folder layouts.
- Exports structured performance matrix (`reports/cross_dataset/cross_dataset_report.md`), machine-readable JSON metrics (`cross_dataset_metrics.json`), and grouped bar chart comparisons (`cross_dataset_comparison.png`).

---

## 6. Project Verification Summary

Every figure below is a measured value taken from a generated artefact. The
per-run numbers live in `reports/` (cross-dataset table, evaluation summary,
speed table, per-domain training histories); the client deliverable
`reports/FINAL_CLIENT_DELIVERABLE.md` is regenerated from them and is checked
by `tests/test_deliverable_provenance.py`, which fails if any number in it is
not present in the artefact it came from.

| Pipeline Stage | Target / Artifact | Status | Verification Evidence |
|---|---|:---:|---|
| **Environment** | Dependencies | PASSED | `torch 2.9.1+cu128`, `ultralytics 8.4.203`, `timm 1.0.31`, `facenet-pytorch 2.6`, `opencv 5.0.0` |
| **Celeb-DF Acquisition** | `scripts/download_celebrity_df.py` | PASSED | 9.29 GB / 6,229 videos (590 real, 5,639 synthesis) from `reubensuju/celeb-df-v2` |
| **Video Extraction** | `extract_faces_from_videos.py` | PASSED | 31,141 Celeb-DF face crops + 2,120 DFDC crops; zero-leakage manifest; identity-disjoint splits |
| **Identity Split** | `split_by_identity.py` | PASSED | 59 primary-identity clusters for Celeb-DF; `zero_leakage_guaranteed: True`; donor-identity overlap measured and disclosed |
| **YOLO Training** | `train_yolo.py` (yolo26n-cls) | PASSED | Per-domain weights for Celeb-DF and DFDC; DFDC val acc 87% |
| **ConvNeXt Training** | `train_timm.py` | PASSED | 27.8M params; Celeb-DF best val acc 93.49%, DFDC 92.19% |
| **EfficientNet Training** | `train_timm.py` | PASSED | 10.7M params; Celeb-DF best val acc 84.27%, DFDC 72.81% |
| **Ensemble Evaluation** | `evaluate_ensemble.py` | PASSED | Per-model + soft-voting metrics; Brier calibration per model |
| **Cross-Dataset Matrix** | `build_cross_table.py` | PASSED | 24/24 cells populated, every row `domain-specific`; in-domain AUC 0.7684–0.9403 (Celeb-DF), 0.6044–0.8678 (DFDC) |
| **Robustness Benchmark** | `robustness_test.py` | PASSED | JPEG (80, 50, 20), Blur (1.5, 3.0), Resize (60%, 30%), Combined |
| **Results Generator** | `generate_reports.py` | PASSED | `results_table.md`, `roc_plots.png`, `precision_recall_curves.png` |
| **Client Deliverable** | `regenerate_report.py` | PASSED | All 6 sections populated from artefacts; `run.ps1 -target deliver` exits 0 |
| **Speed Benchmark** | `benchmark_speed.py` | PASSED | `speed_table.csv`: params / GFLOPs / latency / throughput for 4 architectures |
| **Label Mapping Tests** | `tests/test_label_mapping.py` | PASSED | 6 tests: `fake=0, real=1` across ImageFolder, YOLO, timm, reports |
| **Identity Split Tests** | `tests/test_identity_split.py` | PASSED | 14 tests: Union-Find, cluster assignment, primary-identity mode, benchmark parsers, leakage detection |
| **Metric Orientation Tests** | `tests/test_metric_orientation.py` | PASSED | 13 tests: AUC orientation, prediction direction, ensemble sample alignment, no hardcoded metrics |
| **Deliverable Provenance Tests** | `tests/test_deliverable_provenance.py` | PASSED | 8 tests: every deliverable number traceable to its artefact; tampering fails the suite |
| **Eval Transform Tests** | `tests/test_eval_transform.py` | PASSED | 3 tests: evaluation transform matches the training transform |
| **Offline Dependency Tests** | `tests/test_optional_facenet.py`, `tests/test_local_pretrained.py` | PASSED | 6 tests: pipeline imports and trains without facenet-pytorch / the HuggingFace Hub |
| **Leakage Tests** | `tests/test_leakage.py` | PASSED | 3 tests: on-disk manifest audit + intentional leakage injection fails |
| **Drop Frame Tests** | `tests/test_drop_frames.py` | PASSED | 3 tests: blank frame detection, fallback crop, drop stats in manifest |
| **Calibration Tests** | `tests/test_calibration.py` | PASSED | 3 tests: Brier bounds, probability CSV validity, ensemble_metrics.json |
| **JPEG Augmentation Tests** | `tests/test_jpeg_augmentation.py` | PASSED | 15 tests: stochastic JPEG compression, quality limits, pipeline order |
| **DFDC Downloader Tests** | `tests/test_dfdc_downloader.py` | PASSED | 4 tests: Kaggle auth checking, zip extraction, paired source preservation |
| **Cross-Dataset Tests** | `tests/test_cross_dataset.py` | PASSED | 11 tests: Celeb-DF, DFDC, Real-Life directory structure, resolver, reporting |
| **Bootstrap Guard Tests** | `tests/test_bootstrap_guard.py` | PASSED | 3 tests: cluster bootstrap refuses fewer than 20 clusters |
| **Min Videos Guard Tests** | `tests/test_min_videos_per_class.py` | PASSED | 2 tests: report generation halts below 10 videos per class |
| **Full Test Suite** | `pytest tests/` | PASSED | **142/142 tests** (Windows, Python 3.12.10) |

