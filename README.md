# DivyaDrishti (दिव्यदृष्टि) AI — Deepfake Image Detection Pipeline

**DivyaDrishti AI** (from Sanskrit *दिव्य* 'divine' + *दृष्टि* 'vision/perception') is a modular, production-ready deep learning system designed for detecting manipulated, synthetic, and AI-generated facial media (Deepfakes). Features MTCNN face extraction, zero-leakage identity partitioning, multi-backbone neural architectures (EfficientNet-B3, ConvNeXt-Tiny, YOLO26n-cls, ResNet-18/50), mixed-precision training, soft-voting ensemble fusion, cross-dataset generalization matrices, and comprehensive robustness auditing.

---

## 🚀 Quick Launch: Interactive Web App (Streamlit)

Launch the interactive web UI for drag-and-drop deepfake inspection on images and videos:

```bash
streamlit run app.py
```
- **Real-time Face Cropping**: Automatically identifies, extracts, and centers faces.
- **Deepfake Probability Bar**: Clear probability breakdown and REAL/FAKE verdict.
- **Frame-by-Frame Video Inspection**: Samples frames across video timelines to detect temporal deepfake manipulation.

---

## 📦 Pre-Trained Model Weights (GitHub Releases)

Model checkpoints are hosted on GitHub Releases:
🔗 **[Download DivyaDrishti v1.0.0 Model Checkpoints](https://github.com/Mradul-Consultancy/DivyaDrishti/releases/tag/v1.0.0)**

Place the `.pth` files into `models/checkpoints/`:

| File | Backbone | Trained on | Measured ROC-AUC (Celeb-DF v2 test) |
|---|---|---|---|
| [`efficientnet_b3.ra2_in1k_best.pth`](https://github.com/Mradul-Consultancy/DivyaDrishti/releases/download/v1.0.0/efficientnet_b3.ra2_in1k_best.pth) *(123 MB)* | EfficientNet-B3 | Celeb-DF v2 | **0.7684** |
| [`convnext_tiny.fb_in22k_ft_in1k_best.pth`](https://github.com/Mradul-Consultancy/DivyaDrishti/releases/download/v1.0.0/convnext_tiny.fb_in22k_ft_in1k_best.pth) *(318 MB)* | ConvNeXt-Tiny | Celeb-DF v2 | **0.9196** |
| `resnet18_best.pth` *(128 MB)* | ResNet-18 (lightweight) | Celeb-DF v2 | **0.9717** |

> **Which checkpoint am I getting?** `models/checkpoints/<model>_best.pth` is the *fallback*
> path used when a domain-specific checkpoint is absent, and it now holds the Celeb-DF v2
> weights listed above. The per-domain checkpoints live in `models/checkpoints/celebdf/` and
> `models/checkpoints/dfdc/`; those are what the cross-dataset table below was measured
> with. Every figure here is reproducible with `python scripts/build_cross_table.py`.

---

## 📊 Verified Empirical Benchmark & Reality Check

> ⚠️ **Scientific Transparency Notice**: every figure below is read straight out of a
> generated artefact (`reports/cross_dataset_table.csv`, `reports/evaluation_summary.json`,
> `reports/speed_table.csv`, `reports/training_grid/<domain>/*_history.json`). Nothing is
> hand-written or cherry-picked. `tests/test_deliverable_provenance.py` and
> `tests/test_docs_match_measurements.py` fail if a number in this file or in the client
> deliverable is not present in the artefact it came from.

Regenerate them with:

```bash
python scripts/build_cross_table.py      # -> reports/cross_dataset_table.csv
python main.py benchmark-speed --device cpu   # -> reports/speed_table.csv
python scripts/regenerate_report.py      # -> reports/FINAL_CLIENT_DELIVERABLE.md
```

### 1. Cross-Dataset ROC-AUC Matrix

FAKE is the positive class, so **0.5 is chance** and higher is better. Every row is a
genuine train-domain / test-domain measurement: each model was trained on one domain only,
and the `Checkpoint` column of the generated CSV records that the checkpoint was
domain-specific.

| Train domain | Test domain | EfficientNet-B3 | ConvNeXt-Tiny | YOLO26n-cls | Ensemble |
|---|---|---|---|---|---|
| Celeb-DF v2 | Celeb-DF v2 *(in-domain)* | 0.7684 | 0.9196 | 0.9403 | 0.9140 |
| Celeb-DF v2 | DFDC *(cross)* | 0.3822 | 0.5933 | 0.6489 | 0.5444 |
| Celeb-DF v2 | Real-Life *(cross)* | 0.6262 | 0.6533 | 0.5113 | 0.5971 |
| DFDC | Celeb-DF v2 *(cross)* | 0.6064 | 0.7710 | 0.6697 | 0.7217 |
| DFDC | DFDC *(in-domain)* | 0.6044 | 0.8233 | 0.8678 | 0.8211 |
| DFDC | Real-Life *(cross)* | 0.4497 | 0.3929 | 0.3626 | 0.4014 |

**Reading the matrix.** Each model is strongest on the domain it was trained on, drops when
moved to an unseen dataset, and is weakest on the in-the-wild Real-Life set. Note that
Celeb-DF v2's test split is 91.2% fake, so raw accuracy on that split is inflated by the
majority class — ROC-AUC and Brier are the meaningful columns there. The DFDC test split is
balanced (30 real / 30 fake).

### 2. In-Domain Detail (Trained on Celeb-DF v2, Tested on Celeb-DF v2)

*Test split composition: 4,440 fake, 425 real (4,865 total, 91.26% fake).*
> ⚠️ **Evaluation Integrity Disclosure**: Because 91.26% of the test set is Fake, raw accuracy is strongly biased towards the majority class. **Balanced Accuracy** (average of real and fake recall) provides the true uninflated classification capability:

| Architecture | Raw Accuracy | Balanced Accuracy | Real Recall | Fake Recall | F1-Score | ROC-AUC | Brier Score (lower is better) | Latency (CPU) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO26n-cls** | **95.17%** | **79.06%** | **59.53%** | **98.58%** | **97.39%** | **0.9403** | **0.0399** | 14.18 ms |
| **ConvNeXt-Tiny** | **94.45%** | **71.53%** | **43.76%** | **99.30%** | **97.03%** | **0.9196** | **0.0539** | 90.11 ms |
| **Soft-Voting Ensemble** | **94.68%** | **75.91%** | **53.18%** | **98.65%** | **97.13%** | **0.9140** | **0.0563** | 172.06 ms |
| **EfficientNet-B3** | **81.34%** | **68.39%** | **52.71%** | **84.08%** | **89.16%** | **0.7684** | **0.1449** | 67.77 ms |

### 3. Training Status

From `reports/training_grid/<domain>/*_history.json`:

| Model | Domain | Epochs | Best train acc | Best val acc |
|---|---|---|---|---|
| EfficientNet-B3 | Celeb-DF v2 | 4 | 74.58% | 84.27% |
| ConvNeXt-Tiny | Celeb-DF v2 | 4 | 84.79% | 93.49% |
| EfficientNet-B3 | DFDC | 4 | 67.79% | 72.81% |
| ConvNeXt-Tiny | DFDC | 4 | 80.24% | 92.19% |

### 4. Speed & Complexity

From `reports/speed_table.csv` (measured on CPU):

| Architecture | Parameters (M) | Model Size (MB) | Compute (GFLOPs) | Latency (Batch=1 ms) | P95 Latency (ms) | Throughput (FPS B=32) | Measured On |
|---|---|---|---|---|---|---|---|
| ConvNeXt-Tiny (timm) | 27.82 M | 334.1 MB | 8.93 | 90.11 ms | 95.62 ms | 17.0 fps | CPU |
| EfficientNet-B3 (timm) | 10.70 M | 129.2 MB | 2.03 | 67.77 ms | 74.35 ms | 15.7 fps | CPU |
| YOLO26n-cls (Ultralytics) | 1.53 M | 3.2 MB | 0.40 | 14.18 ms | 16.37 ms | 138.0 fps | CPU |
| Soft-Voting Ensemble (All 3) | 40.05 M | 466.5 MB | 11.36 | 172.06 ms | 186.34 ms | 14.5 fps | CPU |

### 5. Datasets Actually Used

Counts read from the processed directories:

| Dataset | Source | train | val | test |
|---|---|---|---|---|
| Celeb-DF v2 | `reubensuju/celeb-df-v2` (6,229 videos) | 4,378 | 4,655 | 4,865 |
| DFDC | Kaggle mirrors (212 videos) | 1,807 | 320 | 60 |
| Real-Life (in-the-wild) | StyleGAN / Stable Diffusion + FF++ swaps | - | - | 1,200 |

---

## Directory Structure

```text
deepfake/
├── data/
│   ├── raw/                            # Source images / videos before extraction
│   ├── raw_videos/                     # Source videos (real/ & fake/) for identity-aware splitting
│   ├── processed/                      # Extracted face crops, per domain
│   │   ├── celebdf/  (train/ val/ test/ real/ fake/)
│   │   ├── dfdc/     (train/ val/ test/ real/ fake/)
│   │   └── train/ val/ test/          # Combined split
│   ├── real_life/                      # 1,200 in-the-wild evaluation images (real/ & fake/)
│   └── downloads/                      # Downloaded archives (Celeb-DF v2, DFDC subsets)
├── preprocessing/
│   ├── __init__.py
│   ├── face_extractor.py              # MTCNN / Haar facial detection & cropping with margin
│   ├── extract_faces_from_videos.py    # Video -> face crops, identity-aware splitting
│   ├── split_by_identity.py           # Union-Find identity/source clustering + zero-leakage audit
│   ├── split_data.py                  # Balanced train/val/test partitioning (image sets)
│   └── transforms.py                  # Augmentations, JPEG degradation, normalization
├── models/
│   ├── __init__.py
│   ├── detector.py                    # MesoNet-4, ResNet-18/50, and any timm backbone
│   └── checkpoints/                   # Saved .pth weights, per domain:
│       ├── celebdf/  (efficientnet_b3, convnext_tiny)
│       └── dfdc/     (efficientnet_b3, convnext_tiny)
├── training/
│   ├── __init__.py
│   ├── dataset.py                     # PyTorch Dataset and DataLoader loaders
│   ├── train.py                       # Training loop with mixed-precision & early stopping
│   ├── train_timm.py                  # EfficientNet-B3 / ConvNeXt-Tiny fine-tuning
│   ├── train_yolo.py                  # YOLO26n-cls fine-tuning with class weights
│   └── class_weights.py               # Inverse-frequency / effective / sqrt weighting
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py                    # Confusion matrix, ROC-AUC, calibration, CSV
│   ├── evaluate_ensemble.py           # Multi-model + soft-voting ensemble evaluation
│   ├── evaluate_cross_dataset.py      # Train-domain x test-domain matrix
│   ├── generate_reports.py            # Results tables + ROC/PR plots from saved predictions
│   ├── robustness_test.py             # JPEG / blur / downscale degradation benchmark
│   └── video_metrics.py               # Video-level aggregation, cluster bootstrap CIs
├── preprocessing/, scripts/
│   ├── build_cross_table.py           # Cross-dataset matrix + heatmap
│   ├── regenerate_report.py           # Client deliverable from artefacts only
│   ├── train_all_datasets.py          # Training grid, one subprocess per model
│   ├── download_celebrity_df.py       # Celeb-DF v2 acquisition
│   └── download_dfdc_subset.py        # DFDC acquisition
├── pretrained/                        # Vendored ImageNet weights (offline timm training)
├── vendor/                            # Vendored ultralytics + facenet-pytorch (offline sandboxes)
├── reports/                           # Generated plots, logs, and markdown summaries
│   ├── FINAL_CLIENT_DELIVERABLE.md    # Client deliverable (generated, never hand-edited)
│   ├── cross_dataset_table.csv        # Cross-dataset metrics
│   ├── evaluation_summary.json        # Single-model evaluation metrics
│   ├── speed_table.csv                # Parameters / GFLOPs / latency / throughput
│   └── training_grid/<domain>/        # Per-domain training histories and curves
├── config.py                          # Centralized configuration & hyperparameters
├── main.py                            # Unified CLI entrypoint
├── requirements.txt                   # Python dependencies
├── run.ps1                            # PowerShell runner (test/pilot/full/reports/deliver)
└── README.md                          # Project documentation & step-by-step guide
```

---

## Step-by-Step Guide

### Step 1: Environment Setup

1. **Clone or navigate to the repository:**
   ```bash
   cd e:/deepfake
   ```

2. **(Optional) Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux / macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   > **Note on PyTorch with GPU (CUDA):**
   > If you have an NVIDIA GPU, install the CUDA-enabled build of PyTorch:
   > ```bash
   > pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   > ```

---

### Step 2: Preparing Your Dataset

Place your images into `data/raw/`:
- `data/raw/real/`: Authentic, unaltered facial images.
- `data/raw/fake/`: Deepfake or synthetic face images.

#### Dataset Compatibility & Benchmarks:
- **Celeb-DF v2 (Active)**: The full public benchmark (6,229 videos) is downloaded with
  `python scripts/download_celebrity_df.py` and extracted into identity-disjoint
  train/val/test splits — 31,141 face crops in total.
- **DFDC (Active)**: 212 videos from the Kaggle mirrors, ingested with
  `python scripts/download_dfdc_subset.py`, extracted into 2,120 face crops.
- **FaceForensics++ (Supported)**: the preprocessing and splitting tools include a built-in
  parser for the official `train.json` / `val.json` / `test.json` split lists.
- **Real-Life in-the-wild (Active)**: 1,200 images (600 authentic portraits, 600 AI-generated
  faces and FF++ swaps) built by `python scripts/ingest_real_life_generator_faces.py`, used as
  a third, fully unseen evaluation domain.
- **Synthetic demo data (Available for smoke tests)**: `python scripts/generate_sample_data.py`
  and `python scripts/generate_sample_videos.py` create a small set with known actor and source
  IDs, for end-to-end verification without downloading anything.

> **Testing without downloading external datasets?**
> Generate synthetic sample test images and videos with identity/source metadata:
> ```bash
> python main.py demo-data --real 30 --fake 30
> python scripts/generate_sample_videos.py
> ```

---

### Step 3: Video Extraction & Identity-Aware Dataset Splitting

When training deepfake detectors, splitting purely by video is insufficient:
1. Deepfake videos are generated from real source videos. If a fake video is in `test` while its source real video is in `train`, the model memorizes facial identity, background, lighting, and camera sensor characteristics rather than learning manipulation artifacts.
2. The same actor or celebrity appears across multiple videos. If the same identity appears in both `train` and `test`, evaluation scores are artificially inflated.

The pipeline uses **Disjoint-Set (Union-Find) connected-component clustering** with path compression to cluster real and derivative fake videos, multi-actor swaps (`id0_id1`), and shared identities into isolated clusters. Whole clusters are assigned to splits, mathematically guaranteeing:
$$\text{Identities}(\text{Train}) \cap \text{Identities}(\text{Test}) = \emptyset$$
$$\text{Sources}(\text{Train}) \cap \text{Sources}(\text{Test}) = \emptyset$$

```bash
# Run identity- and source-aware extraction and splitting
python main.py extract-videos \
  --split-mode identity \
  --video-dir data/raw_videos \
  --output-dir data/processed \
  --metadata-file data/raw_videos/metadata.json \
  --frames-per-video 10 \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --test-ratio 0.15

# Verify zero leakage mathematically at any time
python main.py verify-leakage \
  --manifest data/processed/video_split_manifest.json \
  --metadata-file data/raw_videos/metadata.json
```

**Official Benchmark Support:**
- **FaceForensics++ (FF++)**: `--ffpp-splits-dir <path>` containing official `train.json`, `val.json`, and `test.json`.
- **Celeb-DF v2**: `--celeb-df-test-list <path>` containing `List_of_testing_videos.txt`.

#### Split modes

| Mode | Guarantee | When to use |
|---|---|---|
| `identity` | Strict: **no** identity (including swap donors) appears in two splits | DFDC, FF++, any dataset whose swap graph is not chained |
| `identity_primary` | Primary identities are strictly disjoint; swap-donor overlap is measured and reported as an advisory residual | Celeb-DF v2, whose swap graph chains identities |
| `video` | None (naive per-video split) | Debugging only |

> **Why Celeb-DF needs `identity_primary`.** Celeb-DF v2's swap graph is *chained*: `id0_id16` donates
> to `id0`, `id16_id17` donates to `id16`, and so on. Under strict union-find this merges all 59
> identities into just 4 giant components. Any component that touches the official test list then
> absorbs the entire dataset — extraction ends up with an empty `train/` and `val/`. Strict
> identity-disjointness is mathematically unachievable on a chained swap graph (keeping a swap's
> donor out of the split while keeping its base in it is a contradiction), so `identity_primary`
> groups videos by their **primary** identity, keeps the donor relationship recorded, and reports
> the residual donor overlap explicitly instead of shipping an unusable split.
>
> The audit still treats primary-identity overlap as fatal in every mode.

```bash
# Celeb-DF v2 (6,229 videos) with primary-identity disjointness
python main.py extract-videos \
  --split-mode identity_primary \
  --video-dir data/raw_videos/celebdf \
  --output-dir data/processed/celebdf \
  --frames-per-video 5 \
  --fallback-center-crop
```
- **DFDC**: `--metadata-file <path>` parsing `"original"` pristine source pointers and `"actor_id"`.

A complete audit record is exported to `data/processed/video_split_manifest.json`.

---

### Step 4: Photo Preprocessing & Face Extraction (For Static Images)

If your raw dataset contains full photo portraits, extract facial regions with contextual boundary margins:

```bash
python preprocessing/face_extractor.py --input data/raw/real --output data/processed_faces/real --margin 0.2
python preprocessing/face_extractor.py --input data/raw/fake --output data/processed_faces/fake --margin 0.2
```
*Note: If your dataset already consists of cropped faces, you can proceed directly to splitting.*

---

### Step 4: Dataset Splitting

Partition your images into balanced `train` (70%), `val` (15%), and `test` (15%) subsets:

```bash
# Using main.py:
python main.py split --source data/raw --dest data/processed --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15

# Or directly:
python preprocessing/split_data.py --source data/raw --dest data/processed
```

---

### Step 5: Model Training

Train a binary classifier. You can choose from multiple architectures:
- `resnet18` (fast, standard baseline)
- `resnet50` (deeper, higher capacity)
- `mesonet` (specialized compact CNN for facial tampering)
- `efficientnet_b0` (parameter-efficient, modern vision backbone)

```bash
# Train using ResNet-18 (default):
python main.py train --model resnet18 --epochs 20 --batch-size 32 --lr 0.0001

# Train using MesoNet-4:
python main.py train --model mesonet --epochs 25 --batch-size 32 --lr 0.001

# Train using EfficientNet-B0:
python main.py train --model efficientnet_b0 --epochs 15 --batch-size 16
```

**What happens during training:**
- Automatic mixed precision (`torch.amp`) when running on CUDA.
- Learning rate is dynamically adjusted on plateau (`ReduceLROnPlateau`).
- Checkpoints are saved to `models/checkpoints/<model>_best.pth`.
- Loss and accuracy curves are saved to `reports/training_curves.png`.
- Training history metrics are exported to `reports/training_history.json`.

---

### Step 5b: Fine-Tuning SOTA Backbones with Class Weights

For imbalanced datasets and modern backbones, dedicated fine-tuning scripts are provided for **YOLO26n-cls (Ultralytics)**, **ConvNeXt-Tiny (timm)**, and **EfficientNet-B3 (timm)** with automated class weighting.

#### 1. Fine-Tuning `yolo26n-cls` (Ultralytics)
Ultralytics YOLO classification model fine-tuning with injected weighted `CrossEntropyLoss`:
```bash
# Direct runner:
python training/train_yolo.py --model yolo26n-cls.pt --epochs 30 --batch-size 32 --lr 0.001

# Or via main.py:
python main.py train-yolo --model yolo26n-cls.pt --epochs 30 --batch-size 32
```
- Automatically scans `data/processed/train` and calculates class weights \(w_i = \frac{N}{C \cdot N_i}\).
- Injects a weighted `CrossEntropyLoss` via the Ultralytics `on_train_start` callback.
- Saves best weights to `runs/yolo_cls/deepfake_yolo/weights/best.pt` and metrics to `training_summary.json`.

#### 2. Fine-Tuning `convnext_tiny` (timm)
ConvNeXt-Tiny architecture pretrained on ImageNet-22k and fine-tuned on 1k:
```bash
# Convenience entry point:
python training/train_convnext_effnet.py convnext --epochs 25 --batch-size 32 --lr 5e-5

# Or via train_timm.py / main.py:
python main.py train-timm --model-name convnext_tiny.fb_in22k_ft_in1k --epochs 25 --lr 5e-5
```

#### 3. Fine-Tuning `efficientnet_b3` (timm)
EfficientNet-B3 architecture (RandAugment recipe) with higher spatial resolution:
```bash
# Convenience entry point:
python training/train_convnext_effnet.py effnet --epochs 25 --batch-size 24 --lr 1e-4

# Or via train_timm.py / main.py:
python main.py train-timm --model-name efficientnet_b3.ra2_in1k --epochs 25 --lr 1e-4
```

#### Class Weighting Strategies ([`training/class_weights.py`](file:///e:/deepfake/training/class_weights.py))
- `inverse_freq` (default): Normalized inverse frequency \(w_i = \frac{\text{total}}{C \cdot \text{count}_i}\)
- `effective`: Class-Balanced Effective Number of Samples (Cui et al., 2019) with \(\beta = 0.9999\)
- `sqrt_inverse`: Smoothed square-root inverse weighting \(\sqrt{\text{total}/\text{count}_i}\)

---

### Step 6: Model Evaluation & Performance Reporting

Evaluate the best model checkpoint on the unseen `test` split:

```bash
python main.py evaluate --checkpoint models/checkpoints/resnet18_best.pth --data-dir data/processed
```

**Generated Evaluation Reports:**
- `reports/confusion_matrix.png`: True Positive, False Positive, True Negative, False Negative counts.
- `reports/roc_curve.png`: Receiver Operating Characteristic curve with AUC score.
- `reports/evaluation_summary.json`: Detailed per-class precision, recall, and F1 score.
- `reports/evaluation_report.md`: Markdown summary table ready for presentation.

---

### Step 6b: Multi-Model Evaluation & Soft-Voting Ensemble

Evaluate **YOLO26n-cls**, **ConvNeXt-Tiny**, and **EfficientNet-B3** side-by-side alongside their combined **Soft-Voting Ensemble**:

```bash
# Direct runner (uniform equal weights [1/3, 1/3, 1/3] baseline):
python evaluation/evaluate_ensemble.py --data-dir data/processed

# Or via main.py:
python main.py evaluate-ensemble

# VALIDATION-ONLY WEIGHT TUNING (Zero Test Leakage Protocol):
# Tunes (w1, w2, w3) on val/ using simplex grid search and evaluates once on test/
python main.py evaluate-ensemble --tune-on-val --tune-metric roc_auc

# Or tune on validation to maximize F1-score:
python main.py evaluate-ensemble --tune-on-val --tune-metric f1

# Custom weighting (e.g., weights for [yolo, convnext, effnet]):
python evaluation/evaluate_ensemble.py --weights 1.0 1.5 1.2 --threshold 0.5

# Load pre-tuned weights from JSON:
python main.py evaluate-ensemble --weights-file models/checkpoints/ensemble_weights.json

# Single-image ensemble prediction with breakdown:
python evaluation/evaluate_ensemble.py --image path/to/face.jpg
```

**Outputs saved to `reports/ensemble/`:**
- `ensemble_roc_curves.png`: High-resolution comparison plot rendering ROC curves and AUC scores for all 3 models + the Soft-Voting Ensemble.
- `ensemble_confusion_matrices.png`: 2x2 comparison grid showing individual model and ensemble confusion heatmaps.
- `ensemble_metrics.json`: Per-model & ensemble precision, recall, F1, accuracy, and ROC-AUC.
- `ensemble_weights.json`: Weight audit file documenting validation scores before/after tuning and zero test leakage guarantee.
- `ensemble_report.md`: Markdown comparative summary table.

---

### Step 6c: Robustness Testing & Perturbation Benchmarking

Test the resilience of individual backbones and the Soft-Voting Ensemble under realistic social media degradations:
- **JPEG Compression**: Re-saving at qualities 90, 75, 50, 30, 15.
- **Gaussian Blur**: Radii 1.0, 2.0, 3.0, 5.0 px.
- **Resolution Downscaling**: Scale factors 0.75, 0.50, 0.35, 0.20.
- **Combined Social Media Simulation**: Scale 50% + Gaussian Blur 1.5px + JPEG Quality 30.

```bash
# Run full robustness benchmark:
python evaluation/robustness_test.py --data-dir data/processed

# Or via main.py:
python main.py test-robustness

# Custom sweep parameters:
python evaluation/robustness_test.py \
  --jpeg-qualities 85 60 30 15 \
  --blur-radii 1.0 2.5 4.0 \
  --resize-scales 0.7 0.4 0.2
```

**Artifacts saved to `reports/robustness/`:**
- `robustness_curves.png`: Performance degradation curves (Accuracy vs Distortion Severity).
- `perturbations_preview.png`: Visual multi-panel preview showing how each distortion modifies sample faces.
- `robustness_report.md`: Detailed markdown tables reporting Accuracy, F1, ROC-AUC and the metric drop ($\Delta$) under each condition.
- `robustness_metrics.json`: Full machine-readable benchmark data.

---

### Step 6d: Generate Results Tables and ROC Plots from Saved Predictions

Generate publication-ready comparison tables and high-resolution ROC/PR plots directly from `reports/saved_predictions.csv`:

```bash
# Direct runner:
python evaluation/generate_reports.py --predictions-file reports/saved_predictions.csv

# Or via unified main.py:
python main.py generate-reports
```

**Artifacts Generated:**
- `reports/results_table.md`: Comprehensive Markdown table detailing Accuracy, Balanced Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, PR-AUC, and Confusion Matrix values.
- `reports/results_summary.csv`: Machine-readable results spreadsheet.
- `reports/roc_plots.png`: Multi-model ROC comparison curves with annotated AUC values.
- `reports/precision_recall_curves.png`: Precision-Recall curves with Average Precision (AP) scores.
- `reports/confusion_matrix_grid.png`: 4-panel visual confusion matrix heatmaps.

---

### Step 7: Single Image & Batch Inference

Run deepfake detection on an individual image or a folder of images:

#### Single Image Prediction:
```bash
python main.py predict \
  --checkpoint models/checkpoints/resnet18_best.pth \
  --image path/to/suspect_face.jpg \
  --output-dir reports/predictions
```

#### Batch Directory Prediction:
```bash
python main.py predict \
  --checkpoint models/checkpoints/resnet18_best.pth \
  --input-dir path/to/images_folder/ \
  --output-dir reports/predictions
```

**Output:**
- Console prints classification (`REAL` or `FAKE`), confidence percentage, and class probabilities.
- If `--output-dir` is specified, it saves an annotated image with a visual classification banner.

---

### Step 8: Unit Testing & Label Mapping Verification

Verify that PyTorch's `ImageFolder`, custom `DeepfakeDataset`, `compute_class_weights`, Ultralytics YOLO, and timm backbones all strictly agree on alphabetical label mapping (`fake=0, real=1`):

```bash
pytest -v tests/test_label_mapping.py
```

---

## 1-Minute End-to-End Demo

Test the entire pipeline right now:

```bash
# 1. Generate sample test images
python main.py demo-data --real 20 --fake 20

# 2. Partition into train / val / test
python main.py split --source data/raw --dest data/processed

# 3. Train for 2 quick epochs
python main.py train --model resnet18 --epochs 2 --batch-size 8

# 4. Evaluate on test set
python main.py evaluate --checkpoint models/checkpoints/resnet18_latest.pth

# 5. Predict a sample image
python main.py predict --checkpoint models/checkpoints/resnet18_latest.pth --image data/processed/test/fake/fake_001.jpg
```

---

## Configuration & Hyperparameters

Central settings can be adjusted in [`config.py`](file:///e:/deepfake/config.py):
- `IMAGE_SIZE`: Default `(224, 224)`
- `BATCH_SIZE`: Default `32`
- `LEARNING_RATE`: Default `1e-4`
- `EPOCHS`: Default `20`
- `EARLY_STOPPING_PATIENCE`: Default `5`
- `NORMALIZE_MEAN` / `NORMALIZE_STD`: ImageNet statistics

