# Deepfake Image Detection Pipeline

A modular, production-ready deep learning project for detecting manipulated and synthetic facial images (Deepfakes). Features face alignment/extraction, multiple CNN backbones (ResNet-18, ResNet-50, MesoNet-4, EfficientNet-B0), mixed-precision training, early stopping, evaluation metrics (ROC-AUC, Confusion Matrix), and single/batch inference with visual overlays.

---

## Directory Structure

```text
deepfake/
├── data/
│   ├── raw/
│   │   ├── real/                     # Original authentic face images
│   │   └── fake/                     # Manipulated / Deepfake images
│   └── processed/
│       ├── train/ (real/ & fake/)     # Training set (after face extraction & split)
│       ├── val/   (real/ & fake/)     # Validation set for model selection
│       └── test/  (real/ & fake/)     # Unseen test set for final evaluation
├── preprocessing/
│   ├── __init__.py
│   ├── face_extractor.py             # Facial detection & cropping with margin
│   ├── split_data.py                 # Balanced train/val/test dataset partitioning
│   └── transforms.py                 # Torchvision augmentations & normalization
├── models/
│   ├── __init__.py
│   ├── detector.py                   # MesoNet-4, ResNet-18/50, and EfficientNet
│   └── checkpoints/                  # Saved .pth model weights
├── training/
│   ├── __init__.py
│   ├── dataset.py                    # PyTorch Dataset and DataLoader loaders
│   └── train.py                      # Training loop with mixed-precision & early stopping
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py                   # Generates Confusion Matrix, ROC-AUC & metrics
│   └── predict.py                    # Inference on single images or full directories
├── reports/                          # Generated plots, logs, and markdown summaries
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── evaluation_summary.json
│   └── evaluation_report.md
├── scripts/
│   └── generate_sample_data.py       # Generates sample portraits for testing
├── config.py                         # Centralized configuration & hyperparameters
├── main.py                           # Unified CLI entrypoint
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation and step-by-step guide
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
- **Local Synthetic Dataset (Active)**: Created via `python scripts/generate_sample_data.py` and `python scripts/generate_sample_videos.py` with known actor and source IDs for end-to-end verification.
- **External Benchmarks Supported by Pipeline**: The preprocessing and splitting tools include built-in split parsers for major public benchmarks (FaceForensics++, Celeb-DF v2, and DFDC). Note that the local test dataset does not include external benchmark files unless you download them separately.

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

