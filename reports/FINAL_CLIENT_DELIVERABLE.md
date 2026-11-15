# DivyaDrishti (दिव्यदृष्टि) AI — Deepfake Detection Client Deliverable
*Auto-generated from result artefacts on 2026-11-15. No hand-written numbers.*

> **Important**: This document is generated automatically from JSON/CSV files produced by
> the evaluation pipeline. Do not edit it manually — re-run `python scripts/regenerate_report.py`
> after each training/evaluation cycle.

---

## 1. Evaluation Metrics (In-Domain)

- **model_name**: efficientnet_b3.ra2_in1k
- **checkpoint**: models/checkpoints/dfdc/efficientnet_b3.ra2_in1k_best.pth
- **total_samples**: 60
- **accuracy**: 0.5000
- **precision**: 0.5000
- **recall**: 0.6333
- **f1_score**: 0.5588
- **roc_auc**: 0.6044
- **confusion_matrix**: [[19, 11], [19, 11]]
- **detailed_report**: {'fake': {'precision': 0.5, 'recall': 0.6333333333333333, 'f1-score': 0.5588235294117647, 'support': 30.0}, 'real': {'precision': 0.5, 'recall': 0.36666666666666664, 'f1-score': 0.4230769230769231, 'support': 30.0}, 'accuracy': 0.5, 'macro avg': {'precision': 0.5, 'recall': 0.5, 'f1-score': 0.4909502262443439, 'support': 60.0}, 'weighted avg': {'precision': 0.5, 'recall': 0.5, 'f1-score': 0.4909502262443439, 'support': 60.0}}

---

## 2. Cross-Dataset Performance Table

| Trained on | Test dataset | Model | Checkpoint | Accuracy (%) | Balanced Accuracy (%) | Recall Fake (%) | Recall Real (%) | F1-Score (%) | ROC-AUC | Brier Score | Threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CELEBDF | CelebDF | EfficientNet-B3 | domain-specific | 63.68 | 70.21 | 62.30 | 78.12 | 75.79 | 0.7682 | 0.1448 | 0.9323 |
| CELEBDF | CelebDF | ConvNeXt-Tiny | domain-specific | 85.86 | 85.12 | 86.01 | 84.24 | 91.74 | 0.9196 | 0.0539 | 0.8183 |
| CELEBDF | CelebDF | YOLO26n-cls | domain-specific | 82.88 | 85.19 | 82.39 | 88.00 | 89.78 | 0.9403 | 0.0399 | 0.9144 |
| CELEBDF | CelebDF | Ensemble | domain-specific | 74.57 | 82.24 | 72.95 | 91.53 | 83.97 | 0.9139 | 0.0562 | 0.8527 |
| CELEBDF | DFDC | EfficientNet-B3 | domain-specific | 46.67 | 46.67 | 6.67 | 86.67 | 11.11 | 0.3811 | 0.4488 | 0.9323 |
| CELEBDF | DFDC | ConvNeXt-Tiny | domain-specific | 50.00 | 50.00 | 0.00 | 100.00 | 0.00 | 0.5933 | 0.2510 | 0.8183 |
| CELEBDF | DFDC | YOLO26n-cls | domain-specific | 50.00 | 50.00 | 33.33 | 66.67 | 40.00 | 0.6489 | 0.2961 | 0.9144 |
| CELEBDF | DFDC | Ensemble | domain-specific | 46.67 | 46.67 | 0.00 | 93.33 | 0.00 | 0.5467 | 0.2726 | 0.8527 |
| CELEBDF | Real-Life | EfficientNet-B3 | domain-specific | 50.92 | 50.92 | 7.33 | 94.50 | 13.00 | 0.6260 | 0.3885 | 0.9323 |
| CELEBDF | Real-Life | ConvNeXt-Tiny | domain-specific | 53.25 | 53.25 | 6.50 | 100.00 | 12.21 | 0.6533 | 0.3527 | 0.8183 |
| CELEBDF | Real-Life | YOLO26n-cls | domain-specific | 52.08 | 52.08 | 6.83 | 97.33 | 12.48 | 0.5114 | 0.4063 | 0.9144 |
| CELEBDF | Real-Life | Ensemble | domain-specific | 53.08 | 53.08 | 6.50 | 99.67 | 12.17 | 0.5968 | 0.3623 | 0.8527 |
| DFDC | CelebDF | EfficientNet-B3 | domain-specific | 88.90 | 53.28 | 96.44 | 10.12 | 94.07 | 0.6064 | 0.1212 | 0.2107 |
| DFDC | CelebDF | ConvNeXt-Tiny | domain-specific | 91.41 | 53.80 | 99.37 | 8.24 | 95.48 | 0.7710 | 0.0737 | 0.5245 |
| DFDC | CelebDF | YOLO26n-cls | domain-specific | 91.18 | 52.83 | 99.30 | 6.35 | 95.36 | 0.6697 | 0.0786 | 0.5406 |
| DFDC | CelebDF | Ensemble | domain-specific | 91.55 | 52.18 | 99.89 | 4.47 | 95.57 | 0.7217 | 0.0750 | 0.5244 |
| DFDC | DFDC | EfficientNet-B3 | domain-specific | 56.67 | 56.67 | 83.33 | 30.00 | 65.79 | 0.6044 | 0.3237 | 0.2107 |
| DFDC | DFDC | ConvNeXt-Tiny | domain-specific | 76.67 | 76.67 | 63.33 | 90.00 | 73.08 | 0.8233 | 0.1597 | 0.5245 |
| DFDC | DFDC | YOLO26n-cls | domain-specific | 73.33 | 73.33 | 90.00 | 56.67 | 77.14 | 0.8667 | 0.1686 | 0.5406 |
| DFDC | DFDC | Ensemble | domain-specific | 66.67 | 66.67 | 80.00 | 53.33 | 70.59 | 0.8211 | 0.1830 | 0.5244 |
| DFDC | Real-Life | EfficientNet-B3 | domain-specific | 52.83 | 52.83 | 95.33 | 10.33 | 66.90 | 0.4496 | 0.4109 | 0.2107 |
| DFDC | Real-Life | ConvNeXt-Tiny | domain-specific | 40.67 | 40.67 | 54.00 | 27.33 | 47.65 | 0.3929 | 0.4354 | 0.5245 |
| DFDC | Real-Life | YOLO26n-cls | domain-specific | 47.25 | 47.25 | 89.33 | 5.17 | 62.87 | 0.3626 | 0.4542 | 0.5406 |
| DFDC | Real-Life | Ensemble | domain-specific | 47.50 | 47.50 | 86.67 | 8.33 | 62.28 | 0.4014 | 0.3872 | 0.5244 |

> **Note on cross-dataset generalisation**: ROC-AUC in this report is computed with **FAKE as the
> positive class**, so a value of 0.5 is chance and higher is better. In published deepfake
> forensics literature, state-of-the-art architectures typically lose 15–35 AUC points when
> evaluated on a dataset they were not trained on. Those published figures are cited as external
> academic reference points for risk modelling, not as results of this project.

> **Note on identity leakage**: DFDC groups fakes with their source video, but the same actor can
> appear in different source videos. Some residual identity overlap may therefore remain even after
> a correct identity-aware split. This is disclosed in the report and noted as a dataset-level
> limitation.

> **Note on the "Trained on" column**: a row marked *shared ckpt* was produced with the
> shared/global checkpoint because no domain-specific checkpoint exists for that training
> domain. Such rows are not independent cross-domain measurements. Every row here is
> marked *domain-specific*, so each is a genuine train-domain / test-domain measurement.

---

## 3. Model Speed & Efficiency

| Architecture | Parameters (M) | Model Size (MB) | Compute (GFLOPs) | Latency (Batch=1 ms) | P95 Latency (ms) | Throughput (FPS B=32) | Measured On |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ConvNeXt-Tiny (timm) | 27.82 M | 334.1 MB | 8.93 | 90.11 ms | 95.62 ms | 17.0 fps | CPU |
| EfficientNet-B3 (timm) | 10.70 M | 129.2 MB | 2.03 | 67.77 ms | 74.35 ms | 15.7 fps | CPU |
| YOLO26n-cls (Ultralytics) | 1.53 M | 3.2 MB | 0.40 | 14.18 ms | 16.37 ms | 138.0 fps | CPU |
| Soft-Voting Ensemble (All 3) | 40.05 M | 466.5 MB | 11.36 | 172.06 ms | 186.34 ms | 14.5 fps | CPU |

---

## 4. Model Training Status

| Model | Epochs | Best Train Acc | Best Val Acc | Status |
|---|---|---|---|---|
| convnext_tiny.fb_in22k_ft_in1k [celebdf] | 4 | 84.79% | 93.49% | Converged |
| efficientnet_b3.ra2_in1k [celebdf] | 4 | 74.58% | 84.27% | Converged |
| resnet18 [celebdf] | 6 | 97.30% | 95.36% | Converged |
| convnext_tiny.fb_in22k_ft_in1k [dfdc] | 4 | 80.24% | 92.19% | Converged |
| efficientnet_b3.ra2_in1k [dfdc] | 4 | 67.79% | 72.81% | Converged |

---

## 5. Operating Points (False-Positive Rates)

| Domain | Model | Threshold | FPR at 0.5 | FPR at threshold | Fake recall at 0.5 | Fake recall at threshold | Real recall at threshold |
|---|---|---:|---:|---:|---:|---:|---:|
| celebdf | convnext_tiny.fb_in22k_ft_in1k | 0.8183 | 75.5% | 28.4% | 99.6% | 87.1% | 71.6% |
| celebdf | efficientnet_b3.ra2_in1k | 0.9323 | 51.6% | 24.5% | 87.4% | 65.8% | 75.5% |
| celebdf | ensemble (soft voting, mean of members) | 0.8527 | 63.9% | 19.7% | 99.3% | 76.3% | 80.3% |
| celebdf | resnet18 | 0.9576 | 46.8% | 10.0% | 99.1% | 86.2% | 90.0% |
| celebdf | yolo26n-cls | 0.9144 | 66.1% | 26.1% | 99.0% | 81.4% | 73.9% |
| dfdc | convnext_tiny.fb_in22k_ft_in1k | 0.5245 | 7.0% | 4.0% | 91.4% | 90.5% | 96.0% |
| dfdc | efficientnet_b3.ra2_in1k | 0.2107 | 35.0% | 41.0% | 75.9% | 87.3% | 59.0% |
| dfdc | ensemble (soft voting, mean of members) | 0.5244 | 13.0% | 11.0% | 91.8% | 91.8% | 89.0% |
| dfdc | yolo26n-cls | 0.5406 | 21.0% | 15.0% | 87.3% | 83.6% | 85.0% |

FPR is the share of **authentic** faces wrongly classified as fake. It is measured on each domain's validation split; the test split is never used for threshold selection. The threshold column is the shipped operating point (the balanced Youden point).

> **Why this matters.** A threshold of 0.5 is only meaningful for a balanced test set.
> On the imbalanced Celeb-DF v2 split it would accuse a large share of authentic
> faces. The table above shows the measured false-positive rate at 0.5 and at the
> shipped operating point, so the trade-off is explicit rather than implicit.

---

## 6. Frequency-Domain (Spectral) Branch

| Evaluation | Model | Neural only | Spectral only | Neural + spectral |
|---|---|---:|---:|---:|
| cross-domain (unseen) | convnext_tiny.fb_in22k_ft_in1k | 0.6533 | 0.1038 | 0.8157 |

ROC-AUC, FAKE as the positive class, fitted on validation and scored on the held-out split. The spectral branch helps on the *unseen* evaluation domain and adds nothing in-domain.

---

## 7. Findings

*Every statement below is derived automatically from the artefacts in Sections 1–4.
No sentence is hand-written, so this section cannot drift away from the results.*

- The evaluated checkpoint produces a **non-degenerate** decision boundary: it makes both kinds of prediction on the test set (accuracy 0.5000, ROC-AUC 0.6044 against 0.5 for chance).
- **Across the cross-dataset matrix** (Table 2), the strongest ROC-AUC is 0.9403 (YOLO26n-cls on CelebDF), and the weakest is 0.3626 (YOLO26n-cls on Real-Life).
- 2 of 24 cells sit within ±0.05 of the 0.50 chance line and 9 clear 0.70, so the detectors discriminate on most train/test combinations but not on all of them.

---

## 8. Limitations & Risks

- **Dataset scope**: Models were trained and tested on DFDC and Celeb-DF only. Performance on
  other deepfake types (e.g., text-driven video synthesis, lip-sync fakes) is unknown.
- **Generalisation gap**: Cross-dataset AUC (Table 2) is lower than in-domain AUC in the
  current run. This is consistent with the academic literature, where published detectors
  typically lose 15–35 AUC points when moved to an unseen dataset, and it is the expected
  behaviour for models trained on a single domain.
- **Real-life performance**: The real-life image set consists of AI-generated faces
  from StyleGAN / Stable Diffusion. Results on live-capture deepfakes may differ.
- **Legal notice (AGPL licensing)**: Any AGPL-licensed component used in this pipeline requires
  that derivative source code be disclosed under the same licence. This is a **risk notice**, not
  legal advice. Independent legal counsel should be consulted before commercial deployment.

---

*Report generated by `scripts/regenerate_report.py`. Source data: `reports/evaluation_summary.json`,
`reports/cross_dataset_table.csv`, `reports/speed_table.csv`, `reports/*_history.json`.*

