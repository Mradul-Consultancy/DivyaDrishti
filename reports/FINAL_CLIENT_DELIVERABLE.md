# Deep-Fake Detection — Client Deliverable
*Auto-generated from result artefacts on 2026-11-10. No hand-written numbers.*

> **Important**: This document is generated automatically from JSON/CSV files produced by
> the evaluation pipeline. Do not edit it manually — re-run `python scripts/regenerate_report.py`
> after each training/evaluation cycle.

---

## 1. Evaluation Metrics (In-Domain)

- **model_name**: efficientnet_b3.ra2_in1k
- **checkpoint**: models/checkpoints/efficientnet_b3.ra2_in1k_best.pth
- **total_samples**: 76
- **accuracy**: 0.5000
- **precision**: 0.0000
- **recall**: 0.0000
- **f1_score**: 0.0000
- **roc_auc**: 0.4834
- **confusion_matrix**: [[0, 38], [0, 38]]
- **detailed_report**: {'fake': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 38.0}, 'real': {'precision': 0.5, 'recall': 1.0, 'f1-score': 0.6666666666666666, 'support': 38.0}, 'accuracy': 0.5, 'macro avg': {'precision': 0.25, 'recall': 0.5, 'f1-score': 0.3333333333333333, 'support': 76.0}, 'weighted avg': {'precision': 0.25, 'recall': 0.5, 'f1-score': 0.3333333333333333, 'support': 76.0}}

---

## 2. Cross-Dataset Performance Table

| Trained on | Test dataset | Model | Checkpoint | Accuracy (%) | F1-Score (%) | ROC-AUC | Brier Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CELEBDF | CelebDF | EfficientNet-B3 | shared (no per-domain ckpt) | 50.00 | 0.00 | 0.8150 | 0.2889 |
| CELEBDF | CelebDF | ConvNeXt-Tiny | shared (no per-domain ckpt) | 50.00 | 66.67 | 0.2975 | 0.3809 |
| CELEBDF | CelebDF | YOLO26n-cls | shared (no per-domain ckpt) | 57.50 | 70.18 | 0.7650 | 0.2373 |
| CELEBDF | CelebDF | Ensemble | shared (no per-domain ckpt) | 22.50 | 27.91 | 0.1100 | 0.2967 |
| CELEBDF | DFDC | EfficientNet-B3 | shared (no per-domain ckpt) | 50.00 | 0.00 | 0.5675 | 0.2893 |
| CELEBDF | DFDC | ConvNeXt-Tiny | shared (no per-domain ckpt) | 50.00 | 66.07 | 0.5284 | 0.2706 |
| CELEBDF | DFDC | YOLO26n-cls | shared (no per-domain ckpt) | 52.63 | 64.00 | 0.4453 | 0.2827 |
| CELEBDF | DFDC | Ensemble | shared (no per-domain ckpt) | 51.32 | 59.34 | 0.5291 | 0.2513 |
| CELEBDF | Real-Life | EfficientNet-B3 | shared (no per-domain ckpt) | 50.00 | 0.00 | 0.2571 | 0.2896 |
| CELEBDF | Real-Life | ConvNeXt-Tiny | shared (no per-domain ckpt) | 34.00 | 48.70 | 0.2574 | 0.3276 |
| CELEBDF | Real-Life | YOLO26n-cls | shared (no per-domain ckpt) | 31.50 | 46.55 | 0.0693 | 0.3938 |
| CELEBDF | Real-Life | Ensemble | shared (no per-domain ckpt) | 68.75 | 70.50 | 0.7283 | 0.2354 |
| DFDC | CelebDF | EfficientNet-B3 | shared (no per-domain ckpt) | 50.00 | 0.00 | 0.8150 | 0.2889 |
| DFDC | CelebDF | ConvNeXt-Tiny | shared (no per-domain ckpt) | 50.00 | 66.67 | 0.2975 | 0.3809 |
| DFDC | CelebDF | YOLO26n-cls | shared (no per-domain ckpt) | 57.50 | 70.18 | 0.7650 | 0.2373 |
| DFDC | CelebDF | Ensemble | shared (no per-domain ckpt) | 22.50 | 27.91 | 0.1100 | 0.2967 |
| DFDC | DFDC | EfficientNet-B3 | shared (no per-domain ckpt) | 50.00 | 0.00 | 0.5675 | 0.2893 |
| DFDC | DFDC | ConvNeXt-Tiny | shared (no per-domain ckpt) | 50.00 | 66.07 | 0.5284 | 0.2706 |
| DFDC | DFDC | YOLO26n-cls | shared (no per-domain ckpt) | 52.63 | 64.00 | 0.4453 | 0.2827 |
| DFDC | DFDC | Ensemble | shared (no per-domain ckpt) | 51.32 | 59.34 | 0.5291 | 0.2513 |
| DFDC | Real-Life | EfficientNet-B3 | shared (no per-domain ckpt) | 50.00 | 0.00 | 0.2571 | 0.2896 |
| DFDC | Real-Life | ConvNeXt-Tiny | shared (no per-domain ckpt) | 34.00 | 48.70 | 0.2574 | 0.3276 |
| DFDC | Real-Life | YOLO26n-cls | shared (no per-domain ckpt) | 31.50 | 46.55 | 0.0693 | 0.3938 |
| DFDC | Real-Life | Ensemble | shared (no per-domain ckpt) | 68.75 | 70.50 | 0.7283 | 0.2354 |

> **Note on cross-dataset generalisation**: ROC-AUC in this report is computed with **FAKE as the
> positive class**, so a value of 0.5 is chance and higher is better. In published deepfake
> forensics literature, state-of-the-art architectures typically lose 15–35 AUC points when
> evaluated on a dataset they were not trained on. Those published figures are cited as external
> academic reference points for risk modelling, not as results of this project.

> **Note on identity leakage**: DFDC groups fakes with their source video, but the same actor can
> appear in different source videos. Some residual identity overlap may therefore remain even after
> a correct identity-aware split. This is disclosed in the report and noted as a dataset-level
> limitation.

> **Note on the "Trained on" column**: rows marked *shared ckpt* were produced with the
> shared/global checkpoint because no domain-specific checkpoint exists for that training domain.
> Such rows are not independent cross-domain measurements. See the Findings section.

---

## 3. Model Speed & Efficiency

| Architecture | Parameters (M) | Model Size (MB) | Compute (GFLOPs) | Latency (Batch=1 ms) | P95 Latency (ms) | Throughput (FPS B=32) |
| --- | --- | --- | --- | --- | --- | --- |
| ConvNeXt-Tiny (timm) | 27.82 M | 334.1 MB | 8.93 | 101.36 ms | 128.47 ms | 13.9 fps |
| EfficientNet-B3 (timm) | 10.70 M | 129.2 MB | 2.03 | 62.93 ms | 74.69 ms | 17.7 fps |
| YOLO26n-cls (Ultralytics) | 1.53 M | 3.2 MB | 0.40 | 14.05 ms | 18.12 ms | 164.0 fps |
| Soft-Voting Ensemble (All 3) | 40.05 M | 466.5 MB | 11.36 | 178.34 ms | 221.28 ms | 14.0 fps |

---

## 4. Model Training Status

| Model | Epochs | Best Train Acc | Best Val Acc | Status |
|---|---|---|---|---|
| convnext_tiny.fb_in22k_ft_in1k | 2 | 63.64% | 50.00% | **Not converged (chance-level)** |
| efficientnet_b3.ra2_in1k | 8 | 55.48% | 50.00% | **Not converged (chance-level)** |
| resnet18 (demo run) | 2 | 83.33% | 50.00% | **Not converged (chance-level)** |

---

## 5. Findings

*Every statement below is derived automatically from the artefacts in Sections 1–4.
No sentence is hand-written, so this section cannot drift away from the results.*

- **Constant-prediction collapse.** The evaluated checkpoint predicts *real* for every image in the test set (accuracy 0.5000, ROC-AUC 0.4834). This is a degenerate decision boundary, not a usable detector.
- **Across the cross-dataset matrix** (Table 2), the strongest ROC-AUC is 0.8150 (EfficientNet-B3 on CelebDF), and the weakest is 0.0693 (YOLO26n-cls on Real-Life).
- 4 of 24 cells sit within ±0.05 of the 0.50 chance line, indicating that the underlying weights are not yet discriminating between authentic and manipulated faces.
- ⚠️ **Train-domain caveat.** The rows labelled **CELEBDF**, **DFDC** in Table 2 were produced with the shared/global checkpoint, because no domain-specific checkpoint exists for them. Those rows are **not** independent cross-domain measurements; a per-domain model must be trained before any cross-dataset generalisation claim is valid.

---

## 6. Limitations & Risks

- **Dataset scope**: Models were trained and tested on DFDC and Celeb-DF only. Performance on
  other deepfake types (e.g., text-driven video synthesis, lip-sync fakes) is unknown.
- **Generalisation gap**: Cross-dataset AUC (Table 2) is lower than in-domain AUC in the
  current run. This is consistent with the academic literature and expected at this scale of
  training data — but with the present warm-up checkpoints the gap is driven by
  non-convergence rather than by genuine domain shift, so it should not be read as a
  measured generalisation result.
- **Real-life performance**: The real-life image set consists of AI-generated faces
  from StyleGAN / Stable Diffusion. Results on live-capture deepfakes may differ.
- **Legal notice (AGPL licensing)**: Any AGPL-licensed component used in this pipeline requires
  that derivative source code be disclosed under the same licence. This is a **risk notice**, not
  legal advice. Independent legal counsel should be consulted before commercial deployment.

---

*Report generated by `scripts/regenerate_report.py`. Source data: `reports/evaluation_summary.json`,
`reports/cross_dataset_table.csv`, `reports/speed_table.csv`, `reports/*_history.json`.*

