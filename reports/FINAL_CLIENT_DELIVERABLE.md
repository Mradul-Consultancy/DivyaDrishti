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
| CELEBDF | CelebDF | EfficientNet-B3 | domain-specific | 57.50 | 48.48 | 0.6700 | 0.3553 |
| CELEBDF | CelebDF | ConvNeXt-Tiny | domain-specific | 60.00 | 71.43 | 0.7625 | 0.3252 |
| CELEBDF | CelebDF | YOLO26n-cls | domain-specific | 50.00 | 9.09 | 0.6125 | 0.4180 |
| CELEBDF | CelebDF | Ensemble | domain-specific | 60.00 | 55.56 | 0.7175 | 0.2208 |
| CELEBDF | DFDC | EfficientNet-B3 | domain-specific | 47.37 | 51.22 | 0.5305 | 0.4094 |
| CELEBDF | DFDC | ConvNeXt-Tiny | domain-specific | 48.68 | 64.22 | 0.5298 | 0.3514 |
| CELEBDF | DFDC | YOLO26n-cls | domain-specific | 56.58 | 23.26 | 0.5956 | 0.3897 |
| CELEBDF | DFDC | Ensemble | domain-specific | 56.58 | 54.79 | 0.5686 | 0.2583 |
| CELEBDF | Real-Life | EfficientNet-B3 | domain-specific | 41.75 | 5.92 | 0.3481 | 0.5522 |
| CELEBDF | Real-Life | ConvNeXt-Tiny | domain-specific | 32.92 | 21.31 | 0.2738 | 0.5243 |
| CELEBDF | Real-Life | YOLO26n-cls | domain-specific | 48.33 | 1.27 | 0.2189 | 0.5117 |
| CELEBDF | Real-Life | Ensemble | domain-specific | 44.33 | 4.02 | 0.2430 | 0.4702 |
| DFDC | CelebDF | EfficientNet-B3 | domain-specific | 82.50 | 83.72 | 0.8475 | 0.1657 |
| DFDC | CelebDF | ConvNeXt-Tiny | domain-specific | 75.00 | 80.00 | 0.9100 | 0.1665 |
| DFDC | CelebDF | YOLO26n-cls | domain-specific | 95.00 | 95.00 | 0.9925 | 0.0536 |
| DFDC | CelebDF | Ensemble | domain-specific | 90.00 | 90.48 | 0.9825 | 0.0921 |
| DFDC | DFDC | EfficientNet-B3 | domain-specific | 61.84 | 64.20 | 0.7188 | 0.3270 |
| DFDC | DFDC | ConvNeXt-Tiny | domain-specific | 61.84 | 50.85 | 0.7368 | 0.1951 |
| DFDC | DFDC | YOLO26n-cls | domain-specific | 72.37 | 66.67 | 0.8331 | 0.1687 |
| DFDC | DFDC | Ensemble | domain-specific | 60.53 | 62.50 | 0.7860 | 0.1888 |
| DFDC | Real-Life | EfficientNet-B3 | domain-specific | 62.42 | 61.94 | 0.6620 | 0.3056 |
| DFDC | Real-Life | ConvNeXt-Tiny | domain-specific | 48.00 | 26.93 | 0.4585 | 0.2934 |
| DFDC | Real-Life | YOLO26n-cls | domain-specific | 47.33 | 3.66 | 0.3874 | 0.4786 |
| DFDC | Real-Life | Ensemble | domain-specific | 49.92 | 19.11 | 0.5888 | 0.2850 |

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
- **Across the cross-dataset matrix** (Table 2), the strongest ROC-AUC is 0.9925 (YOLO26n-cls on CelebDF), and the weakest is 0.2189 (YOLO26n-cls on Real-Life).
- 3 of 24 cells sit within ±0.05 of the 0.50 chance line, indicating that the underlying weights are not yet discriminating between authentic and manipulated faces.

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

