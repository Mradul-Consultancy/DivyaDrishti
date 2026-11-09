# Deepfake Detection — Honest Project Status
*Updated: 2026-11-07*

---

## Reality Check

| Layer | % Done | Evidence |
|---|---|---|
| **Pipeline code** (preprocessing, training, evaluation, CLI) | ~70% | All scripts exist, 55 tests pass, `main.py --help` shows 18 commands |
| **Data** (actual images to train on) | ~10% | Pilot DFDC subset (200 videos) present, Celeb‑DF still missing |
| **Trained models** (real, convergent weights) | 0% | No full‑grid checkpoints yet |
| **Experiments** (cross‑dataset table, real‑life scores) | 0% | No runs have completed |

---

## Data Situation (The Blocker)

### Processed face images today
```
celebdf/train/real   →  10 images   ← toy only
celebdf/train/fake   →  MISSING
celebdf/test/fake    →  20 images
celebdf/test/real    →  20 images

dfdc/train/fake      →  70 images
dfdc/train/real      →  70 images

dfdc/val/{fake,real} →  10 each

dfdc/test/{fake,real}→  30 each
```

### Raw downloads
```
Celeb-real/      →  2 zip files  (id0_0000, id10_0000)
Celeb-synthesis/ →  2 zip files

dfdc_train_part_0→  2 videos + metadata.json
```

Only a handful of raw videos were ever downloaded. Face extraction has only been run in demo mode.

### Checkpoint metadata
All three checkpoints: **`epoch=1, val_acc=50.0`** — coin‑flip accuracy. These are warm‑starts saved after a single epoch on the toy data, not real trained weights.

---

## The Gap
```
                NOW                        GOAL
Raw video       2 Celeb-real zips          ~400 Celeb-real + ~430 Celeb-synthesis
downloads       2 DFDC videos              ~400–1000 DFDC videos

Processed       ~240 face crops total      ~40,000–200,000 face crops
images          across all splits          across all splits

Training        epoch=1, val_acc=50%       20–50 epochs, val_AUC > 0.90
                (untrained)

Evaluation      0 results                  6‑cell cross‑dataset AUC table
                                            + CI bounds + real‑life scores
``` 
---

## Path to Real Experiments
> **All real training must happen on Kaggle** (or similar GPU). CPU training of EfficientNet‑B3 on a real‑sized dataset takes ~8 h per epoch.

### Phase 1 — Get Real Data (Kaggle, ~2–3 hrs wall time)

**CelebDF** — download from `jessicali9530/celeb-df-v2` (or `iabhi/celeb-df-deepfake-dataset`):
- Celeb‑real (~590 MP4s), Celeb‑synthesis (~5639 MP4s), YouTube‑real (~300 MP4s)
- Extract faces → `data/processed/celebdf/{train,val,test}/{real,fake}/`

**DFDC** — `c2-contest/deepfake-detection-challenge`, parts 0–9 for pilot, all 50 for full run:
- Extract faces → `data/processed/dfdc/`

**Real‑life** — populate `data/real_life/{real,fake}/` with 50–200 clips each from sources outside the training sets.

### Phase 2 — Pilot (Kaggle, ~40 min on T4)

Run `notebooks/kaggle_pilot.ipynb` as‑is:
1. Download 200 paired DFDC videos
2. Extract faces (MTCNN)
3. Verify zero‑leakage split
4. Train EfficientNet‑B3 × 5 epochs (smoke test)
5. Evaluate → `saved_predictions.csv`
6. Robustness spot‑check (JPEG q=50)

This is to catch pipeline bugs **before** committing 18 GPU‑hours to the full run.

### Phase 3 — Full Training Grid (~12–18 hrs on T4)

```
For each dataset in {celebdf, dfdc}:
    For each model in {EfficientNet‑B3, ConvNeXt‑Tiny, YOLO26n‑cls}:
        Train 20 epochs, early stopping patience=5
```

```
bash
python main.py train-all-datasets  # already implemented
```

### Phase 4 — Build the Results Table (~1 hr)

```
bash
python main.py build-cross-table   # 6‑cell AUC matrix + CI bounds
python main.py generate-reports    # ROC curves, confusion matrices
python main.py benchmark-speed     # Params, GFLOPs, latency table
```

Expected cross‑dataset table (illustrative — honest expected ranges):

| Train ↓ / Test → | CelebDF | DFDC | Real‑Life |
|---|---|---|---|
| CelebDF | ~0.90–0.95 | **~0.65–0.75** | **~0.58–0.70** |
| DFDC    | **~0.65–0.75** | ~0.88–0.93 | **~0.60–0.72** |

> **NOTE**
> Cross‑domain and real‑life numbers will be significantly lower than in‑domain. The generalisation gap **is** the experiment. It is a real and publishable finding — not a failure.

---

## Outstanding Code Gaps (before full run)

| Gap | Priority |
|---|---|
| `celebdf/train/fake` missing — CelebDF extraction was never completed | 🔴 Blocker |
| `data/processed/celebdf/val/` missing entirely | 🔴 Blocker |
| `data/real_life/` empty | 🔴 For real‑life test |
| `benchmark_speed.py` has never run to completion (no output files yet) | 🟡 |
| Grad‑CAM tested only on a random tensor, not a real face crop | 🟡 |
| YOLO CLI default checkpoint path assumes flat layout, not per‑dataset | 🟡 |

---

## Realistic Timeline

| Task | GPU Time (Kaggle T4) |
|---|---|
| Data download + face extraction (CelebDF + 10 DFDC parts) | 2–3 hrs |
| Pilot run (1 model × 5 epochs, bug verification) | ~40 min |
| Full training grid (6 runs × 20 epochs) | 12–18 hrs |
| Cross‑table + reports + benchmark | ~30 min |
| **Total** | **~16–22 hrs** |

Kaggle gives 30 hrs/week free on T4. The full run comfortably fits in one week of quota.

