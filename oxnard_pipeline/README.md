# Oxnard 2-Stage Weed Stem Detection — Reproducible Pipeline

This folder contains everything needed to reproduce the end-to-end comparison of
**direct keypoint regression** vs **soft-argmax heatmap** as Phase-2 architectures, on
top of two different Phase-1 detectors (YOLO-detect, YOLO-segment), plus a
single-stage YOLO-pose baseline.

## Current best result

| config | mean Dist | median Dist | mean px | median px | n_kept | ms / image |
|---|---|---|---|---|---|---|
| det + keypoint (direct)        | 4.75 | 3.89 | 6.97 | 5.71 | 234 | 42.4 |
| det + heatmap (soft-argmax)    | 3.92 | 3.04 | 5.76 | 4.46 | 234 | 32.6 |
| seg + keypoint (direct)        | 4.22 | 3.19 | 6.19 | 4.68 | 267 | 43.6 |
| **seg + heatmap (soft-argmax)** | **3.77** | **2.83** | **5.54** | **4.16** | **267** | 41.5 |
| pose joint (single-stage)      | 4.84 | 3.94 | 7.11 | 5.78 | 220 | 30.3 |

`Dist = mean(sqrt(squared_px / diag²)) × 1000` with `diag = sqrt(1280² + 720²) = 1468.6`.

Charts: [results/comparison/comparison_chart.png](../results/comparison/comparison_chart.png),
[results/comparison/training_chart.png](../results/comparison/training_chart.png).

## Architecture (winning cell: seg + heatmap)

```
left/<img>.jpg  (1280×720)
   │
   ▼  Phase 1
best_pigweed_145.pt  (YOLO-seg, imgsz=1280, conf=0.45)
   │  → bbox + segmentation mask per detected pigweed
   ▼
Mask the image (background → black) + pad bbox 20 px + crop
   │
   ▼  Phase 2
keypoint_oxnard_heatmap_masked.pt
  MobileNet-v3-small encoder
  → U-Net decoder with 24-ch + 48-ch skip connections
  → 56×56 score map → spatial soft-argmax → (x, y) in 224 space
   │  trained with SmoothL1 loss directly on coordinates
   ▼
Map stem back to full-image coordinates
```

The direct-regression variant uses the same encoder but replaces the decoder with
global-average-pool + MLP → (x, y). The two variants are toggled in
[train_keypoint_v2.py](train_keypoint_v2.py) via `KP_ARCH=heatmap|direct` and
`KP_CROPS=masked|raw`.

## Files in this folder

| File | What it does | When to run |
|------|--------------|-------------|
| [01_prep.py](01_prep.py) | Phase-1 detect + Phase-2 masked-crop dataset prep. Splits 478 left images 394/84/85, recovers per-crop bboxes by re-running `best_pigweed_145.pt`, writes `datasets/oxnard_det/`, `datasets/oxnard_kp/`, `datasets/oxnard_det/crop_offsets.csv`. | Once, after the source data is in place |
| [build_unmasked_crops.py](build_unmasked_crops.py) | Builds `datasets/oxnard_kp_raw/` (same crops, **no** masking — used for the `KP_CROPS=raw` ablation). | Once, after `01_prep.py` |
| [04_phase1_pose_prep.py](04_phase1_pose_prep.py) | Builds `datasets/oxnard_pose/` (YOLO-pose format, single keypoint per bbox) using `crop_offsets.csv`. | Once, before training the pose baseline |
| [train_keypoint_v2.py](train_keypoint_v2.py) | Phase-2 trainer. `KP_ARCH ∈ {heatmap, direct}` × `KP_CROPS ∈ {masked, raw}` → 4 checkpoints + 4 `.train.json` logs. | Run 4× to populate `models/keypoint_oxnard_*` |
| [compare_all.py](compare_all.py) | End-to-end eval of all 5 cells on the held-out test split. Writes `results/comparison/comparison.json`, `comparison_chart.png`, `training_chart.png`. | After the 4 Phase-2 models exist |
| [02_train_keypoint.py](02_train_keypoint.py) | Legacy single-config trainer (kept for reference, not used by `compare_all.py`). | Optional |
| [03_eval.py](03_eval.py) | Legacy single-config eval (seg + heatmap only). | Optional sanity check |

## Inputs the pipeline expects

All paths are absolute (hardcoded in the scripts) and live one level up at
`/home/jaehwan/Desktop/laser-weeding/`:

| What | Path | How to get it |
|---|---|---|
| Raw left-camera images (478×) | `left/` | Source data (manually labelled stems are in `data/keypoint_labels/`, paired crops in `processed_crops/`) |
| Phase-1 segmentation weights | `best_pigweed_145.pt` | Trained on the Roboflow [Oxnard-Pigweed-1](../Oxnard-Pigweed-1/) export — see that folder's README |
| Phase-1 detection weights | `results/oxnard_det/weights/best.pt` | YOLO-detect, trained from `datasets/oxnard_det/oxnard_det.yaml` after `01_prep.py` |
| Phase-1 pose weights | `results/oxnard_pose/weights/best.pt` | YOLO-pose, trained from `datasets/oxnard_pose/oxnard_pose.yaml` after `04_phase1_pose_prep.py` |
| Phase-2 weights (4×) | `models/keypoint_oxnard_{heatmap,direct}_{masked,raw}.pt` | Produced by `train_keypoint_v2.py` |

## Reproduce from scratch

```bash
conda activate yolo26
cd /home/jaehwan/Desktop/laser-weeding

# 0) Data prep — produces detection + masked-crop + raw-crop + pose datasets
python oxnard_pipeline/01_prep.py
python oxnard_pipeline/build_unmasked_crops.py
python oxnard_pipeline/04_phase1_pose_prep.py

# 1) Phase-1 detector (YOLOv8-m, imgsz=1280, ~30 min on A6000)
yolo detect train \
    model=yolov8m.pt \
    data=datasets/oxnard_det/oxnard_det.yaml \
    epochs=100 patience=25 batch=8 imgsz=1280 seed=0 \
    project=results name=oxnard_det

# 2) Phase-1 pose baseline (YOLOv8-m-pose, imgsz=1280, ~40 min on A6000)
yolo pose train \
    model=yolov8m-pose.pt \
    data=datasets/oxnard_pose/oxnard_pose.yaml \
    epochs=120 patience=30 batch=8 imgsz=1280 seed=0 \
    project=results name=oxnard_pose

# 3) Phase-1 segmentation weights are already at best_pigweed_145.pt — see Oxnard-Pigweed-1/README.md to retrain

# 4) Phase-2: 4 keypoint models (each ~3-5 min on A6000)
KP_ARCH=heatmap KP_CROPS=masked python oxnard_pipeline/train_keypoint_v2.py
KP_ARCH=heatmap KP_CROPS=raw    python oxnard_pipeline/train_keypoint_v2.py
KP_ARCH=direct  KP_CROPS=masked python oxnard_pipeline/train_keypoint_v2.py
KP_ARCH=direct  KP_CROPS=raw    python oxnard_pipeline/train_keypoint_v2.py

# 5) End-to-end comparison + figures
python oxnard_pipeline/compare_all.py
```

Outputs:
- `results/comparison/comparison.json` — raw numbers per cell
- `results/comparison/comparison_chart.png` — accuracy + speed + scatter (3 panels)
- `results/comparison/training_chart.png` — val px err / best epoch / training time per Phase-2 variant
- `models/keypoint_oxnard_*.train.json` — per-variant training stats

## Eval protocol

- Phase-1 predictions are matched to GT by IoU ≥ 0.30 against the GT bbox; unmatched detections are dropped (so a false-positive far from any plant doesn't pollute the mean).
- For each kept prediction, the predicted stem is paired with the IoU-best GT stem.
- `Dist` is the same formula as `calculate_mse` in the original paper's `test.py`: per-sample √(squared_px) divided by image diagonal in pixels, averaged, ×1000.

## Architecture notes (why heatmap beats direct)

The Phase-2 heatmap variant is **not** the naive "sigmoid heatmap + hard argmax" — that
loses to direct regression by ~2–3 px because hard argmax has a 1-px quantization
floor and the per-pixel sigmoid loss doesn't shape the readout. The variant in this
repo is:

- 56×56 score map → **spatial softmax** (probability map)
- **soft-argmax**: `E[x] = Σ p(x,y)·x_grid`, fully differentiable, sub-pixel accurate
- **U-Net skip connections** (24-ch and 48-ch from MobileNet-v3 stages) so fine spatial
  detail isn't lost at the 7×7 bottleneck
- **SmoothL1 loss on the (x, y) coordinate**, not on the heatmap shape

Same backbone weights (`mobilenet_v3_small(DEFAULT)`) for both variants, identical
data, identical optimizer/LR/augmentation. The architectural changes are all that
differ. Reference implementation: [scripts/softargmax_skip_noseg_sweep.py](../scripts/softargmax_skip_noseg_sweep.py).

A snapshot of the previous "vanilla heatmap" end-to-end results (sigmoid + hard
argmax, no skips, MSE on heatmap) is preserved at
[results/comparison/comparison_old_heatmap.json](../results/comparison/comparison_old_heatmap.json)
for the architecture ablation in the paper.
