# Oxnard-Pigweed-1 (Phase-1 segmentation training data)

Roboflow export (v1, 2026-03-18) used to train the Phase-1 pigweed segmentation
model that powers the `seg_*` cells of the end-to-end comparison.

| | |
|---|---|
| Source | https://universe.roboflow.com/laser-weeding/oxnard-pigweed (v1) |
| License | CC BY 4.0 |
| Format | YOLO-seg (polygon labels, single class `Pigweed`) |
| Images | 145 total — 115 train / 30 valid (image size 1280×720) |
| Pre-processing | Auto-orientation only; no augmentation |
| Trained checkpoint | `../best_pigweed_145.pt` |
| Used by | `seg_direct` and `seg_heatmap` cells in `../oxnard_pipeline/compare_all.py` |

## Layout

```
Oxnard-Pigweed-1/
├── data.yaml          # YOLO config (paths are relative to this folder)
├── train/{images,labels}/   115 images
├── valid/{images,labels}/    30 images
├── README.dataset.txt        Roboflow auto-generated
├── README.roboflow.txt       Roboflow auto-generated
└── README.md                 (this file)
```

`data.yaml` already points train/val/test paths relative to this folder, so
training works without edits.

## Retraining `best_pigweed_145.pt` from scratch

```bash
conda activate yolo26
cd /home/jaehwan/Desktop/laser-weeding

yolo segment train \
    model=yolov8m-seg.pt \
    data=Oxnard-Pigweed-1/data.yaml \
    epochs=145 imgsz=1280 batch=8 seed=0 \
    project=results name=oxnard_pigweed_seg

# After training, copy the best weights to the path the pipeline expects:
cp results/oxnard_pigweed_seg/weights/best.pt best_pigweed_145.pt
```

## Where this fits in the bigger picture

This dataset trains **only Phase-1 (segmentation)**. The full reproducible
pipeline — keypoint vs heatmap × det vs seg comparison, training scripts, eval,
and chart generation — lives at:

→ [`../oxnard_pipeline/`](../oxnard_pipeline/) — start with that folder's README.

The Phase-1 detection variant (`det_*` cells) uses a different dataset
(`datasets/oxnard_det/`, 478 images split 394/84/85), built by
`oxnard_pipeline/01_prep.py` from the left-camera images in `../left/`. That one
is **not** sourced from this Roboflow export.
