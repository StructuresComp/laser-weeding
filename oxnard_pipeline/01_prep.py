"""
Build a clean train/val/test split of Oxnard left-camera data, with paired
labels for both pipeline phases:

  Phase 1 (detection): full left images + YOLO bbox labels (single class)
  Phase 2 (keypoint):  per-plant crops + stem coords (already exists in
                       processed_crops/, we just symlink & split)

Splits are made at PARENT IMAGE level so no plant from the same image
appears in two splits.

Bbox source: re-run best_pigweed_145.pt with the exact settings of
scripts/segment_processing.py (imgsz=1280, conf=0.45, padding=20, min 400
visible pixels). Each detection's index `i` matches plant_<stem>_<i>.png
in processed_crops/, which is how we link crops to bboxes and stems.

Outputs:
  datasets/oxnard_det/{train,val,test}/{images (symlinks), labels}/
                                        + oxnard_det.yaml
  datasets/oxnard_kp/{train,val,test}/{images (symlinks to crops), labels (symlinks to stems)}/
"""
import os, shutil, cv2, numpy as np, torch
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

ROOT      = '/home/jaehwan/Desktop/laser-weeding'
LEFT_DIR  = os.path.join(ROOT, 'left')
CROPS_DIR = os.path.join(ROOT, 'processed_crops')
KP_LBL_DIR= os.path.join(ROOT, 'data/keypoint_labels')
SEG_W     = os.path.join(ROOT, 'best_pigweed_145.pt')

DET_ROOT  = os.path.join(ROOT, 'datasets/oxnard_det')
KP_ROOT   = os.path.join(ROOT, 'datasets/oxnard_kp')

SEG_IMGSZ = 1280
SEG_CONF  = 0.45
PADDING   = 20
MIN_VISIBLE_PX = 400
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
SEED = 42


def reset(p):
    if os.path.islink(p): os.unlink(p)
    elif os.path.isdir(p): shutil.rmtree(p)


def derive_bboxes_and_stems(seg, fname):
    """Return list of (bbox_xyxy, stem_xy_full_image) for plants in this left
    image that have a labeled stem in keypoint_labels/."""
    stem = os.path.splitext(fname)[0]
    img = cv2.imread(os.path.join(LEFT_DIR, fname))
    if img is None: return None, None
    H, W = img.shape[:2]

    res = seg.predict(source=img, imgsz=SEG_IMGSZ, conf=SEG_CONF, verbose=False)[0]
    if res.masks is None or len(res.masks) == 0: return img, []
    boxes = res.boxes.xyxy.cpu().numpy()
    masks = res.masks.data.cpu().numpy()

    out = []
    for i, (box, m) in enumerate(zip(boxes, masks)):
        mr = cv2.resize(m, (W, H))
        binary = (mr > 0.5).astype(np.uint8) * 255
        masked = cv2.bitwise_and(img, img, mask=binary)
        x1, y1, x2, y2 = map(int, box)
        x1p = max(0, x1 - PADDING); y1p = max(0, y1 - PADDING)
        x2p = min(W, x2 + PADDING); y2p = min(H, y2 + PADDING)
        crop = masked[y1p:y2p, x1p:x2p]
        if crop.size == 0: continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if cv2.countNonZero(gray) <= MIN_VISIBLE_PX: continue
        # crop index i must match plant_<stem>_<i>
        lbl = os.path.join(KP_LBL_DIR, f"plant_{stem}_{i}.txt")
        if not os.path.exists(lbl): continue
        with open(lbl) as f:
            sx_crop, sy_crop = (float(v) for v in f.read().strip().split())
        sx, sy = sx_crop + x1p, sy_crop + y1p
        # bbox here is ORIGINAL detection bbox (no padding) — used as YOLO label
        out.append(((float(x1), float(y1), float(x2), float(y2)), (sx, sy),
                    (x1p, y1p, x2p, y2p)))
    return img, out


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading seg model: {SEG_W}")
    seg = YOLO(SEG_W); seg.to(str(dev))

    # Find all left images that have at least one labeled crop
    labeled = {f for f in os.listdir(KP_LBL_DIR) if f.endswith('.txt')}
    left_files = sorted(f for f in os.listdir(LEFT_DIR) if f.lower().endswith(('.jpg', '.png')))
    eligible = [f for f in left_files
                if any(lbl.startswith(f"plant_{os.path.splitext(f)[0]}_") for lbl in labeled)]
    print(f"Left images with labeled crops: {len(eligible)}  (of {len(left_files)})")

    # Build per-image (bbox, stem) lists
    print("Recovering bboxes by re-running seg model...")
    img_records = {}    # fname -> (img_hw, list of (bbox_xyxy_orig, stem_xy_full, padded_xyxy))
    n_objects = 0
    for idx, fname in enumerate(eligible):
        img, recs = derive_bboxes_and_stems(seg, fname)
        if img is None: continue
        H, W = img.shape[:2]
        if recs:
            img_records[fname] = ((H, W), recs)
            n_objects += len(recs)
        if (idx+1) % 50 == 0:
            print(f"  {idx+1}/{len(eligible)}  matched objects so far: {n_objects}")
    print(f"Total: {len(img_records)} parent images, {n_objects} (bbox, stem) pairs\n")

    # Split parents
    parents = sorted(img_records.keys())
    train, temp = train_test_split(parents, train_size=TRAIN_FRAC, random_state=SEED)
    val, test = train_test_split(temp, train_size=VAL_FRAC/(1-TRAIN_FRAC), random_state=SEED)
    splits = {'train': train, 'val': val, 'test': test}
    for k, v in splits.items():
        n = sum(len(img_records[f][1]) for f in v)
        print(f"{k:>5}: {len(v):3} images, {n:4} objects")

    # Build phase-1 (detection) dataset
    print("\nBuilding detection dataset...")
    for sp, parents in splits.items():
        img_dst = os.path.join(DET_ROOT, sp, 'images')
        lbl_dst = os.path.join(DET_ROOT, sp, 'labels')
        for p in (img_dst, lbl_dst): reset(p)
        os.makedirs(img_dst); os.makedirs(lbl_dst)
        for fname in parents:
            (H, W), recs = img_records[fname]
            os.symlink(os.path.join(LEFT_DIR, fname),
                       os.path.join(img_dst, fname))
            stem = os.path.splitext(fname)[0]
            with open(os.path.join(lbl_dst, stem + '.txt'), 'w') as f:
                for (x1, y1, x2, y2), _, _ in recs:
                    cx, cy = (x1+x2)/2/W, (y1+y2)/2/H
                    bw, bh = (x2-x1)/W, (y2-y1)/H
                    f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    with open(os.path.join(DET_ROOT, 'oxnard_det.yaml'), 'w') as f:
        f.write(
            f"path: {DET_ROOT}\n"
            f"train: train/images\nval: val/images\ntest: test/images\n\n"
            f"nc: 1\nnames:\n  0: pigweed\n"
        )
    print(f"Detection dataset: {DET_ROOT}/oxnard_det.yaml")

    # Build phase-2 (keypoint) dataset (symlink crops + their stem labels by split)
    print("\nBuilding keypoint dataset (per-crop)...")
    for sp, parents in splits.items():
        img_dst = os.path.join(KP_ROOT, sp, 'images')
        lbl_dst = os.path.join(KP_ROOT, sp, 'labels')
        for p in (img_dst, lbl_dst): reset(p)
        os.makedirs(img_dst); os.makedirs(lbl_dst)
        n = 0
        for fname in parents:
            stem = os.path.splitext(fname)[0]
            (_, _), recs = img_records[fname]
            for i, (_, _, _) in enumerate(recs):
                # idx i in recs is enumeration of MATCHED records, but we need
                # the original detection index from the filename — recs already
                # filtered to matched ones, so we re-use that order via the
                # presence of plant_<stem>_<i>.png on disk
                pass
        # Cleaner: just iterate processed_crops by parent
        for fname in parents:
            stem = os.path.splitext(fname)[0]
            for crop_name in os.listdir(CROPS_DIR):
                if not crop_name.startswith(f"plant_{stem}_"): continue
                lbl_name = crop_name.replace('.png', '.txt')
                if not os.path.exists(os.path.join(KP_LBL_DIR, lbl_name)): continue
                os.symlink(os.path.join(CROPS_DIR, crop_name),
                           os.path.join(img_dst, crop_name))
                os.symlink(os.path.join(KP_LBL_DIR, lbl_name),
                           os.path.join(lbl_dst, lbl_name))
                n += 1
        print(f"  {sp:>5}: {n:4} crops")

    # Save metadata so eval can use the same crop offsets
    print("\nSaving crop-offset metadata for eval...")
    import csv
    meta_path = os.path.join(DET_ROOT, 'crop_offsets.csv')
    with open(meta_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['parent_image', 'split', 'crop_index', 'x1p', 'y1p', 'x2p', 'y2p',
                    'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
                    'stem_x_full', 'stem_y_full'])
        for sp, parents in splits.items():
            for fname in parents:
                _, recs = img_records[fname]
                for i, (bbox, stem_xy, padded) in enumerate(recs):
                    w.writerow([fname, sp, i, *padded, *bbox, *stem_xy])
    print(f"  -> {meta_path}")

    print("\nNext steps (run in tmux):")
    print(f"  # Phase 1 — detection")
    print(f"  yolo task=detect mode=train data={DET_ROOT}/oxnard_det.yaml \\")
    print(f"       model=yolov8m.pt imgsz=1280 epochs=100 batch=8 patience=25 \\")
    print(f"       project={ROOT}/results name=oxnard_det")
    print()
    print(f"  # Phase 2 — keypoint")
    print(f"  python scripts/oxnard_train_kp.py")
    print()
    print(f"  # Then evaluation:")
    print(f"  python scripts/oxnard_eval_v2.py")


if __name__ == '__main__':
    main()
