"""
Build the UNMASKED-crops counterpart to datasets/oxnard_kp/.

Same train/val/test split, same bboxes, same stems — but the crops are RAW
bbox content (no background blacking), matching what a YOLO-DETECT-only Phase
1 would feed at inference time.

For each (parent_image, bbox, stem) record in datasets/oxnard_det/crop_offsets.csv:
  - Read the parent image
  - Extract the padded crop directly (NO masking)
  - Stem coords are already in full-image space; convert to crop-local

Outputs:
  datasets/oxnard_kp_raw/{train,val,test}/
    images/  -- per-plant raw crops
    labels/  -- "kx ky" (stem in crop coords)
"""
import os, csv, shutil, cv2

ROOT = '/home/jaehwan/Desktop/laser-weeding'
META = os.path.join(ROOT, 'datasets/oxnard_det/crop_offsets.csv')
LEFT_DIR = os.path.join(ROOT, 'left')
DST = os.path.join(ROOT, 'datasets/oxnard_kp_raw')
PADDING = 20


def main():
    for sp in ('train', 'val', 'test'):
        for sub in ('images', 'labels'):
            p = os.path.join(DST, sp, sub)
            if os.path.isdir(p): shutil.rmtree(p)
            os.makedirs(p)

    counts = {'train': 0, 'val': 0, 'test': 0}
    last_img = None
    img = None
    with open(META) as f:
        for r in csv.DictReader(f):
            sp = r['split']
            fname = r['parent_image']
            if fname != last_img:
                img = cv2.imread(os.path.join(LEFT_DIR, fname))
                last_img = fname
            if img is None: continue
            H, W = img.shape[:2]
            x1p, y1p = int(float(r['x1p'])), int(float(r['y1p']))
            x2p, y2p = int(float(r['x2p'])), int(float(r['y2p']))
            crop = img[y1p:y2p, x1p:x2p]
            if crop.size == 0: continue
            sx, sy = float(r['stem_x_full']) - x1p, float(r['stem_y_full']) - y1p
            stem_img = os.path.splitext(fname)[0]
            i = r['crop_index']
            crop_name = f"plant_{stem_img}_{i}.png"
            cv2.imwrite(os.path.join(DST, sp, 'images', crop_name), crop)
            with open(os.path.join(DST, sp, 'labels', crop_name.replace('.png', '.txt')), 'w') as fout:
                fout.write(f"{sx:.4f} {sy:.4f}\n")
            counts[sp] += 1
    for k, v in counts.items():
        print(f"  {k}: {v} crops")
    print(f"\nUnmasked crops written to {DST}/")


if __name__ == '__main__':
    main()
