"""
Build a YOLO-pose dataset for Oxnard left camera using the (bbox, stem)
pairs already recovered by oxnard_prep.py (crop_offsets.csv).

Output:
  datasets/oxnard_pose/{train,val,test}/{images, labels}/
  datasets/oxnard_pose/oxnard_pose.yaml

Each label line: cls cx cy w h kpt_x kpt_y vis  (all normalized to [0,1])
"""
import os, csv, shutil

ROOT = '/home/jaehwan/Desktop/laser-weeding'
LEFT_DIR = os.path.join(ROOT, 'left')
META = os.path.join(ROOT, 'datasets/oxnard_det/crop_offsets.csv')
POSE_ROOT = os.path.join(ROOT, 'datasets/oxnard_pose')
IMG_W, IMG_H = 1280, 720


def reset(p):
    if os.path.islink(p): os.unlink(p)
    elif os.path.isdir(p): shutil.rmtree(p)


def main():
    # Read meta
    rows_by_split = {'train': {}, 'val': {}, 'test': {}}
    with open(META) as f:
        for r in csv.DictReader(f):
            sp = r['split']
            rows_by_split[sp].setdefault(r['parent_image'], []).append(r)

    for split, files in rows_by_split.items():
        img_dst = os.path.join(POSE_ROOT, split, 'images')
        lbl_dst = os.path.join(POSE_ROOT, split, 'labels')
        for p in (img_dst, lbl_dst): reset(p)
        os.makedirs(img_dst); os.makedirs(lbl_dst)
        n_obj = 0
        for fname, recs in files.items():
            os.symlink(os.path.join(LEFT_DIR, fname),
                       os.path.join(img_dst, fname))
            stem = os.path.splitext(fname)[0]
            with open(os.path.join(lbl_dst, stem + '.txt'), 'w') as f:
                for r in recs:
                    x1 = float(r['bbox_x1']); y1 = float(r['bbox_y1'])
                    x2 = float(r['bbox_x2']); y2 = float(r['bbox_y2'])
                    sx = float(r['stem_x_full']); sy = float(r['stem_y_full'])
                    cx = (x1+x2)/2/IMG_W; cy = (y1+y2)/2/IMG_H
                    bw = (x2-x1)/IMG_W;   bh = (y2-y1)/IMG_H
                    kx = sx/IMG_W;        ky = sy/IMG_H
                    f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} "
                            f"{kx:.6f} {ky:.6f} 2\n")
                    n_obj += 1
        print(f"  {split:>5}: {len(files):>3} images, {n_obj:>4} objects")

    yaml_path = os.path.join(POSE_ROOT, 'oxnard_pose.yaml')
    with open(yaml_path, 'w') as f:
        f.write(
            f"path: {POSE_ROOT}\n"
            f"train: train/images\nval: val/images\ntest: test/images\n\n"
            f"nc: 1\nnames:\n  0: pigweed\n\n"
            f"kpt_shape: [1, 3]\nflip_idx: [0]\n"
        )
    print(f"\nWrote {yaml_path}")
    print("\nTraining command (run in tmux):")
    print(f"  yolo task=pose mode=train \\")
    print(f"       data={yaml_path} \\")
    print(f"       model=yolov8m-pose.pt \\")
    print(f"       imgsz=1280 epochs=120 batch=8 patience=30 \\")
    print(f"       project={ROOT}/results name=oxnard_pose")


if __name__ == '__main__':
    main()
