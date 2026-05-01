"""
Unified comparison eval. Runs every (Phase 1, Phase 2) combo on the SAME test
split, reports Mean Dist + Median Dist + inference latency.

5 cells:
  A. YOLO-detect      + direct-regression   (raw crops)
  B. YOLO-detect      + heatmap-regression  (raw crops)
  C. YOLO-segment     + direct-regression   (masked crops)
  D. YOLO-segment     + heatmap-regression  (masked crops)  ← current best
  E. YOLO-pose joint  (single-stage)

Outputs a JSON per combo and a summary table.
"""
import os, csv, json, time, cv2, numpy as np, torch, torch.nn as nn
from torchvision import models
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import functional as TF
from ultralytics import YOLO

ROOT = '/home/jaehwan/Desktop/laser-weeding'
META = os.path.join(ROOT, 'datasets/oxnard_det/crop_offsets.csv')
LEFT_DIR = os.path.join(ROOT, 'left')
OUT_DIR = os.path.join(ROOT, 'results/comparison')
os.makedirs(OUT_DIR, exist_ok=True)

PHASE1_DET  = os.path.join(ROOT, 'results/oxnard_det/weights/best.pt')
PHASE1_SEG  = os.path.join(ROOT, 'best_pigweed_145.pt')
PHASE1_POSE = os.path.join(ROOT, 'results/oxnard_pose/weights/best.pt')

KP = {
    'direct_raw':    os.path.join(ROOT, 'models/keypoint_oxnard_direct_raw.pt'),
    'direct_masked': os.path.join(ROOT, 'models/keypoint_oxnard_direct_masked.pt'),
    'heatmap_raw':   os.path.join(ROOT, 'models/keypoint_oxnard_heatmap_raw.pt'),
    'heatmap_masked':os.path.join(ROOT, 'models/keypoint_oxnard_heatmap_masked.pt'),
}

DET_CONF, DET_IMGSZ = 0.40, 1280
SEG_CONF, SEG_IMGSZ = 0.45, 1280
POSE_CONF = 0.40
PADDING, MIN_VISIBLE_PX, IOU_MATCH_MIN = 20, 400, 0.30
TRAIN_SIZE = 224
IMG_W, IMG_H = 1280, 720
DIAG = float(np.sqrt(IMG_W**2 + IMG_H**2))
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


SOFT_RES = 56


class HeatmapHead(nn.Module):
    """Soft-argmax + U-Net skip connections — matches train_keypoint_v2.py."""
    def __init__(self, soft_res=SOFT_RES, train_size=TRAIN_SIZE):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights=None).features
        self.enc_stage1 = backbone[:4]
        self.enc_stage2 = backbone[4:9]
        self.enc_stage3 = backbone[9:]
        self.up1 = nn.ConvTranspose2d(576, 256, 4, 2, 1)
        self.dec1 = nn.Sequential(nn.Conv2d(256 + 48, 256, 3, padding=1),
                                  nn.BatchNorm2d(256), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dec2 = nn.Sequential(nn.Conv2d(128 + 24, 128, 3, padding=1),
                                  nn.BatchNorm2d(128), nn.ReLU())
        self.up3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec3 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                  nn.BatchNorm2d(64), nn.ReLU(),
                                  nn.Conv2d(64, 1, 1))
        ys = torch.linspace(0, train_size - 1, soft_res)
        xs = torch.linspace(0, train_size - 1, soft_res)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        self.register_buffer('grid_x', gx.clone())
        self.register_buffer('grid_y', gy.clone())

    def forward(self, x):
        s1 = self.enc_stage1(x)
        s2 = self.enc_stage2(s1)
        bot = self.enc_stage3(s2)
        d1 = self.dec1(torch.cat([self.up1(bot), s2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), s1], dim=1))
        score = self.dec3(self.up3(d2))
        B, _, H, W = score.shape
        probs = torch.softmax(score.view(B, -1), dim=1).view(B, H, W)
        x_pred = (probs * self.grid_x).sum(dim=(1, 2))
        y_pred = (probs * self.grid_y).sum(dim=(1, 2))
        return torch.stack([x_pred, y_pred], dim=1)


class DirectHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.mobilenet_v3_small(weights=None).features
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(576, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 2), nn.Tanh())
    def forward(self, x): return self.head(self.encoder(x))


def letterbox(crop, target=TRAIN_SIZE):
    h, w = crop.shape[:2]; s = target / max(w, h)
    nw, nh = max(1, int(w*s)), max(1, int(h*s)); dx, dy = (target-nw)//2, (target-nh)//2
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    r = cv2.resize(rgb, (nw, nh))
    canvas = np.zeros((target, target, 3), dtype=np.uint8)
    canvas[dy:dy+nh, dx:dx+nw] = r
    return canvas, s, dx, dy


@torch.no_grad()
def stem_in_crop(kp, dev, crop, arch):
    if crop.size == 0 or min(crop.shape[:2]) < 4: return None
    canvas, s, dx, dy = letterbox(crop)
    t = NORMALIZE(TF.to_image(canvas).float()/255.).unsqueeze(0).to(dev)
    if arch == 'heatmap':
        # soft-argmax model returns (x, y) in pixel space
        out = kp(t).cpu().numpy()[0]
        px, py = float(out[0]), float(out[1])
    else:
        out = kp(t).cpu().numpy()[0]
        px = (out[0] + 1) / 2 * (TRAIN_SIZE - 1)
        py = (out[1] + 1) / 2 * (TRAIN_SIZE - 1)
    return ((px - dx) / s, (py - dy) / s)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0


def load_kp(path, arch, dev):
    m = (HeatmapHead() if arch == 'heatmap' else DirectHead()).to(dev)
    state = torch.load(path, map_location=dev)
    if isinstance(state, dict) and 'state_dict' in state: state = state['state_dict']
    m.load_state_dict(state); m.eval()
    return m


def load_test_gt():
    gt_by_img = {}
    with open(META) as f:
        for r in csv.DictReader(f):
            if r['split'] != 'test': continue
            gt_by_img.setdefault(r['parent_image'], []).append({
                'bbox': (float(r['bbox_x1']), float(r['bbox_y1']),
                         float(r['bbox_x2']), float(r['bbox_y2'])),
                'stem': (float(r['stem_x_full']), float(r['stem_y_full']))})
    return gt_by_img


def score(all_pred, all_gt):
    pred = np.array(all_pred); gt = np.array(all_gt)
    px = np.sqrt(((pred - gt) ** 2).sum(axis=1))
    norm = px / DIAG
    return {
        'n_kept': int(len(px)),
        'mean_px': float(px.mean()),
        'median_px': float(np.median(px)),
        'mean_dist': float(norm.mean() * 1000),
        'median_dist': float(np.median(norm) * 1000),
        'raw_mse': float((norm**2).mean()),
    }


def run_2stage(name, phase1_path, phase1_kind, phase2_path, phase2_arch, mask_crops):
    print(f"\n[{name}]  Phase1={os.path.basename(phase1_path)} ({phase1_kind})  "
          f"Phase2={os.path.basename(phase2_path)} ({phase2_arch})  mask_crops={mask_crops}")
    dev = torch.device('cuda')
    yolo = YOLO(phase1_path)
    kp = load_kp(phase2_path, phase2_arch, dev)
    gt_by_img = load_test_gt()
    all_pred, all_gt = [], []
    n_total = 0
    t0 = time.time()
    p1_time, p2_time = 0.0, 0.0
    for fname, gt_list in sorted(gt_by_img.items()):
        img = cv2.imread(os.path.join(LEFT_DIR, fname))
        if img is None: continue
        H, W = img.shape[:2]

        torch.cuda.synchronize(); t1 = time.time()
        if phase1_kind == 'seg':
            res = yolo.predict(source=img, imgsz=SEG_IMGSZ, conf=SEG_CONF, verbose=False)[0]
        else:
            res = yolo.predict(source=img, imgsz=DET_IMGSZ, conf=DET_CONF, verbose=False)[0]
        torch.cuda.synchronize(); p1_time += time.time() - t1
        if res.boxes is None or len(res.boxes) == 0: continue
        boxes = res.boxes.xyxy.cpu().numpy()
        masks = res.masks.data.cpu().numpy() if (mask_crops and res.masks is not None) else None

        for i, box in enumerate(boxes):
            n_total += 1
            x1, y1, x2, y2 = map(int, box)
            if mask_crops and masks is not None:
                m = masks[i]
                mr = cv2.resize(m, (W, H))
                binary = (mr > 0.5).astype(np.uint8) * 255
                src = cv2.bitwise_and(img, img, mask=binary)
            else:
                src = img
            x1p = max(0, x1 - PADDING); y1p = max(0, y1 - PADDING)
            x2p = min(W, x2 + PADDING); y2p = min(H, y2 + PADDING)
            crop = src[y1p:y2p, x1p:x2p]
            if mask_crops:
                if crop.size == 0: continue
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                if cv2.countNonZero(gray) <= MIN_VISIBLE_PX: continue
            pbox = (float(x1), float(y1), float(x2), float(y2))
            best_j, best_iou = -1, 0
            for j, g in enumerate(gt_list):
                v = iou(pbox, g['bbox'])
                if v > best_iou: best_j, best_iou = j, v
            if best_iou < IOU_MATCH_MIN: continue

            torch.cuda.synchronize(); t2 = time.time()
            stem_local = stem_in_crop(kp, dev, crop, phase2_arch)
            torch.cuda.synchronize(); p2_time += time.time() - t2
            if stem_local is None: continue
            all_pred.append((stem_local[0] + x1p, stem_local[1] + y1p))
            all_gt.append(gt_list[best_j]['stem'])

    elapsed = time.time() - t0
    n_imgs = len(gt_by_img)
    s = score(all_pred, all_gt)
    s.update({
        'name': name, 'n_total_phase1': n_total, 'n_imgs': n_imgs,
        'phase1_kind': phase1_kind, 'phase2_arch': phase2_arch,
        'mask_crops': mask_crops, 'phase1_path': phase1_path, 'phase2_path': phase2_path,
        'total_eval_seconds': elapsed,
        'phase1_seconds': p1_time,
        'phase2_seconds': p2_time,
        'ms_per_image': elapsed / n_imgs * 1000,
    })
    return s


def run_pose():
    print(f"\n[pose]  Phase1={os.path.basename(PHASE1_POSE)} (joint)")
    dev = torch.device('cuda')
    yolo = YOLO(PHASE1_POSE)
    gt_by_img = load_test_gt()
    all_pred, all_gt = [], []
    n_total = 0
    t0 = time.time()
    for fname, gt_list in sorted(gt_by_img.items()):
        img = cv2.imread(os.path.join(LEFT_DIR, fname))
        if img is None: continue
        res = yolo.predict(source=img, imgsz=DET_IMGSZ, conf=POSE_CONF, verbose=False)[0]
        if res.keypoints is None or len(res.keypoints) == 0: continue
        kpts = res.keypoints.xy.cpu().numpy()
        boxes = res.boxes.xyxy.cpu().numpy()
        for i in range(len(kpts)):
            n_total += 1
            pbox = tuple(boxes[i])
            best_j, best_iou = -1, 0
            for j, g in enumerate(gt_list):
                v = iou(pbox, g['bbox'])
                if v > best_iou: best_j, best_iou = j, v
            if best_iou < IOU_MATCH_MIN: continue
            all_pred.append(kpts[i, 0]); all_gt.append(gt_list[best_j]['stem'])
    elapsed = time.time() - t0
    n_imgs = len(gt_by_img)
    s = score(all_pred, all_gt)
    s.update({
        'name': 'pose_joint', 'n_total_phase1': n_total, 'n_imgs': n_imgs,
        'phase1_kind': 'pose', 'phase2_arch': 'joint', 'mask_crops': False,
        'phase1_path': PHASE1_POSE, 'phase2_path': PHASE1_POSE,
        'total_eval_seconds': elapsed,
        'phase1_seconds': elapsed,
        'phase2_seconds': 0.0,
        'ms_per_image': elapsed / n_imgs * 1000,
    })
    return s


def main():
    runs = [
        ('det_direct',   PHASE1_DET, 'detect',  KP['direct_raw'],     'direct',  False),
        ('det_heatmap',  PHASE1_DET, 'detect',  KP['heatmap_raw'],    'heatmap', False),
        ('seg_direct',   PHASE1_SEG, 'seg',     KP['direct_masked'],  'direct',  True),
        ('seg_heatmap',  PHASE1_SEG, 'seg',     KP['heatmap_masked'], 'heatmap', True),
    ]
    results = []
    for name, p1, kind, p2, arch, mask in runs:
        if not os.path.exists(p2):
            print(f"SKIP {name}: missing {p2}")
            continue
        results.append(run_2stage(name, p1, kind, p2, arch, mask))
    results.append(run_pose())

    out = os.path.join(OUT_DIR, 'comparison.json')
    with open(out, 'w') as f: json.dump(results, f, indent=2)
    print(f"\nWrote {out}\n")

    # Summary table
    hdr = f"{'config':<14} {'mean_dist':>10} {'median_dist':>12} {'mean_px':>9} {'median_px':>10} {'kept':>5} {'ms/img':>8}"
    print(hdr); print('-' * len(hdr))
    for r in results:
        print(f"{r['name']:<14} {r['mean_dist']:>10.4f} {r['median_dist']:>12.4f} "
              f"{r['mean_px']:>9.2f} {r['median_px']:>10.2f} {r['n_kept']:>5d} {r['ms_per_image']:>8.1f}")

    # Pull training stats from each Phase-2's training json (if present)
    train_stats = {}
    import glob
    for fp in glob.glob(os.path.join(ROOT, 'models/keypoint_oxnard_*.train.json')):
        d = json.load(open(fp))
        train_stats[f"{d['arch']}_{d['crops']}"] = d

    plot_results(results, train_stats)


def plot_results(results, train_stats):
    """Render the comparison as a PNG figure (3 panels) for the paper."""
    import matplotlib.pyplot as plt

    names = [r['name'] for r in results]
    mean_d = [r['mean_dist'] for r in results]
    med_d  = [r['median_dist'] for r in results]
    speed  = [r['ms_per_image'] for r in results]
    kept   = [r['n_kept'] for r in results]

    color_map = {
        'det_direct':  '#3b8ed0',
        'det_heatmap': '#1f6394',
        'seg_direct':  '#48a472',
        'seg_heatmap': '#246b40',
        'pose_joint':  '#d97742',
    }
    colors = [color_map.get(n, '#888') for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: accuracy (mean and median Dist)
    ax = axes[0]
    x = np.arange(len(names))
    w = 0.35
    bars1 = ax.bar(x - w/2, mean_d, w, label='Mean Dist', color=colors, edgecolor='black')
    bars2 = ax.bar(x + w/2, med_d,  w, label='Median Dist', color=colors, alpha=0.55, edgecolor='black')
    for bx, v in zip(x - w/2, mean_d): ax.text(bx, v + 0.08, f"{v:.2f}", ha='center', fontsize=9)
    for bx, v in zip(x + w/2, med_d):  ax.text(bx, v + 0.08, f"{v:.2f}", ha='center', fontsize=9, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha='right')
    ax.set_ylabel('Dist (× 1000, lower is better)')
    ax.set_title('Accuracy: Mean / Median Distance')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Panel 2: inference speed
    ax = axes[1]
    bars = ax.bar(x, speed, color=colors, edgecolor='black')
    for bx, v in zip(x, speed): ax.text(bx, v + 0.5, f"{v:.1f} ms", ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha='right')
    ax.set_ylabel('ms / image (lower is better)')
    ax.set_title('Inference latency (per test image)')
    ax.grid(axis='y', alpha=0.3)

    # Panel 3: accuracy-vs-speed scatter
    ax = axes[2]
    for n, m, s, c, k in zip(names, mean_d, speed, colors, kept):
        ax.scatter(s, m, color=c, s=200, edgecolor='black', zorder=3, label=f"{n}  (n={k})")
        ax.annotate(n, (s, m), xytext=(6, 6), textcoords='offset points', fontsize=9)
    ax.set_xlabel('Inference latency (ms / image)')
    ax.set_ylabel('Mean Dist')
    ax.set_title('Accuracy ↔ speed tradeoff')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')

    plt.suptitle('2-stage vs joint pose: 5-cell ablation on held-out test split',
                 fontsize=13, weight='bold')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'comparison_chart.png')
    plt.savefig(out, dpi=140, bbox_inches='tight'); plt.close()
    print(f"Saved {out}")

    # Second figure: training comparison (val px err, training time per Phase-2 variant)
    if train_stats:
        keys = sorted(train_stats.keys())
        val_err = [train_stats[k]['best_val_px_err'] for k in keys]
        epochs  = [train_stats[k]['best_epoch'] for k in keys]
        train_t = [train_stats[k]['train_seconds']/60 for k in keys]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        cmap2 = ['#3b8ed0', '#1f6394', '#48a472', '#246b40']
        x = np.arange(len(keys))
        for ax, vals, ttl, ylab in [
            (axes[0], val_err, 'Phase-2 best val px error (224 space)', 'pixels'),
            (axes[1], epochs,  'Phase-2 best epoch',                    'epochs'),
            (axes[2], train_t, 'Phase-2 training time',                 'minutes')]:
            ax.bar(x, vals, color=cmap2, edgecolor='black')
            for bx, v in zip(x, vals):
                ax.text(bx, v + max(vals)*0.02, f"{v:.2f}" if isinstance(v, float) else str(v),
                        ha='center', fontsize=9)
            ax.set_xticks(x); ax.set_xticklabels(keys, rotation=15, ha='right')
            ax.set_title(ttl); ax.set_ylabel(ylab); ax.grid(axis='y', alpha=0.3)
        plt.suptitle('Phase-2 keypoint training: 4 variants (architecture × crop style)',
                     fontsize=13, weight='bold')
        plt.tight_layout()
        out2 = os.path.join(OUT_DIR, 'training_chart.png')
        plt.savefig(out2, dpi=140, bbox_inches='tight'); plt.close()
        print(f"Saved {out2}")


if __name__ == '__main__':
    main()
