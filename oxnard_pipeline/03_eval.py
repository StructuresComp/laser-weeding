"""
Fixed 2-stage eval on Oxnard test split.

Difference from oxnard_eval_v2.py: at inference we apply the mask exactly the
way segment_processing.py did when it built processed_crops/. This matches the
distribution Phase 2 was trained on (background blacked out + padding=20).

Phase 1: best_pigweed_145.pt (gives bbox AND mask)
Phase 2: models/keypoint_oxnard.pt   (trained on TRAIN split crops only — no leakage)

Optional: set KP_MODEL_PATH env to evaluate a different keypoint model
(e.g. /home/jaehwan/Desktop/laser-weeding/new_best_targeting_v3.pth, but note
that one was trained on ALL crops including test → LEAKED).
"""
import os, csv, cv2, numpy as np, torch, torch.nn as nn
from torchvision import models
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import functional as TF
from ultralytics import YOLO

ROOT = '/home/jaehwan/Desktop/laser-weeding'
SEG_W = os.path.join(ROOT, 'best_pigweed_145.pt')
KP_W  = os.environ.get('KP_MODEL_PATH', os.path.join(ROOT, 'models/keypoint_oxnard.pt'))
META  = os.path.join(ROOT, 'datasets/oxnard_det/crop_offsets.csv')
LEFT_DIR = os.path.join(ROOT, 'left')
SEG_IMGSZ, SEG_CONF = 1280, 0.45
PADDING = 20
MIN_VISIBLE_PX = 400
TRAIN_SIZE = 224
IOU_MATCH_MIN = 0.30
IMG_W, IMG_H = 1280, 720

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class MeristemPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.mobilenet_v3_small(weights=None).features
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(576, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 1, 3, padding=1), nn.Sigmoid())
    def forward(self, x): return self.decoder(self.encoder(x))


def letterbox(crop, target=TRAIN_SIZE):
    h, w = crop.shape[:2]
    s = target / max(w, h)
    nw, nh = max(1, int(w*s)), max(1, int(h*s))
    dx, dy = (target-nw)//2, (target-nh)//2
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    r = cv2.resize(rgb, (nw, nh))
    canvas = np.zeros((target, target, 3), dtype=np.uint8)
    canvas[dy:dy+nh, dx:dx+nw] = r
    return canvas, s, dx, dy


@torch.no_grad()
def stem_in_crop(kp, dev, crop):
    if crop.size == 0 or min(crop.shape[:2]) < 4: return None
    canvas, s, dx, dy = letterbox(crop)
    t = NORMALIZE(TF.to_image(canvas).float()/255.).unsqueeze(0).to(dev)
    heat = kp(t)[0, 0].cpu().numpy()
    py, px = np.unravel_index(np.argmax(heat), heat.shape)
    return ((px - dx) / s, (py - dy) / s)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Phase 1 (seg + bbox): {SEG_W}")
    seg = YOLO(SEG_W)
    print(f"Phase 2 (keypoint):   {KP_W}")
    kp = MeristemPredictor().to(dev)
    state = torch.load(KP_W, map_location=dev)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    kp.load_state_dict(state); kp.eval()

    # Load GT bbox+stem for test split
    gt_by_img = {}
    with open(META) as f:
        for r in csv.DictReader(f):
            if r['split'] != 'test': continue
            gt_by_img.setdefault(r['parent_image'], []).append({
                'bbox': (float(r['bbox_x1']), float(r['bbox_y1']),
                         float(r['bbox_x2']), float(r['bbox_y2'])),
                'stem': (float(r['stem_x_full']), float(r['stem_y_full']))})
    print(f"Test images: {len(gt_by_img)}, GT stems: {sum(len(v) for v in gt_by_img.values())}")

    all_pred, all_gt = [], []
    n_total, n_kept = 0, 0

    for fname, gt_list in sorted(gt_by_img.items()):
        ip = os.path.join(LEFT_DIR, fname)
        img = cv2.imread(ip)
        if img is None: continue
        H, W = img.shape[:2]

        res = seg.predict(source=img, imgsz=SEG_IMGSZ, conf=SEG_CONF, verbose=False)[0]
        if res.masks is None or len(res.masks) == 0: continue
        boxes = res.boxes.xyxy.cpu().numpy()
        masks = res.masks.data.cpu().numpy()

        for i, (box, m) in enumerate(zip(boxes, masks)):
            n_total += 1
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

            # IoU match against GT bboxes — drop predictions that aren't real
            pbox = (float(x1), float(y1), float(x2), float(y2))
            best_j, best_iou = -1, 0
            for j, g in enumerate(gt_list):
                v = iou(pbox, g['bbox'])
                if v > best_iou: best_j, best_iou = j, v
            if best_iou < IOU_MATCH_MIN: continue

            # Run phase 2 on the MASKED crop (matches train distribution)
            stem_local = stem_in_crop(kp, dev, crop)
            if stem_local is None: continue
            pred_xy = (stem_local[0] + x1p, stem_local[1] + y1p)
            all_pred.append(pred_xy); all_gt.append(gt_list[best_j]['stem'])
            n_kept += 1

    if not all_pred:
        print("No predictions matched."); return
    pred = np.array(all_pred); gt = np.array(all_gt)
    px_err = np.sqrt(((pred - gt) ** 2).sum(axis=1))
    diag = float(np.sqrt(IMG_W**2 + IMG_H**2))
    norm = px_err / diag
    print("\n" + "=" * 70)
    print("OXNARD 2-STAGE EVAL (masked crops, matches Phase-2 train distribution)")
    print("=" * 70)
    print(f"Test images:      {len(gt_by_img)}")
    print(f"Total Phase-1:    {n_total}")
    print(f"Kept (IoU≥{IOU_MATCH_MIN}): {n_kept}")
    print(f"GT stems:         {sum(len(v) for v in gt_by_img.values())}")
    print()
    print(f"Mean px error:    {px_err.mean():.2f}")
    print(f"Median px error:  {np.median(px_err):.2f}")
    print(f"Mean Dist:   {norm.mean()*1000:.4f}")
    print(f"Median Dist: {np.median(norm)*1000:.4f}")
    print(f"Raw MSE:     {(norm**2).mean():.6f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
