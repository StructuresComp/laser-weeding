"""
Unified Phase-2 keypoint trainer.

Supports two architectures and two crop styles via env vars:
  KP_ARCH=heatmap  (default)  — soft-argmax readout + U-Net skip connections,
                                 SmoothL1 loss on (x, y) coordinates (architecture
                                 from scripts/softargmax_skip_sweep.py)
  KP_ARCH=direct              — global-pooled MLP -> (x, y), MSE on normalized coords
  KP_CROPS=masked  (default)  — datasets/oxnard_kp/      (background blacked out)
  KP_CROPS=raw                — datasets/oxnard_kp_raw/  (raw bbox content)

Saves to: models/keypoint_oxnard_<ARCH>_<CROPS>.pt
Also writes models/keypoint_oxnard_<ARCH>_<CROPS>.train.json with timing + best val px err.
"""
import os, json, time, cv2, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import functional as TF

ROOT = '/home/jaehwan/Desktop/laser-weeding'
ARCH = os.environ.get('KP_ARCH', 'heatmap')
CROP_STYLE = os.environ.get('KP_CROPS', 'masked')
KP_ROOT = os.path.join(ROOT, 'datasets',
                       'oxnard_kp' if CROP_STYLE == 'masked' else 'oxnard_kp_raw')
SAVE_TO = os.path.join(ROOT, f'models/keypoint_oxnard_{ARCH}_{CROP_STYLE}.pt')
LOG_TO  = os.path.join(ROOT, f'models/keypoint_oxnard_{ARCH}_{CROP_STYLE}.train.json')

TRAIN_SIZE = 224
SOFT_RES = 56
BATCH = 16
LR = 1e-4
MAX_EPOCHS = 150
PATIENCE = 25


def letterbox_params(w, h, target):
    s = target / max(w, h); nw, nh = int(w*s), int(h*s)
    return s, (target-nw)//2, (target-nh)//2


class CropDataset(Dataset):
    def __init__(self, split, augment=False):
        self.img_dir = os.path.join(KP_ROOT, split, 'images')
        self.lbl_dir = os.path.join(KP_ROOT, split, 'labels')
        self.names = sorted(f for f in os.listdir(self.img_dir) if f.endswith('.png'))
        self.augment = augment
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    def __len__(self): return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img = cv2.imread(os.path.join(self.img_dir, name))
        h, w = img.shape[:2]
        with open(os.path.join(self.lbl_dir, name.replace('.png', '.txt'))) as f:
            kx, ky = (float(v) for v in f.read().strip().split())
        s, dx, dy = letterbox_params(w, h, TRAIN_SIZE)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        r = cv2.resize(rgb, (int(w*s), int(h*s)))
        canvas = np.zeros((TRAIN_SIZE, TRAIN_SIZE, 3), dtype=np.uint8)
        canvas[dy:dy+r.shape[0], dx:dx+r.shape[1]] = r
        tx, ty = kx*s + dx, ky*s + dy
        if self.augment and np.random.rand() < 0.5:
            canvas = canvas[:, ::-1].copy()
            tx = TRAIN_SIZE - 1 - tx
        img_t = self.norm(TF.to_image(canvas).float()/255.)
        if ARCH == 'heatmap':
            # soft-argmax model outputs raw pixel coords in [0, TRAIN_SIZE-1]
            target = torch.tensor([tx, ty], dtype=torch.float32)
        else:
            # direct regression: target normalized to [-1, 1]
            target = torch.tensor([tx / (TRAIN_SIZE-1) * 2 - 1,
                                   ty / (TRAIN_SIZE-1) * 2 - 1], dtype=torch.float32)
        return img_t, target, torch.tensor([tx, ty])


class HeatmapHead(nn.Module):
    """Soft-argmax + U-Net skip connections. Returns (x, y) in pixel space [0, TRAIN_SIZE-1].

    Mirrors MeristemSoftArgmaxSkip from scripts/softargmax_skip_noseg_sweep.py — the
    architecture that beat plain regression on the original sweep.
    """
    def __init__(self, soft_res=SOFT_RES, train_size=TRAIN_SIZE):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights='DEFAULT').features
        self.enc_stage1 = backbone[:4]   # 24ch @ 28x28
        self.enc_stage2 = backbone[4:9]  # 48ch @ 14x14
        self.enc_stage3 = backbone[9:]   # 576ch @ 7x7

        self.up1 = nn.ConvTranspose2d(576, 256, 4, 2, 1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 24, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU())
        self.up3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
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
        score = self.dec3(self.up3(d2))               # 1ch @ 56x56
        B, _, H, W = score.shape
        probs = torch.softmax(score.view(B, -1), dim=1).view(B, H, W)
        x_pred = (probs * self.grid_x).sum(dim=(1, 2))
        y_pred = (probs * self.grid_y).sum(dim=(1, 2))
        return torch.stack([x_pred, y_pred], dim=1)


class DirectHead(nn.Module):
    """Same encoder, replaces decoder with global-pool + MLP -> (x, y) in [-1, 1]."""
    def __init__(self):
        super().__init__()
        self.encoder = models.mobilenet_v3_small(weights='DEFAULT').features
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(576, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 2), nn.Tanh())
    def forward(self, x): return self.head(self.encoder(x))


def evaluate(model, loader, dev):
    model.eval(); errs = []
    with torch.no_grad():
        for img, _, gt in loader:
            img = img.to(dev)
            pred = model(img).cpu().numpy()
            if ARCH == 'heatmap':
                # soft-argmax already returns coords in pixel space
                for b in range(pred.shape[0]):
                    px, py = pred[b, 0], pred[b, 1]
                    gx, gy = gt[b].numpy()
                    errs.append(float(np.sqrt((px-gx)**2 + (py-gy)**2)))
            else:
                for b in range(pred.shape[0]):
                    px = (pred[b, 0] + 1) / 2 * (TRAIN_SIZE-1)
                    py = (pred[b, 1] + 1) / 2 * (TRAIN_SIZE-1)
                    gx, gy = gt[b].numpy()
                    errs.append(float(np.sqrt((px-gx)**2 + (py-gy)**2)))
    return float(np.mean(errs))


def main():
    print(f"ARCH={ARCH}  CROPS={CROP_STYLE}  KP_ROOT={KP_ROOT}")
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = CropDataset('train', augment=True)
    val_ds = CropDataset('val', augment=False)
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)

    model = (HeatmapHead() if ARCH == 'heatmap' else DirectHead()).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    # heatmap (soft-argmax) trains in pixel space with SmoothL1; direct uses MSE on [-1,1]
    loss_fn = nn.SmoothL1Loss() if ARCH == 'heatmap' else nn.MSELoss()
    best_err = float('inf'); patience_left = PATIENCE; best_epoch = 0
    os.makedirs(os.path.dirname(SAVE_TO), exist_ok=True)
    t_start = time.time()

    for epoch in range(1, MAX_EPOCHS+1):
        model.train()
        for img, target, _ in train_dl:
            img, target = img.to(dev), target.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(img), target)
            loss.backward(); opt.step()
        val_err = evaluate(model, val_dl, dev)
        print(f"epoch {epoch:3d}  val_px_err={val_err:.3f}")
        if val_err < best_err:
            best_err, best_epoch = val_err, epoch
            torch.save(model.state_dict(), SAVE_TO)
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"early stop. best val px_err = {best_err:.3f} at epoch {best_epoch}")
                break

    elapsed = time.time() - t_start
    with open(LOG_TO, 'w') as f:
        json.dump({
            'arch': ARCH, 'crops': CROP_STYLE,
            'best_val_px_err': best_err,
            'best_epoch': best_epoch,
            'total_epochs': epoch,
            'train_seconds': elapsed,
            'n_train': len(train_ds),
            'n_val': len(val_ds),
            'weights': SAVE_TO,
        }, f, indent=2)
    print(f"Saved {SAVE_TO}")
    print(f"Train log: {LOG_TO}")


if __name__ == '__main__':
    main()
