"""
Train Phase-2 keypoint model (MeristemPredictor) on the Oxnard train split
built by oxnard_prep.py. Same architecture the user previously used.

Reads:  datasets/oxnard_kp/{train,val}/{images,labels}/
Saves:  models/keypoint_oxnard.pt   (best by val pixel error)
"""
import os, cv2, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import functional as TF

ROOT = '/home/jaehwan/Desktop/laser-weeding'
KP_ROOT = os.path.join(ROOT, 'datasets/oxnard_kp')
SAVE_TO = os.path.join(ROOT, 'models/keypoint_oxnard.pt')
TRAIN_SIZE = 224
SIGMA = 2.0
BATCH = 16
LR = 1e-4
MAX_EPOCHS = 300
PATIENCE = 30


def letterbox_params(w, h, target):
    s = target / max(w, h)
    nw, nh = int(w*s), int(h*s)
    return s, (target-nw)//2, (target-nh)//2


def make_target_heatmap(cx, cy, sigma=SIGMA, size=TRAIN_SIZE):
    y, x = np.ogrid[:size, :size]
    return np.exp(-((x-cx)**2 + (y-cy)**2) / (2*sigma**2)).astype(np.float32)


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
        target = make_target_heatmap(tx, ty)
        img_t = self.norm(TF.to_image(canvas).float()/255.)
        return img_t, torch.from_numpy(target).unsqueeze(0), torch.tensor([tx, ty])


class MeristemPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.mobilenet_v3_small(weights='DEFAULT').features
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(576, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 1, 3, padding=1), nn.Sigmoid())
    def forward(self, x): return self.decoder(self.encoder(x))


def evaluate(model, loader, dev):
    model.eval()
    px_errs = []
    with torch.no_grad():
        for img, _, gt in loader:
            img = img.to(dev)
            heat = model(img).cpu().numpy()  # (B, 1, 224, 224)
            for b in range(heat.shape[0]):
                py, px = np.unravel_index(np.argmax(heat[b, 0]), heat[b, 0].shape)
                gx, gy = gt[b].numpy()
                px_errs.append(float(np.sqrt((px-gx)**2 + (py-gy)**2)))
    return float(np.mean(px_errs))


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = CropDataset('train', augment=True)
    val_ds   = CropDataset('val', augment=False)
    print(f"train crops: {len(train_ds)}, val crops: {len(val_ds)}")
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)

    model = MeristemPredictor().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    best_err = float('inf'); patience_left = PATIENCE
    os.makedirs(os.path.dirname(SAVE_TO), exist_ok=True)

    for epoch in range(1, MAX_EPOCHS+1):
        model.train()
        running = 0.0; n = 0
        for img, target, _ in train_dl:
            img, target = img.to(dev), target.to(dev)
            opt.zero_grad()
            pred = model(img)
            loss = loss_fn(pred, target)
            loss.backward(); opt.step()
            running += loss.item() * img.size(0); n += img.size(0)
        train_loss = running / n
        val_err = evaluate(model, val_dl, dev)
        print(f"epoch {epoch:3d}  train_loss={train_loss:.6f}  val_px_err={val_err:.3f}")
        if val_err < best_err:
            best_err = val_err
            torch.save(model.state_dict(), SAVE_TO)
            patience_left = PATIENCE
            print(f"  -> saved (best px_err={best_err:.3f})")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"early stop. best val px_err = {best_err:.3f}")
                break


if __name__ == '__main__':
    main()
