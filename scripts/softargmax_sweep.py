"""
Data Generalization Sweep — Spatial Soft-Argmax
================================================
Best-of-both-worlds variant per Brian's PDF:
  - The convolutional decoder builds a 2D feature map (preserves spatial info,
    same as a heatmap model).
  - A spatial 2D softmax converts that map into a probability distribution.
  - Coordinate grids and a weighted sum extract continuous (x, y) directly.
  - Loss is computed on COORDINATES (SmoothL1), NOT on a target heatmap shape.

The model still reasons spatially like a heatmap, but the training signal is
the same as a regression model — no arbitrary heatmap shape to match. This is
the proper fix for "the heatmap is over-penalized on shape, not location".

Same fixed test set, same seed as the other sweeps.

Usage:
    python softargmax_sweep.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import models
from torchvision.transforms.v2 import functional as F
from torchvision import transforms as T_v1  # for ColorJitter (image-only)
import torchvision.transforms.v2 as transforms
import csv
import time

# --- CONFIG ---
IMG_DIR = '/home/jaehwan/Desktop/laser-weeding/processed_crops'
LBL_DIR = '/home/jaehwan/Desktop/laser-weeding/data/keypoint_labels'

TRAIN_SIZE = 224
BATCH_SIZE = 8
LR = 1e-4
MAX_EPOCHS = 300
PATIENCE = 50

# Output spatial resolution of the soft-argmax map. Higher = finer precision
# (sub-pixel via weighted sum, but a coarser map limits how sharp the peak can be).
SOFT_RES = 56  # 224 / 4

FIXED_TEST_FRACTION = 0.2
SWEEP_SIZES = [1500, 1000, 700, 500, 300, 200, 100, 50]

# --- GEOMETRY UTILITIES ---
def letterbox_params(w, h, target_size):
    scale = target_size / max(w, h)
    nw, nh = int(w * scale), int(h * scale)
    dx, dy = (target_size - nw) // 2, (target_size - nh) // 2
    return scale, dx, dy

# --- DATASET — returns (image, coords). Same coordinate-aware augmentation as regression. ---
class PigweedCoordDataset(Dataset):
    def __init__(self, names, augment=False):
        self.names = names
        self.augment = augment
        self.color_aug = transforms.ColorJitter(brightness=0.2, contrast=0.2)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self): return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img_bgr = cv2.imread(os.path.join(IMG_DIR, name))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        with open(os.path.join(LBL_DIR, name.replace('.png', '.txt')), 'r') as f:
            x_raw, y_raw = map(float, f.read().split())

        scale, dx, dy = letterbox_params(w, h, TRAIN_SIZE)
        img_res = cv2.resize(img_rgb, (int(w * scale), int(h * scale)))
        canvas = np.zeros((TRAIN_SIZE, TRAIN_SIZE, 3), dtype=np.uint8)
        canvas[dy:dy+img_res.shape[0], dx:dx+img_res.shape[1]] = img_res

        tx, ty = (x_raw * scale) + dx, (y_raw * scale) + dy

        img_tensor = F.to_image(canvas)

        if self.augment:
            if torch.rand(1).item() < 0.5:
                img_tensor = F.horizontal_flip(img_tensor)
                tx = TRAIN_SIZE - 1 - tx
            if torch.rand(1).item() < 0.5:
                img_tensor = F.vertical_flip(img_tensor)
                ty = TRAIN_SIZE - 1 - ty
            angle = (torch.rand(1).item() * 360.0) - 180.0
            img_tensor = F.rotate(img_tensor, angle)
            theta = np.deg2rad(-angle)
            cx, cy = (TRAIN_SIZE - 1) / 2.0, (TRAIN_SIZE - 1) / 2.0
            dx_c, dy_c = tx - cx, ty - cy
            tx = cx + dx_c * np.cos(theta) - dy_c * np.sin(theta)
            ty = cy + dx_c * np.sin(theta) + dy_c * np.cos(theta)
            img_tensor = self.color_aug(img_tensor)

        img_final = self.normalize(img_tensor.float() / 255.0)
        coords = torch.tensor([tx, ty], dtype=torch.float32)
        return img_final, coords

# --- MODEL: Spatial Soft-Argmax ---
class MeristemSoftArgmax(nn.Module):
    """
    MobileNetV3-Small encoder → upsample to SOFT_RES × SOFT_RES → 1×1 conv → 1 channel
    → spatial softmax → expected (x, y) via coordinate grid weighted sum.

    The output is a continuous (x, y) in pixel space [0, TRAIN_SIZE - 1].
    """
    def __init__(self, soft_res=SOFT_RES, train_size=TRAIN_SIZE):
        super().__init__()
        self.soft_res = soft_res
        self.train_size = train_size

        self.encoder = models.mobilenet_v3_small(weights='DEFAULT').features  # 576 ch @ 7x7

        # Upsample 7x7 → 14x14 → 28x28 → 56x56 (= SOFT_RES)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(576, 256, 4, 2, 1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 1, 1),  # 1x1 conv → single-channel score map
        )

        # Coordinate grids in PIXEL space, registered as buffers so they move with .to(device)
        ys = torch.linspace(0, train_size - 1, soft_res)
        xs = torch.linspace(0, train_size - 1, soft_res)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # both (SOFT_RES, SOFT_RES)
        self.register_buffer('grid_x', grid_x.clone())
        self.register_buffer('grid_y', grid_y.clone())

    def forward(self, x):
        feat = self.encoder(x)              # (B, 576, 7, 7)
        score = self.decoder(feat)          # (B, 1, SOFT_RES, SOFT_RES)
        B, _, H, W = score.shape

        # Spatial softmax over the 2D map
        flat = score.view(B, -1)            # (B, H*W)
        probs = torch.softmax(flat, dim=1)  # (B, H*W)
        probs = probs.view(B, H, W)         # (B, H, W)

        # Weighted sum with coordinate grids → expected (x, y)
        x_pred = (probs * self.grid_x).sum(dim=(1, 2))  # (B,)
        y_pred = (probs * self.grid_y).sum(dim=(1, 2))  # (B,)
        return torch.stack([x_pred, y_pred], dim=1)     # (B, 2)

# --- PIXEL ERROR (trivial — output is already in pixel space) ---
def compute_pixel_errors(model, loader, device):
    model.eval()
    errors = []
    with torch.no_grad():
        for imgs, gt_coords in loader:
            preds = model(imgs.to(device)).cpu().numpy()
            gt = gt_coords.numpy()
            err = np.sqrt(((preds - gt) ** 2).sum(axis=1))
            errors.extend(err.tolist())
    errors = np.array(errors)
    return float(errors.mean()), float(np.median(errors))

# --- SINGLE TRAINING RUN ---
def train_single_run(train_names, val_names, test_names, device, run_label):
    train_loader = DataLoader(PigweedCoordDataset(train_names, augment=True),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(PigweedCoordDataset(val_names, augment=False),
                            batch_size=BATCH_SIZE, num_workers=0)
    test_loader = DataLoader(PigweedCoordDataset(test_names, augment=False),
                             batch_size=BATCH_SIZE, num_workers=0)

    model = MeristemSoftArgmax().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.SmoothL1Loss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_state = None
    final_epoch = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_l = 0
        for imgs, coords in train_loader:
            imgs, coords = imgs.to(device), coords.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), coords)
            loss.backward()
            optimizer.step()
            train_l += loss.item()

        model.eval()
        val_l = 0
        with torch.no_grad():
            for v_imgs, v_coords in val_loader:
                val_l += criterion(model(v_imgs.to(device)), v_coords.to(device)).item()
        avg_val = val_l / len(val_loader)
        scheduler.step(avg_val)
        final_epoch = epoch + 1

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                break

        if (epoch + 1) % 25 == 0:
            avg_train = train_l / len(train_loader)
            print(f"  [{run_label}] Epoch {epoch+1:03d} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

    model.load_state_dict(best_state)
    mean_px, median_px = compute_pixel_errors(model, test_loader, device)

    model.eval()
    test_l = 0
    with torch.no_grad():
        for t_imgs, t_coords in test_loader:
            test_l += criterion(model(t_imgs.to(device)), t_coords.to(device)).item()
    avg_test = test_l / len(test_loader)

    return best_val_loss, avg_test, mean_px, median_px, final_epoch

# --- MAIN ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    all_labels = sorted([f for f in os.listdir(LBL_DIR) if f.endswith('.txt')])
    all_img_names = [f.replace('.txt', '.png') for f in all_labels]
    print(f"Total labeled images available: {len(all_img_names)}\n")

    trainval_pool, fixed_test = train_test_split(
        all_img_names, test_size=FIXED_TEST_FRACTION, random_state=42
    )
    pool_size = len(trainval_pool)
    print(f"Fixed test set: {len(fixed_test)} images (same as other sweeps)")
    print(f"Train+val pool: {pool_size} images")
    print(f"Soft-argmax resolution: {SOFT_RES}x{SOFT_RES}\n")

    sizes = [s for s in SWEEP_SIZES if s <= pool_size]
    if pool_size not in sizes:
        sizes.insert(0, pool_size)
    else:
        sizes.sort(reverse=True)

    results = []

    for n in sizes:
        n_train = int(n * 0.75)
        n_val = n - n_train
        print(f"{'='*60}")
        print(f"SWEEP (SOFT-ARGMAX): N={n} (Train={n_train} | Val={n_val} | Test={len(fixed_test)} fixed)")
        print(f"{'='*60}")

        if n < pool_size:
            subset, _ = train_test_split(trainval_pool, train_size=n, random_state=42)
        else:
            subset = trainval_pool

        train_n, val_n = train_test_split(subset, test_size=0.25, random_state=42)
        test_n = fixed_test

        t0 = time.time()
        best_val, test_loss, mean_px, median_px, epochs = train_single_run(
            train_n, val_n, test_n, device, f"N={n}"
        )
        elapsed = time.time() - t0

        results.append({
            'total': n,
            'train': len(train_n),
            'val': len(val_n),
            'test': len(test_n),
            'val_loss': best_val,
            'test_loss': test_loss,
            'mean_px_err': mean_px,
            'median_px_err': median_px,
            'epochs': epochs,
            'time_min': elapsed / 60,
        })

        print(f"\n  Result: Mean Px Err={mean_px:.1f} | Median Px Err={median_px:.1f} | "
              f"Epochs={epochs} | Time={elapsed/60:.1f}min\n")

    print("\n" + "=" * 100)
    print("GENERALIZATION SWEEP RESULTS — SPATIAL SOFT-ARGMAX")
    print("=" * 100)
    header = f"{'Total':>6} | {'Train':>5} | {'Val':>4} | {'Test':>4} | {'Val Loss':>12} | {'Test Loss':>12} | {'Mean Px':>8} | {'Med Px':>7} | {'Epochs':>6} | {'Time':>6}"
    print(header)
    print("-" * 100)
    for r in results:
        print(f"{r['total']:>6} | {r['train']:>5} | {r['val']:>4} | {r['test']:>4} | "
              f"{r['val_loss']:>12.5f} | {r['test_loss']:>12.5f} | "
              f"{r['mean_px_err']:>7.1f}px | {r['median_px_err']:>6.1f}px | "
              f"{r['epochs']:>6} | {r['time_min']:>5.1f}m")
    print("=" * 100)

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "softargmax_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {csv_path}")
