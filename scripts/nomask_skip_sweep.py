"""
Data Generalization Sweep — Heatmap, No Mask + Skip Connections
================================================================
Combines two fixes:
  1. No mask multiplication on ground-truth heatmap (clean Gaussian target)
  2. Skip connections from encoder to decoder (U-Net style)

Same fixed test set, same seed as all other sweeps.

Usage:
    python nomask_skip_sweep.py
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
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import functional as F
from torchvision import tv_tensors
import csv
import time

# --- CONFIG ---
IMG_DIR = '/home/jaehwan/Desktop/yolo26/Laser Weeding Training/processed_crops'
LBL_DIR = '/home/jaehwan/Desktop/yolo26/Laser Weeding Training/keypoint_labels'

TRAIN_SIZE = 224
SIGMA = 2.0
BATCH_SIZE = 8
LR = 1e-4
MAX_EPOCHS = 300
PATIENCE = 50

FIXED_TEST_FRACTION = 0.2
SWEEP_SIZES = [1500, 1000, 700, 500, 300, 200, 100, 50]

# --- GEOMETRY UTILITIES ---
def letterbox_params(w, h, target_size):
    scale = target_size / max(w, h)
    nw, nh = int(w * scale), int(h * scale)
    dx, dy = (target_size - nw) // 2, (target_size - nh) // 2
    return scale, dx, dy

# --- DATASET (clean Gaussian — NO mask) ---
class PigweedTargetingDatasetNoMask(Dataset):
    def __init__(self, names, augment=False):
        self.names = names
        self.augment = augment
        self.aug_pipeline = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])
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

        grid_y, grid_x = np.mgrid[0:TRAIN_SIZE, 0:TRAIN_SIZE]
        heatmap = np.exp(-((grid_x - tx)**2 + (grid_y - ty)**2) / (2 * SIGMA**2))
        # NO mask multiplication

        img_tensor = F.to_image(canvas)
        heatmap_tensor = tv_tensors.Mask(torch.tensor(heatmap).unsqueeze(0).float())

        if self.augment:
            img_tensor, heatmap_tensor = self.aug_pipeline(img_tensor, heatmap_tensor)

        img_final = self.normalize(img_tensor.float() / 255.0)
        target_final = heatmap_tensor.as_subclass(torch.Tensor)
        return img_final, target_final, torch.tensor([tx, ty], dtype=torch.float32)

# --- MODEL: Skip-connection heatmap (from skipconn_sweep.py) ---
class MeristemPredictorSkip(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights='DEFAULT').features
        self.enc_stage1 = backbone[:4]   # 24ch @ 28x28
        self.enc_stage2 = backbone[4:9]  # 48ch @ 14x14
        self.enc_stage3 = backbone[9:]   # 576ch @ 7x7

        self.up1 = nn.ConvTranspose2d(576, 256, 4, 2, 1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.up2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 24, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.up3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        s1 = self.enc_stage1(x)
        s2 = self.enc_stage2(s1)
        bot = self.enc_stage3(s2)

        d1 = self.up1(bot)
        d1 = self.dec1(torch.cat([d1, s2], dim=1))
        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, s1], dim=1))
        d3 = self.up3(d2)
        d3 = self.dec3(d3)
        return self.final(d3)

# --- PIXEL ERROR ---
def compute_pixel_errors(model, loader, device):
    model.eval()
    errors = []
    with torch.no_grad():
        for imgs, targets, gt_coords in loader:
            preds = model(imgs.to(device)).squeeze(1).cpu().numpy()
            gt = gt_coords.numpy()
            for i in range(preds.shape[0]):
                hm = preds[i]
                peak_y, peak_x = np.unravel_index(hm.argmax(), hm.shape)
                err = np.sqrt((peak_x - gt[i, 0])**2 + (peak_y - gt[i, 1])**2)
                errors.append(err)
    errors = np.array(errors)
    return float(errors.mean()), float(np.median(errors))

# --- SINGLE TRAINING RUN ---
def train_single_run(train_names, val_names, test_names, device, run_label):
    train_loader = DataLoader(PigweedTargetingDatasetNoMask(train_names, augment=True),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(PigweedTargetingDatasetNoMask(val_names, augment=False),
                            batch_size=BATCH_SIZE, num_workers=0)
    test_loader = DataLoader(PigweedTargetingDatasetNoMask(test_names, augment=False),
                             batch_size=BATCH_SIZE, num_workers=0)

    model = MeristemPredictorSkip().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_state = None
    final_epoch = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_l = 0
        for imgs, targets, _ in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), targets)
            loss.backward()
            optimizer.step()
            train_l += loss.item()

        model.eval()
        val_l = 0
        with torch.no_grad():
            for v_imgs, v_targets, _ in val_loader:
                val_l += criterion(model(v_imgs.to(device)), v_targets.to(device)).item()

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
            print(f"  [{run_label}] Epoch {epoch+1:03d} | Train: {avg_train:.6f} | Val: {avg_val:.6f}")

    model.load_state_dict(best_state)
    model.eval()
    test_l = 0
    with torch.no_grad():
        for t_imgs, t_targets, _ in test_loader:
            test_l += criterion(model(t_imgs.to(device)), t_targets.to(device)).item()
    avg_test = test_l / len(test_loader)

    mean_px, median_px = compute_pixel_errors(model, test_loader, device)
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
    print(f"Fixed test set: {len(fixed_test)} images")
    print(f"Train+val pool: {pool_size} images\n")

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
        print(f"SWEEP (NO-MASK + SKIP): N={n} (Train={n_train} | Val={n_val} | Test={len(fixed_test)} fixed)")
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
            'total': n, 'train': len(train_n), 'val': len(val_n), 'test': len(test_n),
            'val_loss': best_val, 'test_loss': test_loss,
            'mean_px_err': mean_px, 'median_px_err': median_px,
            'epochs': epochs, 'time_min': elapsed / 60,
        })

        print(f"\n  Result: Mean Px Err={mean_px:.1f} | Median Px Err={median_px:.1f} | "
              f"Epochs={epochs} | Time={elapsed/60:.1f}min\n")

    print("\n" + "=" * 100)
    print("GENERALIZATION SWEEP RESULTS — HEATMAP (NO MASK + SKIP CONNECTIONS)")
    print("=" * 100)
    header = f"{'Total':>6} | {'Train':>5} | {'Val':>4} | {'Test':>4} | {'Val MSE':>12} | {'Test MSE':>12} | {'Mean Px':>8} | {'Med Px':>7} | {'Epochs':>6} | {'Time':>6}"
    print(header)
    print("-" * 100)
    for r in results:
        print(f"{r['total']:>6} | {r['train']:>5} | {r['val']:>4} | {r['test']:>4} | "
              f"{r['val_loss']:>12.7f} | {r['test_loss']:>12.7f} | "
              f"{r['mean_px_err']:>7.1f}px | {r['median_px_err']:>6.1f}px | "
              f"{r['epochs']:>6} | {r['time_min']:>5.1f}m")
    print("=" * 100)

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nomask_skip_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {csv_path}")
