"""
Heatmap Model — Outlier & Error Distribution Analysis
=====================================================
For each chosen training-set size N, this script:
  1. Trains the heatmap MeristemPredictor (same as generalization_sweep.py)
  2. Runs inference on the SAME fixed test set used by the sweep
  3. Computes per-image pixel errors
  4. Saves:
       - error_hist_N{n}.png       histogram + CDF of pixel errors
       - worst_preds_N{n}.png      grid of K worst predictions
       - per_image_errors_N{n}.csv every test image's error
  5. Prints quantile stats so you can see how much the tail dominates the mean

Usage:
    python heatmap_outlier_analysis.py
"""

import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Reuse the model + dataset from the sweep so behavior is identical
from generalization_sweep import (
    MeristemPredictor,
    PigweedTargetingDataset,
    FIXED_TEST_FRACTION,
    TRAIN_SIZE,
    LBL_DIR,
    BATCH_SIZE,
    LR,
    MAX_EPOCHS,
    PATIENCE,
)

# --- CONFIG ---
SIZES_TO_ANALYZE = [100, 200, 300]   # focus on the small-data regime where outliers matter
WORST_K = 12                         # how many worst predictions to visualize per N
OUTPUT_DIR = '/home/jaehwan/Desktop/yolo26/Laser Weeding Training/script/outlier_analysis'

# ImageNet normalization (used by the dataset) — needed to undo for visualization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize(img_tensor):
    """Convert a normalized (3, H, W) tensor back to a uint8 (H, W, 3) image."""
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def train_heatmap_model(train_names, val_names, device, run_label):
    """Train heatmap model and return the best-state model. Mirrors generalization_sweep.py."""
    train_loader = DataLoader(PigweedTargetingDataset(train_names, augment=True),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(PigweedTargetingDataset(val_names, augment=False),
                            batch_size=BATCH_SIZE, num_workers=0)

    model = MeristemPredictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_state = None

    for epoch in range(MAX_EPOCHS):
        model.train()
        for imgs, targets, _ in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), targets)
            loss.backward()
            optimizer.step()

        model.eval()
        val_l = 0
        with torch.no_grad():
            for v_imgs, v_targets, _ in val_loader:
                val_l += criterion(model(v_imgs.to(device)), v_targets.to(device)).item()
        avg_val = val_l / len(val_loader)
        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                break

        if (epoch + 1) % 25 == 0:
            print(f"  [{run_label}] Epoch {epoch+1:03d} | Val: {avg_val:.6f}")

    model.load_state_dict(best_state)
    model.eval()
    return model


def compute_per_image_errors(model, test_names, device):
    """Run model on test set, return list of (filename, gt_xy, pred_xy, error_px, image_tensor)."""
    test_dataset = PigweedTargetingDataset(test_names, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0)

    records = []
    idx = 0
    with torch.no_grad():
        for imgs, targets, gt_coords in test_loader:
            preds = model(imgs.to(device)).squeeze(1).cpu().numpy()  # (B, H, W)
            for i in range(preds.shape[0]):
                hm = preds[i]
                peak_y, peak_x = np.unravel_index(hm.argmax(), hm.shape)
                gx, gy = gt_coords[i].numpy()
                err = float(np.sqrt((peak_x - gx) ** 2 + (peak_y - gy) ** 2))
                records.append({
                    'filename': test_names[idx],
                    'gt_x': float(gx),
                    'gt_y': float(gy),
                    'pred_x': float(peak_x),
                    'pred_y': float(peak_y),
                    'error_px': err,
                    'image': imgs[i],   # keep tensor for later visualization
                })
                idx += 1
    return records


def print_quantile_stats(errors, label):
    e = np.array(errors)
    print(f"\n  --- {label} ---")
    print(f"  N test images:    {len(e)}")
    print(f"  Mean:             {e.mean():.2f} px")
    print(f"  Median (P50):     {np.median(e):.2f} px")
    print(f"  P75:              {np.percentile(e, 75):.2f} px")
    print(f"  P90:              {np.percentile(e, 90):.2f} px")
    print(f"  P95:              {np.percentile(e, 95):.2f} px")
    print(f"  P99:              {np.percentile(e, 99):.2f} px")
    print(f"  Max:              {e.max():.2f} px")
    # How much of the mean is contributed by the top 5%?
    top5_count = max(1, int(0.05 * len(e)))
    top5_contrib = np.sort(e)[-top5_count:].sum()
    print(f"  Top 5% contribute {100 * top5_contrib / e.sum():.1f}% of total error sum")


def plot_error_distribution(errors, n, save_path):
    e = np.array(errors)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Histogram (log y)
    axes[0].hist(e, bins=40, color='steelblue', edgecolor='black', alpha=0.85)
    axes[0].set_yscale('log')
    axes[0].axvline(np.median(e), color='green', linestyle='--', linewidth=2,
                    label=f'Median = {np.median(e):.1f} px')
    axes[0].axvline(e.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean = {e.mean():.1f} px')
    axes[0].axvline(np.percentile(e, 95), color='orange', linestyle=':', linewidth=2,
                    label=f'P95 = {np.percentile(e, 95):.1f} px')
    axes[0].set_xlabel('Pixel error')
    axes[0].set_ylabel('Count (log)')
    axes[0].set_title(f'Heatmap test errors @ N={n} train+val')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # CDF
    sorted_e = np.sort(e)
    cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
    axes[1].plot(sorted_e, cdf, color='steelblue', linewidth=2)
    axes[1].axhline(0.5, color='gray', linestyle=':', alpha=0.6)
    axes[1].axhline(0.95, color='gray', linestyle=':', alpha=0.6)
    axes[1].set_xlabel('Pixel error')
    axes[1].set_ylabel('Cumulative fraction of test images')
    axes[1].set_title(f'CDF of test errors @ N={n}')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_worst_predictions(records, n, k, save_path):
    """Grid of the K worst test predictions, with GT (green) and predicted (red) markers."""
    sorted_records = sorted(records, key=lambda r: -r['error_px'])[:k]

    cols = 4
    rows = (k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

    for ax, rec in zip(axes, sorted_records):
        img = denormalize(rec['image'])
        ax.imshow(img)
        ax.scatter(rec['gt_x'], rec['gt_y'], c='lime', s=120, marker='+',
                   linewidths=2.5, label='GT')
        ax.scatter(rec['pred_x'], rec['pred_y'], c='red', s=120, marker='x',
                   linewidths=2.5, label='Pred')
        # Draw a line from GT to prediction so the miss is obvious
        ax.plot([rec['gt_x'], rec['pred_x']], [rec['gt_y'], rec['pred_y']],
                color='yellow', linewidth=1.0, alpha=0.7)
        ax.set_title(f"err={rec['error_px']:.1f}px\n{rec['filename'][:25]}", fontsize=9)
        ax.axis('off')

    # Hide any unused subplots
    for ax in axes[len(sorted_records):]:
        ax.axis('off')

    # Single legend on the figure
    handles = [
        plt.Line2D([0], [0], marker='+', color='w', markerfacecolor='lime',
                   markeredgecolor='lime', markersize=12, label='Ground truth'),
        plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='red',
                   markeredgecolor='red', markersize=12, label='Prediction'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=11)
    fig.suptitle(f'{k} worst heatmap predictions @ N={n} train+val', fontsize=13)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"  Saved: {save_path}")


def save_per_image_csv(records, save_path):
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'gt_x', 'gt_y', 'pred_x', 'pred_y', 'error_px'])
        for r in sorted(records, key=lambda x: -x['error_px']):
            writer.writerow([r['filename'], r['gt_x'], r['gt_y'],
                             r['pred_x'], r['pred_y'], r['error_px']])
    print(f"  Saved: {save_path}")


# --- MAIN ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Same fixed test split as the sweeps (random_state=42)
    all_labels = sorted([f for f in os.listdir(LBL_DIR) if f.endswith('.txt')])
    all_img_names = [f.replace('.txt', '.png') for f in all_labels]
    trainval_pool, fixed_test = train_test_split(
        all_img_names, test_size=FIXED_TEST_FRACTION, random_state=42
    )
    print(f"Fixed test set: {len(fixed_test)} images")
    print(f"Train+val pool: {len(trainval_pool)} images\n")

    for n in SIZES_TO_ANALYZE:
        print("=" * 70)
        print(f"ANALYZING N={n}")
        print("=" * 70)
        t0 = time.time()

        # Same subsampling as the sweeps
        if n < len(trainval_pool):
            subset, _ = train_test_split(trainval_pool, train_size=n, random_state=42)
        else:
            subset = trainval_pool
        train_n, val_n = train_test_split(subset, test_size=0.25, random_state=42)

        print(f"  Training (train={len(train_n)}, val={len(val_n)})...")
        model = train_heatmap_model(train_n, val_n, device, f"N={n}")

        print(f"  Running inference on {len(fixed_test)} test images...")
        records = compute_per_image_errors(model, fixed_test, device)
        errors = [r['error_px'] for r in records]

        print_quantile_stats(errors, f"N={n}")

        # Save outputs
        plot_error_distribution(errors, n, os.path.join(OUTPUT_DIR, f'error_hist_N{n}.png'))
        plot_worst_predictions(records, n, WORST_K,
                               os.path.join(OUTPUT_DIR, f'worst_preds_N{n}.png'))
        save_per_image_csv(records, os.path.join(OUTPUT_DIR, f'per_image_errors_N{n}.csv'))

        print(f"  Done N={n} in {(time.time() - t0)/60:.1f}min\n")

    print(f"\nAll outputs in: {OUTPUT_DIR}")
