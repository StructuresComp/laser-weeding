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

# --- CONFIG ---
IMG_DIR = r'D:\Oxnard_Pigsweed_3.18\processed_crops'
LBL_DIR = r'D:\Oxnard_Pigsweed_3.18\keypoint_labels'
PRETRAINED_WEIGHTS = r'D:\Oxnard_Pigsweed_3.18\best_targeting_v3.pth'  # Path to old weights

TRAIN_SIZE = 224
SIGMA = 2.0
BATCH_SIZE = 8
LR = 1e-4
MAX_EPOCHS = 500
PATIENCE = 50  # Stop if no improvement after 50 epochs

# --- GEOMETRY UTILITIES ---
def letterbox_params(w, h, target_size):
    scale = target_size / max(w, h)
    nw, nh = int(w * scale), int(h * scale)
    dx, dy = (target_size - nw) // 2, (target_size - nh) // 2
    return scale, dx, dy

# --- DATASET ---
class PigweedTargetingDataset(Dataset):
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

        gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
        heatmap *= (gray > 10).astype(np.float32)

        img_tensor = F.to_image(canvas) 
        heatmap_tensor = tv_tensors.Mask(torch.tensor(heatmap).unsqueeze(0).float())

        if self.augment:
            img_tensor, heatmap_tensor = self.aug_pipeline(img_tensor, heatmap_tensor)

        img_final = self.normalize(img_tensor.float() / 255.0)
        target_final = heatmap_tensor.as_subclass(torch.Tensor)

        return img_final, target_final

# --- MODEL (V3-Small) ---
class MeristemPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.mobilenet_v3_small(weights='DEFAULT').features
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(576, 256, 4, 2, 1), # 14x14
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 28x28
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 56x56
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True), # 224x224
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.decoder(self.encoder(x))

# --- MAIN LOOP ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_labels = [f for f in os.listdir(LBL_DIR) if f.endswith('.txt')]
    img_names = [f.replace('.txt', '.png') for f in all_labels]
    
    if len(img_names) == 0:
        print("ERROR: No labels found. Please run the labeler script first.")
        exit()
        
    # --- 60 / 20 / 20 SPLIT LOGIC ---
    # First split: 60% Train, 40% Temp
    train_n, temp_n = train_test_split(img_names, test_size=0.4, random_state=42)
    # Second split: Divide the 40% Temp exactly in half (20% Val, 20% Test)
    val_n, test_n = train_test_split(temp_n, test_size=0.5, random_state=42)

    print(f"Dataset Split -> Train: {len(train_n)} | Val: {len(val_n)} | Test: {len(test_n)}")

    train_loader = DataLoader(PigweedTargetingDataset(train_n, augment=True), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(PigweedTargetingDataset(val_n, augment=False), batch_size=BATCH_SIZE)
    test_loader = DataLoader(PigweedTargetingDataset(test_n, augment=False), batch_size=BATCH_SIZE)

    model = MeristemPredictor().to(device)
    
    if os.path.exists(PRETRAINED_WEIGHTS):
        print(f"Loading existing weights from: {PRETRAINED_WEIGHTS}")
        model.load_state_dict(torch.load(PRETRAINED_WEIGHTS, map_location=device))
    else:
        print(f"Warning: Weights not found at {PRETRAINED_WEIGHTS}. Starting from scratch.")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    print(f"Starting Training on {device}. Max Epochs: {MAX_EPOCHS} | Patience: {PATIENCE}")

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_l = 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), targets)
            loss.backward()
            optimizer.step()
            train_l += loss.item()

        model.eval()
        val_l = 0
        with torch.no_grad():
            for v_imgs, v_targets in val_loader:
                val_l += criterion(model(v_imgs.to(device)), v_targets.to(device)).item()
        
        avg_train_loss = train_l / len(train_loader)
        avg_val_loss = val_l / len(val_loader)
        
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1:03d} | Train: {avg_train_loss:.7f} | Val: {avg_val_loss:.7f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "new_best_targeting_v3.pth")
            print(f"  --> Best model saved as 'new_best_targeting_v3.pth' (Loss: {best_val_loss:.7f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\n[EARLY STOP] No improvement for {PATIENCE} epochs. Ending training.")
                break

    print(f"\nTraining Finished. Best Val Loss: {best_val_loss:.7f}")

    # --- FINAL TEST EVALUATION ---
    print("\nEvaluating on held-out Test Set...")
    # Load the best weights we just found
    model.load_state_dict(torch.load("new_best_targeting_v3.pth", map_location=device))
    model.eval()
    test_l = 0
    with torch.no_grad():
        for t_imgs, t_targets in test_loader:
            test_l += criterion(model(t_imgs.to(device)), t_targets.to(device)).item()
            
    avg_test_loss = test_l / len(test_loader)
    print(f"Final Test Loss (MSE): {avg_test_loss:.7f}")