import torch
import torch.nn as nn
import cv2
import os
import numpy as np
from torchvision import models, transforms

# --- CONFIG ---
MODEL_PATH = "new_best_targeting_v3.pth"
IMAGE_DIR = r'D:\Oxnard_Pigsweed_3.18\processed_crops'
TRAIN_SIZE = 224

class MeristemPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.mobilenet_v3_small(weights=None).features
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(576, 256, 4, 2, 1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.decoder(self.encoder(x))

def get_both_targets(heatmap, mask):
    """Calculates BOTH the Centroid and the Peak for visual comparison."""
    masked = heatmap * mask
    
    # 1. Get the absolute Peak
    _, max_val, _, max_loc = cv2.minMaxLoc(masked)
    peak_target = max_loc
    
    # 2. Get the Centroid (Center of Mass)
    M = cv2.moments(masked)
    if M["m00"] > 0.001:
        centroid_target = (M["m10"] / M["m00"], M["m01"] / M["m00"])
    else:
        centroid_target = None # Fallback if no mass exists
        
    return centroid_target, peak_target, max_val

# --- SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MeristemPredictor().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

images = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.png')])
idx = len(images) - 1

while True:
    img_name = images[idx]
    orig_bgr = cv2.imread(os.path.join(IMAGE_DIR, img_name))
    h_orig, w_orig = orig_bgr.shape[:2]
    
    # 1. Pre-process
    scale = TRAIN_SIZE / max(h_orig, w_orig)
    nw, nh = int(w_orig * scale), int(h_orig * scale)
    dx, dy = (TRAIN_SIZE - nw) // 2, (TRAIN_SIZE - nh) // 2
    
    img_res = cv2.resize(orig_bgr, (nw, nh))
    canvas = np.zeros((TRAIN_SIZE, TRAIN_SIZE, 3), dtype=np.uint8)
    canvas[dy:dy+nh, dx:dx+nw] = img_res
    
    # Generate MASK from the segmented plant pixels
    mask = (cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 10).astype(np.float32)

    # 2. Inference
    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    input_t = transforms.ToTensor()(img_rgb)
    input_t = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(input_t).unsqueeze(0).to(device)
    
    with torch.no_grad():
        heatmap = model(input_t).squeeze().cpu().numpy()

    # 3. GET BOTH TARGETS
    centroid_tgt, peak_tgt, max_val = get_both_targets(heatmap, mask)

    # 4. Global Coordinate Recovery & UI Drawing
    viz = orig_bgr.copy()

    # Draw Peak (Blue Tilted Cross)
    p_x = int((peak_tgt[0] - dx) / scale)
    p_y = int((peak_tgt[1] - dy) / scale)
    cv2.drawMarker(viz, (p_x, p_y), (255, 0, 0), cv2.MARKER_TILTED_CROSS, 20, 2)

    # Draw Centroid (Green Standard Cross)
    if centroid_tgt:
        c_x = int((centroid_tgt[0] - dx) / scale)
        c_y = int((centroid_tgt[1] - dy) / scale)
        cv2.drawMarker(viz, (c_x, c_y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

    # 5. Display UI
    display = cv2.resize(viz, (512, 512), interpolation=cv2.INTER_NEAREST)
    
    cv2.putText(display, f"MAX CONF: {max_val:.4f}", (10, 30), 1, 1.2, (255, 255, 255), 2)
    cv2.putText(display, "GREEN + : Centroid", (10, 60), 1, 1.2, (0, 255, 0), 2)
    cv2.putText(display, "BLUE  X : Peak", (10, 90), 1, 1.2, (255, 0, 0), 2)
    cv2.putText(display, "LASER STATUS: ARMED", (10, 120), 1, 1.2, (0, 0, 255), 2)
    
    cv2.imshow("SCI Lab: Mask-Enforced Targeting", display)
    
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'): break
    elif key == ord('d'): idx = max(0, idx - 1)
    elif key == ord('a'): idx = min(len(images)-1, idx + 1)

cv2.destroyAllWindows()