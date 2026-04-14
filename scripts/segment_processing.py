import cv2
import os
import numpy as np
from ultralytics import YOLO

# --- CONFIG ---
# Using the Small .pt model for maximum mask accuracy
model_path = r'D:\Oxnard_Pigsweed_3.18\best_pigweed_145.pt'
input_folder = r'D:\Oxnard_Pigsweed_3.18\left'
output_folder = r'D:\Oxnard_Pigsweed_3.18\processed_crops'
padding = 20  # Slightly more padding for MobileNet context

# Strict pixel count to ensure the plant is actually visible
min_visible_pixels = 400 

os.makedirs(output_folder, exist_ok=True)

# Load the Small PyTorch model
print(f"Loading Full-Precision Model: {model_path}")
model = YOLO(model_path, task='segment')

# This moves the .pt model to the GPU for faster (but still precise) processing
model.to('cuda')

print("Starting Master Dataset Extraction. Priority: Quality over Speed.")

for img_name in os.listdir(input_folder):
    if not img_name.endswith(('.jpg', '.png')): continue
    
    img = cv2.imread(os.path.join(input_folder, img_name))
    
    # Run inference at 1280px. .pt is slower but far more precise on masks.
    results = model(img, imgsz=1280, conf=0.45, verbose=False)[0]

    if results.masks is None: continue

    boxes = results.boxes.xyxy.cpu().numpy()
    masks = results.masks.data.cpu().numpy()

    for i, (box, mask) in enumerate(zip(boxes, masks)):
        # 1. Resize mask to original high-res image dimensions
        mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
        
        # 2. Thresholding at 0.5 (Standard "Plant/Not-Plant" line)
        binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
        
        # 3. Apply Mask (Black out the Oxnard soil)
        masked_img = cv2.bitwise_and(img, img, mask=binary_mask)

        # 4. Define the crop area with padding
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(img.shape[1], x2 + padding), min(img.shape[0], y2 + padding)
        
        crop = masked_img[y1:y2, x1:x2]

        if crop.size > 0:
            # --- FINAL VALIDATION ---
            # Verify we actually have a plant and not just a black square
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            actual_pixel_count = cv2.countNonZero(gray_crop)

            if actual_pixel_count > min_visible_pixels:
                save_name = f"plant_{img_name[:-4]}_{i}.png"
                cv2.imwrite(os.path.join(output_folder, save_name), crop)
            else:
                print(f"Skipping empty detection in {img_name}")

print(f"\nSUCCESS: High-quality dataset created at {output_folder}")