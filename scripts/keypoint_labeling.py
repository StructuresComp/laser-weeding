import cv2
import os

# --- CONFIG ---
image_folder = r'D:\Oxnard_Pigsweed_3.18\processed_crops'
label_folder = r'D:\Oxnard_Pigsweed_3.18\keypoint_labels'
os.makedirs(label_folder, exist_ok=True)

# State variables for mouse callback
clicked_point = None
temp_point = None

def mouse_callback(event, x, y, flags, param):
    global temp_point
    if event == cv2.EVENT_LBUTTONDOWN:
        temp_point = (x, y)

# 1. Get sorted list of images (sorting ensures consistent order)
images = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
total_imgs = len(images)

# 2. Find the last labeled image to resume from
existing_labels = sorted([f for f in os.listdir(label_folder) if f.endswith('.txt')])
start_idx = 0

if existing_labels:
    # Get the alphabetically last label you created
    last_label = existing_labels[-1]
    last_img_name = last_label.replace('.txt', '.png')
    
    # Find where that image is in our master list, and start at the NEXT index
    if last_img_name in images:
        start_idx = images.index(last_img_name) + 1
        print(f"Found existing labels. Resuming from image {start_idx + 1}...")
else:
    print("No existing labels found. Starting from the beginning.")

cv2.namedWindow("SCI Lab Labeler", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("SCI Lab Labeler", mouse_callback)

print("--- CONTROLS ---")
print("'Space' or 'Enter' : Confirm and Save")
print("'s'                : Skip (bad segment)")
print("'q'                : Quit Labeling")
print("----------------")

# 3. Start loop explicitly from the resume index
for i in range(start_idx, total_imgs):
    img_name = images[i]
    label_path = os.path.join(label_folder, f"{os.path.splitext(img_name)[0]}.txt")
    
    # Fallback check (in case you skipped an image earlier in the sequence)
    if os.path.exists(label_path):
        continue

    img_path = os.path.join(image_folder, img_name)
    raw_img = cv2.imread(img_path)
    if raw_img is None: continue

    temp_point = None # Reset for new image
    
    while True:
        display_img = raw_img.copy()
        
        # UI Overlay (Status text will accurately reflect the total list progress)
        status_text = f"[{i+1}/{total_imgs}] {img_name}"
        cv2.putText(display_img, status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw the current selection
        if temp_point:
            cv2.drawMarker(display_img, temp_point, (0, 0, 255), cv2.MARKER_CROSS, 10, 2)
            cv2.circle(display_img, temp_point, 3, (255, 255, 255), -1)

        cv2.imshow("SCI Lab Labeler", display_img)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Quitting...")
            cv2.destroyAllWindows()
            exit()
            
        elif key == ord('s'):
            print(f"Skipped: {img_name}")
            break # Move to next image without saving
            
        elif (key == ord(' ') or key == 13) and temp_point:
            # SAVE COORDINATES
            with open(label_path, 'w') as f:
                f.write(f"{temp_point[0]} {temp_point[1]}")
            print(f"Saved: {img_name} -> {temp_point}")
            break

cv2.destroyAllWindows()
print("All images processed!")