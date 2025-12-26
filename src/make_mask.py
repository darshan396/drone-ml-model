import cv2
import numpy as np
import os

# --- CONFIGURATION ---
# The name of your helipad image
FILENAME = "../data/dataset/train/image.png" 

# Where to save the mask
MASK_DIR = "../data/dataset/train/masks"

def create_safe_mask():
    # 1. Create the folder if it doesn't exist
    if not os.path.exists(MASK_DIR):
        os.makedirs(MASK_DIR)
        print(f"📂 Created folder: {MASK_DIR}")

    # 2. Create a pure black image (256x256)
    # 0 = Background (Safe)
    # Shape: (Height, Width) -> Standard for this project is 256x256
    black_mask = np.zeros((256, 256), dtype=np.uint8)

    # 3. Save the mask
    save_path = os.path.join(MASK_DIR, FILENAME)
    cv2.imwrite(save_path, black_mask)

    print(f"✅ Created Safe Mask: {save_path}")
    print("   -> All pixels are 0 (Black).")
    print("   -> This tells the AI: 'Ignore the white paint, this entire area is safe.'")

if __name__ == "__main__":
    create_safe_mask()