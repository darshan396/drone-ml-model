import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import sys

# Ensure python can find your src modules
sys.path.append(os.getcwd())

# Correct imports based on your folder structure
from unet import RoverLanding
from preprocessing import preprocess

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Point to the checkpoint folder
MODEL_PATH = "rover_model_latest.pth"
# Use the dataset structure we defined
TEST_DIR = "../data/dataset/test_images"  
OUTPUT_FILE = "prediction_results.png"

def visualize_prediction():
    # 1. Validation
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file '{MODEL_PATH}' not found. Run train.py first!")
        return

    # Find images automatically using glob
    image_paths = glob.glob(os.path.join(TEST_DIR, "*.*"))
    # Filter for valid image extensions
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_paths:
        print(f"❌ No images found in {TEST_DIR}")
        print(f"   -> Please add some .png or .jpg files there to test.")
        return

    # Limit to first 3 images to keep the plot readable
    image_paths = image_paths[:3] 
    print(f"📊 Visualizing: {[os.path.basename(p) for p in image_paths]}")

    # 2. Load Model
    model = RoverLanding(n_classes=4).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 3. Setup Plot
    n_images = len(image_paths)
    fig, axes = plt.subplots(n_images, 3, figsize=(15, 5 * n_images))
    
    # Handle single image case (matplotlib quirk where axes isn't a list)
    if n_images == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, img_path in enumerate(image_paths):
        # Preprocess
        data = preprocess(img_path)
        if data is None: continue
        
        # Prepare tensor
        x = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        # Inference
        with torch.no_grad():
            seg_logits, safety_score = model(x)
            
            # Process outputs
            prob = torch.sigmoid(safety_score).item()
            mask = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy()

        # --- DISPLAY RESULTS ---
        
        # Column 1: Input (Showing Channel 0 - Albedo/Intensity)
        # Note: It will look normalized (gray), which is what the model sees.
        axes[idx, 0].imshow(data[:, :, 0], cmap='gray')
        axes[idx, 0].set_title(f"Input: {os.path.basename(img_path)}")
        axes[idx, 0].axis('off')

        # Column 2: Segmentation Mask
        # 0=Background (Blue), 1=Hazard (Red/Yellow)
        axes[idx, 1].imshow(mask, cmap='jet', vmin=0, vmax=3)
        axes[idx, 1].set_title("Hazard Map (AI Vision)")
        axes[idx, 1].axis('off')

        # Column 3: Safety Score & Verdict
        verdict = "SAFE" if prob > 0.6 else "UNSAFE"
        color = "green" if verdict == "SAFE" else "red"
        
        text = f"Safety Score:\n{prob:.2%}\n\nVerdict:\n{verdict}"
        axes[idx, 2].text(0.5, 0.5, text, fontsize=18, ha='center', va='center', color=color, fontweight='bold')
        axes[idx, 2].axis('off')

    plt.tight_layout()
    
    # Save instead of show (better for headless servers)
    plt.savefig(OUTPUT_FILE)
    print(f"✅ Visualization saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    visualize_prediction()