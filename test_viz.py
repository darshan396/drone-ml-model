import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import sys
import joblib 

sys.path.append(os.getcwd())
from src.unet import RoverLanding
from src.preprocessing import preprocess

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UNET_PATH = "src/rover_model_latest.pth"
ADABOOST_PATH = "src/rover_adaboost_pure.pkl"
TEST_DIR = "data/dataset/test_images"  
OUTPUT_FILE = "prediction_results_pure_boost.png"

def visualize():
    unet = RoverLanding(n_classes=4).to(DEVICE)
    unet.load_state_dict(torch.load(UNET_PATH, map_location=DEVICE))
    unet.eval()
    adaboost = joblib.load(ADABOOST_PATH)

    image_paths = glob.glob(os.path.join(TEST_DIR, "*.*"))[:3]
    if not image_paths: print("❌ No images found."); return

    n_images = len(image_paths)
    fig, axes = plt.subplots(n_images, 3, figsize=(15, 5 * n_images))
    if n_images == 1: axes = np.expand_dims(axes, axis=0)

    for idx, img_path in enumerate(image_paths):
        data = preprocess(img_path)
        if data is None: continue
        x = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            seg_logits, _ = unet(x)
            mask = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy()
            
            features = unet.get_features(x).reshape(1, -1)

        verdict_class = adaboost.predict(features)[0] 
        confidence = adaboost.predict_proba(features)[0][1]

        axes[idx, 0].imshow(data[:, :, 0], cmap='gray')
        axes[idx, 0].set_title(f"Input: {os.path.basename(img_path)}")
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(mask, cmap='jet', vmin=0, vmax=3)
        axes[idx, 1].set_title("Hazard Map")
        axes[idx, 1].axis('off')

        verdict = "SAFE" if verdict_class == 1 else "UNSAFE"
        color = "green" if verdict == "SAFE" else "red"
        text = f"Landing Confidence:\n{confidence:.2%}\n\nResult:\n{verdict}"
        axes[idx, 2].text(0.5, 0.5, text, fontsize=16, ha='center', va='center', color=color, fontweight='bold')
        axes[idx, 2].axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE)
    print(f"✅ Visualization saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    visualize()