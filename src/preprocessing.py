import cv2
import torch
import numpy as np
import os
from unet import RoverLanding

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- PREPROCESS ----------------
def preprocess(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return None

    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Channel 1: local normalized intensity
    mu = cv2.GaussianBlur(gray, (31, 31), 0)
    sigma = cv2.GaussianBlur((gray - mu) ** 2, (31, 31), 0)
    sigma = np.sqrt(sigma) + 1e-6
    norm = (gray - mu) / sigma

    # Channel 2: gradient magnitude (roughness)
    gx = cv2.Sobel(norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(norm, cv2.CV_32F, 0, 1, ksize=3)
# OLD:
# grad = np.sqrt(gx**2 + gy**2)

# NEW: (Clip extremely high contrast values which are usually paint or shadows)
    grad = np.sqrt(gx**2 + gy**2)
    grad = np.clip(grad, 0, 3.0) # Cap the roughness score

    # Channel 3: laplacian (edges / pits)
    lap = cv2.Laplacian(norm, cv2.CV_32F, ksize=3)

    # Channel 4: local variance (texture)
    mean = cv2.GaussianBlur(norm, (15, 15), 0)
    sq_mean = cv2.GaussianBlur(norm**2, (15, 15), 0)
    var = sq_mean - mean**2

    return np.stack([norm, grad, lap, var], axis=-1).astype(np.float32)


# ---------------- INFERENCE ----------------
if __name__ == "__main__":

    model = RoverLanding(n_classes=4).to(DEVICE)
    model.eval()

    weights_path = "rover_model_latest.pth"
    if not os.path.exists(weights_path):
        print("❌ Model weights not found")
        exit()

    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    print(f"Loaded trained brain from: {weights_path}")

    test_images = [
        "image.png",
        "image1.png",
        "image_copy.png"
    ]

# ... inside your loop ...

    for img_path in test_images:
        data = preprocess(img_path)
        if data is None:
            continue

        x = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # Update: Capture the trained safety_score
            seg_logits, safety_score = model(x)
            
            # Get probability directly from the model
            # Applying sigmoid to squash output between 0 and 1
            predicted_prob = torch.sigmoid(safety_score).item()

        print("-" * 40)
        print(f"Image: {img_path}")
        print(f"Model Predicted Safety: {predicted_prob:.2%}")
        
        # Optional: Compare with your manual calculation to verify segmentation quality
        seg_map = seg_logits.argmax(dim=1).squeeze(0)
        hazard_ratio = (seg_map != 0).float().mean().item()
        print(f"Manual Calc (Check): {(1.0 - hazard_ratio):.2%}")
        
        print("SAFE" if predicted_prob > 0.5 else "UNSAFE")
