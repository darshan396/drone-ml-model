import torch
import os
import glob
from unet import RoverLanding
from src.preprocessing import preprocess

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "checkpoints/rover_model_latest.pth"
TEST_DIR = "dataset/test_images"

def predict():
    if not os.path.exists(MODEL_PATH): return print("❌ Train model first!")
    
    model = RoverLanding(n_classes=4).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print(f"{'IMAGE':<30} | {'SCORE':<8} | VERDICT")
    print("-" * 50)

    for path in glob.glob(os.path.join(TEST_DIR, "*.*")):
        data = preprocess(path)
        if data is None: continue
        x = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            _, safety_logit = model(x)
            prob = torch.sigmoid(safety_logit).item()
        
        status = "SAFE" if prob > 0.7 else "UNSAFE"
        print(f"{os.path.basename(path):<30} | {prob*100:>6.1f}% | {status}")

if __name__ == "__main__":
    predict()