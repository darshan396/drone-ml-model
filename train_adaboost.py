import torch
import numpy as np
import os
import sys
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
import joblib

# Ensure we look in the current directory for modules
sys.path.append(os.getcwd())
from src.unet import RoverLanding
from src.preprocessing import preprocess

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# FIX: Point to the 'checkpoints' folder, NOT 'src'
UNET_PATH = "src/rover_model_latest.pth" 
ADABOOST_PATH = "src/rover_adaboost_pure.pkl"
DATASET_ROOT = "data/dataset"

def train_booster():
    print(f"🚀 Loading U-Net Brain from {UNET_PATH}...")
    
    # 1. Load U-Net
    model = RoverLanding(n_classes=4).to(DEVICE)
    
    if not os.path.exists(UNET_PATH):
        print(f"❌ Error: File not found at {UNET_PATH}")
        print("   -> Did you run 'python train.py'?")
        return

    model.load_state_dict(torch.load(UNET_PATH, map_location=DEVICE))
    model.eval()

    # 2. Extract Features
    print("🔍 Extracting physics features from training set...")
    labels_path = os.path.join(DATASET_ROOT, "train", "labels.csv")
    
    if not os.path.exists(labels_path):
        print(f"❌ Error: Labels not found at {labels_path}")
        return

    df = pd.read_csv(labels_path)
    X, y = [], []
    
    for idx, row in df.iterrows():
        img_name = row['filename']
        label_score = float(row['safety_label'])
        
        # Find image
        img_path = os.path.join(DATASET_ROOT, "train", "images", img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(DATASET_ROOT, "unlabeled_images", img_name)
        
        if not os.path.exists(img_path): continue

        data = preprocess(img_path)
        if data is None: continue
        
        tensor_img = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            features = model.get_features(tensor_img)
        
        X.append(features)
        y.append(1 if label_score > 0.6 else 0)

    if len(X) == 0:
        print("❌ Error: No features extracted. Check your dataset paths.")
        return

    X = np.array(X)
    y = np.array(y)
    
    print(f"✅ Extracted features for {len(X)} images.")

    # 3. Train AdaBoost
    print("⚡ Training AdaBoost...")
    clf = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
    clf.fit(X, y)
    
    joblib.dump(clf, ADABOOST_PATH)
    print(f"✅ AdaBoost Model Saved: {ADABOOST_PATH}")

if __name__ == "__main__":
    train_booster()