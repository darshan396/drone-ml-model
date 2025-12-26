import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

# Ensure src modules are found
sys.path.append(os.getcwd())
# FIX: Changed 'src.dataset' to 'dataset' if files are in root, 
# or 'src.unet' based on your uploaded file structure.
# Assuming standard structure:
from src.dataset import RoverDataset
from src.unet import RoverLanding

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-4
BATCH_SIZE = 4
EPOCHS = 40  

# ---------------- PATH FIXES ----------------
# FIX 1: Match the path used in train_adaboost.py
DATASET_ROOT = "data/dataset"  

# FIX 2: Ensure checkpoints folder exists
CHECKPOINT_DIR = "src" 
PRETRAINED_PATH = os.path.join(CHECKPOINT_DIR, "rover_pretrained_10k.pth") # Matches pretrain.py output
SAVE_PATH = os.path.join(CHECKPOINT_DIR, "rover_model_latest.pth")

def train():
    print(f"🚀 Initializing Training on {DEVICE}...")
    
    # 1. Verify Paths Before Starting
    train_dir = os.path.join(DATASET_ROOT, "train")
    mask_check = os.path.join(train_dir, "masks")
    
    if not os.path.exists(mask_check):
        print(f"❌ CRITICAL ERROR: Mask folder not found at {mask_check}")
        print("   -> Your model will learn NOTHING without masks.")
        print("   -> Check your folder structure: is it 'dataset/' or 'data/dataset/'?")
        return

    # 2. Initialize Model
    model = RoverLanding(n_classes=4).to(DEVICE)

    # Load Pre-trained Weights
    if os.path.exists(PRETRAINED_PATH):
        print(f"✅ Loading Geological Intuition from {PRETRAINED_PATH}...")
        try:
            pretrained_dict = torch.load(PRETRAINED_PATH, map_location=DEVICE)
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                               if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        except:
            print("⚠️ Warning: Weights mismatch. Training from scratch.")
    else:
        print(f"⚠️ Warning: {PRETRAINED_PATH} not found. Training from scratch.")

    # 3. Dataset
    dataset = RoverDataset(
        image_dir=os.path.join(DATASET_ROOT, "unlabeled_images"), 
        mask_dir=os.path.join(train_dir, "masks"),
        label_file=os.path.join(train_dir, "labels.csv")
    )

    if len(dataset) == 0:
        print("❌ Error: Dataset is empty."); return

    print(f"✅ Found {len(dataset)} training samples.")
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # --- PARANOIA WEIGHTS (To fix Purple Map) ---
    # Background: 0.1, Hazards: 10.0
    class_weights = torch.tensor([0.1, 10.0, 10.0, 10.0]).to(DEVICE)
    
    criterion_seg = nn.CrossEntropyLoss(weight=class_weights)
    criterion_safety = nn.MSELoss() 

    model.train()
    if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        
        for batch_idx, (images, masks, labels) in enumerate(loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1) 

            optimizer.zero_grad()
            seg_logits, safety_score = model(images)
            
            # Loss Calculation
            loss_seg = criterion_seg(seg_logits, masks)
            loss_safety = criterion_safety(torch.sigmoid(safety_score), labels)
            
            total_loss = loss_seg + (5.0 * loss_safety)

            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / len(loader)
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {avg_loss:.4f}")
        
        torch.save(model.state_dict(), SAVE_PATH)

    print(f"✅ Training finished. Model saved to {SAVE_PATH}")
    
if __name__ == "__main__":
    train()