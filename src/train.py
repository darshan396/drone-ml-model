import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

# Ensure src modules are found
sys.path.append(os.getcwd())
from dataset import RoverDataset
from unet import RoverLanding

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-4
BATCH_SIZE = 4
EPOCHS = 50  # Increased from 40 to give it more time to learn

# Paths
DATASET_ROOT = "../data/dataset"
PRETRAINED_PATH = ("rover_pretrained_10k.pth")
SAVE_PATH = ("rover_model_latest.pth")

def train():
    print(f"🚀 Initializing Training on {DEVICE}...")
    
    model = RoverLanding(n_classes=4).to(DEVICE)

    # Load Pre-trained Weights
    if os.path.exists(PRETRAINED_PATH):
        print(f"✅ Loading Geological Intuition from {PRETRAINED_PATH}...")
        pretrained_dict = torch.load(PRETRAINED_PATH, map_location=DEVICE)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                           if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        print(f"⚠️ Warning: {PRETRAINED_PATH} not found.")

    # Dataset
    train_dir = os.path.join(DATASET_ROOT, "train")
    if not os.path.exists(train_dir):
        print(f"❌ Error: {train_dir} missing."); return

    dataset = RoverDataset(
        image_dir=os.path.join(DATASET_ROOT, "unlabeled_images"), 
        mask_dir=os.path.join(train_dir, "masks"),
        label_file=os.path.join(train_dir, "labels.csv")
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # --- FIX 1: RELAX THE PARANOIA ---
    # Old: [0.1, 2.0, 2.0, 2.0] -> Caused noise everywhere.
    # New: [0.5, 1.2, 1.2, 1.2] -> Tells model "Background is important too!"
    # This will stop it from marking grass/paint as "Death Rocks".
    class_weights = torch.tensor([0.5, 1.2, 1.2, 1.2]).to(DEVICE)
    criterion_seg = nn.CrossEntropyLoss(weight=class_weights)
    criterion_safety = nn.MSELoss() 

    model.train()

    print(f"   Training on {len(dataset)} samples for {EPOCHS} epochs...")

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
            
            # --- FIX 2: FORCE DECISIVENESS ---
            # Using Sigmoid to ensure 0-1 range match
            safety_prob = torch.sigmoid(safety_score)
            loss_safety = criterion_safety(safety_prob, labels)
            
            # Increased Multiplier from 10.0 -> 25.0
            # This forces the model to prioritize the final "Safe/Unsafe" score
            # over drawing perfect pixel maps.
            total_loss = loss_seg + (25.0 * loss_safety)

            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / len(loader)
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {avg_loss:.4f}")
        
        torch.save(model.state_dict(), SAVE_PATH)

    print(f"✅ Training finished. Optimized model saved to {SAVE_PATH}")
    
if __name__ == "__main__":
    train()