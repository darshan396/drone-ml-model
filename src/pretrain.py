import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast 
import glob
import os
import numpy as np
from tqdm import tqdm 
from preprocessing import preprocess
from unet import RoverLanding

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-3
BATCH_SIZE = 8        
EPOCHS = 30            
NUM_WORKERS = 4        
IMAGE_DIR = "../data/dataset/unlabeled_images"

class BigUnlabeledDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = glob.glob(os.path.join(image_dir, "*.*"))
        print(f"✅ Found {len(self.image_paths)} images in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        data = preprocess(path)
        
        if data is None: 
            return self.__getitem__((idx + 1) % len(self))
            
        return torch.from_numpy(data).permute(2, 0, 1).float()

def pretrain_large():
    torch.backends.cudnn.benchmark = True 
    
    model = RoverLanding(n_classes=4).to(DEVICE)
    dataset = BigUnlabeledDataset(image_dir=IMAGE_DIR)
    
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        drop_last=True
    )
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scaler = GradScaler() 

    print(f"🚀 Starting Large Scale Pre-training on {DEVICE}")
    print(f"   Images: {len(dataset)} | Batch: {BATCH_SIZE} | Epochs: {EPOCHS}")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        loop = tqdm(loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=True)

        for imgs in loop:
            imgs = imgs.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True) 

            with autocast():
                reconstruction, _ = model(imgs)
                loss = criterion(reconstruction, imgs)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            
            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(loader)
        print(f"   Avg Loss: {avg_loss:.4f}")
        
        torch.save(model.state_dict(), f"rover_10k_checkpoint.pth")

    print("✅ 10k Training Complete!")
    torch.save(model.state_dict(), "rover_pretrained_10k.pth")

if __name__ == "__main__":
    pretrain_large()