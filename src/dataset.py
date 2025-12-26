import os
import torch
import cv2
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset
from preprocessing import preprocess  

class RoverDataset(Dataset):
    def __init__(self, image_dir, mask_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.labels_df = pd.read_csv(label_file)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_name = row['filename']
        safety_label = row['safety_label']
        
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # 1. Load Data (Physics Tensor: Albedo, Roughness, Fractures, Texture)
        processed_data = preprocess(img_path)
        if processed_data is None:
            return self.__getitem__((idx + 1) % len(self))

        # --- NEW: PHYSICS ROBUSTNESS AUGMENTATION ---
        # Since processed_data is float32 (normalized), we use numpy math.
        
        # A. Sensor Noise Injection (Simulate camera grain)
        if random.random() > 0.5:
            noise = np.random.normal(0, 0.05, processed_data.shape).astype(np.float32)
            processed_data = processed_data + noise

        # B. Signal Scaling (Simulate varying contrast/surface reflectivity)
        if random.random() > 0.5:
            # Scale intensity by ±10%
            alpha = random.uniform(0.9, 1.1) 
            processed_data = processed_data * alpha
        # ---------------------------------------------

        # Convert to Tensor (Channels, Height, Width)
        image = torch.from_numpy(processed_data).permute(2, 0, 1).float()

        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, 0)
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            mask = torch.from_numpy(mask).long()
        else:
            mask = torch.zeros((256, 256)).long()

        # --- GEOMETRIC AUGMENTATION (Existing) ---
        # Randomly flip horizontal
        if random.random() > 0.5:
            image = torch.flip(image, [2])
            mask = torch.flip(mask, [1])

        # Randomly flip vertical
        if random.random() > 0.5:
            image = torch.flip(image, [1])
            mask = torch.flip(mask, [0])
            
        # Random 90-degree rotation
        k = random.randint(0, 3)
        image = torch.rot90(image, k, [1, 2])
        mask = torch.rot90(mask, k, [0, 1])
        # -----------------------------------------

        label = torch.tensor(safety_label, dtype=torch.float32)

        return image, mask, label