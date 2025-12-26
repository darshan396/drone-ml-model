import cv2
import numpy as np
import os

def preprocess(image_path, img_size=(256, 256)):
    """
    Converts image to 4-channel Physics Tensor:
    Ch1: Normalized Intensity (Albedo)
    Ch2: Sobel Gradient (Roughness)
    Ch3: Laplacian (Fractures/Edges)
    Ch4: Local Variance (Granularity)
    """
    if not os.path.exists(image_path):
        return None

    try:
        # Load image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None: 
            return None
            
        img = cv2.resize(img, img_size)
        img = img.astype(np.float32)

        # 1. Normalization (Albedo)
        mu = cv2.GaussianBlur(img, (31, 31), 0)
        sigma = cv2.GaussianBlur((img - mu) ** 2, (31, 31), 0)
        sigma = np.sqrt(sigma) + 1e-6
        norm = (img - mu) / sigma

        # 2. Gradient Magnitude (Roughness)
        gx = cv2.Sobel(norm, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(norm, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx**2 + gy**2)

        # 3. Laplacian (Fractures)
        lap = cv2.Laplacian(norm, cv2.CV_32F, ksize=3)

        # 4. Local Variance (Texture)
        mean = cv2.GaussianBlur(norm, (15, 15), 0)
        sq_mean = cv2.GaussianBlur(norm**2, (15, 15), 0)
        var = sq_mean - mean**2

        # Stack into (H, W, 4) tensor
        # We generally expect (H, W, C) here, usually 256x256x4
        return np.stack([norm, grad, lap, var], axis=-1).astype(np.float32)

    except Exception as e:
        print(f"Error preprocessing {image_path}: {e}")
        return None