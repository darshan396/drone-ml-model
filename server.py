import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import torch
import sqlite3
import shutil
import os
import sys
import datetime
import cv2
import numpy as np

# --- CONFIGURATION ---
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from unet import RoverLanding
    from src.preprocessing import preprocess
    from config.config import CHECKPOINT_DIR, LOG_DIR, PROJECT_ROOT
except ImportError:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DB_PATH = os.path.join(LOG_DIR, "rover_history.db")
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")
# Filename defined here for easy updates
MODEL_FILENAME = "rover_pretrained_final.pth"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")

# --- DATABASE ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS inspections 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, score REAL, status TEXT, date TEXT)''')
    conn.commit()
    conn.close()

init_db()

# --- MODEL LOADING ---
def load_model():
    # Priority 1: Checkpoints folder
    path = os.path.join(CHECKPOINT_DIR, MODEL_FILENAME)
    
    # Priority 2: Root folder (fallback)
    if not os.path.exists(path): 
        print(f"⚠️ Warning: Model not found in {CHECKPOINT_DIR}. Checking root...")
        path = MODEL_FILENAME
    
    if os.path.exists(path):
        print(f"✅ AI Engine: Loading model from {path}...")
        try:
            model = RoverLanding(n_classes=4).to(DEVICE)
            model.load_state_dict(torch.load(path, map_location=DEVICE), strict=False)
            model.eval()
            return model
        except Exception as e:
            print(f"❌ Error loading weights: {e}")
            return None
    else:
        print(f"❌ AI Engine: Model weights ({MODEL_FILENAME}) NOT FOUND.")
        return None

model = load_model()

# --- HELPER: ROAD DETECTION HEURISTIC ---
def detect_road_markings(image_path):
    """
    Classical CV to detect lane markings.
    Returns a 'Safety Boost' factor if yellow/white lines are found.
    """
    try:
        img = cv2.imread(image_path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 1. Detect Yellow (Lane Paint)
        lower_yellow = np.array([15, 80, 80])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 2. Detect White (Lane Paint) - High Brightness, Low Saturation
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 50, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        # Combine masks
        combined_lanes = cv2.bitwise_or(mask_yellow, mask_white)
        
        # Calculate percentage of image that is "Lane Marking"
        total_pixels = img.shape[0] * img.shape[1]
        lane_pixels = cv2.countNonZero(combined_lanes)
        ratio = lane_pixels / total_pixels

        # If > 1% of the image is lane markings, it's likely a road
        if ratio > 0.01: 
            return 0.4  # Boost safety score by 40%
        return 0.0
        
    except Exception as e:
        print(f"Road detection error: {e}")
        return 0.0

# --- ROUTES ---

@app.get("/")
def read_root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return "❌ Error: index.html not found."

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    if not model: return {"error": "Model not loaded. Check server logs."}

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_filename = f"img_{timestamp}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, clean_filename)
    
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 1. Deep Learning Analysis
    data = preprocess(filepath)
    if data is None: return {"error": "Processing failed"}

    x = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        seg_logits, safety_score = model(x)
        raw_prob = torch.sigmoid(safety_score).item()

    # 2. Hybrid Correction (The Road Patch)
    road_boost = detect_road_markings(filepath)
    final_prob = min(0.99, raw_prob + road_boost)

    # Debug log for verification
    if road_boost > 0:
        print(f"🛣️ Road detected! Score boosted: {raw_prob:.2f} -> {final_prob:.2f}")

    # Generate Visualization (Mask)
    mask = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    mask_filename = f"mask_{timestamp}_{file.filename}"
    mask_path = os.path.join(UPLOAD_FOLDER, mask_filename)
    
    color_mask = np.zeros((256, 256, 3), dtype=np.uint8)
    color_mask[mask != 0] = [20, 20, 255] # Red hazards
    cv2.imwrite(mask_path, color_mask)

    # 3. Final Decision
    score = round(final_prob * 100, 2)
    status = "SAFE" if final_prob > 0.7 else "UNSAFE"
    time_str = datetime.datetime.now().strftime("%H:%M:%S")
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO inspections (filename, score, status, date) VALUES (?, ?, ?, ?)",
                 (clean_filename, score, status, time_str))
    conn.commit()
    conn.close()

    return {"status": status, "score": score}

@app.get("/history")
def get_history():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT id, filename, score, status, date FROM inspections ORDER BY id DESC LIMIT 50").fetchall()
    conn.close()
    return [{"id": r[0], "filename": r[1], "score": r[2], "status": r[3], "date": r[4]} for r in rows]

@app.delete("/history/{item_id}")
def delete_history_item(item_id: int):
    print(f"🗑️ Deleting Item ID: {item_id}")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM inspections WHERE id=?", (item_id,))
    conn.commit()
    conn.close()
    return {"message": "Deleted"}

if __name__ == "__main__":
    print("🚀 Server is running! Go to http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)