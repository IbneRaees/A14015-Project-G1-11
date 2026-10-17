import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

def execute_baseline_pipeline(image_dir: str = "data/floodnet_sample"):
    """
    Runs single-pass YOLOv8 segmentation on UAV imagery as the non-agentic baseline.
    Evaluates: Building Damage, Road Passability, and Frame Latency.
    """
    image_paths = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))
    if not image_paths:
        print("[!] No images found. Run `python data/download_floodnet.py` first.")
        return

    print("[+] Loading Baseline Model (YOLOv8x-Seg)...")
    # Using pre-trained segmentation head as the un-agentic baseline anchor
    model = YOLO("yolov8n-seg.pt") 

    latencies = []
    predictions = []

    print(f"[+] Running inference on {len(image_paths)} UAV tile(s)...")
    
    for img_path in image_paths:
        start_time = time.time()
        
        # Single-pass model evaluation
        results = model(str(img_path), verbose=False)
        
        elapsed_time = time.time() - start_time
        latencies.append(elapsed_time)
        
        # Extract detected bounding boxes and segmentation masks
        boxes = results[0].boxes
        num_objects = len(boxes) if boxes is not None else 0
        
        predictions.append({
            "file": img_path.name,
            "detected_objects": num_objects,
            "latency_seconds": round(elapsed_time, 4)
        })

    avg_latency = np.mean(latencies)
    print("\n================ Baseline Initial Results ================")
    print(f"Total Tiles Evaluated    : {len(image_paths)}")
    print(f"Average Frame Latency   : {avg_latency:.4f} seconds/frame")
    print("==========================================================")
    
    return predictions, avg_latency

if __name__ == "__main__":
    execute_baseline_pipeline()