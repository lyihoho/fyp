import cv2
import numpy as np
import os

def strip_light_watermarks(image_path, output_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return
    
    # -------------------------------------------------------------------------
    # 🎯 TARGETED BINARY THRESHOLD CONVERSION
    # -------------------------------------------------------------------------
    # Replacing the convertScaleAbs linear math with a definitive binary split.
    # Any pixel lighter than 195 (the watermark lines) flatlines to pure white (255).
    # Only pristine, dark machine-printed text survives the cutoff.
    threshold_value = 150
    _, cleaned = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cleaned)
    print(f"✨ Cleaned watermark from: {os.path.basename(image_path)}")

# Run it across your 10 synthetic downloads
SYNTHETIC_DIR = "data/raw_images/train3_synthetic_10"
CLEANED_DIR = "data/raw_images/synthetic3_10_cleaned"

if os.path.exists(SYNTHETIC_DIR):
    for f in os.listdir(SYNTHETIC_DIR):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            strip_light_watermarks(os.path.join(SYNTHETIC_DIR, f), os.path.join(CLEANED_DIR, f))
else:
    print(f"⚠️ Directory '{SYNTHETIC_DIR}' not found. Please verify your data directory path layout.")