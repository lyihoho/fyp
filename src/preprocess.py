import cv2
import numpy as np
import os
import sys

def deskew_image(gray_img):
    """
    Computes the structural text line orientation angle using Hough Line Transform
    and applies an affine rotation matrix to align the canvas back to a true 0-degree baseline.
    """
    # Use Canny edge detection to highlight text contours
    edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
    
    # Run Hough Lines to isolate dominant horizontal linear orientations
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            # Filter out extreme vertical angles to focus purely on text row tilt
            if -45 < angle < 45:
                angles.append(angle)
                
    # If a valid skew angle is captured, rotate the entire document frame
    if len(angles) > 0:
        median_angle = np.median(angles)
        if abs(median_angle) > 0.5:  # Only rotate if the displacement is significant
            h, w = gray_img.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated_img = cv2.warpAffine(gray_img, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated_img
            
    return gray_img

def enhance_full_frame_with_geometry(image_path):
    """
    Applies an advanced multi-stage geometric and structural conditioning pipeline:
    Grayscale -> Multi-Line Deskewing -> Bi-Cubic Scale Standardization.
    Preserves organic textures while bringing characters to ideal OCR dimensions.
    """
    # Load raw input image matrix from the file path
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        print(f"  [WARNING] Unable to read image matrix: {image_path}")
        return None
        
    # --- STEP 1: CHROMATIC REDUCTION (GRAYSCALE) ---
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    
    # --- STEP 2: MULTI-LINE ORIENTATION DESKEWING (UPGRADED) ---
    # Dynamically straightens slanted or tilted handheld captures
    deskewed = deskew_image(gray)

    # --- STEP 3: BI-CUBIC RESIZING AND SPATIAL SCALE STANDARDIZATION (UPGRADED) ---
    # Standardizes the input canvas to an optimal resolution scale for OCR text recognition.
    # We upscale the image smoothly by a scale factor of 1.5x using high-quality cubic interpolation.
    h, w = deskewed.shape[:2]
    target_width = int(w * 1.5)
    target_height = int(h * 1.5)
    resized_grayscale = cv2.resize(deskewed, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    
    return resized_grayscale

def execute_batch_processing_pipeline(input_folder, output_folder):
    """
    Traverses the specified source directory, executes the full geometric preprocessing
    suite on valid image matrices, and exports uniform grayscale assets.
    """
    if not os.path.exists(input_folder):
        print(f"[FATAL DIRECTORY ERROR] Source directory not found at: {input_folder}")
        sys.exit(1)
        
    os.makedirs(output_folder, exist_ok=True)
    
    valid_extensions = (".png", ".jpg", ".jpeg")
    files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)])
    
    print("="*80)
    print(f"PIPELINE INITIATED: Processing {len(files)} source documents...")
    print(f"Executing: Grayscale -> Hough Deskew -> Bi-Cubic Resizing (Geometry Active)")
    print("="*80)
    
    for idx, filename in enumerate(files):
        source_path = os.path.join(input_folder, filename)
        destination_path = os.path.join(output_folder, filename)
        
        output_matrix = enhance_full_frame_with_geometry(source_path)
        
        if output_matrix is not None:
            cv2.imwrite(destination_path, output_matrix)
            print(f" [{idx+1}/{len(files)}] Geometrically Standardized: {filename}")
            
    print("\n" + "="*80)
    print(f"SUCCESS: Batch preprocessing and spatial alignment finalized.")
    print(f"Aligned and standardized grayscale images exported to: {output_folder}")
    print("="*80)

if __name__ == "__main__":
    INPUT_DIR_TARGET = "data/raw_images/train_15"
    OUTPUT_DIR_TARGET = "data/processed_data/train_15"
    
    execute_batch_processing_pipeline(INPUT_DIR_TARGET, OUTPUT_DIR_TARGET)