import os
import sys
import cv2
import numpy as np
import base64

# DEEP LEARNING TEXT COMPARISON
def calculate_text_similarity(text1, text2):
    """
    Computes a clean similarity percentage (0% to 100%) between two text blocks
    using the Levenshtein distance ratio. Perfect for catching textual duplicates.
    """
    str1 = "".join(text1.lower().split())
    str2 = "".join(text2.lower().split())
    
    if not str1 or not str2:
        return 0.0
        
    if str1 == str2:
        return 100.0

    if len(str1) < len(str2):
        str1, str2 = str2, str1
        
    distances = range(len(str1) + 1)
    for i2, c2 in enumerate(str2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(str1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
        
    lev_dist = distances[-1]
    max_len = max(len(str1), len(str2))
    
    return (1.0 - (lev_dist / max_len)) * 100

# COMPUTER VISION ORB CODES & DATABASE SERIALIZERS (PHYSICAL FOLDS)
def serialize_descriptors(descriptors):
    """Converts raw OpenCV ORB matrices to a base64 text string for SQLite."""
    if descriptors is None:
        return ""
    binary_data = descriptors.tobytes()
    text_string = base64.b64encode(binary_data).decode('utf-8')
    return text_string

def deserialize_descriptors(text_string):
    """Rebuilds the absolute binary matrix OpenCV needs from an SQLite string."""
    if not text_string:
        return None
    binary_data = base64.b64decode(text_string.encode('utf-8'))
    descriptors = np.frombuffer(binary_data, dtype=np.uint8).reshape(-1, 32)
    return descriptors

def check_physical_duplicate(img_path1, img_path2):
    """Compares the active physical micro-shadow landmarks between two images."""
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        return 0, 0

    orb = cv2.ORB_create(nfeatures=1500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0, len(kp1) if kp1 else 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = [m for m in matches if m.distance < 40]
    
    return len(good_matches), len(kp1)


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    TARGET_IMG_DIR = os.path.join(BASE_DIR, "data", "processed_data", "combined_train")
    TARGET_TEXT_DIR = os.path.join(BASE_DIR, "data", "extracted_text", "extracted_combined_train")
    
    file1_img = os.path.join(TARGET_IMG_DIR, "r_01.jpeg")
    file2_img = os.path.join(TARGET_IMG_DIR, "r_03.jpeg") 
    
    file1_txt = os.path.join(TARGET_TEXT_DIR, "r_01.txt")
    file2_txt = os.path.join(TARGET_TEXT_DIR, "r_03.txt") 

    print("="*80)
    print("🚀 Running Comprehensive Dual-Engine (Textual + Geometric) Verification...")
    print("="*80)
    
    # 1. Text Check Gateway
    if os.path.exists(file1_txt) and os.path.exists(file2_txt):
        with open(file1_txt, "r", encoding="utf-8") as f: text_receipt_1 = f.read()
        with open(file2_txt, "r", encoding="utf-8") as f: text_receipt_2 = f.read()
        
        match_percentage = calculate_text_similarity(text_receipt_1, text_receipt_2)
        print(f"Evaluated Text-Signature Overlap: {match_percentage:.2f}%")
    else:
        print("Skipping text verification (.txt logs missing)")

    print("-"*20)

    # 2. Geometric Folds Gate
    if os.path.exists(file1_img) and os.path.exists(file2_img):
        good_matches, total_kp = check_physical_duplicate(file1_img, file2_img)
        print(f"Base Image Features Found: {total_kp}")
        print(f"Cross-Matched Keypoints   : {good_matches}")
        
        if good_matches > 50:
            print("ALERT: High physical alignment! Exact same paper creases or angles.")
        else:
            print("SAFE: Distinct physical surfaces.")
    else:
        print("Skipping geometric validation (Image assets missing)")
        
    print("="*20)