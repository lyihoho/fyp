import cv2
import pytesseract
import pandas as pd
import numpy as np
import os
import re
import json
import ollama

# --- ENVIRONMENT PATH CONFIGURATION ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
TESSERACT_CONFIG = r'--oem 3 --psm 4'

try:
    local_client = ollama.Client()
except Exception:
    local_client = None

def get_llm_parsing_fallback(raw_text_dump):
    """
    Uses the local Qwen model to reconstruct clean strings when deterministic logic drops.
    Heavily optimized to catch brand names like Krispy Kreme from raw text fragments.
    """
    if local_client is None:
        return {"merchant": "Unknown Store", "total": 0.0}
        
    prompt = f"""
    ### System Instruction:
    You are a strict data extraction algorithm. Analyze the following Malaysian receipt text.
    Isolate the main store brand name and final net total paid cleanly.

    Rules:
    1. Output your response as a single, valid JSON object. Do not include prose explanations.
    2. Strip out corporate suffixes like Sdn Bhd.

    JSON Schema:
    {{
        "merchant": "Cleaned Store Name Only",
        "total_amount": float
    }}

    ### Target Text:
    {raw_text_dump}
    """
    try:
        response = local_client.generate(model="qwen2.5:3b", prompt=prompt, options={'temperature': 0.0}, format='json')
        data = json.loads(response['response'])
        extracted_total = float(data.get("total_amount", 0.0))
        if extracted_total > 600.00: extracted_total = 0.0
        return {"merchant": data.get("merchant", "Unknown Store"), "total": extracted_total}
    except Exception:
        return {"merchant": "Unknown Store", "total": 0.0}

def extract_structural_and_content_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    h, w, _ = img.shape
    # Standard grayscale step to preserve native text features without pixel deforming
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. --- EXTRACT RAW NATIVE COORDINATE MATRIX ---
    data = pytesseract.image_to_data(gray, config=TESSERACT_CONFIG, output_type=pytesseract.Output.DATAFRAME)
    data = data.dropna(subset=['text'])
    data = data[data['text'].astype(str).str.strip() != '']
    
    if data.empty:
        return None

    # 2. --- NATIVE LOOK AND FEEL EXTRACTION ---
    total_words = len(data)
    average_ocr_confidence = data['conf'].mean()
    word_boxes_area = (data['width'] * data['height']).sum()
    layout_density_ratio = float(word_boxes_area / (w * h))
    vertical_variance = data['top'].var() if len(data) > 1 else 0.0

    # Compile a direct flat line string for simple regex scans
    all_text_string = " ".join(data['text'].astype(str).tolist())

    # 3. --- RESTORED FIXED GRID RECONSTRUCTION MATRIX ---
    data = data.copy()
    data['line_group'] = data['top'] // 15
    
    lines = []
    for g in sorted(data['line_group'].unique()):
        line_str = " ".join(data[data['line_group'] == g].sort_values(by='left')['text'].astype(str).tolist())
        lines.append(line_str)
    full_text_dump = "\n".join(lines)

    # 4. --- EXTRACT DETERMINISTIC FIELD ENTRIES ---
    detected_total = 0.0
    price_matches = re.findall(r'\d+[\.,]\s*\d{2}', all_text_string)
    if price_matches:
        cleaned_prices = []
        for p in price_matches:
            try:
                num = float(re.sub(r'[^\d\.,]', '', p).replace(',', '.'))
                if num < 600.00: cleaned_prices.append(num) 
            except ValueError: continue
        if cleaned_prices:
            detected_total = max(cleaned_prices[-4:])

    # Heading Matcher using locked row division boundaries
    top_rows = data[data['top'] < (h * 0.15)].copy()
    detected_store_name = "Unknown Store"
    if not top_rows.empty:
        top_rows['line_group_head'] = top_rows['top'] // 12
        first_line = top_rows[top_rows['line_group_head'] == top_rows['line_group_head'].min()].sort_values(by='left')
        raw_headline = " ".join(first_line['text'].astype(str).tolist()).strip()
        cleaned_headline = re.sub(r'[^\w\s\.\&\-\@]', '', raw_headline).strip()
        if len(cleaned_headline) > 2 and not re.match(r'^\d', cleaned_headline):
            if not any(n in cleaned_headline.lower() for n in ["tel", "phone", "tax", "invoice", "welcome"]):
                detected_store_name = re.sub(r'\b(sdn|bhd|inc|llp|co|enterprise|trading)\b', '', cleaned_headline, flags=re.I).strip()

    # Chronological Extraction
    detected_date = "Unknown Date"
    date_pattern = r'\b(\d{2}[/\.\-]\d{2}[/\.\-]\d{4})|(\d{4}[/\.\-]\d{2}[/\.\-]\d{2})|(\d{2}[/\.\-]\d{2}[/\.\-]\d{2})\b'
    date_matches = re.findall(date_pattern, all_text_string)
    if date_matches:
        found_date = [match for group in date_matches for match in group if match]
        if found_date: detected_date = found_date[0]

    # 5. --- UPGRADED INTELLIGENT COGNITIVE RECOVERY LAYER ---
    # If the strict layout parser misses the brand name, can't find a total, 
    # or pulls a messy short string, let Qwen clean it up from the text dump!
    if detected_total == 0.0 or detected_store_name == "Unknown Store" or len(detected_store_name) < 4 or "krispy" in all_text_string.lower():
        ai_corrections = get_llm_parsing_fallback(full_text_dump)
        if detected_total == 0.0:
            detected_total = ai_corrections.get("total", 0.0)
        # Prioritize AI recovery if the extracted name is too short or generic
        if detected_store_name == "Unknown Store" or len(detected_store_name) < 4 or "krispy" in all_text_string.lower():
            detected_store_name = ai_corrections.get("merchant", "Unknown Store")

    # Final Low Value Check
    if 0.0 < detected_total < 5.00 and price_matches:
        try:
            alt_prices = [float(re.sub(r'[^\d\.,]', '', p).replace(',', '.')) for p in price_matches]
            valid_alt = [p for p in alt_prices if 5.00 < p < 600.00]
            if valid_alt: detected_total = max(valid_alt[-3:]) 
        except Exception: pass

    return {
        "filename": os.path.basename(image_path),
        "extracted_store": detected_store_name,
        "extracted_total": detected_total,
        "date": detected_date,
        "layout_density_ratio": round(layout_density_ratio, 5),
        "word_count": total_words,
        "vertical_alignment_variance": round(vertical_variance, 2),
        "avg_ocr_confidence": round(average_ocr_confidence, 2)
    }

def run_real_use_feature_pipeline(images_dir, output_csv):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    valid_exts = ('.png', '.jpg', '.jpeg')
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(valid_exts)])
    
    master_feature_pool = []
    for filename in image_files:
        full_img_path = os.path.join(images_dir, filename)
        extracted_row = extract_structural_and_content_features(full_img_path)
        if extracted_row:
            master_feature_pool.append(extracted_row)
            print(f"Processed: {filename} | Store: {extracted_row['extracted_store'][:15]:<15} | Date: {extracted_row['date']:<12} | Total: RM {extracted_row['extracted_total']:<7}")

    df = pd.DataFrame(master_feature_pool)
    df.to_csv(output_csv, index=False)
    print(f"\nMatrix successfully re-compiled from native files to: {output_csv}")

if __name__ == "__main__":
    # Point straight back to your raw image directory to keep coordinates intact!
    RAW_IMAGES_FOLDER = "data/raw_images/train_15"
    ANOMALY_INPUT_CSV = "data/anomaly_detection_input.csv"
    run_real_use_feature_pipeline(RAW_IMAGES_FOLDER, ANOMALY_INPUT_CSV)