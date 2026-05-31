# parsing.py
import cv2
import pytesseract
import pandas as pd
import numpy as np
import os
import re
import json
import ollama
from sqlalchemy import text
from database import SessionLocal, Receipt, Base, engine

Base.metadata.create_all(bind=engine)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
TESSERACT_CONFIG = r'--oem 3 --psm 4'

try:
    local_client = ollama.Client()
except Exception:
    local_client = None

def get_llm_parsing_fallback(raw_text_dump):
    if local_client is None:
        return {"merchant": "Unknown Store", "total": 0.0}
    prompt = f"### System Instruction:\nYou are a strict data extraction algorithm. Analyze the following Malaysian receipt text. Isolate the main store brand name and final net total paid cleanly.\n\nJSON Schema:\n{{\n    \"merchant\": \"Cleaned Store Name Only\",\n    \"total_amount\": float\n}}\n\n### Target Text:\n{raw_text_dump}"
    try:
        response = local_client.generate(model="qwen2.5:3b", prompt=prompt, options={'temperature': 0.0}, format='json')
        data = json.loads(response['response'])
        extracted_total = float(data.get("total_amount", 0.0))
        if extracted_total > 2000.00: extracted_total = 0.0
        return {"merchant": data.get("merchant", "Unknown Store"), "total": extracted_total}
    except Exception:
        return {"merchant": "Unknown Store", "total": 0.0}

def extract_structural_and_content_features(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
        
    h, w, _ = img.shape
    aspect_ratio = float(w / h) # Calculate structural shape factor
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    data = pytesseract.image_to_data(gray, config=TESSERACT_CONFIG, output_type=pytesseract.Output.DATAFRAME)
    data = data.dropna(subset=['text'])
    data = data[data['text'].astype(str).str.strip() != '']
    if data.empty: return None

    # Calculate Look & Feel distributions
    total_words = len(data)
    average_ocr_confidence = data['conf'].mean()
    word_boxes_area = (data['width'] * data['height']).sum()
    layout_density_ratio = float(word_boxes_area / (w * h))
    vertical_variance = data['top'].var() if len(data) > 1 else 0.0

    # Calculate Character Spacing Variance (Tampering indicator sub-metric)
    # Measures how erratic the horizontal gaps are between word bounding boxes
    if len(data) > 1:
        data_sorted = data.sort_values(by=['top', 'left'])
        horizontal_gaps = data_sorted['left'].diff().dropna()
        char_spacing_variance = float(horizontal_gaps.var()) if len(horizontal_gaps) > 1 else 0.0
    else:
        char_spacing_variance = 0.0

    all_text_string = " ".join(data['text'].astype(str).tolist())

    # Fixed grid line reconstruction
    data = data.copy()
    data['line_group'] = data['top'] // 15
    lines = []
    for g in sorted(data['line_group'].unique()):
        line_str = " ".join(data[data['line_group'] == g].sort_values(by='left')['text'].astype(str).tolist())
        lines.append(line_str)
    full_text_dump = "\n".join(lines)

    # Price and math parsing calculations
    detected_total = 0.0
    price_matches = re.findall(r'\d+[\.,]\s*\d{2}', all_text_string)
    math_valid = 1
    
    if price_matches:
        cleaned_prices = []
        for p in price_matches:
            try:
                num = float(re.sub(r'[^\d\.,]', '', p).replace(',', '.'))
                if num < 2000.00: cleaned_prices.append(num) 
            except ValueError: continue
        if cleaned_prices:
            detected_total = max(cleaned_prices[-4:])
            # Quick Cross-Field Math Validation Sub-metric: 
            # If the max price isn't roughly equal to the sum of items or balances, flag it
            if len(cleaned_prices) >= 3:
                sorted_prices = sorted(cleaned_prices)
                if abs(sorted_prices[-1] - (sorted_prices[-2] + sorted_prices[-3])) > 50.00:
                    math_valid = 0 # Outrageous price math layout gap detected!

    # Merchant extraction matching
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

    detected_date = "Unknown Date"
    date_pattern = r'\b(\d{2}[/\.\-]\d{2}[/\.\-]\d{4})|(\d{4}[/\.\-]\d{2}[/\.\-]\d{2})|(\d{2}[/\.\-]\d{2}[/\.\-]\d{2})\b'
    date_matches = re.findall(date_pattern, all_text_string)
    if date_matches:
        found_date = [match for group in date_matches for match in group if match]
        if found_date: detected_date = found_date[0]

    if detected_total == 0.0 or detected_store_name == "Unknown Store" or len(detected_store_name) < 4 or "krispy" in all_text_string.lower():
        ai_corrections = get_llm_parsing_fallback(full_text_dump)
        if detected_total == 0.0: detected_total = ai_corrections.get("total", 0.0)
        if detected_store_name == "Unknown Store" or len(detected_store_name) < 4 or "krispy" in all_text_string.lower():
            detected_store_name = ai_corrections.get("merchant", "Unknown Store")

    return {
        "filename": os.path.basename(image_path),
        "extracted_store": detected_store_name,
        "extracted_total": detected_total,
        "date": detected_date,
        "layout_density_ratio": round(layout_density_ratio, 5),
        "word_count": total_words,
        "line_count": len(lines),  
        "vertical_alignment_variance": round(vertical_variance, 2),
        "avg_ocr_confidence": round(average_ocr_confidence, 2),
        "aspect_ratio": round(aspect_ratio, 4),
        "math_valid": math_valid,
        "char_spacing_variance": round(char_spacing_variance, 2)
    }

def run_real_use_feature_pipeline(images_dir, output_csv):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    valid_exts = ('.png', '.jpg', '.jpeg')
    if not os.path.exists(images_dir): return

    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(valid_exts)])
    master_feature_pool = []
    db_session = SessionLocal()

    try:
        db_session.query(Receipt).delete()
        db_session.commit()
        try:
            db_session.execute(text("DELETE FROM sqlite_sequence WHERE name='receipts';"))
            db_session.commit()
        except Exception: db_session.rollback()

        for filename in image_files:
            full_img_path = os.path.join(images_dir, filename)
            extracted_row = extract_structural_and_content_features(full_img_path)
            
            if extracted_row:
                master_feature_pool.append(extracted_row)
                print(f"Parsing: {filename} | Aspect Ratio: {extracted_row['aspect_ratio']} | Math Valid: {extracted_row['math_valid']}")
                
                new_receipt = Receipt(
                    filename=extracted_row['filename'],
                    merchant=extracted_row['extracted_store'],
                    date=extracted_row['date'],
                    total_amount=float(extracted_row['extracted_total']),
                    receipt_length=int(extracted_row['word_count']), 
                    num_lines=int(extracted_row['line_count']),
                    layout_density_ratio=float(extracted_row['layout_density_ratio']),
                    vertical_alignment_variance=float(extracted_row['vertical_alignment_variance']),
                    avg_ocr_confidence=float(extracted_row['avg_ocr_confidence']),
                    aspect_ratio=float(extracted_row['aspect_ratio']),
                    math_valid_flag=int(extracted_row['math_valid']),
                    character_spacing_var=float(extracted_row['char_spacing_variance'])
                )
                db_session.add(new_receipt)
        db_session.commit()
        print("\n💾 [SQLITE SUCCESS] Extended look-and-feel data stored cleanly.")
    except Exception as e:
        db_session.rollback()
        print(f"❌ Error: {str(e)}")
    finally: db_session.close()

    if master_feature_pool:
        pd.DataFrame(master_feature_pool).to_csv(output_csv, index=False)

if __name__ == "__main__":
    RAW_IMAGES_FOLDER = "data/processed_data/combined_train"
    ANOMALY_INPUT_CSV = "data/anomaly_detection_input.csv"
    run_real_use_feature_pipeline(RAW_IMAGES_FOLDER, ANOMALY_INPUT_CSV)