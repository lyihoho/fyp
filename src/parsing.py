# parsing.py
import cv2
import pytesseract
import pandas as pd
import numpy as np
import os
import re
import json
import ollama

# --- IMPORT SQLALCHEMY OBJECTS AND TEXT WRAPPER ---
from sqlalchemy import text
from database import SessionLocal, Receipt, Base, engine

# Force table instantiation matrix verification checks straight away
Base.metadata.create_all(bind=engine)

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
    Heavily optimized to catch brand names from raw text fragments.
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
    3. Only extract the total amount. If there are words like "total" or "cash" or similar in one receipt, extract the amount beside "total".

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
        
        # Safe high-limit ceiling boundary cap verification step
        if extracted_total > 2000.00: 
            extracted_total = 0.0
        return {"merchant": data.get("merchant", "Unknown Store"), "total": extracted_total}
    except Exception:
        return {"merchant": "Unknown Store", "total": 0.0}

def extract_structural_and_content_features(image_path):
    """
    Core extraction function. Analyzes optical textures, character coordinates, 
    and layouts to generate structural metrics vectors natively.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. --- EXTRACT RAW NATIVE COORDINATE MATRIX ---
    data = pytesseract.image_to_data(gray, config=TESSERACT_CONFIG, output_type=pytesseract.Output.DATAFRAME)
    data = data.dropna(subset=['text'])
    data = data[data['text'].astype(str).str.strip() != '']
    
    if data.empty:
        return None

    # 2. --- LOOK AND FEEL GEOMETRIC CALCULATIONS ---
    total_words = len(data)
    average_ocr_confidence = data['conf'].mean()
    word_boxes_area = (data['width'] * data['height']).sum()
    layout_density_ratio = float(word_boxes_area / (w * h))
    vertical_variance = data['top'].var() if len(data) > 1 else 0.0

    all_text_string = " ".join(data['text'].astype(str).tolist())

    # 3. --- FIXED GRID LINE RECONSTRUCTION MATRIX ---
    data = data.copy()
    data['line_group'] = data['top'] // 15
    
    lines = []
    for g in sorted(data['line_group'].unique()):
        line_str = " ".join(data[data['line_group'] == g].sort_values(by='left')['text'].astype(str).tolist())
        lines.append(line_str)
    full_text_dump = "\n".join(lines)

    # 4. --- AMOUNT AND MERCHANT REGEX PATTERN SEARCHING ---
    detected_total = 0.0
    price_matches = re.findall(r'\d+[\.,]\s*\d{2}', all_text_string)
    if price_matches:
        cleaned_prices = []
        for p in price_matches:
            try:
                num = float(re.sub(r'[^\d\.,]', '', p).replace(',', '.'))
                if num < 2000.00: 
                    cleaned_prices.append(num) 
            except ValueError: 
                continue
        if cleaned_prices:
            detected_total = max(cleaned_prices[-4:])

    # Heading Matcher using upper bounding boundary thresholds
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

    # Date Chronological Extraction
    detected_date = "Unknown Date"
    date_pattern = r'\b(\d{2}[/\.\-]\d{2}[/\.\-]\d{4})|(\d{4}[/\.\-]\d{2}[/\.\-]\d{2})|(\d{2}[/\.\-]\d{2}[/\.\-]\d{2})\b'
    date_matches = re.findall(date_pattern, all_text_string)
    if date_matches:
        found_date = [match for group in date_matches for match in group if match]
        if found_date: 
            detected_date = found_date[0]

    # 5. --- COGNITIVE AI RECOVERY FALLBACK LAYER ---
    if detected_total == 0.0 or detected_store_name == "Unknown Store" or len(detected_store_name) < 4 or "krispy" in all_text_string.lower():
        ai_corrections = get_llm_parsing_fallback(full_text_dump)
        if detected_total == 0.0:
            detected_total = ai_corrections.get("total", 0.0)
        if detected_store_name == "Unknown Store" or len(detected_store_name) < 4 or "krispy" in all_text_string.lower():
            detected_store_name = ai_corrections.get("merchant", "Unknown Store")

    # Final Low Value Check
    if 0.0 < detected_total < 5.00 and price_matches:
        try:
            alt_prices = [float(re.sub(r'[^\d\.,]', '', p).replace(',', '.')) for p in price_matches]
            valid_alt = [p for p in alt_prices if 5.00 < p < 2000.00]
            if valid_alt: 
                detected_total = max(valid_alt[-3:]) 
        except Exception: 
            pass

    return {
        "filename": os.path.basename(image_path),
        "extracted_store": detected_store_name,
        "extracted_total": detected_total,
        "date": detected_date,
        "layout_density_ratio": round(layout_density_ratio, 5),
        "word_count": total_words,
        "line_count": len(lines),  
        "vertical_alignment_variance": round(vertical_variance, 2),
        "avg_ocr_confidence": round(average_ocr_confidence, 2)
    }

def run_real_use_feature_pipeline(images_dir, output_csv):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    valid_exts = ('.png', '.jpg', '.jpeg')
    
    if not os.path.exists(images_dir):
        print(f"❌ Error: Image folder directory '{images_dir}' not found.")
        return

    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(valid_exts)])
    master_feature_pool = []
    
    db_session = SessionLocal()
    print("🌲 [PIPELINE START] Instantiating ingestion loop directly into SQLAlchemy layout...")

    try:
        # --- ⚡ FRESH PURGE COMPLETE DATABASE ---
        db_session.query(Receipt).delete()
        db_session.commit()
        try:
            db_session.execute(text("DELETE FROM sqlite_sequence WHERE name='receipts';"))
            db_session.commit()
        except Exception:
            db_session.rollback()
            
        print("🧹 [DATABASE FLUSHED] Old database tables purged and sequence tracking reset back to 1.")

        for filename in image_files:
            full_img_path = os.path.join(images_dir, filename)
            extracted_row = extract_structural_and_content_features(full_img_path)
            
            if extracted_row:
                master_feature_pool.append(extracted_row)
                print(f"Saving Asset Data -> {filename} | Merchant: {extracted_row['extracted_store'][:12]:<12} | Density: {extracted_row['layout_density_ratio']} | Variance: {extracted_row['vertical_alignment_variance']}")
                
                # --- 💾 COMMIT TOTAL METRICS STRUCT INTO LOCAL SQLITE FIELDS DIRECTLY ---
                new_receipt = Receipt(
                    filename=extracted_row['filename'],
                    merchant=extracted_row['extracted_store'],
                    date=extracted_row['date'],
                    total_amount=float(extracted_row['extracted_total']),
                    receipt_length=int(extracted_row['word_count']), 
                    num_lines=int(extracted_row['line_count']),
                    layout_density_ratio=float(extracted_row['layout_density_ratio']),
                    vertical_alignment_variance=float(extracted_row['vertical_alignment_variance']),
                    avg_ocr_confidence=float(extracted_row['avg_ocr_confidence'])
                )
                db_session.add(new_receipt)
        
        db_session.commit()
        print("\n💾 [SQLITE SUCCESS] All records and look-and-feel geometric metrics successfully committed to receipts.db!")

    except Exception as e:
        db_session.rollback()
        print(f"❌ Critical Error running loop ingestion pipeline: {str(e)}")
    finally:
        db_session.close()

    if master_feature_pool:
        df = pd.DataFrame(master_feature_pool)
        df.to_csv(output_csv, index=False)
        print(f"📝 CSV Backup asset saved to: {output_csv}")

if __name__ == "__main__":
    RAW_IMAGES_FOLDER = "data/processed_data/combined_train"
    ANOMALY_INPUT_CSV = "data/anomaly_detection_input.csv"
    run_real_use_feature_pipeline(RAW_IMAGES_FOLDER, ANOMALY_INPUT_CSV)