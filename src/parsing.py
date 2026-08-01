import cv2
import pandas as pd
import numpy as np
import os
import re
import json
import ollama
import joblib 
import base64
from sqlalchemy import text
from database import SessionLocal, Receipt, Base, engine
from paddleocr import PaddleOCR 

# Levenshtein similarity and ORB serializers to load values to separate columns
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

def serialize_descriptors(descriptors):
    if descriptors is None:
        return ""
    binary_data = descriptors.tobytes()
    text_string = base64.b64encode(binary_data).decode('utf-8')
    return text_string

def deserialize_descriptors(text_string):
    if not text_string:
        return None
    binary_data = base64.b64decode(text_string.encode('utf-8'))
    descriptors = np.frombuffer(binary_data, dtype=np.uint8).reshape(-1, 32)
    return descriptors

Base.metadata.create_all(bind=engine)

# Initialize deep learning ocr
ocr_model = PaddleOCR(lang='en')

try:
    local_client = ollama.Client()
except Exception:
    local_client = None

def get_llm_parsing_fallback(raw_text_dump):
    if local_client is None:
        return {"merchant": "Unknown Store", "total": 0.0, "date": "Unknown Date"}
    prompt = f"""### System Instruction:
You are a strict data extraction algorithm. Analyze the following Malaysian receipt text. Isolate the main store brand name, the final net total paid, and the transaction date cleanly.

JSON Schema: {{
    "merchant": "Cleaned Store Name Only",
    "date": "DD/MM/YYYY",
    "total_amount": float
}}

### Target Text:
{raw_text_dump}"""
    
    try:
        response = local_client.generate(model="qwen2.5:3b", prompt=prompt, options={'temperature': 0.0}, format='json')
        data = json.loads(response['response'])
        extracted_total = float(data.get("total_amount", 0.0))
        if extracted_total > 2000.00: extracted_total = 0.0
        return {
            "merchant": data.get("merchant", "Unknown Store"), 
            "total": extracted_total,
            "date": data.get("date", "Unknown Date")
        }
    except Exception:
        return {"merchant": "Unknown Store", "total": 0.0, "date": "Unknown Date"}

def extract_structural_and_content_features(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
        
    h, w, _ = img.shape
    aspect_ratio = float(w / h)
    
    result = ocr_model.ocr(image_path)
    
    parsed_words = []
    if result and isinstance(result, list):
        for block in result:
            if block is None: continue
            for line in block:
                box = line[0]        
                text_str = line[1][0] 
                conf = line[1][1]     
                
                box_w = abs(box[1][0] - box[0][0])
                box_h = abs(box[2][1] - box[0][1])
                
                parsed_words.append({
                    'text': text_str,
                    'conf': conf,
                    'top': box[0][1],
                    'left': box[0][0],
                    'width': box_w,
                    'height': box_h
                })
                
    data = pd.DataFrame(parsed_words)
    
    # Check for empty data
    if data.empty or len(data) < 5 or data['conf'].mean() < 0.82: 
        return {
            "filename": os.path.basename(image_path),
            "extracted_store": "Unknown Store",
            "extracted_total": 0.0,
            "date": "Unknown Date",
            "layout_density_ratio": 0.0,
            "word_count": 0,
            "line_count": 0,  
            "vertical_alignment_variance": 0.0,
            "avg_ocr_confidence": round(float(data['conf'].mean() * 100), 2) if not data.empty else 0.0,
            "aspect_ratio": round(aspect_ratio, 4),
            "math_valid": 0,
            "char_spacing_variance": 0.0,
            "unreadable_gate_flag": True,
            "full_raw_text": ""
        }

    total_words = len(data)
    average_ocr_confidence = data['conf'].mean() * 100 
    word_boxes_area = (data['width'] * data['height']).sum()
    layout_density_ratio = float(word_boxes_area / (w * h))
    vertical_variance = data['top'].var() if len(data) > 1 else 0.0

    if len(data) > 1:
        data_sorted = data.sort_values(by=['top', 'left'])
        horizontal_gaps = data_sorted['left'].diff().dropna()
        char_spacing_variance = float(horizontal_gaps.var()) if len(horizontal_gaps) > 1 else 0.0
    else:
        char_spacing_variance = 0.0

    all_text_string = " ".join(data['text'].astype(str).tolist())

    data = data.copy()
    data['line_group'] = data['top'] // 15
    lines = []
    for g in sorted(data['line_group'].unique()):
        line_str = " ".join(data[data['line_group'] == g].sort_values(by='left')['text'].astype(str).tolist())
        lines.append(line_str)
    full_text_dump = "\n".join(lines)

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
            
            # Check for prepended digit tampering (e.g. adding 1 in front of total)
            if detected_total > 0.0:
                t_str = f"{detected_total:.2f}"
                t_int_str, t_dec_str = t_str.split('.')
                for p in set(cleaned_prices):
                    if p >= detected_total or p <= 0.01:
                        continue
                    p_str = f"{p:.2f}"
                    p_int_str, p_dec_str = p_str.split('.')
                    if p_dec_str == t_dec_str:
                        if t_int_str.endswith(p_int_str) and len(p_int_str) < len(t_int_str):
                            diff = detected_total - p
                            for place in [10.0, 100.0, 1000.0]:
                                ratio = diff / place
                                if abs(ratio - round(ratio)) < 0.01 and round(ratio) in range(1, 10):
                                    math_valid = 0
                                    break

    top_rows = data[data['top'] < (h * 0.15)].copy()
    detected_store_name = "Unknown Store"
    if not top_rows.empty:
        top_rows['line_group_head'] = top_rows['top'] // 12
        first_line = top_rows[top_rows['line_group_head'] == top_rows['line_group_head'].min()].sort_values(by='left')
        raw_headline = " ".join(first_line['text'].astype(str).tolist()).strip()
        cleaned_headline = re.sub(r'[^\w\s\.\&\-\@]', '', raw_headline).strip()
        if len(cleaned_headline) > 2 and not re.match(r'^\d', cleaned_headline):
            detected_store_name = re.sub(r'\b(sdn|bhd|inc|llp|co|enterprise|trading)\b', '', cleaned_headline, flags=re.I).strip()

    detected_date = "Unknown Date"
    date_pattern = r'\b(\d{2}[/\.\-]\d{2}[/\.\-]\d{4})|(\d{4}[/\.\-]\d{2}[/\.\-]\d{2})\b'
    date_matches = re.findall(date_pattern, all_text_string)
    if date_matches:
        found_date = [match for group in date_matches for match in group if match]
        if found_date: detected_date = found_date[0]

    if detected_total == 0.0 or detected_store_name == "Unknown Store" or len(detected_store_name) < 4:
        ai_corrections = get_llm_parsing_fallback(full_text_dump)
        if detected_total == 0.0: detected_total = ai_corrections.get("total", 0.0)
        if detected_store_name == "Unknown Store" or len(detected_store_name) < 4:
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
        "avg_ocr_confidence": float(average_ocr_confidence),
        "aspect_ratio": round(aspect_ratio, 4),
        "math_valid": math_valid,
        "char_spacing_variance": round(char_spacing_variance, 2),
        "unreadable_gate_flag": False,
        "full_raw_text": full_text_dump
    }

def run_real_use_feature_pipeline(images_dir, output_csv):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
    if not os.path.exists(images_dir): return

    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(valid_exts)])
    master_feature_pool = []
    db_session = SessionLocal()

    try:
        print("\n" + "="*80 + "\n📥 [INGESTION PHASE] SCREENING DOCUMENT MATRIX AT THE GATE\n" + "="*80)

        orb = cv2.ORB_create(nfeatures=1500)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        for filename in image_files:
            full_img_path = os.path.join(images_dir, filename)
            
            # Protect pipeline from internal runtime crashes
            try:
                # EXTRACT GENERAL LOOK & FEEL VIA OCR
                extracted_row = extract_structural_and_content_features(full_img_path)
                
                if not extracted_row:
                    continue

                master_feature_pool.append(extracted_row)
                
                # COMPUTE AND ENCODE THE LIVE ORB DESCRIPTORS
                img_gray = cv2.imread(full_img_path, cv2.IMREAD_GRAYSCALE)
                current_serialized_orb = ""
                live_des = None
                
                if img_gray is not None and not extracted_row['unreadable_gate_flag']:
                    _, live_des = orb.detectAndCompute(img_gray, None)
                    current_serialized_orb = serialize_descriptors(live_des)

                is_textual_duplicate = False
                is_physical_duplicate = False
                highest_text_score = 0.0
                highest_orb_matches = 0

                if not extracted_row['unreadable_gate_flag']:
                    current_text_signature = extracted_row['full_raw_text']
                    past_records = db_session.query(Receipt).all()
                    
                    for record in past_records:
                        # Text Similarity Engine Checking
                        past_text = record.full_raw_text
                        if past_text and current_text_signature:
                            t_score = calculate_text_similarity(current_text_signature, past_text)
                            if t_score > highest_text_score:
                                highest_text_score = t_score
                            if t_score >= 95.0:
                                is_textual_duplicate = True

                        # Base64 Geometric Matrix Checking
                        past_orb_b64 = record.feature_descriptors
                        if past_orb_b64 and live_des is not None:
                            reconstructed_past_des = deserialize_descriptors(past_orb_b64)
                            if reconstructed_past_des is not None:
                                matches = bf.match(live_des, reconstructed_past_des)
                                good_matches = [m for m in matches if m.distance < 40]
                                orb_count = len(good_matches)
                                if orb_count > highest_orb_matches:
                                    highest_orb_matches = orb_count
                                if orb_count > 120:
                                    is_physical_duplicate = True

                # MULTI-MODAL SECURITY ROUTING
                if extracted_row['unreadable_gate_flag']:
                    final_label = "REJECTED (IMAGE UNREADABLE - PROMPT RE-UPLOAD)"
                    final_score = "0%"
                else:
                    if is_physical_duplicate and not is_textual_duplicate:
                        final_label = "PHYSICAL TEMPLATE FRAUD"
                        final_score = f"ORB Match Block ({highest_orb_matches} Pts)"
                    elif is_textual_duplicate and is_physical_duplicate:
                        final_label = "STANDARD DUPLICATE"
                        final_score = f"Exact Clone ({highest_text_score:.1f}% Text / {highest_orb_matches} Pts)"
                    elif is_textual_duplicate and not is_physical_duplicate:
                        final_label = "TEXT DATA REUSE CLASH"
                        final_score = f"Text Hijack Overlap ({highest_text_score:.1f}%)"
                    else:
                        final_label = "pending"
                        final_score = "pending"

                    print(f"Parsing: {filename:<12} | Sim Track: [Text: {highest_text_score:.1f}% | ORB: {highest_orb_matches} Pts] -> Verdict: {final_label}")
                
                # WRITE TO THEIR STANDALONE COLUMNS
                try:
                    # Target the local src/models directory natively
                    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
                    lf_scaler = joblib.load(os.path.join(models_dir, "look_feel_scaler.pkl"))
                    lf_forest = joblib.load(os.path.join(models_dir, "look_feel_forest.pkl"))
                    sf_scaler = joblib.load(os.path.join(models_dir, "structure_scaler.pkl"))
                    sf_forest = joblib.load(os.path.join(models_dir, "structure_forest.pkl"))
                    
                    # Compute Look & Feel via Isolation Forest models
                    lf_vector = np.array([[extracted_row['layout_density_ratio'], extracted_row['word_count'], extracted_row['line_count'], extracted_row['aspect_ratio']]])
                    scaled_lf = lf_scaler.transform(lf_vector)
                    lf_anomaly_score = lf_forest.score_samples(scaled_lf)[0]
                    lf_score = round(float(np.clip((lf_anomaly_score + 0.8) / 0.5 * 100, 10, 98)), 1)
                    
                    # Compute Structure & Format via Isolation Forest models
                    total_chars = len(extracted_row['full_raw_text'])
                    safe_lines = float(extracted_row['line_count']) if extracted_row['line_count'] > 0 else 1.0
                    chars_per_line = total_chars / safe_lines
                    
                    sf_vector = np.array([[float(extracted_row['line_count']), float(extracted_row['vertical_alignment_variance']), chars_per_line]])
                    scaled_sf = sf_scaler.transform(sf_vector)
                    sf_anomaly_score = sf_forest.score_samples(scaled_sf)[0]
                    sf_score = round(float(np.clip((sf_anomaly_score + 0.8) / 0.5 * 100, 20, 95)), 1)
                except Exception:
                    # Secure fallback defaults if models are missing
                    lf_score, sf_score = 80.0, 80.0

                ca_score = 100.0 if extracted_row['math_valid'] == 1 else 30.0
                ti_score = float(extracted_row['avg_ocr_confidence'])

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
                    character_spacing_var=float(extracted_row['char_spacing_variance']),
                    
                    full_raw_text=extracted_row['full_raw_text'],
                    feature_descriptors=current_serialized_orb, 
                    
                    fraud_label=final_label,
                    fraud_score=final_score,

                    # Direct mapping to store actual metrics
                    score_look_feel=lf_score if not extracted_row['unreadable_gate_flag'] else 0.0,
                    score_structure_format=sf_score if not extracted_row['unreadable_gate_flag'] else 0.0,
                    score_content_accuracy=ca_score,
                    score_text_integrity=ti_score if not extracted_row['unreadable_gate_flag'] else 0.0
                )
                db_session.add(new_receipt)
                
                # Commit progressively file-by-file 
                db_session.commit()

            except Exception as loop_error:
                db_session.rollback()
                print(f"Skipped corrupted image [{filename}] due to ingestion engine error: {loop_error}")
                continue
                
        print("\nDeep Learning & Computer Vision metrics ledger initialized completely.")
    except Exception as e:
        db_session.rollback()
        print(f"Ingestion Error: {str(e)}")
    finally: db_session.close()

    if master_feature_pool:
        df = pd.DataFrame(master_feature_pool)
        if 'full_raw_text' in df.columns:
            df = df.drop(columns=['full_raw_text'])
        df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    RAW_IMAGES_FOLDER = "data/processed_data/combined_train"
    ANOMALY_INPUT_CSV = "data/anomaly_detection_input.csv"
    run_real_use_feature_pipeline(RAW_IMAGES_FOLDER, ANOMALY_INPUT_CSV)