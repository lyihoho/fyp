import pandas as pd
from database import SessionLocal, Receipt

db_session = SessionLocal()
try:
    # Query all processed receipts from the database
    all_records = db_session.query(Receipt).all()
    
    # Convert database objects to a list of dicts
    data_list = []
    for r in all_records:
        data_list.append({
            "filename": r.filename,
            "extracted_store": r.merchant,
            "extracted_total": r.total_amount,
            "date": r.date,
            "layout_density_ratio": r.layout_density_ratio,
            "word_count": r.receipt_length,
            "line_count": r.num_lines,
            "vertical_alignment_variance": r.vertical_alignment_variance,
            "avg_ocr_confidence": r.avg_ocr_confidence,
            "aspect_ratio": r.aspect_ratio,
            "math_valid": r.math_valid_flag,
            "char_spacing_variance": r.character_spacing_var,
            "unreadable_gate_flag": False if r.fraud_label != "REJECTED (IMAGE UNREADABLE - PROMPT RE-UPLOAD)" else True
        })
        
    df = pd.DataFrame(data_list)
    df.to_csv("data/anomaly_detection_input.csv", index=False)
    print("🏆 Success! Full data matrix written to CSV from database.")
finally:
    db_session.close()