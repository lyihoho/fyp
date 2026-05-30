import os
import json
import pandas as pd
import sys

# Define path locations
text_folder = "data/extracted_text/extracted_train_15"
labeled_csv = "data/raw_images/ground_truth.csv"
output_json = "data/train_dataset.json"

if not os.path.exists(labeled_csv):
    print(f"[ERROR] Can't find your spreadsheet at: {labeled_csv}")
    sys.exit(1)

# Read your spreadsheet entries
df_labels = pd.read_csv(labeled_csv)
final_training_pool = []

print("="*60)
print("Compiling Dataset: Merging labels with raw OCR text dumps...")
print("="*60)

for idx, row in df_labels.iterrows():
    filename = str(row["filename"]).strip()
    
    # SYSTEM SHIELD: Natively repair missing file extensions from spreadsheet rows
    if not filename.lower().endswith(".txt"):
        filename = f"{filename}.txt"
        
    text_path = os.path.join(text_folder, filename)
    
    if os.path.exists(text_path):
        with open(text_path, "r", encoding="utf-8") as f:
            raw_ocr_content = f.read()
            
        training_entry = {
            "text": raw_ocr_content.strip(),
            "output": {
                "merchant": str(row["merchant"]).strip(),
                "date": str(row["date"]).strip(),
                "total_amount": float(row["total"]),
            }
        }
        final_training_pool.append(training_entry)
        print(f" -> Successfully merged raw text for: {filename}")
    else:
        print(f" [WARNING] Text file missing at {text_path}. Skipped.")

with open(output_json, "w", encoding="utf-8") as out:
    json.dump(final_training_pool, out, indent=2)

print("\n" + "="*60)
print(f"SUCCESS: Generated '{output_json}' automatically!")
print(f"Total compiled training entries: {len(final_training_pool)}")
print("="*60)