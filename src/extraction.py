import cv2
import pytesseract
import os
import re
import pandas as pd

# If not added to PATH, uncomment next line
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- SECTION 3.7: DATA PARSING & CLEANING FUNCTIONS ---

def clean_ocr_text(text):
    """Fixes common Tesseract character hallucinations before running regex."""
    # Swap 'O' or 'o' to '0' if it sits directly inside a number (e.g., 1O.5O -> 10.50)
    text = re.sub(r'(?<=\d)[Oo](?=\d)', '0', text)
    return text

def extract_total(text):
    text = clean_ocr_text(text)
    
    # Target common financial keywords
    keywords = r'(total|amount due|net|grand total|cash|paid|rm|amount)'
    lines = text.split('\n')
    possible_totals = []
    
    for line in lines:
        if re.search(keywords, line.lower()):
            # Find standardized decimal structures (e.g., 12.50)
            numbers = re.findall(r'\d+\.\d{2}', line)
            if numbers:
                possible_totals.append(float(numbers[0]))
                
    if possible_totals:
        return max(possible_totals)
    
    # Fallback: Grab the largest decimal number printed anywhere on the page
    all_decimals = re.findall(r'\d+\.\d{2}', text)
    if all_decimals:
        return max([float(x) for x in all_decimals])
        
    return 0.0  # Return 0.0 if nothing resembling a price structure can be found

def extract_date(text):
    # Match standard numeric date formats: DD/MM/YYYY, YYYY-MM-DD, DD-MM-YYYY, etc.
    date_pattern = r'(\d{1,4}[-/.]\d{1,2}[-/.]\d{2,4})'
    matches = re.findall(date_pattern, text)
    if matches:
        return matches[0]
    return "Unknown"

# --- MAIN EXECUTION PIPELINE ---

if __name__ == "__main__":
    input_folder = "data/processed_data/processed_batch_1"
    output_text_folder = "data/extracted_text/extracted_train_15"
    
    os.makedirs(output_text_folder, exist_ok=True)
    print("Pipeline Initiated. Processing images...")

    # Array to compile structured data for the final dataset output
    parsed_dataset = []

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(input_folder, filename)

            # 1. Read Processed Image
            img = cv2.imread(image_path)
            if img is None:
                continue

            # 2. Section 3.6: Raw OCR Extraction
            extracted_text = pytesseract.image_to_string(img)

            # 3. Save Raw Text Document (For verification / audit trail)
            text_filename = os.path.splitext(filename)[0] + ".txt"
            output_text_path = os.path.join(output_text_folder, text_filename)
            with open(output_text_path, "w", encoding="utf-8") as f:
                f.write(extracted_text)

            # 4. Section 3.7: Apply Regex Parsers to the Messy Text String
            predicted_total = extract_total(extracted_text)
            predicted_date = extract_date(extracted_text)

            # 5. Append Results to Dataset Array
            parsed_dataset.append({
                "filename": filename,
                "predicted_date": predicted_date,
                "predicted_total": predicted_total
            })

            print(f"Processed: {filename} -> Total: {predicted_total} | Date: {predicted_date}")

    # 6. Save Extracted Data Directly to a CSV for Anomaly Detection (Section 3.8)
    df_output = pd.DataFrame(parsed_dataset)
    output_csv_path = "data/extracted_text/parsed_predictions.csv"
    df_output.to_csv(output_csv_path, index=False)
    
    print("\n" + "="*40)
    print(f"SUCCESS: Pipeline execution complete.")
    print(f"Structured metrics written to: {output_csv_path}")
    print("="*40)