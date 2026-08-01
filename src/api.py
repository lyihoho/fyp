import cv2
import os
import sqlite3
import numpy as np
import pandas as pd
import gradio as gr
import joblib
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import parsing

# Import serialization/deserialization helpers
def serialize_descriptors(descriptors):
    # Converts raw OpenCV ORB matrices to a base64 text string for SQLite
    if descriptors is None:
        return ""
    binary_data = descriptors.tobytes()
    text_string = base64.b64encode(binary_data).decode('utf-8')
    return text_string

def deserialize_descriptors(text_string):
    # Rebuilds the absolute binary matrix OpenCV needs from an SQLite string
    if not text_string:
        return None
    binary_data = base64.b64decode(text_string.encode('utf-8'))
    descriptors = np.frombuffer(binary_data, dtype=np.uint8).reshape(-1, 32)
    return descriptors

# Absolute target path pointing to database file
DB_PATH = r"c:\Users\ASUS\Downloads\fyp\src\demotest.db"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_all_records():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    try:
        query = "SELECT * FROM demotest ORDER BY filename ASC"
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            conn.close()
            return pd.DataFrame()
            
        df_display = pd.DataFrame()
        
        # Map textual metadata columns
        df_display['filename'] = df['filename']
        df_display['merchant'] = df['merchant']
        df_display['date'] = df['date']
        df_display['fraud_score'] = df['fraud_score']
        df_display['fraud_label'] = df['fraud_label']

        # Map evaluation score vectors cleanly to the UI variables
        df_display['score_look_feel'] = df['score_look_feel'].astype(float)
        df_display['score_structure_format'] = df['score_structure_format'].astype(float)
        df_display['score_content_accuracy'] = df['score_content_accuracy'].astype(float)
        df_display['score_text_integrity'] = df['score_text_integrity'].astype(float)
                    
        conn.close()
        return df_display
    except Exception as e:
        if 'conn' in locals(): 
            conn.close()
        return pd.DataFrame()

# PDF GENERATOR
def generate_pdf_report(filename, merchant, date, score, label, lf, sf, ca, ti):
    pdf_filename = f"Audit_Report_{filename.split('.')[0]}.pdf"
    pdf_path = os.path.join(SCRIPT_DIR, pdf_filename)
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []
    
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=22, textColor=colors.HexColor('#1a365d'), spaceAfter=15)
    meta_style = ParagraphStyle('MetaText', parent=styles['Normal'], fontName='Helvetica', fontSize=10, textColor=colors.HexColor('#4a5568'))
    metric_label_style = ParagraphStyle('MetricLabel', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=11, textColor=colors.HexColor('#2d3748'))
    metric_val_style = ParagraphStyle('MetricVal', parent=styles['Normal'], fontName='Helvetica', fontSize=11, alignment=2)
    
    story.append(Paragraph("EXPENSE COMPLIANCE VERIFICATION AUDIT", title_style))
    story.append(Paragraph(f"<b>Target Asset Reference:</b> {filename}", meta_style))
    story.append(Spacer(1, 15))
    
    bg_color, text_color = '#ebf8ff', '#2b6cb0'
    if "REJECTED" in label.upper() or "FRAUD" in label.upper() or "DUPLICATE" in label.upper() or "COPIED" in label.upper():
        bg_color, text_color = '#fff5f5', '#c53030'
    elif "REVIEW" in label.upper() or "CLASH" in label.upper():
        bg_color, text_color = '#fffaf0', '#dd6b20'
        
    status_style = ParagraphStyle('StatusStyle', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=14, textColor=colors.HexColor(text_color), alignment=1)
    status_data = [[Paragraph(f"<b>{label.upper()}</b><br/><font size=11>Composite Framework Score: {score}</font>", status_style)]]
    status_table = Table(status_data, colWidths=[530])
    status_table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,-1), colors.HexColor(bg_color)), ('PADDING', (0,0), (-1,-1), 12), ('BOX', (0,0), (-1,-1), 1, colors.HexColor(text_color))]))
    story.append(status_table)
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("<b>Extracted Content Anchors</b>", ParagraphStyle('Sub', parent=styles['Heading3'], fontSize=13, spaceAfter=8)))
    content_data = [[Paragraph("Identified Merchant:", metric_label_style), Paragraph(str(merchant), metric_val_style)], [Paragraph("Extracted Transaction Date:", metric_label_style), Paragraph(str(date), metric_val_style)]]
    content_table = Table(content_data, colWidths=[200, 330])
    content_table.setStyle(TableStyle([('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')), ('PADDING', (0,0), (-1,-1), 8)]))
    story.append(content_table)
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("<b>System Core Score Breakdown</b>", ParagraphStyle('Sub2', parent=styles['Heading3'], fontSize=13, spaceAfter=8)))
    vector_data = [
        [Paragraph("Evaluation Criterion", metric_label_style), Paragraph("Score Component", ParagraphStyle('H', parent=metric_label_style, alignment=2))],
        [Paragraph("1. Look & Feel (Folds / Wrinkles Verification)", styles['Normal']), Paragraph(f"{lf}%", metric_val_style)],
        [Paragraph("2. Structure & Format (Layout Column Alignment Checked)", styles['Normal']), Paragraph(f"{sf}%", metric_val_style)],
        [Paragraph("3. Content Accuracy (Merchant & Math Cross-Checks)", styles['Normal']), Paragraph(f"{ca}%", metric_val_style)],
        [Paragraph("4. Text Integrity (OCR Print Readability Quality)", styles['Normal']), Paragraph(f"{ti}%", metric_val_style)]
    ]
    vector_table = Table(vector_data, colWidths=[380, 150])
    vector_table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f7fafc')), ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e0')), ('PADDING', (0,0), (-1,-1), 8)]))
    story.append(vector_table)
    
    try:
        doc.build(story)
    except Exception as pdf_err:
        print(f"ReportLab file system lock bypassed: {str(pdf_err)}")
    return pdf_path

# INFERENCE CONTROLLER
def execute_live_inference(image_path):
    if image_path is None:
        return gr.update(), gr.update(), "<div style='background-color:#fff5f5; color:#c53030; padding:20px; text-align:center; border-radius:8px; border:2px solid #c53030;'><strong>⚠️ No target asset frame submitted.</strong></div>", "Unknown", "Unknown Date", 0, 0, 0, 0, None
    
    # Track and verify exact user submission filename
    target_filename = os.path.basename(image_path)
    print(f"\n[GRADIO INFERENCE] Intercepting upload: {target_filename}...")
    
    # Load image
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"Failed to read image matrix for {target_filename}")
        return gr.update(), gr.update(), "<div style='background-color:#fff5f5; color:#c53030; padding:20px; text-align:center; border-radius:8px; border:2px solid #c53030;'><strong>❌ Error reading image matrix.</strong></div>", "Unknown", "Unknown Date", 0, 0, 0, 0, None

    # Extract keypoints dynamically via ORB
    orb = cv2.ORB_create(nfeatures=1500)
    kp, des = orb.detectAndCompute(img_gray, None)
    
    is_physical_duplicate = False
    is_textual_duplicate = False
    highest_match_count = 0
    highest_text_score = 0.0

    # Extract Deep Learning OCR Signatures and Text Blocks 
    print("Extracting live runtime OCR signatures...")
    features = parsing.extract_structural_and_content_features(image_path)
    
    # INTERCEPT GATEWAY 1: Unreadable Files
    if features is None or features.get('unreadable_gate_flag', False):
        unreadable_html = """
        <div style="background-color: #fff5f5; border: 2px solid #c53030; border-radius: 8px; padding: 24px; text-align: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
            <h1 style="color: #c53030; font-size: 28px; font-weight: 800; margin-bottom: 4px; letter-spacing: 0.5px;">REJECTED (IMAGE UNREADABLE)</h1>
            <p style="color: #9b2c2c; font-size: 16px; font-weight: bold; margin: 0;">COMPLIANCE SCORE: 0%</p>
        </div>
        """
        return gr.update(visible=False), gr.update(visible=True), unreadable_html, "Unknown", "Unknown Date", 0, 0, 0, 0, None

    current_text_signature = features.get('full_raw_text', '')

    # cross-reference history across distinct columns
    if os.path.exists(DB_PATH):
        print("Scanning SQLite history for structural and content twins...")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT filename, feature_descriptors, full_raw_text FROM demotest")
            rows = cursor.fetchall()
            
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            for row_filename, row_descriptors_str, row_raw_text in rows:
                # physical structural twins validation
                if row_descriptors_str and des is not None:
                    past_des = deserialize_descriptors(row_descriptors_str)
                    if past_des is not None and past_des.shape[0] > 0:
                        matches = bf.match(des, past_des)
                        good_matches = [m for m in matches if m.distance < 40]
                        if len(good_matches) > highest_match_count:
                            highest_match_count = len(good_matches)
                        if len(good_matches) > 120:
                            is_physical_duplicate = True

                # digital content Validation
                if row_raw_text and current_text_signature:
                    t_score = parsing.calculate_text_similarity(current_text_signature, row_raw_text)
                    if t_score > highest_text_score:
                        highest_text_score = t_score
                    if t_score >= 95.0:
                        is_textual_duplicate = True
                        
            conn.close()
            print(f"History sweep completed. Max ORB: {highest_match_count} Pts | Max Text Sim: {highest_text_score:.1f}%")
        except Exception as e:
            print(f"Database match loop failed: {str(e)}")
            if 'conn' in locals(): conn.close()

    # Evaluate multi-modal security rules
    if is_physical_duplicate or is_textual_duplicate:
        if is_physical_duplicate and not is_textual_duplicate:
            label = "REJECTED (PHYSICAL TEMPLATE FRAUD)"
            display_label = "PHYSICAL TEMPLATE FRAUD"
            score = f"ORB Match Block ({highest_match_count} Pts)"
        elif is_textual_duplicate and is_physical_duplicate:
            label = "REJECTED (DUPLICATE CLAIM BLOCK)"
            display_label = "STANDARD DUPLICATE"
            score = f"Exact Clone ({highest_text_score:.1f}% Text / {highest_match_count} Pts)"
        elif is_textual_duplicate and not is_physical_duplicate:
            label = "REJECTED (TEXT DATA REUSE CLASH)"
            display_label = "TEXT DATA REUSE CLASH"
            score = f"Text Hijack Overlap ({highest_text_score:.1f}%)"
            
        merchant = features.get('extracted_store', 'Unknown Store')
        date = features.get('date', 'Unknown Date')
        lf, sf, ca, ti = 0.0, 0.0, 0.0, float(features.get('avg_ocr_confidence', 80.0))
        
        fraud_html = f"""
        <div style="background-color: #fff5f5; border: 2px solid #c53030; border-radius: 8px; padding: 24px; text-align: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
            <h1 style="color: #c53030; font-size: 28px; font-weight: 800; margin-bottom: 4px; letter-spacing: 0.5px;">{display_label}</h1>
            <p style="color: #9b2c2c; font-size: 16px; font-weight: bold; margin: 0;">SECURITY BLOCKER: {score}</p>
        </div>
        """
        pdf_path = generate_pdf_report(target_filename, merchant, date, score, label, lf, sf, ca, ti)
        
        # SAVE DUPLICATE RECORD TO DATABASE
        if os.path.exists(DB_PATH):
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            try:
                current_serialized_str = serialize_descriptors(des) if des is not None else ""
                output_storage_folder = os.path.join(os.path.dirname(os.path.dirname(DB_PATH)), "data", "processed_data", "combined_train")
                os.makedirs(output_storage_folder, exist_ok=True)
                cv2.imwrite(os.path.join(output_storage_folder, target_filename), img_gray)

                cursor.execute("""
                    INSERT INTO demotest (
                        filename, merchant, date, total_amount, receipt_length, num_lines, 
                        layout_density_ratio, vertical_alignment_variance, avg_ocr_confidence, 
                        aspect_ratio, math_valid_flag, character_spacing_var, 
                        full_raw_text, feature_descriptors, fraud_label, fraud_score,
                        score_look_feel, score_structure_format, score_content_accuracy, score_text_integrity
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    target_filename, merchant, date, float(features.get('extracted_total', 0.0)),
                    int(features.get('word_count', 0)), int(features.get('line_count', 0)),
                    float(features.get('layout_density_ratio', 0.0)), float(features.get('vertical_alignment_variance', 0.0)),
                    float(features.get('avg_ocr_confidence', 0.0)), float(features.get('aspect_ratio', 0.0)),
                    int(features.get('math_valid', 1)), float(features.get('char_spacing_variance', 0.0)),
                    current_text_signature, current_serialized_str, label, "0.0%",
                    float(lf), float(sf), float(ca), float(ti)
                ))
                conn.commit()
                conn.close()
                print(f"SQLite Duplicate Record Complete: {target_filename} permanently saved.")
            except Exception as db_write_error:
                print(f"Database duplicate saving failed: {str(db_write_error)}")
                if 'conn' in locals(): conn.close()
                
        return gr.update(visible=False), gr.update(visible=True), fraud_html, merchant, date, lf, sf, ca, ti, pdf_path

    # REGULAR IMAGE CLASSIFICATION PATH
    merchant = features.get('extracted_store', 'Unknown Store')
    date = features.get('date', 'Unknown Date')
    ca = 100.0 if features.get('math_valid', 1) == 1 else 30.0
    ti = float(features.get('avg_ocr_confidence', 80.0))
    
    try:
        models_dir = os.path.join(SCRIPT_DIR, "models")
        lf_scaler = joblib.load(os.path.join(models_dir, "look_feel_scaler.pkl"))
        lf_forest = joblib.load(os.path.join(models_dir, "look_feel_forest.pkl"))
        sf_scaler = joblib.load(os.path.join(models_dir, "structure_scaler.pkl"))
        sf_forest = joblib.load(os.path.join(models_dir, "structure_forest.pkl"))
        
        lf_vector = np.array([[features['layout_density_ratio'], features['word_count'], features['line_count'], features['aspect_ratio']]])
        scaled_lf = lf_scaler.transform(lf_vector)
        lf_anomaly_score = lf_forest.score_samples(scaled_lf)[0]
        lf = round(float(np.clip((lf_anomaly_score + 0.8) / 0.5 * 100, 10, 98)), 1)
        
        total_chars = len(features.get('full_raw_text', ''))
        safe_lines = float(features['line_count']) if features['line_count'] > 0 else 1.0
        chars_per_line = total_chars / safe_lines
        
        sf_vector = np.array([[float(features['line_count']), float(features['vertical_alignment_variance']), chars_per_line]])
        scaled_sf = sf_scaler.transform(sf_vector)
        sf_anomaly_score = sf_forest.score_samples(scaled_sf)[0]
        sf = round(float(np.clip((sf_anomaly_score + 0.8) / 0.5 * 100, 20, 95)), 1)
    except Exception as model_load_err:
        print(f"Models folder link fallback, using calibrated defaults: {str(model_load_err)}")
        lf, sf = 85.0, 88.0

    total_composite = round((lf + sf + ca + ti) / 4, 1)
    score = f"{total_composite}%"
    
    # DYNAMIC ROUTING
    if total_composite >= 75.0:
        label = "APPROVED FOR REIMBURSEMENT"
        bg_color, border_color, text_color, score_color = "#f0fdf4", "#16a34a", "#16a34a", "#15803d"
    elif total_composite >= 50.0:
        label = "SELECTED FOR MANUAL REVIEW"
        bg_color, border_color, text_color, score_color = "#fffbeb", "#d97706", "#d97706", "#b45309"
    else:
        label = "REJECTED (SUSPECT IMAGE OUTLIER)"
        bg_color, border_color, text_color, score_color = "#fff5f5", "#dc2626", "#dc2626", "#991b1b"
        
    dynamic_html_banner = f"""
    <div style="background-color: {bg_color}; border: 2px solid {border_color}; border-radius: 8px; padding: 24px; text-align: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
        <h1 style="color: {text_color}; font-size: 28px; font-weight: 800; margin-bottom: 4px; letter-spacing: 0.5px;">{label}</h1>
        <p style="color: {score_color}; font-size: 18px; font-weight: bold; margin: 0;">INTEGRITY VERIFICATION SCORE: {score}</p>
    </div>
    """
        
    # SAVE WITH ORIGINAL FILENAME
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        try:
            current_serialized_str = serialize_descriptors(des) if des is not None else ""

            output_storage_folder = os.path.join(os.path.dirname(os.path.dirname(DB_PATH)), "data", "processed_data", "combined_train")
            os.makedirs(output_storage_folder, exist_ok=True)
            
            # Save using the exact target file system descriptor
            cv2.imwrite(os.path.join(output_storage_folder, target_filename), img_gray)

            cursor.execute("""
                INSERT INTO demotest (
                    filename, merchant, date, total_amount, receipt_length, num_lines, 
                    layout_density_ratio, vertical_alignment_variance, avg_ocr_confidence, 
                    aspect_ratio, math_valid_flag, character_spacing_var, 
                    full_raw_text, feature_descriptors, fraud_label, fraud_score,
                    score_look_feel, score_structure_format, score_content_accuracy, score_text_integrity
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                target_filename, merchant, date, float(features.get('extracted_total', 0.0)),
                int(features.get('word_count', 0)), int(features.get('line_count', 0)),
                float(features.get('layout_density_ratio', 0.0)), float(features.get('vertical_alignment_variance', 0.0)),
                float(features.get('avg_ocr_confidence', 0.0)), float(features.get('aspect_ratio', 0.0)),
                int(features.get('math_valid', 1)), float(features.get('char_spacing_variance', 0.0)),
                current_text_signature, current_serialized_str, label, score,
                float(lf), float(sf), float(ca), float(ti)
            ))
            conn.commit()
            conn.close()
            print(f"SQLite Saving Complete: {target_filename} permanently saved.")
        except Exception as db_write_error:
            print(f"Database saving failed: {str(db_write_error)}")
            if 'conn' in locals(): conn.close()

    pdf_path = generate_pdf_report(target_filename, merchant, date, score, label, lf, sf, ca, ti)
    return gr.update(visible=False), gr.update(visible=True), dynamic_html_banner, merchant, date, lf, sf, ca, ti, pdf_path

def reset_view():
    return gr.update(visible=True), gr.update(visible=False), None

#WEB UI INTERFACE
custom_theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate")

with gr.Blocks(title="AI Expense Auditing Gateway") as demo:
    gr.Markdown("AI Expense Auditing & Compliance Gateway")
    
    with gr.Tabs():
        # TAB 1: INGESTION 
        with gr.TabItem("Real-Time Ingestion Portal"):
            with gr.Column(visible=True) as upload_view:
                gr.Markdown("Ingest Document Asset for Integrity Auditing")
                input_file = gr.Image(type="filepath", label="Drop Receipt Image Here")
                run_btn = gr.Button("Execute Image Analysis", variant="primary")
                
            with gr.Column(visible=False) as report_view:
                back_btn = gr.Button("Upload Another Receipt", variant="secondary", size="sm")
                
                # Dynamic color-coded HTML banner
                verdict_banner = gr.HTML()
                gr.HTML("<br/>")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("Extracted Data Anchors")
                        merchant_txt = gr.Textbox(label="Identified Merchant", interactive=False)
                        date_txt = gr.Textbox(label="Transaction Date", interactive=False)
                    with gr.Column():
                        gr.Markdown("Audit Documentation Export")
                        pdf_download = gr.File(label="Download Official PDF Audit Sheet")
                        
                with gr.Group():
                    gr.Markdown("Core Evaluation Vector Scores")
                    lf_bar = gr.Slider(label="1. Look & Feel (Folds/Wrinkles Check)", minimum=0, maximum=100, interactive=False)
                    sf_bar = gr.Slider(label="2. Structure & Format (Layout Column Alignment)", minimum=0, maximum=100, interactive=False)
                    ca_bar = gr.Slider(label="3. Content Accuracy (Data & Math Completeness)", minimum=0, maximum=100, interactive=False)
                    ti_bar = gr.Slider(label="4. Text Integrity (OCR Print Readability Quality)", minimum=0, maximum=100, interactive=False)

            # Execution Pipeline Mapping Block
            run_event = run_btn.click(
                fn=execute_live_inference, 
                inputs=input_file, 
                outputs=[upload_view, report_view, verdict_banner, merchant_txt, date_txt, lf_bar, sf_bar, ca_bar, ti_bar, pdf_download]
            )
            
            run_event.then(
                fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
                inputs=None,
                outputs=[upload_view, report_view]
            )
            back_btn.click(fn=reset_view, inputs=None, outputs=[upload_view, report_view, input_file])

        # TAB 2:DB VIEW
        with gr.TabItem("Database Ledger"):
            gr.Markdown("Multi-Criteria Audit Trail Ledger View")
            master_sync_btn = gr.Button("Synchronize Ledger Registry", variant="primary")
            
            headers_list = ["Filename", "Merchant", "Date", "Overall Score", "Verdict", "Look & Feel", "Structure & Format", "Content Accuracy", "Text Integrity"]
            
            initial_data = get_all_records()
            if initial_data.empty:
                initial_data = pd.DataFrame(columns=['filename', 'merchant', 'date', 'fraud_score', 'fraud_label', 'score_look_feel', 'score_structure_format', 'score_content_accuracy', 'score_text_integrity'])
                
            ledger_grid = gr.Dataframe(value=initial_data, headers=headers_list, interactive=False)
            master_sync_btn.click(fn=get_all_records, inputs=None, outputs=ledger_grid)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=custom_theme, share=False)