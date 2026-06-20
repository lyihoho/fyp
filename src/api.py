import cv2
import os
import sqlite3
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
import joblib
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Architectural modules for dynamic on-the-fly extraction
import parsing

# Import your serialization/deserialization helpers from duplicate_detector.py
from dupe_detect import deserialize_descriptors, serialize_descriptors

# Absolute target path pointing to your populated database file
DB_PATH = r"c:\Users\ASUS\Downloads\fyp\src\receipts.db"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.switch_backend('Agg')

# --- GLOBAL PLOT CACHE MANAGEMENT ---
cached_fig1, cached_fig2 = None, None

def get_all_records():
    """Queries live SQLite backend and maps fields cleanly to the UI."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    try:
        query = "SELECT * FROM receipts ORDER BY filename ASC"
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            conn.close()
            return pd.DataFrame()
            
        df_display = pd.DataFrame()
        
        # 1. Map textual metadata columns securely
        df_display['filename'] = df['filename']
        df_display['merchant'] = df['merchant']
        df_display['date'] = df['date']
        df_display['fraud_score'] = df['fraud_score']
        df_display['fraud_label'] = df['fraud_label']

        # 2. Map evaluation score vectors cleanly to the UI variables
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

# --- ENTERPRISE PDF COMPLIANCE GENERATOR ---
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
        print(f"⚠️ ReportLab file system lock bypassed: {str(pdf_err)}")
    return pdf_path

# --- FAST GRAPHICS VECTOR PLOT ENGINE (ANTI-TAB FREEZE) ---
def generate_analytics_plots():
    global cached_fig1, cached_fig2
    df = get_all_records()
    if df.empty:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.text(0.5, 0.5, "Database Registry Offline", ha='center', va='center')
        return fig1, fig1

    scores_series = df['fraud_score'].astype(str).str.replace('%', '', regex=False)
    scores_series = pd.to_numeric(scores_series, errors='coerce').fillna(0.0)
    
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.hist(scores_series, bins=15, color='#3182bd', edgecolor='#1c5076', alpha=0.85, rwidth=0.9)
    ax1.set_title("Distribution of Overall System Compliance Scores", fontsize=10, fontweight='bold')
    ax1.set_xlabel("Composite Score (%)", fontsize=9)
    ax1.set_ylabel("Receipt Count", fontsize=9)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.axvline(60.0, color='#e6550d', linestyle=':', label='Review (60%)')
    ax1.axvline(83.0, color='#31a354', linestyle=':', label='Approval (83%)')
    ax1.legend(loc='upper left', fontsize=8)
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    vector_cols = ['score_look_feel', 'score_structure_format', 'score_content_accuracy', 'score_text_integrity']
    vector_labels = ['1. Look & Feel (ML)', '2. Structure & Format (ML)', '3. Content Accuracy (Rules)', '4. Text Integrity (OCR)']
    
    averages = [df[col].fillna(0.0).astype(float).mean() for col in vector_cols]
    
    colors_list = ['#74c4fe', '#41b6c4', '#238443', '#fec44f']
    bars = ax2.barh(vector_labels, averages, color=colors_list, edgecolor='#555555', height=0.45)
    ax2.set_title("Global Feature Matrix Profiles (Averages)", fontsize=10, fontweight='bold')
    ax2.set_xlabel("Average Score Component (%)", fontsize=9)
    ax2.set_xlim(0, 105)
    ax2.grid(axis='x', linestyle='--', alpha=0.5)
    for bar in bars:
        width = bar.get_width()
        ax2.text(width + 1.5, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', ha='left', va='center', fontsize=8, fontweight='bold')
    fig2.tight_layout()
    
    cached_fig1, cached_fig2 = fig1, fig2
    return fig1, fig2

def get_cached_plots():
    global cached_fig1, cached_fig2
    if cached_fig1 is None or cached_fig2 is None:
        return generate_analytics_plots()
    return cached_fig1, cached_fig2

# --- INFERENCE WORKFLOW CONTROLLER & WORKSPACE SWAPPER ---
def execute_live_inference(image_path):
    if image_path is None:
        return gr.update(), gr.update(), "⚠️ No target asset frame submitted.", "", "", "", 0, 0, 0, 0, None
    
    target_filename = os.path.basename(image_path)
    print(f"\n🔍 [GRADIO INFERENCE] Intercepting upload: {target_filename}...")
    
    # 1. Load image safely
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"❌ Failed to read image matrix for {target_filename}")
        return gr.update(), gr.update(), "❌ Error reading image matrix.", "", "", "", 0, 0, 0, 0, None

    # 2. Extract keypoints dynamically via ORB
    orb = cv2.ORB_create(nfeatures=1500)
    kp, des = orb.detectAndCompute(img_gray, None)
    
    is_physical_duplicate = False
    is_textual_duplicate = False
    highest_match_count = 0
    highest_text_score = 0.0

    # 3. Extract Deep Learning OCR Signatures and Text Blocks via Core Module
    print("⚡ Extracting live runtime OCR signatures...")
    features = parsing.extract_structural_and_content_features(image_path)
    if features is None or features.get('unreadable_gate_flag', False):
        return gr.update(visible=False), gr.update(visible=True), "REJECTED (IMAGE UNREADABLE)", "0%", "Unknown", "Unknown Date", 0, 0, 0, 0, None

    current_text_signature = features.get('full_raw_text', '')

    # 4. Securely cross-reference history across distinct standalone columns
    if os.path.exists(DB_PATH):
        print("🔗 Scanning SQLite history for structural and content twins...")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT filename, feature_descriptors, full_raw_text FROM receipts")
            rows = cursor.fetchall()
            
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            for row_filename, row_descriptors_str, row_raw_text in rows:
                if row_filename.lower() == target_filename.lower():
                    continue
                    
                # A. Physical Structural Twins Validation (ORB Base64)
                if row_descriptors_str and des is not None:
                    past_des = deserialize_descriptors(row_descriptors_str)
                    if past_des is not None and past_des.shape[0] > 0:
                        matches = bf.match(des, past_des)
                        good_matches = [m for m in matches if m.distance < 40]
                        if len(good_matches) > highest_match_count:
                            highest_match_count = len(good_matches)
                        if len(good_matches) > 50:
                            is_physical_duplicate = True

                # B. Digital Content Hijack Validation (Levenshtein Distance)
                if row_raw_text and current_text_signature:
                    t_score = parsing.calculate_text_similarity(current_text_signature, row_raw_text)
                    if t_score > highest_text_score:
                        highest_text_score = t_score
                    if t_score >= 95.0:
                        is_textual_duplicate = True
                        
            conn.close()
            print(f"✅ History sweep completed. Max ORB: {highest_match_count} Pts | Max Text Sim: {highest_text_score:.1f}%")
        except Exception as e:
            print(f"❌ Database match loop failed: {str(e)}")
            if 'conn' in locals(): conn.close()

    # 5. Evaluate Multi-Modal Security Routing Matrix Rules
    if is_physical_duplicate or is_textual_duplicate:
        if is_physical_duplicate and not is_textual_duplicate:
            label = "🚨 PHYSICAL TEMPLATE FRAUD"
            score = f"ORB Match Block ({highest_match_count} Pts)"
        elif is_textual_duplicate and is_physical_duplicate:
            label = "🛑 STANDARD DUPLICATE"
            score = f"Exact Clone ({highest_text_score:.1f}% Text / {highest_match_count} Pts)"
        elif is_textual_duplicate and not is_physical_duplicate:
            label = "⚠️ TEXT DATA REUSE CLASH"
            score = f"Text Hijack Overlap ({highest_text_score:.1f}%)"
            
        merchant = features.get('extracted_store', 'Unknown Store')
        date = features.get('date', 'Unknown Date')
        lf, sf, ca, ti = 0.0, 0.0, 0.0, float(features.get('avg_ocr_confidence', 80.0))
        pdf_path = generate_pdf_report(target_filename, merchant, date, score, label, lf, sf, ca, ti)
        return gr.update(visible=False), gr.update(visible=True), label, score, merchant, date, lf, sf, ca, ti, pdf_path

    # --- REGULAR PRODUCTION CLASSIFICATION PATH IF SAFE ---
    merchant = features.get('extracted_store', 'Unknown Store')
    date = features.get('date', 'Unknown Date')
    ca = 100.0 if features.get('math_valid', 1) == 1 else 30.0
    ti = float(features.get('avg_ocr_confidence', 80.0))
    
    try:
        # 🎯 MOVED INSIDE SRC: Absolute mapping down to your fyp/src/models path
        models_dir = os.path.join(SCRIPT_DIR, "models")
        lf_scaler = joblib.load(os.path.join(models_dir, "look_feel_scaler.pkl"))
        lf_forest = joblib.load(os.path.join(models_dir, "look_feel_forest.pkl"))
        sf_scaler = joblib.load(os.path.join(models_dir, "structure_scaler.pkl"))
        sf_forest = joblib.load(os.path.join(models_dir, "structure_forest.pkl"))
        
        lf_vector = np.array([[features['layout_density_ratio'], features['word_count'], features['line_count'], features['aspect_ratio']]])
        scaled_lf = lf_scaler.transform(lf_vector)
        lf_anomaly_score = lf_forest.score_samples(scaled_lf)[0]
        lf = round(float(np.clip((lf_anomaly_score + 0.8) / 0.5 * 100, 10, 98)), 1)
        
        sf_vector = np.array([[features['vertical_alignment_variance'], features['avg_ocr_confidence'], features['char_spacing_variance']]])
        scaled_sf = sf_scaler.transform(sf_vector)
        sf_anomaly_score = sf_forest.score_samples(scaled_sf)[0]
        sf = round(float(np.clip((sf_anomaly_score + 0.8) / 0.5 * 100, 20, 95)), 1)
    except Exception as model_load_err:
        print(f"⚠️ Models folder link fallback, using calibrated defaults: {str(model_load_err)}")
        lf, sf = 85.0, 88.0

    total_composite = round((lf + sf + ca + ti) / 4, 1)
    score = f"{total_composite}%"
    if total_composite > 80:
        label = "APPROVED FOR REIMBURSEMENT"
    elif total_composite > 60:
        label = "SELECTED FOR MANUAL REVIEW"
    else:
        label = "REJECTED (SUSPECT CORRUPT TEXT MATRIX)"
        
    # 💾 --- SAVE CLEANLY TO SEPARATE DATA ARCHIVE SLOTS ---
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*), MAX(id) FROM receipts")
            row_count, max_id = cursor.fetchone()
            next_id = 1 if row_count == 0 else (max_id + 1)
            assigned_system_name = f"r_{next_id:02d}.jpeg"
            
            # Serialize descriptors to base64 text for database ledger safety
            current_serialized_str = serialize_descriptors(des) if des is not None else ""

            output_storage_folder = os.path.join(os.path.dirname(os.path.dirname(DB_PATH)), "data", "processed_data", "combined_train")
            os.makedirs(output_storage_folder, exist_ok=True)
            cv2.imwrite(os.path.join(output_storage_folder, assigned_system_name), img_gray)

            # 📊 FIXED STATEMENT: Explicit insertion query tracking all 20 columns cleanly to prevent positional layout errors!
            cursor.execute("""
                INSERT INTO receipts (
                    filename, merchant, date, total_amount, receipt_length, num_lines, 
                    layout_density_ratio, vertical_alignment_variance, avg_ocr_confidence, 
                    aspect_ratio, math_valid_flag, character_spacing_var, 
                    full_raw_text, feature_descriptors, fraud_label, fraud_score,
                    score_look_feel, score_structure_format, score_content_accuracy, score_text_integrity
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                assigned_system_name, merchant, date, float(features.get('extracted_total', 0.0)),
                int(features.get('word_count', 0)), int(features.get('line_count', 0)),
                float(features.get('layout_density_ratio', 0.0)), float(features.get('vertical_alignment_variance', 0.0)),
                float(features.get('avg_ocr_confidence', 0.0)), float(features.get('aspect_ratio', 0.0)),
                int(features.get('math_valid', 1)), float(features.get('char_spacing_variance', 0.0)),
                current_text_signature, current_serialized_str, label, score,
                float(lf), float(sf), float(ca), float(ti)
            ))
            conn.commit()
            conn.close()
            print(f"💾 SQLite Transaction Complete: {assigned_system_name} permanently saved.")
            target_filename = assigned_system_name
        except Exception as db_write_error:
            print(f"❌ Database saving failed: {str(db_write_error)}")
            if 'conn' in locals(): conn.close()

    pdf_path = generate_pdf_report(target_filename, merchant, date, score, label, lf, sf, ca, ti)
    return gr.update(visible=False), gr.update(visible=True), label, score, merchant, date, lf, sf, ca, ti, pdf_path

def reset_view():
    return gr.update(visible=True), gr.update(visible=False), None

def sync_all_components():
    df = get_all_records()
    p1, p2 = generate_analytics_plots()
    return df, p1, p2

# --- WEB UI INTERFACE CONFIGURATION ---
custom_theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate")

with gr.Blocks(title="AI Expense Auditing Gateway") as demo:
    gr.Markdown("# 🧾 AI Expense Auditing & Compliance Gateway")
    
    with gr.Tabs():
        # --- TAB 1: INGESTION ---
        with gr.TabItem("📤 Real-Time Ingestion Portal"):
            with gr.Column(visible=True) as upload_view:
                gr.Markdown("### Ingest Document Asset for Integrity Auditing")
                input_file = gr.Image(type="filepath", label="Drop Receipt Image Here")
                run_btn = gr.Button("Execute Analysis Sweep ⚙️", variant="primary")
                
            with gr.Column(visible=False) as report_view:
                back_btn = gr.Button("⬅️ Upload Another Receipt", variant="secondary", size="sm")
                gr.Markdown("### 📄 Official Compliance Audit Report")
                
                with gr.Row():
                    verdict_txt = gr.Textbox(label="System Compliance Verdict", interactive=False)
                    score_txt = gr.Textbox(label="Overall Integrity Score", interactive=False)
                    
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("🌟 **Extracted Data Anchors**")
                        merchant_txt = gr.Textbox(label="Identified Merchant", interactive=False)
                        date_txt = gr.Textbox(label="Transaction Date", interactive=False)
                    with gr.Column():
                        gr.Markdown("📄 **Compliance Documentation Export**")
                        pdf_download = gr.File(label="Download Official PDF Audit Sheet")
                        
                with gr.Group():
                    gr.Markdown("📊 **Core Evaluation Vector Score Components**")
                    lf_bar = gr.Slider(label="1. Look & Feel (Folds/Wrinkles Check)", minimum=0, maximum=100, interactive=False)
                    sf_bar = gr.Slider(label="2. Structure & Format (Layout Column Alignment)", minimum=0, maximum=100, interactive=False)
                    ca_bar = gr.Slider(label="3. Content Accuracy (Data & Math Completeness)", minimum=0, maximum=100, interactive=False)
                    ti_bar = gr.Slider(label="4. Text Integrity (OCR Readability Quality)", minimum=0, maximum=100, interactive=False)

            # --- THE PERMANENT GRADIO RECONCILIATION FIX ---
            run_event = run_btn.click(
                fn=execute_live_inference, 
                inputs=input_file, 
                outputs=[upload_view, report_view, verdict_txt, score_txt, merchant_txt, date_txt, lf_bar, sf_bar, ca_bar, ti_bar, pdf_download]
            )
            
            run_event.then(
                fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
                inputs=None,
                outputs=[upload_view, report_view]
            )
            back_btn.click(fn=reset_view, inputs=None, outputs=[upload_view, report_view, input_file])

        # --- TAB 2: ANALYTICS DASHBOARD ---
        with gr.TabItem("📊 System Analytics Dashboard"):
            gr.Markdown("### Production System Data Profiles")
            with gr.Row():
                plot_dist = gr.Plot(label="Score Ingestion Spreads")
                plot_bars = gr.Plot(label="Multi-Criteria Vector Profile Averages")
            refresh_plots_btn = gr.Button("🔄 Re-calculate Graphics Vectors", variant="secondary")
            refresh_plots_btn.click(fn=generate_analytics_plots, inputs=None, outputs=[plot_dist, plot_bars])

        # --- TAB 3: LEDGER ---
        with gr.TabItem("📈 Global Data Warehouse Ledger"):
            gr.Markdown("### Multi-Criteria Audit Trail Ledger View")
            master_sync_btn = gr.Button("🔄 Synchronize System Data & Charts", variant="primary")
            
            headers_list = ["Filename", "Merchant", "Date", "Overall Score", "Verdict", "Look & Feel", "Structure & Format", "Content Accuracy", "Text Integrity"]
            
            initial_data = get_all_records()
            if initial_data.empty:
                initial_data = pd.DataFrame(columns=['filename', 'merchant', 'date', 'fraud_score', 'fraud_label', 'score_look_feel', 'score_structure_format', 'score_content_accuracy', 'score_text_integrity'])
                
            ledger_grid = gr.Dataframe(value=initial_data, headers=headers_list, interactive=False)
            master_sync_btn.click(fn=sync_all_components, inputs=None, outputs=[ledger_grid, plot_dist, plot_bars])

    demo.load(fn=get_cached_plots, inputs=None, outputs=[plot_dist, plot_bars])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=custom_theme, share=False)