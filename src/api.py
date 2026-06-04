# api.py
import os
import sqlite3
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Force Python to find assets relative to where this script is saved
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "receipts.db")

# Force matplotlib into non-interactive backend mode to maintain background thread stability
plt.switch_backend('Agg')

# --- GLOBAL PLOT CACHE MANAGEMENT ---
cached_fig1, cached_fig2 = None, None

def get_all_records():
    """Queries live SQLite backend and normalizes legacy schema mapping to the web interface."""
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
        
        # 1. Map textual reporting metadata
        df_display['filename'] = df['filename']
        df_display['merchant'] = df['merchant']
        df_display['date'] = df['date']
        df_display['fraud_score'] = df['fraud_score']
        df_display['fraud_label'] = df['fraud_label']

        # 2. Map evaluation vectors (Translating 'score_machine_purity' -> Structure & Format UI)
        df_display['score_look_feel'] = df['score_look_feel'].astype(float)
        df_display['score_structure_format'] = df['score_machine_purity'].astype(float)
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
    """Compiles an official corporate compliance verification audit certificate sheet."""
    pdf_filename = f"Audit_Report_{filename.split('.')[0]}.pdf"
    pdf_path = os.path.join(SCRIPT_DIR, pdf_filename)
    
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []
    
    # Typography configurations
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=22, textColor=colors.HexColor('#1a365d'), spaceAfter=15)
    meta_style = ParagraphStyle('MetaText', parent=styles['Normal'], fontName='Helvetica', fontSize=10, textColor=colors.HexColor('#4a5568'))
    metric_label_style = ParagraphStyle('MetricLabel', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=11, textColor=colors.HexColor('#2d3748'))
    metric_val_style = ParagraphStyle('MetricVal', parent=styles['Normal'], fontName='Helvetica', fontSize=11, alignment=2)
    
    story.append(Paragraph("EXPENSE COMPLIANCE VERIFICATION AUDIT", title_style))
    story.append(Paragraph(f"<b>Target Asset Reference:</b> {filename}", meta_style))
    story.append(Spacer(1, 15))
    
    # Compute system status banner coloring dynamically based on evaluation strings
    bg_color, text_color = '#ebf8ff', '#2b6cb0'
    if "REJECTED" in label.upper():
        bg_color, text_color = '#fff5f5', '#c53030'
    elif "REVIEW" in label.upper():
        bg_color, text_color = '#fffaf0', '#dd6b20'
        
    status_style = ParagraphStyle('StatusStyle', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=14, textColor=colors.HexColor(text_color), alignment=1)
    status_data = [[Paragraph(f"<b>{label.upper()}</b><br/><font size=11>Composite Framework Score: {score}</font>", status_style)]]
    status_table = Table(status_data, colWidths=[530])
    status_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(bg_color)),
        ('PADDING', (0,0), (-1,-1), 12),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor(text_color))
    ]))
    story.append(status_table)
    story.append(Spacer(1, 20))
    
    # Core semantic data block
    story.append(Paragraph("<b>Extracted Content Anchors</b>", ParagraphStyle('Sub', parent=styles['Heading3'], fontSize=13, spaceAfter=8)))
    content_data = [
        [Paragraph("Identified Merchant:", metric_label_style), Paragraph(str(merchant), metric_val_style)],
        [Paragraph("Extracted Transaction Date:", metric_label_style), Paragraph(str(date), metric_val_style)]
    ]
    content_table = Table(content_data, colWidths=[200, 330])
    content_table.setStyle(TableStyle([('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')), ('PADDING', (0,0), (-1,-1), 8)]))
    story.append(content_table)
    story.append(Spacer(1, 20))
    
    # Core multi-criteria scoring matrix block
    story.append(Paragraph("<b>System Core Score Breakdown</b>", ParagraphStyle('Sub2', parent=styles['Heading3'], fontSize=13, spaceAfter=8)))
    vector_data = [
        [Paragraph("Evaluation Criterion", metric_label_style), Paragraph("Score Component", ParagraphStyle('H', parent=metric_label_style, alignment=2))],
        [Paragraph("1. Look & Feel (Macro-Geometry Anomaly Forest)", styles['Normal']), Paragraph(f"{lf}%", metric_val_style)],
        [Paragraph("2. Structure & Format (Micro-Alignment Anomaly Forest)", styles['Normal']), Paragraph(f"{sf}%", metric_val_style)],
        [Paragraph("3. Content Accuracy (Deterministic Verification Engine)", styles['Normal']), Paragraph(f"{ca}%", metric_val_style)],
        [Paragraph("4. Text Integrity (OCR Print Confidence Matrix)", styles['Normal']), Paragraph(f"{ti}%", metric_val_style)]
    ]
    vector_table = Table(vector_data, colWidths=[380, 150])
    vector_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f7fafc')),
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e0')),
        ('PADDING', (0,0), (-1,-1), 8)
    ]))
    story.append(vector_table)
    
    doc.build(story)
    return pdf_path

# --- FAST GRAPHICS VECTOR PLOT ENGINE (ANTI-TAB FREEZE) ---
def generate_analytics_plots():
    global cached_fig1, cached_fig2
    df = get_all_records()
    if df.empty:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.text(0.5, 0.5, "Database Registry Offline", ha='center', va='center')
        return fig1, fig1

    scores_series = df['fraud_score'].astype(str).str.replace('%', '', regex=False).astype(float)
    
    # Plot 1: Composite Score Distribution Histogram
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

    # Plot 2: Global Vector Metric Component Averages
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    vector_cols = ['score_look_feel', 'score_structure_format', 'score_content_accuracy', 'score_text_integrity']
    vector_labels = ['1. Look & Feel (ML)', '2. Structure & Format (ML)', '3. Content Accuracy (Rules)', '4. Text Integrity (OCR)']
    averages = [df[col].astype(float).mean() for col in vector_cols]
    colors = ['#74c4fe', '#41b6c4', '#238443', '#fec44f']
    bars = ax2.barh(vector_labels, averages, color=colors, edgecolor='#555555', height=0.45)
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
    df = get_all_records()
    if df.empty:
        return gr.update(), gr.update(), "⚠️ Database offline.", "", "", "", 0, 0, 0, 0, None
        
    match = df[df['filename'].str.lower() == target_filename.lower()]
    
    if not match.empty:
        rec = match.iloc[0]
        merchant = rec.get('merchant', 'Unknown Store')
        date = rec.get('date', 'Unknown Date')
        label = str(rec.get('fraud_label', 'UNKNOWN')).upper()
        score = str(rec.get('fraud_score', '0.0%'))
        
        lf = float(rec.get('score_look_feel', 0.0))
        sf = float(rec.get('score_structure_format', 0.0))
        ca = float(rec.get('score_content_accuracy', 0.0))
        ti = float(rec.get('score_text_integrity', 0.0))
        
        pdf_path = generate_pdf_report(target_filename, merchant, date, score, label, lf, sf, ca, ti)
        
        # Flips interface visibility parameters to switch layout modes full-screen
        return (
            gr.update(visible=False), # Instantly hide the Step 1 Dropzone workspace
            gr.update(visible=True),  # Instantly display the step 2 Report card workspace full-width
            label, score, merchant, date, lf, sf, ca, ti, pdf_path
        )
    else:
        warning_msg = f"⚠️ File '{target_filename}' unrecognized as baseline asset. Ensure it is stored in pipeline folders and execution arrays are synchronized."
        return gr.update(visible=True), gr.update(visible=False), warning_msg, "", "", "", 0, 0, 0, 0, None

def reset_view():
    """Flips layout state properties back to the fresh ingestion workspace view frame."""
    return gr.update(visible=True), gr.update(visible=False), None

def sync_all_components():
    df = get_all_records()
    p1, p2 = generate_analytics_plots()
    return df, p1, p2

# --- WEB UI INTERFACE CONFIGURATION ---
custom_theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate")
generate_analytics_plots() # Run warming pre-render sequences once on backend initiation

with gr.Blocks(title="AI Expense Auditing Gateway") as demo:
    gr.Markdown("# 🧾 AI Expense Auditing & Compliance Gateway")
    
    with gr.Tabs():
        # =====================================================================
        # TAB 1: DYNAMIC INGESTION WORKSPACE (THE FLIP WORKFLOW)
        # =====================================================================
        with gr.TabItem("📤 Real-Time Ingestion Portal"):
            
            # WORKSPACE VIEW A: SINGLE-COLUMN IMAGE INPUT ZONE
            with gr.Column(visible=True) as upload_view:
                gr.Markdown("### Ingest Document Asset for Integrity Auditing")
                input_file = gr.Image(type="filepath", label="Drop Receipt Image Here")
                run_btn = gr.Button("Execute Analysis Sweep ⚙️", variant="primary")
                
            # WORKSPACE VIEW B: FULL-WIDTH CLEAN AUDIT SHEET REPORT CARD
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

            # Route functional mappings to execute view transformations
            run_btn.click(
                fn=execute_live_inference, 
                inputs=input_file, 
                outputs=[upload_view, report_view, verdict_txt, score_txt, merchant_txt, date_txt, lf_bar, sf_bar, ca_bar, ti_bar, pdf_download]
            )
            back_btn.click(fn=reset_view, inputs=None, outputs=[upload_view, report_view, input_file])

        # =====================================================================
        # TAB 2: PRODUCTION GRAPHICAL VECTOR METRICS DASHBOARD
        # =====================================================================
        with gr.TabItem("📊 System Analytics Dashboard"):
            gr.Markdown("### Production System Data Profiles")
            with gr.Row():
                plot_dist = gr.Plot(label="Score Ingestion Spreads")
                plot_bars = gr.Plot(label="Multi-Criteria Vector Profile Averages")
            refresh_plots_btn = gr.Button("🔄 Re-calculate Graphics Vectors", variant="secondary")
            refresh_plots_btn.click(fn=generate_analytics_plots, inputs=None, outputs=[plot_dist, plot_bars])

        # =====================================================================
        # TAB 3: MASTER LOGICAL AUDIT SHEET RELATIONAL REGISTRY
        # =====================================================================
        with gr.TabItem("📈 Global Data Warehouse Ledger"):
            gr.Markdown("### Multi-Criteria Audit Trail Ledger View")
            master_sync_btn = gr.Button("🔄 Synchronize System Data & Charts", variant="primary")
            initial_data = get_all_records()
            headers_list = ["Filename", "Merchant", "Date", "Overall Score", "Verdict", "Look & Feel", "Structure & Format", "Content Accuracy", "Text Integrity"]
            ledger_grid = gr.Dataframe(value=initial_data, headers=headers_list[:len(initial_data.columns)], interactive=False)
            master_sync_btn.click(fn=sync_all_components, inputs=None, outputs=[ledger_grid, plot_dist, plot_bars])

    # Connect initialization hooks to stream pre-cached Matplotlib images smoothly
    demo.load(fn=get_cached_plots, inputs=None, outputs=[plot_dist, plot_bars])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=custom_theme, share=False)