# evaluate_performance.py
import os
import pandas as pd
from database import SessionLocal, Receipt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def run_academic_performance_audit(ground_truth_csv_path):
    print("=" * 80)
    print("📊 AI AUDITING COMPLIANCE PIPELINE | PERFORMANCE VALIDATION ARCHIVE")
    print("=" * 80)
    
    if not os.path.exists(ground_truth_csv_path):
        print(f"❌ Error: Ground truth file missing at {ground_truth_csv_path}")
        return

    # 1. Load Ground Truth Labels
    df_gt = pd.read_csv(ground_truth_csv_path)
    # Standardize filename strings to eliminate spacing mismatches
    df_gt['filename'] = df_gt['filename'].astype(str).str.strip().str.lower()
    
    # 2. Extract Live Predictions from SQLite Database
    session = SessionLocal()
    try:
        db_records = session.query(Receipt).all()
        if not db_records:
            print("❌ Error: No database entries found in receipts.db. Run pipeline scripts first.")
            return
            
        predictions_pool = []
        for r in db_records:
            # Map your textual verdicts to standard binary classes (0 = Normal, 1 = Anomaly/Suspicious)
            # Both "REJECTED" and "MANUAL REVIEW" are flagged as anomaly alerts in a corporate audit setup
            verdict = str(r.fraud_label).upper()
            if "REJECTED" in verdict or "MANUAL REVIEW" in verdict or "SUSPECT" in verdict:
                predicted_binary = 1
            else:
                predicted_binary = 0
                
            predictions_pool.append({
                "filename": str(r.filename).strip().lower(),
                "predicted_label": predicted_binary,
                "text_verdict": r.fraud_label,
                "composite_score": r.fraud_score
            })
        df_pred = pd.DataFrame(predictions_pool)
    finally:
        session.close()

    # 3. Merge Ground Truth with Live Production Scores
    df_merged = pd.merge(df_gt, df_pred, on='filename', how='inner')
    
    if df_merged.empty:
        print("❌ Alignment Error: Could not match database filenames with ground_truth rows.")
        return

    y_true = df_merged['is_anomaly'].astype(int).tolist()
    y_pred = df_merged['predicted_label'].astype(int).tolist()

    # 4. Compute Statistical Confusion Matrix Data
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 5. Output Academic Validation Charts to Console
    print(f"🔄 Successfully aligned {len(df_merged)} processed assets against validation parameters.\n")
    print("-" * 50)
    print("📈 CORE MACHINE LEARNING PERFORMANCE METRICS")
    print("-" * 50)
    print(f" ├─ System Precision Score : {precision * 100:.2f}% (Ability to minimize false fraud alarms)")
    print(f" ├─ System Recall Score    : {recall * 100:.2f}% (Sensitivity to catching real anomalies)")
    print(f" ├─ Overall F1-Score       : {f1 * 100:.2f}% (Harmonic balancing metric)")
    print(f" └─ Global Model Accuracy  : {accuracy * 100:.2f}%\n")

    print("-" * 50)
    print("📋 CONFUSION MATRIX DISTRIBUTION TELEMETRY")
    print("-" * 50)
    print(f"  [True Negatives  (TN)] : {tn:<3} | Clean receipts accurately passed automatically.")
    print(f"  [False Positives (FP)] : {fp:<3} | Clean receipts routed to Manual Review/Rejection.")
    print(f"  [False Negatives (FN)] : {fn:<3} | Anomalies/Corrupt files missed by the pipeline.")
    print(f"  [True Positives  (TP)] : {tp:<3} | Severe anomalies successfully flagged/intercepted.")
    print("-" * 50)
    print("🎉 PERFORMANCE AUDIT COMPLETE. DATA READY FOR DISSERTATION CHAPTER 4.")
    print("=" * 80)

if __name__ == "__main__":
    GROUND_TRUTH_FILE = "ground_truth.xlsx - ground_truth.csv"
    run_academic_performance_audit(GROUND_TRUTH_FILE)