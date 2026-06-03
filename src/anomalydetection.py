# anomaly_detection.py
import os
import joblib
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from database import SessionLocal, Receipt

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
LF_MODEL_PATH = os.path.join(MODELS_DIR, "look_feel_forest.pkl")
SF_MODEL_PATH = os.path.join(MODELS_DIR, "structure_forest.pkl")
LF_SCALER_PATH = os.path.join(MODELS_DIR, "look_feel_scaler.pkl")
SF_SCALER_PATH = os.path.join(MODELS_DIR, "structure_scaler.pkl")

class MultiCriteriaAnomalyEngine:
    def __init__(self):
        if not all(os.path.exists(p) for p in [LF_MODEL_PATH, SF_MODEL_PATH, LF_SCALER_PATH, SF_SCALER_PATH]):
            raise FileNotFoundError("❌ Trained model files or scalers missing! Run train_model.py first.")
        
        self.look_feel_model = joblib.load(LF_MODEL_PATH)
        self.structure_model = joblib.load(SF_MODEL_PATH)
        self.look_feel_scaler = joblib.load(LF_SCALER_PATH)
        self.structure_scaler = joblib.load(SF_SCALER_PATH)

    def calculate_four_scores(self, target, history_df):
        # --- METRIC 4: TEXT INTEGRITY & TAMPER SCORE ---
        s_integrity = 100.0
        ocr_conf = float(target.get("avg_ocr_confidence", 100.0))
        spacing_var = float(target.get("character_spacing_var", 0.0))

        if ocr_conf < 82.0: 
            s_integrity -= ((82.0 - ocr_conf) * 1.5)
        if spacing_var > 30000.0: 
            s_integrity -= 15.0
        
        s_integrity = max(0.0, min(100.0, s_integrity))

        # --- METRIC 1: LOOK & FEEL SCORE ---
        raw_lf = np.array([[float(target["layout_density_ratio"]), float(target["receipt_length"]), float(target["aspect_ratio"])]])
        scaled_lf = self.look_feel_scaler.transform(raw_lf)
        lf_ml_score = self.look_feel_model.decision_function(scaled_lf)[0]
        s_look_feel = max(0.0, min(100.0, (lf_ml_score + 0.45) * 200.0))

        # --- METRIC 2: STRUCTURE & FORMAT SCORE ---
        raw_sf = np.array([[float(target["num_lines"]), float(target["vertical_alignment_variance"])]])
        scaled_sf = self.structure_scaler.transform(raw_sf)
        sf_ml_score = self.structure_model.decision_function(scaled_sf)[0]
        s_structure = max(0.0, min(100.0, (sf_ml_score + 0.45) * 200.0))

        # --- METRIC 3: CONTENT ACCURACY SCORE ---
        s_content = 100.0
        new_merchant = str(target.get("merchant", "")).lower().strip()
        new_total = float(target.get("total_amount", 0.0))
        new_date = str(target.get("date", "")).lower().strip()

        missing_total = (new_total <= 0.0)
        missing_merchant = ("unknown" in new_merchant)
        missing_date = ("unknown" in new_date)

        if missing_total: s_content -= 15.0
        if missing_merchant: s_content -= 10.0
        if missing_date: s_content -= 5.0
        if int(target.get("math_valid_flag", 1)) == 0: s_content -= 15.0

        # The Conditional Cascade Penalty Matrix
        if missing_total and missing_date:
            s_content -= 30.0  
        elif missing_merchant and missing_total:
            s_content -= 25.0
        elif missing_merchant and missing_date:
            s_content -= 20.0

        s_content = max(0.0, min(100.0, s_content))

        for _, row in history_df.iterrows():
            if abs(new_total - float(row.get("total_amount", 0.0))) < 0.01:
                text_sim = SequenceMatcher(None, new_merchant, str(row.get("merchant", "")).lower().strip()).ratio()
                if text_sim > 0.85 and str(row.get("date", "")).strip().lower() == new_date:
                    s_content = 0.0  
                    break

        return round(s_look_feel, 1), round(s_structure, 1), round(s_content, 1), round(s_integrity, 1)

def run_evaluation_suite():
    print("🚨 [EVALUATION ENGINE RUNNING] Pulling database records...")
    session = SessionLocal()
    
    try:
        records = session.query(Receipt).all()
        if not records:
            print("❌ Error: No parsed records found in the database. Run parsing.py first.")
            return

        data_pool = []
        for r in records:
            row_dict = r.__dict__.copy()  
            row_dict.pop('_sa_instance_state', None)
            data_pool.append(row_dict)
            
        df_master = pd.DataFrame(data_pool)
        engine = MultiCriteriaAnomalyEngine()
        
        print("\n" + "="*80 + "\n⚙️ RUNTIME EVALUATION: AUDIT COMPLIANCE SWITCHBOARD\n" + "="*80)

        for r in records:
            # Clean architectural intercept: Check if ingestion gatekeeper flagged it unreadable
            if getattr(r, 'fraud_label', '') == "REJECTED (IMAGE UNREADABLE - PROMPT RE-UPLOAD)":
                s_lf, s_sf, s_ca, s_ti = 0.0, 0.0, 0.0, 0.0
                composite_score = 0.0
                verdict_label = "REJECTED (IMAGE UNREADABLE - PROMPT RE-UPLOAD)"
            else:
                current_target = r.__dict__.copy()
                current_target.pop('_sa_instance_state', None)
                df_background = df_master[df_master['filename'] != r.filename]
                
                s_lf, s_sf, s_ca, s_ti = engine.calculate_four_scores(current_target, df_background)
                composite_score = (s_lf + s_sf + s_ca + s_ti) / 4.0
                
                if s_ca == 0.0:
                    verdict_label = "REJECTED (DUPLICATE TRANS CLONE)"
                elif composite_score >= 83.0: 
                    verdict_label = "APPROVED FOR REIMBURSEMENT"
                elif 60.0 <= composite_score < 83.0:
                    verdict_label = "SELECTED FOR MANUAL REVIEW"
                else:
                    verdict_label = "REJECTED (SUSPECT PROFILE OUTLIER)"

            r.score_look_feel = s_lf
            r.score_structure_format = s_sf  
            r.score_content_accuracy = s_ca
            r.score_text_integrity = s_ti
            r.fraud_score = f"{composite_score:.1f}%"
            r.fraud_label = verdict_label

            print(f"📄 File: {r.filename:<12} | Merchant: {str(r.merchant)[:18]:<18}")
            print(f" 🎚️ [OVERALL SCALE]: {r.fraud_score} -> *** {verdict_label} ***")
            print(f" ├─ 1. Look & Feel Score        : {s_lf}%")
            print(f" ├─ 2. Structure & Format Score  : {s_sf}%")
            print(f" ├─ 3. Content Accuracy Score    : {s_ca}%")
            print(f" └─ 4. Text Integrity Score      : {s_ti}%")
            print("-" * 80)

        session.commit()
        print("💾 [SQLITE SUCCESS] Multi-criteria dynamic evaluation synced to database tables!")
        
    except Exception as e:
        session.rollback()
        print(f"❌ Critical Evaluation Error: {str(e)}")
    finally:
        session.close()

if __name__ == "__main__":
    run_evaluation_suite()