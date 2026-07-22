# anomalydetection.py
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
            raise FileNotFoundError("Trained model files or scalers missing! Run train_model.py first.")
        
        self.look_feel_model = joblib.load(LF_MODEL_PATH)
        self.structure_model = joblib.load(SF_MODEL_PATH)
        self.look_feel_scaler = joblib.load(LF_SCALER_PATH)
        self.structure_scaler = joblib.load(SF_SCALER_PATH)

    def calculate_four_scores(self, target, history_df):
        full_text = str(target.get("full_raw_text", "")).strip()
        full_text_lower = full_text.lower()
        
        words = full_text_lower.split()
        total_words = len(words) if len(words) > 0 else 1
        total_chars = len(full_text) if len(full_text) > 0 else 1

        # METRIC 4: TEXT INTEGRITY & TAMPER SCORE
        ocr_conf = float(target.get("avg_ocr_confidence", 100.0))
        spacing_var = float(target.get("character_spacing_var", 0.0))

        s_integrity = ocr_conf 

        if spacing_var > 0:
            spacing_penalty = min(25.0, (spacing_var / 2000.0)) 
            s_integrity -= spacing_penalty

        num_lines = float(target.get("num_lines", 1.0))
        num_lines = num_lines if num_lines > 0 else 1.0
        char_per_line_ratio = total_chars / num_lines
        
        if char_per_line_ratio > 80.0:  
            s_integrity -= min(30.0, (char_per_line_ratio - 80.0) * 0.5)

        s_integrity = max(0.0, min(100.0, s_integrity))

        # METRIC 1 & 2: MACHINE LEARNING SPATIAL LAYOUTS
        raw_lf = np.array([[float(target["layout_density_ratio"]), float(target["receipt_length"]), float(target["num_lines"]), float(target["aspect_ratio"])]])
        scaled_lf = self.look_feel_scaler.transform(raw_lf)
        lf_ml_score = self.look_feel_model.decision_function(scaled_lf)[0]

        raw_sf = np.array([[float(target["num_lines"]), float(target["vertical_alignment_variance"]), char_per_line_ratio]])
        scaled_sf = self.structure_scaler.transform(raw_sf)
        sf_ml_score = self.structure_model.decision_function(scaled_sf)[0]

        # CHARACTER METRIC EXTREME SHIELD
        if len(full_text) > 15000:
            return 0.0, 0.0, 0.0, 0.0

        s_look_feel = max(0.0, min(100.0, (lf_ml_score + 0.35) * 175.0))
        s_structure = max(0.0, min(100.0, (sf_ml_score + 0.35) * 175.0))

        # METRIC 3: CONTENT ACCURACY SCORE
        receipt_anchors = ["total", "amount", "rm", "cash", "change", "tax", "subtotal", "inv", "thank", "qty", "price", "item"]
        keyword_count = sum(full_text_lower.count(anchor) for anchor in receipt_anchors)
        keyword_density = (keyword_count / total_words) * 100.0
        
        s_content = min(100.0, keyword_density * 10.0) 

        new_merchant = str(target.get("merchant", "")).lower().strip()
        new_date = str(target.get("date", "")).lower().strip()
        
        try:
            new_total = float(target.get("total_amount", 0.0))
        except (ValueError, TypeError):
            new_total = 0.0

        if new_total <= 0.0: s_content -= 15.0
        if "unknown" in new_merchant: s_content -= 10.0
        if "unknown" in new_date: s_content -= 5.0
        if int(target.get("math_valid_flag", 1)) == 0: s_content = 30.0

        s_content = max(0.0, min(100.0, s_content))



        return round(s_look_feel, 1), round(s_structure, 1), round(s_content, 1), round(s_integrity, 1)
    
def run_evaluation_suite():
    print("\nRunning Evaluation, Pulling database records...")
    session = SessionLocal()
    
    try:
        records = session.query(Receipt).all()
        if not records:
            print("Error: No parsed records found in the database. Run parsing.py first.")
            return

        data_pool = []
        for r in records:
            row_dict = r.__dict__.copy()  
            row_dict.pop('_sa_instance_state', None)
            data_pool.append(row_dict)
            
        df_master = pd.DataFrame(data_pool)
        engine = MultiCriteriaAnomalyEngine()
        
        print("\n" + "="*80 + "\nRUNTIME EVALUATION OUTPUT\n" + "="*20)

        for r in records:
            current_label = getattr(r, 'fraud_label', '')
            
            # GATE 1: INGESTION ENGINE FLAGS 
            if current_label == "REJECTED (IMAGE UNREADABLE - PROMPT RE-UPLOAD)":
                s_lf, s_sf, s_ca, s_ti = 0.0, 0.0, 0.0, 0.0
                composite_score = 0.0
                verdict_label = "REJECTED (IMAGE UNREADABLE - PROMPT RE-UPLOAD)"
                
            elif current_label == "STANDARD DUPLICATE":
                s_lf, s_sf, s_ca, s_ti = 0.0, 0.0, 0.0, 0.0
                composite_score = 0.0
                verdict_label = "REJECTED (DUPLICATE CLAIM BLOCK)"
                
            elif current_label == "TEXT DATA REUSE CLASH":
                s_lf, s_sf, s_ca, s_ti = 0.0, 0.0, 0.0, 0.0
                composite_score = 0.0
                verdict_label = "REJECTED (TEXT DATA REUSE CLASH)"
                
            elif current_label == "PHYSICAL TEMPLATE FRAUD":
                current_target = r.__dict__.copy()
                current_target.pop('_sa_instance_state', None)
                df_background = df_master[df_master['filename'] != r.filename]
                
                s_lf, s_sf, s_ca, s_ti = engine.calculate_four_scores(current_target, df_background)
                composite_score = (s_lf + s_sf + s_ca + s_ti) / 4.0
                
                target_date = str(r.date).strip().lower()
                try:
                    target_total = float(r.total_amount)
                except (ValueError, TypeError):
                    target_total = 0.0
                
                strict_clash = df_background[
                    (df_background['date'].str.strip().str.lower() == target_date) & 
                    (df_background['total_amount'].astype(float) == target_total)
                ]
                
                if not strict_clash.empty and target_date != "unknown date" and target_total > 0.0:
                    verdict_label = "SELECTED FOR MANUAL REVIEW (PHYSICAL TEMPLATE FRAUD)"
                else:
                    if s_ti < 25.0 or s_sf < 20.0:
                        composite_score = 0.0
                        verdict_label = "REJECTED (CORRUPT TEXT)"
                    elif s_ca == 0.0:
                        verdict_label = "REJECTED (DUPLICATE/NON-RECEIPT)"
                    elif composite_score >= 50:
                        verdict_label = "SELECTED FOR MANUAL REVIEW"
                    else:
                        verdict_label = "REJECTED (SUSPECT IMAGE OUTLIER)"

            # --- ENGINE GATE 2: STANDARD EVALUATION ---
            else:
                current_target = r.__dict__.copy()
                current_target.pop('_sa_instance_state', None)
                df_background = df_master[df_master['filename'] != r.filename]
                
                s_lf, s_sf, s_ca, s_ti = engine.calculate_four_scores(current_target, df_background)
                composite_score = (s_lf + s_sf + s_ca + s_ti) / 4.0
                
                if s_ti < 25.0 or s_sf < 20.0 or len(str(r.full_raw_text)) > 15000:
                    composite_score = 0.0
                    verdict_label = "REJECTED (CORRUPT TEXT)"
                elif s_ca == 0.0:
                    verdict_label = "REJECTED (DUPLICATE)"
                elif composite_score >= 75.0: 
                    verdict_label = "APPROVED FOR REIMBURSEMENT"
                elif 50.0 <= composite_score < 75.0:
                    verdict_label = "SELECTED FOR MANUAL REVIEW"
                else:
                    verdict_label = "REJECTED (SUSPECT IMAGE OUTLIER)"

            # Update row fields
            r.score_look_feel = s_lf
            r.score_structure_format = s_sf  
            r.score_content_accuracy = s_ca
            r.score_text_integrity = s_ti
            r.fraud_score = f"{composite_score:.1f}%"
            r.fraud_label = verdict_label


            # Progressive inline commits block silent background crash rollbacks
            try:
                session.commit()
            except Exception as row_err:
                session.rollback()
                print(f"Skipping lock on record {r.filename}: {row_err}")

        print("Multi-criteria dynamic evaluation synced to database")
        
    except Exception as e:
        session.rollback()
        print(f"Critical Evaluation Error: {str(e)}")
    finally:
        session.close()

if __name__ == "__main__":
    run_evaluation_suite()