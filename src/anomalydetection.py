import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from difflib import SequenceMatcher

class MultiCriteriaAnomalyEngine:
    def __init__(self, contamination=0.05, random_state=42):
        # Two separate Isolation Forests to isolate Look & Feel from Structure format
        self.look_feel_model = IsolationForest(contamination=contamination, random_state=random_state)
        self.structure_model = IsolationForest(contamination=contamination, random_state=random_state)
        self.fitted = False

    def train_on_baseline_data(self, database_df):
        """
        TRAINING PHASE: Feeds your historical database rows into the models 
        so they can learn the baseline geometric boundaries of your receipts.
        """
        # Train Look & Feel on Density and Word Count
        X_look_feel = database_df[["layout_density_ratio", "word_count"]].values
        self.look_feel_model.fit(X_look_feel)
        
        # Train Structure & Format on Line Variance
        X_structure = database_df[["vertical_alignment_variance"]].values
        self.structure_model.fit(X_structure)
        
        self.fitted = True
        print(f"--> [TRAINING COMPLETE] Machine learning models trained on {len(database_df)} receipt baselines.")

    def calculate_section_scores(self, target_receipt, database_df):
        """
        EVALUATION PHASE: Uses the trained ML models and text logic to output
        4 separate individual section scores (0.0% to 100.0%).
        """
        if not self.fitted:
            raise ValueError("The machine learning models must be trained via train_on_baseline_data() first!")

        # ---------------------------------------------------------------------
        # SECTION 1: LOOK & FEEL SCORE (Powered by Trained Isolation Forest)
        # ---------------------------------------------------------------------
        lf_features = np.array([[target_receipt["layout_density_ratio"], target_receipt["word_count"]]])
        # decision_function returns a score where negative = anomaly outlier
        lf_ml_score = self.look_feel_model.decision_function(lf_features)[0]
        
        # Map the raw ML score to a clean 0-100% presentation rating
        s_look_feel = max(0.0, min(100.0, (lf_ml_score + 0.5) * 200.0))

        # ---------------------------------------------------------------------
        # SECTION 2: STRUCTURE & FORMAT SCORE (Powered by Trained Isolation Forest)
        # ---------------------------------------------------------------------
        sf_features = np.array([[target_receipt["vertical_alignment_variance"]]])
        sf_ml_score = self.structure_model.decision_function(sf_features)[0]
        
        s_structure = max(0.0, min(100.0, (sf_ml_score + 0.5) * 200.0))

        # ---------------------------------------------------------------------
        # SECTION 3: CONTENT ACCURACY SCORE (Rule-Based Field Validation)
        # ---------------------------------------------------------------------
        s_content = 100.0
        if float(target_receipt.get("extracted_total", 0.0)) <= 0.0:
            s_content -= 50.0  
        if "unknown" in str(target_receipt.get("extracted_store", "")).lower():
            s_content -= 30.0
        if "unknown" in str(target_receipt.get("date", "")).lower():
            s_content -= 20.0

        # ---------------------------------------------------------------------
        # SECTION 4: HANDWRITING / ALTERATION SECURITY SCORE (OCR Confidence)
        # ---------------------------------------------------------------------
        s_handwriting = 100.0
        avg_ocr_conf = float(target_receipt.get("avg_ocr_confidence", 100.0))
        
        if avg_ocr_conf < 75.0:
            confidence_deficit = 75.0 - avg_ocr_conf
            s_handwriting -= (confidence_deficit * 3.5)
            
        s_handwriting = max(0.0, min(100.0, s_handwriting))

        # ---------------------------------------------------------------------
        # GUARDRAIL: CLONE DUPLICATE HARD-DROP
        # ---------------------------------------------------------------------
        # If an incoming receipt is a perfect spatial clone clone of a database file,
        # we bypass the average and instantly flag it as a duplicate threat.
        new_store = str(target_receipt.get("extracted_store", "")).lower().strip()
        new_total = float(target_receipt.get("extracted_total", 0.0))
        new_date = str(target_receipt.get("date", "")).strip().lower()

        for idx, row in database_df.iterrows():
            total_matches = abs(new_total - float(row.get("extracted_total", 0.0))) < 0.01
            stored_date = str(row.get("date", "")).strip().lower()
            
            if total_matches:
                if "unknown" not in new_date and "unknown" not in stored_date and new_date != stored_date:
                    continue # Different transaction dates = Safe recurring buy!
                
                stored_store = str(row.get("extracted_store", "")).lower().strip()
                text_sim = SequenceMatcher(None, new_store, stored_store).ratio()
                
                # If store names match and layout features are tightly mirrored
                if text_sim > 0.80 and abs(target_receipt["layout_density_ratio"] - row["layout_density_ratio"]) < 0.005:
                    return round(s_look_feel, 1), round(s_structure, 1), round(s_content, 1), 0.0 # Force handwriting/security to 0%

        return round(s_look_feel, 1), round(s_structure, 1), round(s_content, 1), round(s_handwriting, 1)


if __name__ == "__main__":
    DATABASE_CSV = "data/anomaly_detection_input.csv"
    
    print("="*80)
    print("TRAINING & RUNNING RUNTIME MULTI-CRITERIA AI AUDIT SUITE")
    print("="*80)
    
    try:
        db_df = pd.read_csv(DATABASE_CSV)
    except FileNotFoundError:
        print(f"Error: {DATABASE_CSV} not found. Run parsing.py first!")
        exit()
        
    # Initialize our engine
    engine = MultiCriteriaAnomalyEngine(contamination=0.05)
    
    # 1. RUN TRAINING PHASE NATIVELY
    engine.train_on_baseline_data(db_df)
    print("-" * 80)
    
    # 2. EVALUATE TRACKS
    for idx, target_row in db_df.iterrows():
        background_db = db_df.drop(idx)
        current_features = target_row.to_dict()
        
        # Compute individual metrics via trained trees
        s_lf, s_sf, s_ca, s_hw = engine.calculate_section_scores(current_features, background_db)
        
        # Calculate final composite score
        final_score = (s_lf + s_sf + s_ca + s_hw) / 4.0
        
        print(f"File Reference: {current_features['filename']}")
        print(f"↳ Merchant: {str(current_features['extracted_store'])[:15]:<15} | Price: RM {current_features['extracted_total']}")
        print(f"  [EVALUATION SCORECARD]:")
        print(f"  ├─ 1. Look & Feel Score (ML)  : {s_lf}%")
        print(f"  ├─ 2. Structure & Format (ML) : {s_sf}%")
        print(f"  ├─ 3. Content Accuracy Score  : {s_ca}%")
        print(f"  └─ 4. Handwriting/Scribble Sec: {s_hw}%")
        
        if s_hw == 0.0:
            print(f"  🚨 [FINAL VERDICT]: {final_score:.1f}% -> CRITICAL IDENTICAL CLONE SCAM DETECTED")
        elif final_score < 75.0:
            print(f"  ⚠️  [FINAL VERDICT]: {final_score:.1f}% -> SUSPECT PROFILE WARNING (Outlier Metrics)")
        else:
            print(f"  ✅ [FINAL VERDICT]: {final_score:.1f}% -> DOCUMENT VERIFIED CLEAN AND AUTHENTIC")
        print("-" * 80)