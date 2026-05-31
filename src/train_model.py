# train_model.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler  # CamScanner-style Math Normalizer
from database import SessionLocal, Receipt

# Configuration for file outputs
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
LF_MODEL_PATH = os.path.join(MODELS_DIR, "look_feel_forest.pkl")
SF_MODEL_PATH = os.path.join(MODELS_DIR, "structure_forest.pkl")

# New Blueprint Paths to freeze feature scale properties
LF_SCALER_PATH = os.path.join(MODELS_DIR, "look_feel_scaler.pkl")
SF_SCALER_PATH = os.path.join(MODELS_DIR, "structure_scaler.pkl")

def train_and_export_models():
    print("🌲 [MODEL TRAINING] Connecting to receipts.db via SQLAlchemy...")
    session = SessionLocal()
    
    try:
        receipts = session.query(Receipt).all()
        if len(receipts) < 5:
            print(f"❌ Error: Found only {len(receipts)} records. You need more rows before training.")
            return

        # Convert to DataFrame cleanly for matrix extraction slicing
        data_pool = [r.__dict__.copy() for r in receipts]
        for d in data_pool: d.pop('_sa_instance_state', None)
        df = pd.DataFrame(data_pool)

        os.makedirs(MODELS_DIR, exist_ok=True)

        # ---------------------------------------------------------------------
        # LAYER 1: SCALING & TRAINING LOOK & FEEL FORREST
        # ---------------------------------------------------------------------
        X_lf = df[["layout_density_ratio", "receipt_length", "aspect_ratio"]].values
        print(f"📊 Standardizing & Training Look & Feel Forest on {len(X_lf)} rows...")
        
        # Instantiate and fit the mathematical normalization criteria matrix
        scaler_lf = StandardScaler()
        X_lf_scaled = scaler_lf.fit_transform(X_lf)
        
        lf_model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        lf_model.fit(X_lf_scaled)
        
        # Export frozen matrix artifacts
        joblib.dump(lf_model, LF_MODEL_PATH)
        joblib.dump(scaler_lf, LF_SCALER_PATH)

        # ---------------------------------------------------------------------
        # LAYER 2: SCALING & TRAINING STRUCTURE & FORMAT FORREST
        # ---------------------------------------------------------------------
        X_sf = df[["num_lines", "vertical_alignment_variance"]].values
        print(f"📐 Standardizing & Training Structure Forest on {len(X_sf)} rows...")
        
        # Instantiate and fit variance normalizer to protect against handheld distance skewing
        scaler_sf = StandardScaler()
        X_sf_scaled = scaler_sf.fit_transform(X_sf)
        
        sf_model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        sf_model.fit(X_sf_scaled)
        
        # Export frozen matrix artifacts
        joblib.dump(sf_model, SF_MODEL_PATH)
        joblib.dump(scaler_sf, SF_SCALER_PATH)

        print("\n📦 [EXPORT SUCCESS] All 2 Models and 2 Scalers successfully frozen as binaries!")
        print(f" 👉 Model Directory Assets: {MODELS_DIR}")

    except Exception as e:
        print(f"❌ Critical Error during model training: {str(e)}")
    finally:
        session.close()

if __name__ == "__main__":
    train_and_export_models()