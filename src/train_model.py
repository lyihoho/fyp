# train_model.py
import os
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sqlalchemy import text

# Direct imports from your existing architecture setup
from database import SessionLocal, Receipt

MODEL_OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "isolation_forest_model.pkl")

def train_and_update_anomalies():
    print("🌲 [MODEL TRAINING] Connecting to receipts.db via SQLAlchemy...")
    session = SessionLocal()
    
    try:
        # 1. Pull all 25 records cleanly stored inside your database table
        receipts = session.query(Receipt).all()
        
        if len(receipts) < 5:
            print(f"❌ Error: Found only {len(receipts)} records. You need more rows before training.")
            return

        # 2. Extract your exact structural look-and-feel metrics for the matrix
        features = []
        for r in receipts:
            features.append([
                float(r.receipt_length or 0),
                float(r.num_lines or 0),
                float(r.total_amount or 0.0)
            ])
            
        X = np.array(features)

        print(f"📊 Training Isolation Forest on {len(X)} records across structural dimensions...")

        # 3. Initialize and fit the Unsupervised Isolation Forest Model
        # contamination=0.10 assumes roughly 10% of the dataset will fall out as an anomaly (like your BIFI slip)
        model = IsolationForest(n_estimators=100, contamination=0.06, random_state=42)
        model.fit(X)

        # 4. Calculate continuous spatial scores and classifications
        predictions = model.predict(X)         # Returns 1 for normal, -1 for anomaly
        raw_scores = model.decision_function(X)   # Continuous float (lower/more negative = more anomalous)

        print("🔄 Syncing calculated anomaly signatures back into database schema tracking columns...")
        
        # 5. Map the model outputs back to your existing columns
        for idx, r in enumerate(receipts):
            r.fraud_score = float(raw_scores[idx])
            
            # Map the predictive label string based on the model's tree splits
            if predictions[idx] == -1:
                r.fraud_label = "rejected"  # Flagged structural outlier
            else:
                r.fraud_label = "approved"  # Verified baseline match
                
        session.commit()
        print("✅ [SQLITE SUCCESS] Table records successfully updated with structural anomaly labels.")

        # 6. Save the trained model file permanently as a binary asset
        os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
        joblib.dump(model, MODEL_OUTPUT_PATH)
        print(f"📦 [EXPORT SUCCESS] Trained `.pkl` model file safely generated at:\n👉 {MODEL_OUTPUT_PATH}")

    except Exception as e:
        session.rollback()
        print(f"❌ Critical Error during model runtime execution: {str(e)}")
    finally:
        session.close()

if __name__ == "__main__":
    train_and_update_anomalies()