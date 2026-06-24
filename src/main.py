# main.py
import os
import sys
import time
from database import init_db, SessionLocal, Receipt

# Import your architectural modules
import parsing
import train_model
import anomalydetection
#import report_generator

def execute_pipeline(images_directory, output_csv_backup, production_mode=False):
    print("=" * 40)
    print(f"PIPELINE STARTING | MODE: {'PRODUCTION' if production_mode else 'BASELINE TRAINING'}")
    print("=" * 40)
    start_time = time.time()

    # Step 0: Ensure database infrastructure slots are ready
    init_db()

    # ---------------------------------------------------------------------
    # SYSTEM MODE A: BASELINE TRAINING
    # ---------------------------------------------------------------------
    if not production_mode:
        print("\n[DEVELOPMENT STEP 1] Parsing image directory batch into database rows...")
        parsing.run_real_use_feature_pipeline(images_directory, output_csv_backup)

        print("\n[DEVELOPMENT STEP  2] Re-fitting StandardScaler matrices & Isolation Forests...")
        train_model.train_and_export_models()

        print("\n[DEVELOPMENT STEP 3] Re-scoring whole database pool against new baseline...")
        anomalydetection.run_evaluation_suite()
        
        # Keep PDF generation turned off during heavy ingestion rounds to save speed
        print("\n[DEVELOPMENT STEP 4] Skipping PDF generation for batch speed.")

    # ---------------------------------------------------------------------
    # SYSTEM MODE B: PRODUCTION PIPELINE
    # ---------------------------------------------------------------------
    else:
        print("\n[PRODUCTION STEP 1] Scanning new production upload slot...")
        # In a live app, images_directory would point to the single uploaded file
        # parsing.py extracts it, checks uniqueness against DB, and creates a raw row entry.
        parsing.run_real_use_feature_pipeline(images_directory, output_csv_backup)

        print("\nTraining skipped. Using frozen .pkl model weights.")
    

        print("\n[PRODUCTION STEP 2] Evaluating fresh row entry against central database metrics...")
        anomalydetection.run_evaluation_suite()

        #print("\n[PRODUCTION STEP 4] Compiling downloadable Turnitin-style PDF audit sheet...")
        #report_generator.generate_all_pending_reports()

    duration = time.time() - start_time
    print("=" * 40)
    print(f"🎉 PIPELINE RUN COMPLETED SUCCESSFULLY IN {duration:.2f} SECONDS")
    print("=" * 40)

if __name__ == "__main__":
    RAW_RECEIPTS_FOLDER = "data/test_2"
    BACKUP_DATA_CSV = "data/anomaly_detection_input.csv"
    
    # 🛠️ YOUR MASTER SWITCH:
    # Set to False right now to train system
    # Set to True later for actual development to lock the models and generate pdf report
    execute_pipeline(RAW_RECEIPTS_FOLDER, BACKUP_DATA_CSV, production_mode=True)