import time
from database import init_db, SessionLocal, Receipt
import parsing
import train_model
import anomalydetection

def execute_pipeline(images_directory, output_csv_backup, production_mode=False):
    print("=" * 20)
    print(f"PIPELINE STARTING | MODE: {'PRODUCTION' if production_mode else 'TRAINING'}")
    print("=" * 20)
    start_time = time.time()

    # ensure database infrastructure slots are ready
    init_db()

    # TRAINING MODE
    if not production_mode:
        print("\nParsing image batch into database...")
        #parsing.run_real_use_feature_pipeline(images_directory, output_csv_backup)

        print("\nTraining and exporting Isolation Forest models...")
        train_model.train_and_export_models()

        print("\nScoring and filtering rows...")
        anomalydetection.run_evaluation_suite()


    # PRODUCTION MODE
    else:
        print("\nScanning new upload...")
        # In gradio, images_directory would point to the uploaded file
        # parsing.py checks for dupe and creates a raw row entry.
        parsing.run_real_use_feature_pipeline(images_directory, output_csv_backup)

        print("\nTraining skipped. Using frozen .pkl files.")
    

        print("\nEvaluating fresh row entry against metrics...")
        anomalydetection.run_evaluation_suite()

    duration = time.time() - start_time
    print("=" * 20)
    print(f"PIPELINE RUN COMPLETED SUCCESSFULLY IN {duration:.2f} SECONDS")
    print("=" * 20)

if __name__ == "__main__":
    RAW_RECEIPTS_FOLDER = "data/demotest"
    BACKUP_DATA_CSV = "data/anomaly_detection_input.csv"
    
    # Set to False train system
    # Set to True for actual development to lock the models
    execute_pipeline(RAW_RECEIPTS_FOLDER, BACKUP_DATA_CSV, production_mode=True)