# AI-Driven Automated Expense Claim Verification & Anomaly Detection System

An automated financial compliance routing pipeline developed as a BSc (Honours) in Computer Science Final Year Project (FYP) at Sunway University. The system combines computer vision, layout extraction, deterministic arithmetic checks, and machine learning anomaly detection models to replace slow and insecure manual expense claim auditing.

---

# Key Features

* **Visual & Textual Duplicate Check (Gate 1)**: Runs parallel dual-engine verification. Visually compares receipt layout matrices using **OpenCV ORB** matching (Hamming distance) and textually using **Levenshtein Edit Distance** to block recaptured photo duplicates.
* **Tampering & Math Auditor (Gate 2)**: Scans transaction totals and individual item lists. Utilizes place-value math rules to detect manual pen alterations (e.g., prepending or changing digits).
* **Isolation Forest Anomaly Scoring (Gate 3)**: Leverages dual unsupervised decision forests (**Scikit-Learn**) to evaluate global geometries (aspect ratio, margins) and structural density metrics (column alignment, characters per line).
* **Gradio Web Interface**: Provides a user-friendly local web dashboard for claims submission, live evaluation, database ledger review, and automated vector PDF compliance report downloads.

---

# Repository Directory Layout

fyp/
├── data/
│   ├── test/                  # SROIE test receipt image dataset (383 files)
│   ├── demotest/              # Live UAT evaluation receipt images (7 files)
│   └── processed_data/        # Extracted combined training feature arrays
├── src/
│   ├── models/                # Frozen trained Isolation Forest model binaries (.pkl)
│   ├── api.py                 # Live Gradio User Interface (UI) web server
│   ├── main.py                # Command-Line batch execution script (for Viva Demo)
│   ├── anomalydetection.py    # Feature engineering and Isolation Forest ML scoring
│   ├── parsing.py             # OCR text extraction layer and place-value math checker
│   ├── database.py            # SQLite database schema, CRUD logs, and duplicate checking
│   ├── train_model.py         # Retraining script for Isolation Forest models
│   ├── demotest.db            # SQLite database containing UAT evaluation records
│   └── receipts.db            # SQLite database containing training feature tables
├── .gitignore                 # Configured directory exclusions
└── README                     # Project setup and execution manual


---

# Setup & Installation

# 1. Clone the repository
git clone <your-repository-url>
cd fyp

# 2. Activate the Virtual Environment
Ensure you are using **Windows PowerShell**:
.\venv\Scripts\Activate.ps1

# 3. Install Project Dependencies
Install dependencies including OpenCV, PaddleOCR, and Scikit-Learn:
pip install -r requirements.txt
*(Note: If you do not have a `requirements.txt` file, install the main packages manually: `pip install opencv-python paddleocr scikit-learn pandas joblib pypdf reportlab openpyxl`)*

---

# How to Run the System

# Option A: Run the Live Gradio Web UI (api.py)
To start the interactive web application, run:
python src/api.py
1. Open the local address in your browser: `http://127.0.5.1:7860`.
2. Upload a receipt from `data/demotest/` (e.g., `normal1.jpeg` or `tampered.jpeg`).
3. View the color-coded verdict banner, download the audit PDF report, and click **"Synchronize Ledger Registry"** to see live SQL records synced to `demotest.db`.

# Option B: Run the Command-Line Batch Demo (main.py)
To run the automated batch pipeline across all test assets in the terminal without opening a browser:
python src/main.py
This runs the entire verification suite sequentially across `data/demotest/` and outputs the logs, database writes, and verdicts directly to the terminal console.

# Option C: Retrain the Machine Learning Models (train_model.py)
To retrain the Isolation Forest models on the dataset stored in `receipts.db` and output new `.pkl` files:
python src/train_model.py

---

# Attribution
Developed by **Lim Yi** (ID: 22078430) as a capstone project for the Department of Computing and Information Systems, Sunway University.
