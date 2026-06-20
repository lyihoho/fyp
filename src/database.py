# database.py
import os
from sqlalchemy import Column, Integer, String, Float, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Resolve absolute path to project directory cleanly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'receipts.db')}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Receipt(Base):
    __tablename__ = "receipts"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, nullable=False)
    merchant = Column(String, default="Unknown Store")
    date = Column(String, default="Unknown Date")
    total_amount = Column(Float, default=0.0)
    
    # --- ISOLATION FOREST STRUCTURAL ATOMS ---
    receipt_length = Column(Integer, default=0)
    num_lines = Column(Integer, default=0)
    layout_density_ratio = Column(Float, default=0.0)
    vertical_alignment_variance = Column(Float, default=0.0)
    avg_ocr_confidence = Column(Float, default=0.0)
    aspect_ratio = Column(Float, default=0.0)
    math_valid_flag = Column(Integer, default=1)
    character_spacing_var = Column(Float, default=0.0)
    
    # --- EVALUATION ENGINE RESULTS ---
    fraud_score = Column(String, default="pending")
    fraud_label = Column(String, default="pending")

    # --- THE ACADEMIC DUAL-SECURITY FIELDS ---
    full_raw_text = Column(Text, nullable=True)
    feature_descriptors = Column(Text, nullable=True)

    # ==========================================================================
    # 📊 THE FIXED FOUR CORE VECTOR SCORE METRICS COLUMNS
    # ==========================================================================
    score_look_feel = Column(Float, nullable=True)         # Metric 1
    score_content_accuracy = Column(Float, nullable=True)   # Metric 2
    score_text_integrity = Column(Float, nullable=True)     # Metric 3
    score_structure_format = Column(Float, nullable=True)   # Metric 4

def init_db():
    """Initializes the database tables completely from SQLAlchemy metadata definitions."""
    Base.metadata.create_all(bind=engine)
    print("✅ Database schema initialized completely via init_db().")

if __name__ == "__main__":
    init_db()