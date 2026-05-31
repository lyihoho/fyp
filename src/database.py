# database.py
import os
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'receipts.db')}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Receipt(Base):
    __tablename__ = "receipts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String, nullable=True)
    merchant = Column(String, nullable=True)
    date = Column(String, nullable=True)
    total_amount = Column(Float, nullable=True)
    
    # Raw Sub-Metrics for the 4 Core Categories
    receipt_length = Column(Integer, nullable=True)  # Word count
    num_lines = Column(Integer, nullable=True)       # Line count
    layout_density_ratio = Column(Float, nullable=True)
    vertical_alignment_variance = Column(Float, nullable=True)
    avg_ocr_confidence = Column(Float, nullable=True)
    aspect_ratio = Column(Float, nullable=True)      # New: Width/Height shape metric
    math_valid_flag = Column(Integer, default=1)     # New: 1 if subtotal matches total, 0 if mismatch
    character_spacing_var = Column(Float, nullable=True) # New: Character box gap variance
    
    # YOUR 4 FINAL TARGET CONFIDENCE SCORES
    score_look_feel = Column(Float, nullable=True)       # Metric 1
    score_content_accuracy = Column(Float, nullable=True) # Metric 2
    score_text_integrity = Column(Float, nullable=True)   # Metric 3
    score_machine_purity = Column(Float, nullable=True)   # Metric 4
    
    # Overall System Decisions
    fraud_score = Column(String, default="pending")      # Weighted composite %
    fraud_label = Column(String, default="pending")      # System Verdict String

def init_db():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init_db()
    print("✅ database.py updated with the 4-Metric architecture slots!")