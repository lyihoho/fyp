# database.py
import os
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Establish absolute database storage path mapping
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
    
    # Structural Metrics
    receipt_length = Column(Integer, nullable=True)  # True word count mapping
    num_lines = Column(Integer, nullable=True)       # True line/row count mapping
    
    # Core Geometric Look-and-Feel Metrics
    layout_density_ratio = Column(Float, nullable=True)
    vertical_alignment_variance = Column(Float, nullable=True)
    avg_ocr_confidence = Column(Float, nullable=True)
    
    # Validation Tracking Columns
    fraud_score = Column(String, default="pending")
    fraud_label = Column(String, default="pending")

def init_db():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init_db()
    print("✅ receipts.db initialized successfully with complete feature metrics tables!")