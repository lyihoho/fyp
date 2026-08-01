import os
from sqlalchemy import Column, Integer, String, Float, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Point absolute path to project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'demotest.db')}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Receipt(Base):
    __tablename__ = "demotest"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    merchant = Column(String, default="Unknown Store")
    date = Column(String, default="Unknown Date")
    total_amount = Column(Float, default=0.0)
    
    # ISOLATION FOREST COLUMNS
    receipt_length = Column(Integer, default=0)
    num_lines = Column(Integer, default=0)
    layout_density_ratio = Column(Float, default=0.0)
    vertical_alignment_variance = Column(Float, default=0.0)
    avg_ocr_confidence = Column(Float, default=0.0)
    aspect_ratio = Column(Float, default=0.0)
    math_valid_flag = Column(Integer, default=1)
    character_spacing_var = Column(Float, default=0.0)
    
    # EVALUATION ENGINE RESULTS
    fraud_score = Column(String, default="pending")
    fraud_label = Column(String, default="pending")

    # DUPE CHECKING FIELDS
    full_raw_text = Column(Text, nullable=True)
    feature_descriptors = Column(Text, nullable=True)

    # FOUR SCORE METRICS COLUMNS
    score_look_feel = Column(Float, nullable=True)         
    score_content_accuracy = Column(Float, nullable=True)   
    score_text_integrity = Column(Float, nullable=True)     
    score_structure_format = Column(Float, nullable=True)   

def init_db():
    Base.metadata.create_all(bind=engine)
    print("Database schema initialized")

if __name__ == "__main__":
    init_db()