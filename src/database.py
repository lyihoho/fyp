# database.py
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()
DB_URL = "sqlite:///receipts.db"

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

class Receipt(Base):
    __tablename__ = "receipts"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    merchant = Column(String)
    date = Column(String)
    total_amount = Column(Float)
    receipt_length = Column(Integer)
    num_lines = Column(Integer)
    num_amounts = Column(Integer)
    avg_item_price = Column(Float)
    contains_tax = Column(Integer)
    fraud_score = Column(Float)
    fraud_label = Column(String, default="pending")  # approved/rejected/pending

def init_db():
    Base.metadata.create_all(bind=engine)
