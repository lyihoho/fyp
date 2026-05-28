import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from database import init_db, SessionLocal, Receipt
from parsing import parse_receipt
from anomalydetection import AnomalyDetection

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Claim Fraud Detection API")
init_db()

#Initialize model
detector = AnomalyDetection(contamination=0.1)
detector.fitted = True #bypass fitting atm

#Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/upload_receipt")
async def upload_receipt(file: UploadFile = File(...)):
    #save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    #parse receipt
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    parsed = parse_receipt(text)

    #predict anomaly score and label
    score = detector.predict_score(parsed)
    label = "Fraud" if detector.predict_label(parsed) else "Normal"

    #save to db
    db: Session = next(get_db())
    receipt_db = Receipt(
        filename=file.filename,
        merchant=parsed["merchant"],
        date=parsed["date"],
        total_amount=parsed["total_amount"],
        receipt_length=parsed["receipt_length"],
        num_lines=parsed["num_lines"],
        num_amounts=parsed["num_amounts"],
        avg_item_price=parsed["avg_item_price"],
        contains_tax=parsed["contains_tax"],
        fraud_score=score,
        fraud_label=label
    )
    db.add(receipt_db)
    db.commit()
    db.refresh(receipt_db)

    return JSONResponse({
        "filename": file.filename,
        "parsed_data": parsed,
        "fraud_score": score,
        "fraud_label": label
    })

@app.get("/history")
def get_history():
    db: Session = next(get_db())
    receipts = db.query(Receipt).all()
    data = [
        {
            "id": r.id,
            "filename": r.filename,
            "merchant": r.merchant,
            "date": r.date,
            "total_amount": r.total_amount,
            "fraud_score": r.fraud_score,
            "fraud_label": r.fraud_label
        }
        for r in receipts
    ]
    return JSONResponse(data)
