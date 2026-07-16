"""
main.py
========
FastAPI application for Toxic Content Classification.

Endpoints:
1. POST /predict-text
2. POST /predict-image
3. GET /database
"""

import os
import shutil
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from imagecaption import generate_caption
from text_classifier import classify_text
from database import initialize_database, save_record, read_database

app = FastAPI(
    title="Toxic Content Classification API",
    version="1.0"
)

# Create the CSV database automatically if it doesn't exist
initialize_database()


class TextRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "Toxic Content Classification API is running."
    }


# ---------------------------------------------------------
# Text Classification
# ---------------------------------------------------------
@app.post("/predict-text")
def predict_text(request: TextRequest):

    result = classify_text(request.text)

    # Save to CSV
    save_record(
        request.text,
        result["label"]
    )

    return {
        "input": request.text,
        "prediction": result
    }


# ---------------------------------------------------------
# Image Classification
# ---------------------------------------------------------
@app.post("/predict-image")
def predict_image(file: UploadFile = File(...)):

    temp_path = "temp_image.jpg"

    try:
        # Save uploaded image
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Generate caption
        caption = generate_caption(temp_path)

        # Classify caption
        result = classify_text(caption)

        # Save to CSV
        save_record(
            caption,
            result["label"]
        )

        return {
            "caption": caption,
            "prediction": result
        }

    finally:
        # Delete temporary image
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ---------------------------------------------------------
# Database Viewer
# ---------------------------------------------------------
@app.get("/database")
def database():

    return {
        "total_records": len(read_database()),
        "records": read_database()
    }