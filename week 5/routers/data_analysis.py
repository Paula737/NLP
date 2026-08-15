"""
routers/data_analysis.py
FastAPI endpoints for the voice-based data analysis feature:
dataset upload, audio transcription, SQL generation/execution,
and the full orchestrated pipeline.
"""

import os
import shutil
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from modules.db_manager import load_file_to_dataframe, store_dataframe_in_sqlite, get_current_schema
from modules.transcriber import transcribe_audio
from modules.sql_generator import generate_sql
from modules.sql_executor import execute_sql
from modules.explainer import explain_results

router = APIRouter(prefix="/data", tags=["Data Analysis"])


class SQLRequest(BaseModel):
    question: str


class ExecuteSQLRequest(BaseModel):
    sql: str


@router.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".csv", ".xlsx", ".xls"):
        raise HTTPException(status_code=400, detail="Only .csv, .xlsx, .xls files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        df = load_file_to_dataframe(tmp_path)
        schema_info = store_dataframe_in_sqlite(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process file: {e}")
    finally:
        os.remove(tmp_path)

    return {"message": f"Dataset '{file.filename}' loaded successfully.", "schema": schema_info}


@router.get("/schema")
def schema():
    try:
        return get_current_schema()
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = transcribe_audio(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Transcription failed: {e}")
    finally:
        os.remove(tmp_path)

    return result


@router.post("/generate-sql")
def generate_sql_endpoint(req: SQLRequest):
    try:
        schema_info = get_current_schema()
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))

    sql = generate_sql(req.question, schema_info)
    return {"question": req.question, "sql": sql}


@router.post("/execute-query")
def execute_query_endpoint(req: ExecuteSQLRequest):
    result = execute_sql(req.sql)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Full pipeline: audio in -> transcription -> SQL generation ->
    execution -> explanation out. Requires a dataset already uploaded.
    """
    try:
        schema_info = get_current_schema()
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))

    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        transcription = transcribe_audio(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Transcription failed: {e}")
    finally:
        os.remove(tmp_path)

    question = transcription["text"]
    sql = generate_sql(question, schema_info)
    result = execute_sql(sql)

    if not result["success"]:
        return {
            "transcription": transcription,
            "sql": sql,
            "success": False,
            "error": result["error"],
        }

    explanation = explain_results(question, sql, result["rows"])

    return {
        "transcription": transcription,
        "sql": sql,
        "rows": result["rows"],
        "columns": result["columns"],
        "explanation": explanation,
    }