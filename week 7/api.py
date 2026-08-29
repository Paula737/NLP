"""
FastAPI backend for the Agentic RAG system.

Endpoints:
  GET  /                -> serves the frontend (static/index.html)
  GET  /api/documents   -> list currently ingested documents
  POST /api/ingest      -> upload one or more files, ingest into the vector DB
  POST /api/ask         -> ask a question; streams live pipeline progress via
                            Server-Sent Events (Retriever -> Analyst -> Answer)

Run with:
    uvicorn api:app --reload
Then open http://127.0.0.1:8000 in a browser.
"""

import json
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from pipeline.document_pipeline import ingest_files
from orchestrator import ask_stream

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "data" / "uploads"
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

app = FastAPI(title="Agentic RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    conversation_context: str = ""


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found at static/index.html")
    return index_path.read_text(encoding="utf-8")


@app.get("/api/documents")
def list_documents():
    """List files currently sitting in data/uploads (i.e. available to be
    cited/retrieved from). Doesn't guarantee they've been ingested into the
    vector DB - just that they've been uploaded."""
    if not UPLOADS_DIR.exists():
        return {"documents": []}
    docs = sorted(f.name for f in UPLOADS_DIR.iterdir() if f.is_file())
    return {"documents": docs}


@app.post("/api/ingest")
async def ingest_endpoint(files: list[UploadFile] = File(...)):
    """Accept one or more uploaded files, save them to data/uploads, and run
    them through the full document pipeline (load -> clean -> chunk ->
    embed -> store in Chroma)."""
    saved_paths = []
    skipped = []

    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            skipped.append(f.filename)
            continue
        dest = UPLOADS_DIR / f.filename
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved_paths.append(str(dest))

    if not saved_paths:
        raise HTTPException(
            status_code=400,
            detail=f"No supported files provided. Allowed types: {sorted(ALLOWED_EXTENSIONS)}",
        )

    try:
        ingest_files(saved_paths)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}") from e

    return {
        "ingested": [Path(p).name for p in saved_paths],
        "skipped_unsupported": skipped,
    }


@app.post("/api/ask")
async def ask_endpoint(payload: AskRequest):
    """Ask a question against the ingested documents. Streams progress
    events as Server-Sent Events so the frontend can show live pipeline
    status (Retriever -> Analyst -> Answer, including feedback loop count)
    rather than a blind spinner."""

    def event_stream():
        try:
            for event in ask_stream(payload.question, payload.conversation_context):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'stage': 'error', 'status': 'done', 'message': str(e)})}\n\n"
        yield "event: end\ndata: {}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
