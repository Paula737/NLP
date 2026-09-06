"""
FastAPI backend for the Agentic RAG system.

Endpoints:
  GET    /                    -> serves the frontend (static/index.html)
  GET    /api/documents       -> list currently ingested documents
  POST   /api/ingest          -> upload one or more files, ingest into the vector DB
  DELETE /api/documents/{name}-> remove one document from the knowledge base
  DELETE /api/documents       -> clear the entire knowledge base
  POST   /api/ask             -> ask a question (optionally scoped to one
                                  document via source_filter); streams live
                                  pipeline progress via Server-Sent Events
  POST   /api/transcribe      -> transcribe a recorded voice clip into text
                                  using Groq's Whisper API (voice queries)

Run with:
    uvicorn api:app --reload
Then open http://127.0.0.1:8000 in a browser.
"""

import json
import os
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import unquote

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from pydantic import BaseModel
from groq import Groq

from pipeline.document_pipeline import ingest_files, delete_document, clear_all_documents
from orchestrator import ask_stream
from agents.report_agent import generate_report

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "data" / "uploads"
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

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
    source_filter: str = ""  # if set to a filename, scopes the answer to only that document

class ReportRequest(BaseModel):
    question: str
    answer: str  # the full final answer text, including inline citations and the Sources footer

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


@app.delete("/api/documents/{filename}")
def delete_document_endpoint(filename: str):
    """Remove one document from the knowledge base (vector DB + uploaded file)."""
    filename = unquote(filename)
    try:
        removed_chunks = delete_document(filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {e}") from e
    return {"deleted": filename, "chunks_removed": removed_chunks}


@app.delete("/api/documents")
def clear_all_documents_endpoint():
    """Wipe the entire knowledge base. Cannot be undone."""
    try:
        removed_files = clear_all_documents()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear: {e}") from e
    return {"files_removed": removed_files}


@app.post("/api/ask")
async def ask_endpoint(payload: AskRequest):
    """Ask a question against the ingested documents. Streams progress
    events as Server-Sent Events so the frontend can show live pipeline
    status (Retriever -> Analyst -> Answer, including feedback loop count)
    rather than a blind spinner. If source_filter is set, both the initial
    retrieval and the Analyst's feedback loop stay scoped to that one document."""

    def event_stream():
        try:
            for event in ask_stream(payload.question, payload.conversation_context, payload.source_filter):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'stage': 'error', 'status': 'done', 'message': str(e)})}\n\n"
        yield "event: end\ndata: {}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.post("/api/transcribe")
async def transcribe_endpoint(audio: UploadFile = File(...)):
    """Transcribe a recorded voice clip into text using Groq's Whisper API,
    so the user can ask a question by speaking instead of typing. Returns
    the transcribed text for the frontend to drop into the question box."""
    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set in the environment.")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio data received.")

    try:
        client = Groq()
        transcription = client.audio.transcriptions.create(
            file=(audio.filename or "recording.webm", audio_bytes),
            model="whisper-large-v3-turbo",
            response_format="json",
            temperature=0.0,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}") from e

    return {"text": transcription.text.strip()}

def _parse_sources_footer(answer: str) -> list[dict]:
    """Parse the '**Sources**\\n- doc.pdf (page 1)' footer format that
    answer_agent.py's _render_sources_section() produces, into a list of
    {"document": str, "pages": List[int]} dicts for the report agent."""
    sources = []
    match = re.search(r"\*\*Sources\*\*\s*\n(.+)", answer, re.DOTALL)
    if not match:
        return sources
    for line in match.group(1).strip().split("\n"):
        m = re.match(r"-\s*(.+?)(?:\s*\(pages?\s*([\d,\s]+)\))?\s*$", line.strip())
        if not m:
            continue
        doc = m.group(1).strip()
        pages_str = m.group(2)
        pages = [int(p.strip()) for p in pages_str.split(",")] if pages_str else []
        sources.append({"document": doc, "pages": pages})
    return sources


@app.post("/api/report")
async def generate_report_endpoint(payload: ReportRequest):
    """Generate a polished PDF report from a finished, cited answer. The
    Report Generator Agent decides the styling (font/color theme) and
    structure itself. Returns the PDF file for direct download."""
    sources = _parse_sources_footer(payload.answer)
    answer_body = re.split(r"\*\*Sources\*\*", payload.answer)[0].strip()

    try:
        pdf_path = generate_report(payload.question, answer_body, sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}") from e

    return FileResponse(pdf_path, media_type="application/pdf", filename=Path(pdf_path).name)