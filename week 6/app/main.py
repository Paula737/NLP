import os
import shutil
import uuid
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.ingestion.pipeline import ingest_file, ingest_url, ingest_wikipedia, ingest_audio
from app.workflow.orchestrator import run_workflow
from app.cache.redis_cache import cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluator_generator_rag")

app = FastAPI(title="Evaluator-Generator RAG Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

EXT_TO_TYPE = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "txt",
    ".py": "code", ".js": "code", ".ts": "code", ".java": "code",
    ".cpp": "code", ".c": "code", ".go": "code", ".rb": "code",
    ".pptx": "pptx", ".ppt": "pptx",
}


class AskRequest(BaseModel):
    session_id: str
    question: str


class UrlRequest(BaseModel):
    url: str


class WikiRequest(BaseModel):
    topic: str


@app.get("/health")
async def health():
    return {"status": "ok", "redis": cache.ping()}


@app.post("/ingest/file")
async def ingest_file_endpoint(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    dest = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")

    try:
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)

        if ext == ".wav":
            result = ingest_audio(dest)
        else:
            file_type = EXT_TO_TYPE.get(ext)
            if not file_type:
                raise HTTPException(status_code=400, detail=f"Unsupported file extension: {ext}")
            result = ingest_file(dest, file_type)

        logger.info("Ingested file %s -> %s", file.filename, result)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Ingestion failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ingest/url")
async def ingest_url_endpoint(req: UrlRequest):
    try:
        result = ingest_url(req.url)
        logger.info("Ingested URL %s -> %s", req.url, result)
        return result
    except Exception as exc:
        logger.exception("URL ingestion failed for %s", req.url)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ingest/wikipedia")
async def ingest_wikipedia_endpoint(req: WikiRequest):
    try:
        result = ingest_wikipedia(req.topic)
        logger.info("Ingested Wikipedia topic %s -> %s", req.topic, result)
        return result
    except Exception as exc:
        logger.exception("Wikipedia ingestion failed for %s", req.topic)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ask")
async def ask_endpoint(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = run_workflow(req.session_id, req.question)
        logger.info("Answered session=%s status=%s loops=%s",
                     req.session_id, result["status"], result["loops_used"])
        return result
    except Exception as exc:
        logger.exception("Workflow failed for session=%s", req.session_id)
        raise HTTPException(status_code=500, detail=str(exc))


# Serve the simple frontend last, so /health, /ingest/*, /ask take priority
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
