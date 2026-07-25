"""
api.py
------
FastAPI backend for the personal RAG chatbot.

Run it with:
    uvicorn api:app --reload

Then open http://127.0.0.1:8000/docs for the interactive Swagger UI,
or POST to http://127.0.0.1:8000/ask with a JSON body:
    {"question": "What do you do for work?"}

Before running this, make sure you've already:
    1) Put your personal files (pdf/docx/txt) in the data/ folder
    2) Run: python ingest.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag_pipeline import answer_question

app = FastAPI(
    title="Personal RAG Chatbot API",
    description="Ask any question about the person, answered using RAG over their personal documents.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask about the person")
    top_k: int = Field(3, ge=1, le=10, description="How many context chunks to retrieve")


class Source(BaseModel):
    source: str
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["ui"], include_in_schema=False)
def serve_ui():
    """Serves the chat frontend. Visit this in your browser."""
    return FileResponse("frontend/index.html")


@app.get("/api", tags=["health"])
def api_root():
    return {"status": "ok", "message": "Personal RAG Chatbot API is running. See /docs for usage."}


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse, tags=["chat"])
def ask(request: AskRequest):
    """
    Ask a question about the person. The answer is generated using RAG:
    relevant chunks are retrieved from the vector store built by ingest.py,
    then a local LLM (flan-t5-base) generates the answer from that context.
    """
    try:
        result = answer_question(request.question, top_k=request.top_k)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=400,
            detail=f"{e} Run 'python ingest.py' first after adding your files to data/.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {e}")

    return AskResponse(answer=result["answer"], sources=result["sources"])