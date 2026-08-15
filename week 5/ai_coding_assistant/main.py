"""
main.py
Unified FastAPI app: coding assistant (RAG-based) + voice-based data
analysis, mounted as separate routers. Also serves the frontend.
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from routers import coding_assistant, data_analysis

app = FastAPI(title="AI Assistant Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # fine for local dev; restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(coding_assistant.router)
app.include_router(data_analysis.router)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return {"status": "ok", "message": "AI Assistant Platform is running. Frontend at /static/index.html"}