"""
routers/coding_assistant.py
FastAPI endpoints for the coding assistant feature:
chat (explain/generate), feedback learning, code execution, memory view.
"""

import time
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from modules.code_router import route_request
from modules.memory import (
    get_or_create_memory, add_turn, get_context_string,
    update_preferences, set_last_code, get_last_code,
)
from modules.code_runner import run_code
from modules.code_runner import extract_code_block
from modules.rag import add_document

router = APIRouter(prefix="/assistant", tags=["Coding Assistant"])


class ChatRequest(BaseModel):
    session_id: str
    message: str
    code_snippet: Optional[str] = ""
    language: Optional[str] = None
    framework: Optional[str] = None


class ChatResponse(BaseModel):
    intent: str
    type: str
    answer: str
    sources: Optional[List[str]] = None
    needs_feedback: bool = False


class ExecuteRequest(BaseModel):
    session_id: str
    code: Optional[str] = None  # if omitted, runs the session's last generated code


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    memory = get_or_create_memory(req.session_id)

    if req.language or req.framework:
        update_preferences(req.session_id, language=req.language, framework=req.framework)

    # If this session is waiting for a taught solution (Step 7 feedback loop)
    if memory["pending_feedback"]:
        doc_id = f"user_feedback_{int(time.time())}"
        add_document(
            text=req.message,
            metadata={
                "source": "user_feedback",
                "original_query": memory["pending_query"],
                "task_id": doc_id,
            },
            doc_id=doc_id,
        )
        answer = (
            "✅ Thanks! I've stored that solution in my knowledge base (Chroma) "
            "so I can use it for similar requests in the future."
        )
        add_turn(req.session_id, "user", req.message)
        add_turn(req.session_id, "assistant", answer)
        memory["pending_feedback"] = False
        memory["pending_query"] = None

        return ChatResponse(intent="feedback", type="feedback_stored", answer=answer)

    context_str = get_context_string(req.session_id)
    full_query = f"{context_str}\n\nCurrent request: {req.message}" if context_str else req.message

    result = route_request(full_query, code_snippet=req.code_snippet or "")

    add_turn(req.session_id, "user", req.message, intent=result.get("intent"))
    add_turn(req.session_id, "assistant", result["answer"], intent=result.get("intent"))

    if result["type"] == "needs_feedback":
        memory["pending_feedback"] = True
        memory["pending_query"] = req.message

    if result["type"] == "generated_code":
        code = extract_code_block(result["answer"])
        set_last_code(req.session_id, code)

    return ChatResponse(
        intent=result["intent"],
        type=result["type"],
        answer=result["answer"],
        sources=result.get("sources"),
        needs_feedback=(result["type"] == "needs_feedback"),
    )


@router.post("/execute")
def execute(req: ExecuteRequest):
    code = req.code or get_last_code(req.session_id)
    if not code:
        raise HTTPException(
            status_code=400,
            detail="No code provided, and no previously generated code found for this session.",
        )
    return run_code(code)


@router.get("/memory/{session_id}")
def view_memory(session_id: str):
    return get_or_create_memory(session_id)