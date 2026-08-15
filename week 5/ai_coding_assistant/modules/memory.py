"""
modules/memory.py
Session-based conversation memory for FastAPI.
Each session_id gets its own memory dict, held in an in-process store.

NOTE: in-memory only — restarting the server clears all sessions.
Fine for coursework; swap the _SESSIONS dict for Redis/SQLite later
if you need persistence across restarts.
"""

from typing import Optional

_SESSIONS: dict[str, dict] = {}


def get_or_create_memory(session_id: str) -> dict:
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = {
            "history": [],
            "preferences": {"language": None, "framework": None},
            "last_generated_code": None,
            "pending_feedback": False,
            "pending_query": None,
        }
    return _SESSIONS[session_id]


def add_turn(session_id: str, role: str, content: str, intent: Optional[str] = None):
    memory = get_or_create_memory(session_id)
    memory["history"].append({"role": role, "content": content, "intent": intent})


def update_preferences(session_id: str, language: str = None, framework: str = None):
    memory = get_or_create_memory(session_id)
    if language:
        memory["preferences"]["language"] = language
    if framework:
        memory["preferences"]["framework"] = framework


def set_last_code(session_id: str, code: str):
    get_or_create_memory(session_id)["last_generated_code"] = code


def get_last_code(session_id: str) -> Optional[str]:
    return get_or_create_memory(session_id).get("last_generated_code")


def get_context_string(session_id: str, max_turns: int = 6) -> str:
    memory = get_or_create_memory(session_id)
    parts = []

    prefs = memory["preferences"]
    if prefs["language"] or prefs["framework"]:
        parts.append(
            f"User preferences — language: {prefs['language'] or 'unspecified'}, "
            f"framework: {prefs['framework'] or 'unspecified'}."
        )

    recent = memory["history"][-max_turns:]
    if recent:
        parts.append("Recent conversation:")
        for turn in recent:
            parts.append(f"{turn['role']}: {turn['content'][:300]}")

    return "\n".join(parts)