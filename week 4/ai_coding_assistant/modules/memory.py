"""
modules/memory.py
Simple in-session conversation memory.
Streamlit reruns the whole script on every interaction, so this is
designed to be stored in st.session_state (wired up in app.py) —
this module just defines the data structure and helper functions.
"""


def init_memory() -> dict:
    """
    Call once per session to create a fresh memory object.
    """
    return {
        "history": [],           # list of {"role": "user"/"assistant", "content": str, "intent": str}
        "preferences": {
            "language": None,    # e.g. "Python"
            "framework": None,   # e.g. "Flask"
        },
        "last_generated_code": None,
    }


def add_turn(memory: dict, role: str, content: str, intent: str = None):
    memory["history"].append({"role": role, "content": content, "intent": intent})


def update_preferences(memory: dict, language: str = None, framework: str = None):
    if language:
        memory["preferences"]["language"] = language
    if framework:
        memory["preferences"]["framework"] = framework


def set_last_code(memory: dict, code: str):
    memory["last_generated_code"] = code


def get_context_string(memory: dict, max_turns: int = 6) -> str:
    """
    Formats recent history + preferences into a string to prepend to LLM prompts,
    so every call stays context-aware (spec requirement).
    """
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