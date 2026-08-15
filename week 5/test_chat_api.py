# test_chat_api.py
import requests

BASE = "http://127.0.0.1:8000"
session_id = "test-session-1"

r1 = requests.post(f"{BASE}/assistant/chat", json={
    "session_id": session_id,
    "message": "Explain this code",
    "code_snippet": "def add(a, b):\n    return a + b"
})
print("EXPLAIN:", r1.status_code, r1.json()["intent"])

r2 = requests.post(f"{BASE}/assistant/chat", json={
    "session_id": session_id,
    "message": "Write a function that checks if a list has close elements"
})
print("GENERATE:", r2.status_code, r2.json()["intent"], "sources:", r2.json().get("sources"))

r3 = requests.get(f"{BASE}/assistant/memory/{session_id}")
print("MEMORY turns:", len(r3.json()["history"]))