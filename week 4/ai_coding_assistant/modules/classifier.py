"""
modules/classifier.py
LLM-based intent classifier.
Classifies a user query into exactly one of two classes:
    - "explain"  -> user wants existing code explained
    - "generate" -> user wants new code written
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

GROQ_MODEL = "llama-3.1-8b-instant"

CLASSIFIER_PROMPT = """You are an intent classifier for a coding assistant.

Classify the user's request into EXACTLY one of these two categories:

- "explain": the user wants an explanation of existing code (e.g. "explain this code",
  "what does this function do", "why is this loop incorrect", "explain line by line")
- "generate": the user wants new code written (e.g. "write python code",
  "generate a CNN model", "build a Flask API", "create a sorting algorithm")

User request:
\"\"\"{query}\"\"\"

Reply with EXACTLY one word, lowercase, nothing else: explain or generate
"""


def classify_intent(user_query: str) -> str:
    """
    Returns "explain" or "generate".
    Defaults to "generate" if the model output is ambiguous (safer fallback,
    since generate path includes the relevance/RAG safety net anyway).
    """
    prompt = CLASSIFIER_PROMPT.format(query=user_query)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    result = response.choices[0].message.content.strip().lower()

    if "explain" in result:
        return "explain"
    return "generate"