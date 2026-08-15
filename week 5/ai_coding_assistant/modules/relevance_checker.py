"""
modules/relevance_checker.py
Uses an LLM as a judge to decide whether retrieved context is
sufficiently relevant to answer the user's code-generation request.

Combines a cheap numeric pre-filter (distance threshold) with an
LLM judge for borderline / final decision, per the project spec.
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

GROQ_MODEL = "llama-3.1-8b-instant"  # fast + free, plenty for this task

# Below this distance, we trust the match without even asking the LLM
DISTANCE_AUTO_ACCEPT = 0.35
# Above this distance, we reject without asking the LLM (saves API calls)
DISTANCE_AUTO_REJECT = 0.9


def check_relevance(user_query: str, retrieved: list[dict], formatted_context: str) -> dict:
    """
    Returns:
        {
            "relevant": bool,
            "reason": str,
            "method": "distance_auto" | "llm_judge"
        }
    """
    if not retrieved:
        return {"relevant": False, "reason": "No documents retrieved.", "method": "distance_auto"}

    best_distance = min(item["distance"] for item in retrieved)

    # Fast path: obviously good match
    if best_distance <= DISTANCE_AUTO_ACCEPT:
        return {"relevant": True, "reason": f"Best distance {best_distance:.3f} is well within threshold.", "method": "distance_auto"}

    # Fast path: obviously bad match
    if best_distance >= DISTANCE_AUTO_REJECT:
        return {"relevant": False, "reason": f"Best distance {best_distance:.3f} is far outside threshold.", "method": "distance_auto"}

    # Borderline: ask the LLM judge
    judge_prompt = f"""You are a strict relevance evaluator for a code-generation assistant.

User request:
\"\"\"{user_query}\"\"\"

Retrieved context:
\"\"\"{formatted_context}\"\"\"

Question: Is the retrieved context sufficiently relevant to help correctly
answer the user's request?

Reply with EXACTLY one word on the first line: "Relevant" or "NotRelevant".
On the second line, give a one-sentence reason.
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0
    )

    text = response.choices[0].message.content.strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]  # drop empty lines
    verdict = lines[0].lower() if lines else ""
    reason = lines[1] if len(lines) > 1 else "No reason provided by model."

    is_relevant = verdict.startswith("relevant") and not verdict.startswith("notrelevant")

    return {"relevant": is_relevant, "reason": reason, "method": "llm_judge"}