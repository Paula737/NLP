"""
modules/explainer.py
Turns SQL query results into a plain-language explanation via Groq.
Constrained to stick strictly to the actual data — no embellishment,
no invented details, no restating numbers in a way that could distort them.
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def explain_results(question: str, sql: str, rows: list[dict]) -> str:
    if not rows:
        return "The query returned no results."

    preview = rows[:10]

    prompt = f"""A user asked: "{question}"

This SQL query was run: {sql}

Here are the exact results (showing up to 10 rows):
{preview}

Write a short, factual summary of these results in plain language.

Strict rules:
- State ONLY facts directly present in the data above. Do not infer, guess,
  or add any detail not explicitly in the rows (e.g. don't invent how long
  someone has worked somewhere, don't estimate anything not given).
- Do not restate a number in a different, potentially confusing way
  (e.g. don't turn an age into a range or a tenure estimate).
- If listing multiple rows, use a simple list: name and the relevant
  fields from the query, nothing more.
- Keep it concise — a few sentences or a short list, not a narrative.
- Respond in the same language as the user's question.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,  # deterministic, less room for creative drift
    )

    return response.choices[0].message.content