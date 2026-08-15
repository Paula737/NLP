"""
modules/sql_generator.py
Uses Groq to turn a natural-language question (Arabic or English)
into a SQL query, given the dataset's schema.
"""

import os
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

GROQ_MODEL = "llama-3.1-8b-instant"


def generate_sql(question: str, schema: dict) -> str:
    columns_desc_parts = []
    for c in schema["columns"]:
        desc = f"{c['name']} ({c['type']})"
        if c.get("sample_values"):
            desc += f" — example values: {c['sample_values']}"
        columns_desc_parts.append(desc)
    columns_desc = "\n".join(columns_desc_parts)

    prompt = f"""You are a SQL expert. Convert the user's natural-language question
into a single valid SQLite query.

The question may be in Arabic or English — understand it regardless of language.
The DATA ITSELF may be stored in a different language/script than the question
(e.g. the question asks about "القاهرة" but the column actually stores "Cairo").
Use the example values shown below to find the ACTUAL stored value that
matches what the user means, and use that exact stored value/spelling in
your WHERE clause — translate/transliterate as needed to find the match.

Table name: {schema['table_name']}
Columns:
{columns_desc}

User question: \"\"\"{question}\"\"\"

Rules:
- Output ONLY the SQL query, nothing else — no explanation, no markdown fences.
- Use only SELECT statements. Never use INSERT, UPDATE, DELETE, DROP, ALTER.
- Reference only the columns listed above.
- For text comparisons, always use LOWER(column) = LOWER('value') or LIKE,
  never case-sensitive =.
- Match against the actual example values shown above, not a literal
  translation of the user's words.

SQL query:"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    sql = response.choices[0].message.content.strip()
    sql = re.sub(r"^```sql\s*|^```\s*|```$", "", sql, flags=re.MULTILINE).strip()
    return sql