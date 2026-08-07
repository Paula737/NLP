"""
modules/router.py
Routes a classified request to the correct pipeline.
This module has NO LLM logic itself — it just wires the pieces
built in previous steps together, per the spec's two routes:

Route 1 (explain): User -> Classifier -> Explain -> LLM -> Answer
Route 2 (generate): User -> Classifier -> Generate -> RAG -> Relevance -> LLM -> Code
"""

from modules.classifier import classify_intent
from modules.rag import retrieve_context, format_context_for_llm, add_document
from modules.relevance_checker import check_relevance


def handle_explain(user_query: str, code_snippet: str = "") -> dict:
    """
    Route 1: Explain Code. NO RAG, NO retrieval, NO embeddings.
    """
    from groq import Groq
    import os
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""You are an expert programming tutor.
Explain the following code clearly, step by step.

User request: {user_query}

Code:
{code_snippet}
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "type": "explanation",
        "answer": response.choices[0].message.content
    }


def handle_generate(user_query: str) -> dict:
    """
    Route 2: Generate Code. RAG -> Relevance Check -> LLM (or ask user).
    """
    retrieved = retrieve_context(user_query, top_k=3)
    context = format_context_for_llm(retrieved)

    verdict = check_relevance(user_query, retrieved, context)

    if not verdict["relevant"]:
        return {
            "type": "needs_feedback",
            "answer": (
                "I couldn't find relevant knowledge for your request. "
                "Could you provide the correct solution or reference so I can learn it "
                "for future interactions?"
            ),
            "reason": verdict["reason"]
        }

    # Relevant -> generate code using retrieved context
    from groq import Groq
    import os
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""You are an expert Python developer.
Use the reference material below ONLY if it's helpful. Write complete,
correct, well-commented code with error handling.

User request: {user_query}

Reference material:
{context}

Respond with the code in a single Markdown code block.
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "type": "generated_code",
        "answer": response.choices[0].message.content,
        "sources": [item["metadata"]["task_id"] for item in retrieved]
    }


def route_request(user_query: str, code_snippet: str = "") -> dict:
    """
    Main entry point: classify -> route -> return result dict.
    """
    intent = classify_intent(user_query)

    if intent == "explain":
        result = handle_explain(user_query, code_snippet)
    else:
        result = handle_generate(user_query)

    result["intent"] = intent
    return result