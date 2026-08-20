"""
Evaluator LLM
--------------
Judges the Generator's answer on accuracy, relevance, completeness,
consistency with context, grounding, and absence of unsupported claims.
Returns a structured decision: accept or revise (+ feedback). Writes
only to EvaluatorMemory.

Some providers occasionally return an empty or malformed completion
under strict JSON mode. Rather than let that crash the whole /ask
request, we retry once, and if it still fails, fall back to a
"revise" decision with generic feedback so the loop continues instead
of erroring out.
"""
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from app.llm_provider import get_chat_model
from app.memory.evaluator_memory import EvaluatorMemory
from app.cache.redis_cache import cache, NS_EVALUATION

logger = logging.getLogger("evaluator_generator_rag.evaluator")

llm = get_chat_model(temperature=0, json_mode=True)

EVALUATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are the Evaluator LLM in an Evaluator-Generator RAG system.\n"
     "Judge the Generator's answer against: accuracy, relevance to the "
     "question, completeness, consistency with the given context, "
     "grounding in retrieved information, absence of unsupported claims, "
     "and overall quality.\n"
     "You must always return a non-empty JSON object, even for a short or "
     "simple answer. Respond with STRICT JSON only, no markdown fences, "
     "no preamble, matching exactly:\n"
     '{{"decision": "accept" or "revise", "score": <integer 0-10>, '
     '"feedback": "<concrete, actionable feedback, empty string if accept>"}}'),
    ("human",
     "Question: {question}\n\n"
     "Context available to the Generator:\n{context}\n\n"
     "Generator's Answer:\n{answer}\n\n"
     "Evaluate now. Return the JSON object and nothing else."),
])

# LCEL chain: prompt -> model -> JSON parser
evaluator_chain = EVALUATOR_PROMPT | llm | JsonOutputParser()

FALLBACK_RESULT = {
    "decision": "revise",
    "score": 5,
    "feedback": (
        "The evaluator could not produce a structured verdict for this answer "
        "(provider returned an invalid response). Please strengthen grounding, "
        "accuracy, and completeness, and try again."
    ),
}


def _invoke_with_retry(payload: dict, attempts: int = 2) -> dict:
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return evaluator_chain.invoke(payload)
        except (OutputParserException, ValueError) as exc:
            last_exc = exc
            logger.warning("Evaluator returned invalid JSON on attempt %s/%s: %s",
                            attempt, attempts, exc)
        except Exception as exc:  # provider-level errors (e.g. Groq's json_validate_failed)
            last_exc = exc
            logger.warning("Evaluator call failed on attempt %s/%s: %s",
                            attempt, attempts, exc)
    logger.error("Evaluator failed after %s attempts (%s); using fallback verdict.",
                 attempts, last_exc)
    return dict(FALLBACK_RESULT)


def evaluate_answer(session_id: str, question: str, answer: str, context: str) -> dict:
    memory = EvaluatorMemory(session_id)

    cache_payload = f"{question}::{answer}"
    cached = cache.get(NS_EVALUATION, cache_payload)
    if cached:
        result = cached
    else:
        result = _invoke_with_retry({
            "question": question,
            "context": context,
            "answer": answer,
        })
        if result != FALLBACK_RESULT:
            cache.set(NS_EVALUATION, cache_payload, result)

    memory.add_evaluation(question, answer, result)
    return result
