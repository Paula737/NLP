"""
Generator LLM
--------------
Retrieves context from the External Knowledge layer and answers the
question using LCEL (prompt | llm | parser). Grounding rule: never
invent information — if the context doesn't support an answer, say so.
Writes only to GeneratorMemory.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.llm_provider import get_chat_model
from app.vectorstore.store import retrieve
from app.memory.generator_memory import GeneratorMemory
from app.cache.redis_cache import cache, NS_LLM_RESPONSE

llm = get_chat_model(temperature=0.2)

GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are the Generator LLM in an Evaluator-Generator RAG system.\n"
     "Answer the user's question using ONLY the provided context.\n"
     "If the context does not contain enough information to answer, "
     "explicitly say the required information is not available in the "
     "provided sources. Never invent or hallucinate facts.\n"
     "If previous evaluator feedback is provided, revise your answer to "
     "directly address every point raised."),
    ("human",
     "Question: {question}\n\n"
     "Retrieved Context:\n{context}\n\n"
     "Previous Evaluator Feedback (if any): {feedback}\n\n"
     "Write the best possible grounded answer."),
])

# LCEL chain: prompt -> model -> string parser
generator_chain = GENERATOR_PROMPT | llm | StrOutputParser()


def generate_answer(session_id: str, question: str, feedback: str = "") -> dict:
    memory = GeneratorMemory(session_id)

    docs = retrieve(question)
    context = "\n\n".join(d.page_content for d in docs) if docs else ""
    context = context or "No relevant context was found in the external knowledge base."

    # Only cache/reuse first-pass answers (revisions must always be fresh)
    cache_payload = f"{question}::{context[:800]}"
    if not feedback:
        cached = cache.get(NS_LLM_RESPONSE, cache_payload)
        if cached:
            answer = cached["answer"]
            memory.add_turn(question, answer, context)
            return {"answer": answer, "context": context,
                    "sources": [d.metadata for d in docs], "cache_hit": True}

    answer = generator_chain.invoke({
        "question": question,
        "context": context,
        "feedback": feedback or "None",
    })

    if not feedback:
        cache.set(NS_LLM_RESPONSE, cache_payload, {"answer": answer})
        memory.add_turn(question, answer, context)
    else:
        memory.add_improvement_attempt(feedback, answer)

    return {"answer": answer, "context": context,
            "sources": [d.metadata for d in docs], "cache_hit": False}
