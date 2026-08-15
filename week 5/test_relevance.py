# test_relevance.py
from modules.rag import retrieve_context, format_context_for_llm
from modules.relevance_checker import check_relevance

query = "write a function that checks if a list has close elements"
retrieved = retrieve_context(query, top_k=3)
context = format_context_for_llm(retrieved)

verdict = check_relevance(query, retrieved, context)
print(verdict)

# Now try something totally unrelated to HumanEval's 164 problems
query2 = "build a Flask REST API with JWT authentication"
retrieved2 = retrieve_context(query2, top_k=3)
context2 = format_context_for_llm(retrieved2)

verdict2 = check_relevance(query2, retrieved2, context2)
print(verdict2)