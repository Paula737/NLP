# test_rag.py
from modules.rag import retrieve_context, format_context_for_llm

results = retrieve_context("write a function that checks if a list has close elements", top_k=3)

for r in results:
    print(r["metadata"]["task_id"], "-> distance:", r["distance"])

print("\n--- Formatted context ---\n")
print(format_context_for_llm(results))