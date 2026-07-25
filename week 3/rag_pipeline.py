"""
rag_pipeline.py
----------------
Contains all the RAG logic:
    1) Retrieval: takes the user's question, embeds it, and searches
       FAISS for the closest chunks from your personal files.
    2) Generation: builds a prompt with the question + the retrieved chunks,
       and feeds it to a local, free model (flan-t5-base) to produce an answer.

Both models are loaded only once (lazy loading + caching) for better performance.
"""

import pickle
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Same settings as ingest.py
# ---------------------------------------------------------------------------
VECTOR_STORE_DIR = Path("vector_store")
INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"
CHUNKS_PATH = VECTOR_STORE_DIR / "chunks.pkl"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Text generation model: flan-t5-large gives noticeably better instruction-following
# and synthesis than flan-t5-base, at the cost of a bigger download (~3GB) and
# slightly slower inference. Still fully local/free, no API key needed.
# If your machine is very limited on RAM, you can fall back to "google/flan-t5-base".
GENERATION_MODEL_NAME = "google/flan-t5-large"

TOP_K = 6  # number of chunks to retrieve as context per question


# ---------------------------------------------------------------------------
# Loading models and the vector store (only once, via caching)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def get_generator():
    # Loaded directly (instead of via pipeline()) so this works regardless of
    # which pipeline "tasks" a given transformers version registers.
    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
    # IMPORTANT: truncate from the LEFT, not the right. Our prompt puts the
    # instructions + question AFTER the context, so if the whole prompt is too
    # long, default (right-side) truncation would cut off the question itself —
    # leaving the model with nothing but raw context, which makes it just echo
    # the context back instead of answering. Left truncation drops the least
    # useful leading context instead.
    tokenizer.truncation_side = "left"
    model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL_NAME)
    return tokenizer, model


@lru_cache(maxsize=1)
def load_vector_store():
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "Vector store not found. Run 'python ingest.py' first "
            "after adding your files to the data/ folder."
        )
    index = faiss.read_index(str(INDEX_PATH))
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Returns the top_k closest chunks to the question, each as:
        {"text": ..., "source": ..., "score": ...}
    """
    index, chunks = load_vector_store()
    embedder = get_embedder()

    query_vec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        results.append({"text": chunk["text"], "source": chunk["source"], "score": float(score)})
    return results


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def build_prompt(query: str, context_chunks: list[dict]) -> str:
    context_text = "\n---\n".join(c["text"] for c in context_chunks)
    prompt = (
        "Read the context below and answer the question in your own words. "
        "The context may mention several relevant items (experiences, skills, or "
        "projects) scattered across different parts of it. You MUST scan the ENTIRE "
        "context from start to finish and list EVERY relevant item you find — do not "
        "stop after the first one, two, or three items. Missing an item is a mistake. "
        "Format your answer as a bullet list, one line per item, each starting with '- '. "
        "Do not copy long passages verbatim; summarize each item briefly. "
        "If the answer isn't in the context at all, say you don't have that information.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n"
        "Complete bullet-list answer:"
    )
    return prompt


def generate_answer(query: str, context_chunks: list[dict]) -> str:
    tokenizer, model = get_generator()
    prompt = build_prompt(query, context_chunks)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
    output_ids = model.generate(**inputs, max_new_tokens=400, do_sample=False)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer.strip()


# ---------------------------------------------------------------------------
# Main function used by api.py
# ---------------------------------------------------------------------------
def answer_question(query: str, top_k: int = TOP_K) -> dict:
    """
    Returns a dict:
        {"answer": text, "sources": list of chunks used}
    """
    context_chunks = retrieve(query, top_k=top_k)

    if not context_chunks:
        return {
            "answer": "I don't have enough information to answer that question.",
            "sources": [],
        }

    answer = generate_answer(query, context_chunks)
    return {"answer": answer, "sources": context_chunks}


if __name__ == "__main__":
    # Quick manual test from the terminal
    while True:
        user_query = input("\nAsk a question (or type 'exit' to quit): ")
        if user_query.strip().lower() == "exit":
            break
        result = answer_question(user_query)
        print(f"\nAnswer: {result['answer']}")
        print("\nSources used:")
        for src in result["sources"]:
            print(f"  - {src['source']} (score={src['score']:.3f})")