"""
modules/rag.py
Retrieval-Augmented Generation module.
Given a user query, searches the Chroma 'coding_knowledge' collection
and returns the top-k most relevant documents + their metadata.
"""

import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "coding_knowledge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ---------------------------------------------------------
# Set up client + collection (loaded once, reused across calls)
# ---------------------------------------------------------
_client = chromadb.PersistentClient(path=CHROMA_PATH)

_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)

_collection = _client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=_embedding_fn
)


def retrieve_context(query: str, top_k: int = 3) -> list[dict]:
    """
    Searches Chroma for the top_k documents most relevant to the query.

    Returns a list of dicts like:
        {
            "document": "...",
            "metadata": {...},
            "distance": 0.23
        }
    Lower distance = more similar.
    """
    results = _collection.query(
        query_texts=[query],
        n_results=top_k
    )

    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append({
            "document": doc,
            "metadata": meta,
            "distance": dist
        })

    return retrieved


def format_context_for_llm(retrieved: list[dict]) -> str:
    """
    Turns retrieved docs into a single context string to inject into the
    code-generation prompt.
    """
    if not retrieved:
        return ""

    blocks = []
    for i, item in enumerate(retrieved, start=1):
        task_id = item["metadata"].get("task_id", f"doc_{i}")
        blocks.append(f"--- Reference {i} ({task_id}) ---\n{item['document']}")

    return "\n\n".join(blocks)


def add_document(text: str, metadata: dict, doc_id: str):
    """
    Inserts a new document into Chroma (used by the human-feedback
    learning step later — Step 7 of the spec).
    """
    _collection.add(
        documents=[text],
        metadatas=[metadata],
        ids=[doc_id]
    )