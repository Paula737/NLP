"""
Retriever Agent
===============
Job: find relevant evidence from the vector DB. It NEVER answers the question
itself — it only returns an "evidence bundle" (chunks + metadata + scores)
for the Analyst Agent to reason over.

Tools implemented (matching the project diagram):
  1. Query Rewriter      - clarifies/expands vague or pronoun-heavy questions
  2. Semantic Search     - embedding similarity search over the vector DB
  3. Keyword Search      - BM25 lexical search (good for exact terms/IDs)
  4. Metadata Filter     - restrict search to a specific document/page/etc.
  5. Reranker            - re-scores/re-orders candidates with an LLM judge
  6. Context Selector    - picks a compact, diverse, high-quality final set
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from rank_bm25 import BM25Okapi

from pipeline.document_pipeline import get_vectorstore, get_embedding_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


class EvidenceChunk(TypedDict):
    document: str
    page: Optional[int]
    score: float
    content: str
    chunk_id: str


def _llm():
    """Lazily create the Groq chat model (keeps import-time side effects low)."""
    return ChatGroq(model=GROQ_MODEL, temperature=0)


def _doc_to_evidence(doc: Document, score: float) -> EvidenceChunk:
    return {
        "document": doc.metadata.get("source", "unknown"),
        "page": doc.metadata.get("page"),
        "score": round(float(score), 4),
        "content": doc.page_content,
        "chunk_id": doc.metadata.get("chunk_id", ""),
    }


# ---------------------------------------------------------------------------
# 1. QUERY REWRITER
# ---------------------------------------------------------------------------
@tool
def query_rewriter(question: str, conversation_context: str = "") -> str:
    """Rewrite a vague, conversational, or pronoun-heavy user question into a
    clear, retrieval-friendly query. Expands abbreviations and clarifies
    terminology. Pass prior conversation context (e.g. the last question)
    when the current question uses pronouns like 'it' or 'that'.

    Args:
        question: The user's raw question.
        conversation_context: Optional prior turn(s) for pronoun resolution.
    """
    prompt = f"""You are a query rewriting assistant for a retrieval system.
Rewrite the user's question into a single, clear, self-contained search
query. Expand abbreviations, resolve pronouns using the context provided,
and make implicit terms explicit. Return ONLY the rewritten query, nothing
else.

Conversation context (may be empty): {conversation_context}
User question: {question}

Rewritten query:"""
    response = _llm().invoke(prompt)
    return response.content.strip()


# ---------------------------------------------------------------------------
# 2. SEMANTIC SEARCH
# ---------------------------------------------------------------------------
@tool
def semantic_search(query: str, k: int = 20) -> List[EvidenceChunk]:
    """Search the vector database by meaning (embedding similarity) rather
    than exact words. Returns the top-k most semantically similar chunks
    with their similarity scores and source metadata.

    Args:
        query: The (ideally rewritten) search query.
        k: Number of candidate chunks to retrieve.
    """
    vs = get_vectorstore()
    results = vs.similarity_search_with_relevance_scores(query, k=k)
    return [_doc_to_evidence(doc, score) for doc, score in results]


# ---------------------------------------------------------------------------
# 3. KEYWORD SEARCH (BM25)
# ---------------------------------------------------------------------------
def _load_all_chunks() -> List[Document]:
    """Pull every chunk out of Chroma to build a BM25 index over them."""
    vs = get_vectorstore()
    raw = vs.get(include=["documents", "metadatas"])
    docs = []
    for text, meta in zip(raw["documents"], raw["metadatas"]):
        docs.append(Document(page_content=text, metadata=meta or {}))
    return docs


@tool
def keyword_search(query: str, k: int = 20) -> List[EvidenceChunk]:
    """Search for exact words/phrases (lexical match) using BM25. Best for
    technical terms, model names, IDs, or equations where exact wording
    matters more than semantic meaning (e.g. 'BERT-base', 'Equation 12').

    Args:
        query: Keywords or phrase to search for.
        k: Number of top results to return.
    """
    docs = _load_all_chunks()
    if not docs:
        return []

    tokenized_corpus = [d.page_content.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:k]
    max_score = max((s for _, s in ranked), default=1.0) or 1.0
    return [_doc_to_evidence(doc, score / max_score) for doc, score in ranked if score > 0]


# ---------------------------------------------------------------------------
# 4. METADATA FILTER
# ---------------------------------------------------------------------------
@tool
def metadata_filter(query: str, source: str = "", page: Optional[int] = None, k: int = 20) -> List[EvidenceChunk]:
    """Restrict search to chunks matching specific metadata, such as a
    document filename or page number. Use when the user references a
    specific document or section (e.g. 'according to Chapter 4 of paper.pdf').

    Args:
        query: The search query to run within the filtered subset.
        source: Exact filename to restrict to (e.g. 'paper.pdf'). Empty = no filter.
        page: Specific page number to restrict to. None = no filter.
        k: Number of results to return.
    """
    vs = get_vectorstore()
    where = {}
    if source:
        where["source"] = source
    if page is not None:
        where["page"] = page

    filter_arg = where if where else None
    results = vs.similarity_search_with_relevance_scores(query, k=k, filter=filter_arg)
    return [_doc_to_evidence(doc, score) for doc, score in results]


# ---------------------------------------------------------------------------
# 5. RERANKER
# ---------------------------------------------------------------------------
@tool
def reranker(query: str, candidates: List[EvidenceChunk], top_n: int = 8) -> List[EvidenceChunk]:
    """Re-score and re-order a list of candidate chunks by true relevance to
    the query using an LLM judge. Use this after semantic_search and/or
    keyword_search return a larger candidate pool (e.g. 20-50 chunks) to
    surface only the best few before passing to the Analyst Agent.

    Args:
        query: The original (or rewritten) user query.
        candidates: List of evidence chunks to rerank (from prior searches).
        top_n: How many top chunks to keep after reranking.
    """
    if not candidates:
        return []

    numbered = "\n\n".join(
        f"[{i}] (doc={c['document']}, page={c['page']}) {c['content'][:500]}"
        for i, c in enumerate(candidates)
    )
    prompt = f"""Query: {query}

Below are candidate text chunks. Score each chunk's relevance to the query
from 0.0 (irrelevant) to 1.0 (highly relevant). Respond with ONLY a comma
separated list of "index:score" pairs, e.g. "0:0.9,1:0.2,2:0.7", covering
every index from 0 to {len(candidates) - 1}.

Chunks:
{numbered}

Scores:"""
    response = _llm().invoke(prompt).content.strip()

    scores = {}
    try:
        for pair in response.split(","):
            idx_str, score_str = pair.split(":")
            scores[int(idx_str.strip())] = float(score_str.strip())
    except (ValueError, IndexError):
        # Fallback: keep original order/scores if the LLM output was malformed
        return sorted(candidates, key=lambda c: c["score"], reverse=True)[:top_n]

    rescored = []
    for i, c in enumerate(candidates):
        new_score = scores.get(i, c["score"])
        rescored.append({**c, "score": round(new_score, 4)})

    rescored.sort(key=lambda c: c["score"], reverse=True)
    return rescored[:top_n]


# ---------------------------------------------------------------------------
# 6. CONTEXT SELECTOR
# ---------------------------------------------------------------------------
@tool
def context_selector(candidates: List[EvidenceChunk], max_chunks: int = 6) -> List[EvidenceChunk]:
    """Select the final, compact set of high-quality chunks to hand to the
    Analyst Agent. Removes near-duplicate/redundant chunks and favors
    coverage across different source documents, keeping within a small
    budget so the Analyst isn't overloaded with context.

    Args:
        candidates: Reranked evidence chunks to choose from.
        max_chunks: Maximum number of chunks to keep in the final bundle.
    """
    if not candidates:
        return []

    selected: List[EvidenceChunk] = []
    seen_docs = set()

    # Pass 1: take the best chunk from each distinct document first (diversity)
    for c in sorted(candidates, key=lambda x: x["score"], reverse=True):
        if len(selected) >= max_chunks:
            break
        if c["document"] not in seen_docs:
            selected.append(c)
            seen_docs.add(c["document"])

    # Pass 2: fill remaining budget with next-best chunks, skipping near-duplicates
    def _is_near_duplicate(a: str, b: str) -> bool:
        a_words, b_words = set(a.lower().split()), set(b.lower().split())
        if not a_words or not b_words:
            return False
        overlap = len(a_words & b_words) / len(a_words | b_words)
        return overlap > 0.85

    for c in sorted(candidates, key=lambda x: x["score"], reverse=True):
        if len(selected) >= max_chunks:
            break
        if c in selected:
            continue
        if any(_is_near_duplicate(c["content"], s["content"]) for s in selected):
            continue
        selected.append(c)

    return selected[:max_chunks]


# ---------------------------------------------------------------------------
# Orchestration helper: run the full retriever pipeline in one call
# ---------------------------------------------------------------------------
RETRIEVER_TOOLS = [
    query_rewriter,
    semantic_search,
    keyword_search,
    metadata_filter,
    reranker,
    context_selector,
]


def retrieve_evidence(
    question: str,
    conversation_context: str = "",
    source_filter: str = "",
    page_filter: Optional[int] = None,
    pool_size: int = 20,
    final_size: int = 6,
) -> List[EvidenceChunk]:
    """
    Deterministic, non-agentic pipeline that runs all 6 retriever tools in a
    sensible fixed order. Use this for a straightforward call; use the
    RETRIEVER_TOOLS list with a LangChain agent if you want the LLM to decide
    dynamically which tools to call (e.g. skip keyword search if not needed).
    """
    rewritten = query_rewriter.invoke(
        {"question": question, "conversation_context": conversation_context}
    )

    if source_filter or page_filter is not None:
        candidates = metadata_filter.invoke(
            {"query": rewritten, "source": source_filter, "page": page_filter, "k": pool_size}
        )
    else:
        sem_results = semantic_search.invoke({"query": rewritten, "k": pool_size})
        kw_results = keyword_search.invoke({"query": rewritten, "k": pool_size})
        # merge, de-duplicating by chunk_id, keeping the higher score
        merged = {}
        for c in sem_results + kw_results:
            key = c["chunk_id"]
            if key not in merged or c["score"] > merged[key]["score"]:
                merged[key] = c
        candidates = list(merged.values())

    reranked = reranker.invoke({"query": rewritten, "candidates": candidates, "top_n": pool_size // 2 or 1})
    final = context_selector.invoke({"candidates": reranked, "max_chunks": final_size})
    return final


if __name__ == "__main__":
    import json
    import sys as _sys

    q = _sys.argv[1] if len(_sys.argv) > 1 else "What tools does the Retriever Agent use?"
    evidence = retrieve_evidence(q)
    print(json.dumps(evidence, indent=2))
