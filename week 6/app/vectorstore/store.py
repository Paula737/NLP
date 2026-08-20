"""
LLM External Knowledge layer.
A Chroma vector store whose embedding function is wrapped with Redis
caching, so re-embedding identical text (e.g. re-ingesting the same
document, or repeated chunks) is skipped.
"""
from langchain_chroma import Chroma
from app.config import settings
from app.llm_provider import get_embeddings
from app.cache.redis_cache import cache, NS_EMBEDDING


class CachedEmbeddings:
    """Duck-types LangChain's Embeddings interface with a Redis cache in front."""

    def __init__(self):
        self._embeddings = get_embeddings()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results: list = [None] * len(texts)
        to_compute, indices = [], []

        for i, text in enumerate(texts):
            cached_vec = cache.get(NS_EMBEDDING, text)
            if cached_vec is not None:
                results[i] = cached_vec
            else:
                to_compute.append(text)
                indices.append(i)

        if to_compute:
            computed = self._embeddings.embed_documents(to_compute)
            for idx, text, vec in zip(indices, to_compute, computed):
                cache.set(NS_EMBEDDING, text, vec)
                results[idx] = vec

        return results

    def embed_query(self, text: str) -> list[float]:
        cached_vec = cache.get(NS_EMBEDDING, text)
        if cached_vec is not None:
            return cached_vec
        vec = self._embeddings.embed_query(text)
        cache.set(NS_EMBEDDING, text, vec)
        return vec


embedding_function = CachedEmbeddings()

vector_store = Chroma(
    collection_name="external_knowledge",
    embedding_function=embedding_function,
    persist_directory=settings.chroma_persist_dir,
)


def retrieve(query: str, k: int = None):
    return vector_store.similarity_search(query, k=k or settings.retrieval_k)
