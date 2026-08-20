"""
Chunk + Embed + Store stage of the ingestion pipeline.
Ties loaders -> text splitter -> vector store together, and uses Redis
to skip re-processing a file that has already been ingested (same
content hash -> cache hit, no re-embedding).
"""
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.ingestion import loaders
from app.vectorstore.store import vector_store
from app.cache.redis_cache import cache, NS_DOCUMENT

LOADER_MAP = {
    "pdf": loaders.load_pdf,
    "docx": loaders.load_docx,
    "txt": loaders.load_txt,
    "code": loaders.load_code,
    "pptx": loaders.load_pptx,
}

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


def _file_hash(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


def ingest_file(path: str, file_type: str) -> dict:
    if file_type not in LOADER_MAP:
        raise ValueError(f"Unsupported file type: {file_type}")

    file_hash = _file_hash(path)
    cached = cache.get(NS_DOCUMENT, file_hash)
    if cached:
        return {"status": "cache_hit", "chunks": cached["chunk_count"], "source": path}

    docs = LOADER_MAP[file_type](path)
    chunks = splitter.split_documents(docs)
    if chunks:
        vector_store.add_documents(chunks)

    cache.set(NS_DOCUMENT, file_hash, {"chunk_count": len(chunks), "source": path})
    return {"status": "ingested", "chunks": len(chunks), "source": path}


def ingest_audio(path: str) -> dict:
    file_hash = _file_hash(path)
    cached = cache.get(NS_DOCUMENT, file_hash)
    if cached:
        return {"status": "cache_hit", "chunks": cached["chunk_count"], "source": path}

    docs = loaders.load_audio(path)
    chunks = splitter.split_documents(docs)
    if chunks:
        vector_store.add_documents(chunks)

    cache.set(NS_DOCUMENT, file_hash, {"chunk_count": len(chunks), "source": path})
    return {"status": "ingested", "chunks": len(chunks), "source": path}


def ingest_url(url: str) -> dict:
    cached = cache.get(NS_DOCUMENT, url)
    if cached:
        return {"status": "cache_hit", "chunks": cached["chunk_count"], "source": url}

    docs = loaders.load_webpage(url)
    chunks = splitter.split_documents(docs)
    if chunks:
        vector_store.add_documents(chunks)

    cache.set(NS_DOCUMENT, url, {"chunk_count": len(chunks), "source": url})
    return {"status": "ingested", "chunks": len(chunks), "source": url}


def ingest_wikipedia(topic: str) -> dict:
    cached = cache.get(NS_DOCUMENT, f"wiki:{topic}")
    if cached:
        return {"status": "cache_hit", "chunks": cached["chunk_count"], "source": topic}

    docs = loaders.load_wikipedia(topic)
    chunks = splitter.split_documents(docs)
    if chunks:
        vector_store.add_documents(chunks)

    cache.set(NS_DOCUMENT, f"wiki:{topic}", {"chunk_count": len(chunks), "source": topic})
    return {"status": "ingested", "chunks": len(chunks), "source": topic}
