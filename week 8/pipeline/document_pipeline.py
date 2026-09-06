"""
Document Preparation Pipeline
==============================
Implements: Upload -> Loader -> Parser -> Cleaning -> Chunking -> Embeddings -> Vector DB

This mirrors the "Document Pipeline" block in the project diagram. It ingests
PDFs, DOCX, TXT, or images (via OCR), cleans and chunks the text, embeds the
chunks with a local FastEmbed model, and persists everything into a Chroma
vector database with rich metadata (source filename, page number) for later
citation.
"""

import os
import re
import shutil
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from PIL import Image
import pytesseract

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VECTOR_DB_DIR = str(Path(__file__).resolve().parent.parent / "data" / "vectordb")
COLLECTION_NAME = "documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # small, fast, good quality
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
# Languages Tesseract will attempt, combined. Add more with '+', e.g.
# "eng+ara+fra". Each language needs its .traineddata file installed
# (see _configure_tesseract_path note below / the RuntimeError message).
OCR_LANGUAGES = os.environ.get("OCR_LANGUAGES", "eng+ara")


def _configure_tesseract_path():
    """pytesseract needs to find the Tesseract binary. If it's already on
    PATH, do nothing. Otherwise, try the default Windows install location
    so users don't have to manually configure this."""
    if shutil.which("tesseract") is not None:
        return
    default_windows_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if Path(default_windows_path).exists():
        pytesseract.pytesseract.tesseract_cmd = default_windows_path


_configure_tesseract_path()


# ---------------------------------------------------------------------------
# 1. LOADER  -- reads raw files into LangChain Document objects (with page metadata)
# ---------------------------------------------------------------------------
def load_image(file_path: str) -> List[Document]:
    """OCR an image file (PNG/JPG/etc.) into a single-page Document using
    Tesseract. Converts to grayscale first, which typically improves OCR
    accuracy on photos/screenshots."""
    filename = Path(file_path).name
    image = Image.open(file_path)
    if image.mode != "L":
        image = image.convert("L")

    try:
        extracted_text = pytesseract.image_to_string(image, lang=OCR_LANGUAGES)
    except pytesseract.TesseractNotFoundError as e:
        raise RuntimeError(
            "Tesseract OCR engine is not installed or not found on PATH. "
            "Download and install it from: "
            "https://github.com/UB-Mannheim/tesseract/wiki (Windows), then "
            "restart your terminal. If it's installed somewhere non-standard, "
            "set pytesseract.pytesseract.tesseract_cmd in document_pipeline.py "
            "to the full path of tesseract.exe."
        ) from e
    except pytesseract.TesseractError as e:
        raise RuntimeError(
            f"OCR failed (tried languages: '{OCR_LANGUAGES}'). This usually "
            f"means a language pack is missing. Download the missing "
            f"<lang>.traineddata file from "
            f"https://github.com/tesseract-ocr/tessdata and place it in "
            f"Tesseract's 'tessdata' folder (commonly "
            f"C:\\Program Files\\Tesseract-OCR\\tessdata\\). "
            f"Original error: {e}"
        ) from e

    if len(extracted_text.strip()) < 5:
        print(f"[pipeline] Warning: OCR extracted very little text from "
              f"{filename} ({len(extracted_text.strip())} chars) - the image "
              f"may be blurry, low-resolution, or contain no readable text.")

    return [Document(
        page_content=extracted_text,
        metadata={"source": filename, "page": 1, "is_ocr": True},
    )]


def load_document(file_path: str) -> List[Document]:
    """Load a single file (.pdf, .docx, .txt, or an image via OCR) into
    LangChain Documents.

    Each Document keeps a 'page' number (when available) in its metadata,
    which is essential for citations later in the Answer Agent.
    """
    ext = Path(file_path).suffix.lower()
    filename = Path(file_path).name

    if ext == ".pdf":
        docs = PyPDFLoader(file_path).load()
        for d in docs:
            d.metadata["source"] = filename
            # PyPDFLoader is 0-indexed; make it human-friendly (1-indexed)
            page = d.metadata.get("page")
            d.metadata["page"] = (page + 1) if page is not None else None
    elif ext == ".docx":
        docs = Docx2txtLoader(file_path).load()
        for d in docs:
            d.metadata["source"] = filename
            d.metadata.setdefault("page", None)
    elif ext == ".txt":
        docs = TextLoader(file_path, encoding="utf-8").load()
        for d in docs:
            d.metadata["source"] = filename
            d.metadata.setdefault("page", None)
    elif ext in IMAGE_EXTENSIONS:
        docs = load_image(file_path)  # already sets source/page=1 correctly
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return docs



# ---------------------------------------------------------------------------
# 2. PARSER + 3. CLEANING -- normalize whitespace, strip noise/artifacts
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Remove common PDF/DOCX extraction noise: extra whitespace, hyphenation
    breaks, repeated newlines, and non-printable characters."""
    if not text:
        return ""

    # Fix hyphenated line breaks: "informa-\ntion" -> "information"
    text = re.sub(r"-\n(?=[a-z])", "", text)

    # Collapse multiple newlines/spaces into single space, but keep paragraph breaks
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n(?!\n)", " ", text)  # single newlines within a paragraph -> space

    # Strip non-printable / control characters
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", "", text)

    return text.strip()


def clean_documents(docs: List[Document]) -> List[Document]:
    cleaned = []
    for d in docs:
        text = clean_text(d.page_content)
        if text:  # drop empty pages
            cleaned.append(Document(page_content=text, metadata=d.metadata))
    return cleaned


# ---------------------------------------------------------------------------
# 4. CHUNKING -- split into overlapping, model-friendly chunks
# ---------------------------------------------------------------------------
def chunk_documents(
    docs: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Give every chunk a stable id + chunk index, useful for metadata filtering
    # and for deduplication in the Context Selector later.
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = f"{c.metadata.get('source', 'doc')}::{i}"
    return chunks


# ---------------------------------------------------------------------------
# 5. EMBEDDINGS + 6. VECTOR DB -- embed chunks and persist to Chroma
# ---------------------------------------------------------------------------
# Cached as singletons: recreating the embedding model / Chroma client on
# every retrieval call (previously happening up to 3x per single question,
# once each from semantic_search, keyword_search, and metadata_filter) was
# pure wasted overhead with zero benefit - this was a real perf bug.
_embedding_model_singleton: FastEmbedEmbeddings | None = None
_vectorstore_singleton: Chroma | None = None


def get_embedding_model() -> FastEmbedEmbeddings:
    global _embedding_model_singleton
    if _embedding_model_singleton is None:
        _embedding_model_singleton = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
    return _embedding_model_singleton


def get_vectorstore(embeddings: FastEmbedEmbeddings = None) -> Chroma:
    """Return a cached handle to the persistent Chroma store, creating it
    only once per process instead of on every call."""
    global _vectorstore_singleton
    if _vectorstore_singleton is None:
        embeddings = embeddings or get_embedding_model()
        _vectorstore_singleton = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=VECTOR_DB_DIR,
        )
    return _vectorstore_singleton


def ingest_files(file_paths: List[str]) -> Chroma:
    """
    Full pipeline entrypoint: Upload -> Loader -> Parser/Cleaning -> Chunking
    -> Embeddings -> Vector DB.

    Returns the Chroma vectorstore handle, ready for the Retriever Agent.
    """
    global _vectorstore_singleton
    all_chunks: List[Document] = []

    for path in file_paths:
        print(f"[pipeline] Loading: {path}")
        raw_docs = load_document(path)

        print(f"[pipeline] Cleaning {len(raw_docs)} page(s)...")
        cleaned_docs = clean_documents(raw_docs)

        print(f"[pipeline] Chunking...")
        chunks = chunk_documents(cleaned_docs)
        print(f"[pipeline] -> {len(chunks)} chunks from {Path(path).name}")

        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError("No content extracted from provided files.")

    print(f"[pipeline] Embedding {len(all_chunks)} total chunks and writing to Chroma...")
    embeddings = get_embedding_model()
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_DB_DIR,
    )
    # Refresh the cached singleton so subsequent get_vectorstore() calls
    # (e.g. the next question asked) see the newly ingested documents.
    _vectorstore_singleton = vectorstore
    print(f"[pipeline] Done. Vector DB persisted at: {VECTOR_DB_DIR}")
    return vectorstore


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python document_pipeline.py <file1> [file2] ...")
        sys.exit(1)

    ingest_files(sys.argv[1:])
    
def delete_document(filename: str) -> int:
    """Remove a document's chunks from the vector DB and delete its file
    from data/uploads. Returns the number of chunks removed."""
    vs = get_vectorstore()
    existing = vs.get(where={"source": filename})
    count = len(existing.get("ids", []))
    if count:
        vs.delete(where={"source": filename})

    file_path = Path(__file__).resolve().parent.parent / "data" / "uploads" / filename
    if file_path.exists():
        file_path.unlink()

    return count


def clear_all_documents() -> int:
    """Wipe every document from the vector DB and data/uploads. Returns the
    number of files removed. Use with care - this cannot be undone."""
    global _vectorstore_singleton
    uploads_dir = Path(__file__).resolve().parent.parent / "data" / "uploads"
    removed = 0
    for f in uploads_dir.iterdir() if uploads_dir.exists() else []:
        if f.is_file():
            f.unlink()
            removed += 1

    vs = get_vectorstore()
    all_ids = vs.get().get("ids", [])
    if all_ids:
        vs.delete(ids=all_ids)

    return removed