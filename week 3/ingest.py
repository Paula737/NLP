"""
ingest.py
---------
Reads every file inside the data/ folder (txt, pdf, docx),
splits them into small chunks, embeds those chunks,
and stores them in a FAISS vector store so the RAG pipeline
can use them later.

Usage:
    1) Put your personal files (pdf / docx / txt) inside the data/ folder
    2) Run: python ingest.py
"""

import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from docx import Document as DocxDocument
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
VECTOR_STORE_DIR = Path("vector_store")
INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"
CHUNKS_PATH = VECTOR_STORE_DIR / "chunks.pkl"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 900       # number of characters per chunk
CHUNK_OVERLAP = 150     # overlap between chunks so context isn't lost


# ---------------------------------------------------------------------------
# Reading different file types
# ---------------------------------------------------------------------------
def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    text_parts = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        text_parts.append(extracted)
    return "\n".join(text_parts)


def read_docx(path: Path) -> str:
    doc = DocxDocument(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def load_documents(data_dir: Path) -> list[dict]:
    """
    Returns a list of dicts, one per file: {"source": file name, "text": content}
    """
    documents = []
    supported_extensions = {".txt": read_txt, ".pdf": read_pdf, ".docx": read_docx}

    if not data_dir.exists():
        raise FileNotFoundError(f"Folder '{data_dir}' does not exist. Create it and add your files inside.")

    files = [f for f in data_dir.iterdir() if f.suffix.lower() in supported_extensions]

    if not files:
        raise ValueError(
            f"No supported files (txt/pdf/docx) found in '{data_dir}'. "
            "Add at least one file with information about you."
        )

    for file_path in files:
        reader_fn = supported_extensions[file_path.suffix.lower()]
        try:
            text = reader_fn(file_path)
            if text.strip():
                documents.append({"source": file_path.name, "text": text})
                print(f"  ✔ Read: {file_path.name} ({len(text)} characters)")
            else:
                print(f"  ⚠ File is empty or text couldn't be extracted: {file_path.name}")
        except Exception as e:
            print(f"  ✘ Error while reading {file_path.name}: {e}")

    return documents


# ---------------------------------------------------------------------------
# Splitting text into chunks
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = " ".join(text.split())  # clean up extra whitespace
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Building the vector store
# ---------------------------------------------------------------------------
def build_vector_store():
    print(f"1) Reading files from '{DATA_DIR}' ...")
    documents = load_documents(DATA_DIR)

    print("\n2) Splitting files into small chunks ...")
    all_chunks = []  # each item: {"text": ..., "source": ...}
    for doc in documents:
        pieces = chunk_text(doc["text"])
        for piece in pieces:
            all_chunks.append({"text": piece, "source": doc["source"]})
    print(f"   Total number of chunks: {len(all_chunks)}")

    if not all_chunks:
        raise ValueError("Not enough content was split into chunks. Make sure your files contain text.")

    print(f"\n3) Loading the embedding model ({EMBEDDING_MODEL_NAME}) ...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("4) Creating embeddings for all chunks ...")
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    # Normalize the vectors so we can use cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    print("5) Building the FAISS index ...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product = cosine similarity after normalization
    index.add(embeddings)

    print("6) Saving the index and chunks to disk ...")
    VECTOR_STORE_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"\n✅ Done! The vector store was saved in '{VECTOR_STORE_DIR}/'")
    print(f"   - {INDEX_PATH}")
    print(f"   - {CHUNKS_PATH}")


if __name__ == "__main__":
    build_vector_store()