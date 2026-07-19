"""
src/upload.py

Runtime ingestion of a single uploaded document into an ISOLATED per-session
index set (chroma_db / bm25_index / graph_db under a session directory), so one
user's uploads never leak into another user's answers.

Fully local — no Gemini/LLM calls (embeddings + BM25 + spaCy graph only).
"""

import os
from pathlib import Path
from typing import Dict, Optional

from .ingestion import load_pdf, load_markdown, chunk_documents
from .vector_store import VectorStore
from .bm25_retriever import BM25Retriever
from .graph_store import GraphStore

SUPPORTED = {".pdf", ".md", ".markdown", ".txt"}


def session_dirs(session_root: str) -> Dict[str, str]:
    """The three isolated index directories for one session."""
    return {
        "chroma": os.path.join(session_root, "chroma_db"),
        "bm25":   os.path.join(session_root, "bm25_index"),
        "graph":  os.path.join(session_root, "graph_db"),
    }


def _load_one(file_path: str) -> Dict:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        pages = load_pdf(file_path)          # list[str], one per page
        text  = "\n".join(pages)
        dtype = "pdf"
    elif ext in {".md", ".markdown", ".txt"}:
        text  = load_markdown(file_path)
        pages = None
        dtype = "text"
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if not text.strip():
        raise ValueError("Document has no extractable text.")

    return {
        "text":     text,
        "pages":    pages,
        "source":   str(Path(file_path).resolve()),
        "filename": Path(file_path).name,
        "type":     dtype,
    }


def ingest_upload(
    file_path: str,
    session_root: str,
    chunk_size: int = 250,
    overlap: int = 50,
) -> Dict:
    """
    Ingest ONE file into a session's isolated indexes. Returns stats dict:
        {filename, chunks, pages, dirs}
    Reuses the exact same chunking + page-tracking + 3-index build as the
    batch ingestion, just scoped to the session directory.
    """
    dirs = session_dirs(session_root)

    doc    = _load_one(file_path)
    chunks = chunk_documents([doc], chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError("No chunks produced from document.")

    # Vector + BM25 + graph, all written under the session directory.
    VectorStore(persist_directory=dirs["chroma"]).add_chunks(chunks)
    BM25Retriever(index_path=dirs["bm25"]).build_index(chunks)
    gs = GraphStore(graph_db_dir=dirs["graph"])
    gs.build(chunks)
    gs.save()

    n_pages: Optional[int] = len(doc["pages"]) if doc["pages"] is not None else None
    return {
        "filename": doc["filename"],
        "chunks":   len(chunks),
        "pages":    n_pages,
        "dirs":     dirs,
    }
