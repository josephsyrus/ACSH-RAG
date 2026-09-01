"""
src/upload.py

Runtime ingestion of a single uploaded document into an ISOLATED per-session
index set (chroma_db / bm25_index / graph_db under a session directory), so one
user's uploads never leak into another user's answers.

Fully local — no Gemini/LLM calls (embeddings + BM25 + spaCy graph only).
"""

import os
import time
import shutil
from pathlib import Path
from typing import Dict, Optional

from .ingestion import load_pdf, load_markdown, chunk_documents
from .ocr import OcrEngine
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


def _load_one(file_path: str, display_filename: Optional[str] = None) -> Dict:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        # load_pdf returns both page text and per-page OCR flags.  Keep the
        # latter so chunk_documents can preserve its OCR provenance metadata.
        # Supplying the engine makes image-only/scanned pages usable in uploads,
        # just as they already are during batch ingestion.
        ocr_engine = OcrEngine()
        ocr_available = ocr_engine.available()
        pages, page_ocr = load_pdf(file_path, ocr_engine=ocr_engine)
        text  = "\n".join(pages)
        dtype = "pdf"
    elif ext in {".md", ".markdown", ".txt"}:
        text  = load_markdown(file_path)
        pages = None
        page_ocr = None
        dtype = "text"
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if not text.strip() and ext == ".pdf" and not ocr_available:
        raise ValueError(
            "Document has no embedded text and local OCR is unavailable. "
            "Install the OCR dependencies with `pip install -r requirements_web.txt`."
        )
    if not text.strip():
        raise ValueError("Document has no extractable text.")

    return {
        "text":     text,
        "pages":    pages,
        "page_ocr": page_ocr,
        "source":   str(Path(file_path).resolve()),
        "filename": display_filename or Path(file_path).name,
        "type":     dtype,
    }


def ingest_upload(
    file_path: str,
    session_root: str,
    chunk_size: int = 250,
    overlap: int = 50,
    display_filename: Optional[str] = None,
) -> Dict:
    """
    Ingest ONE file into a session's isolated indexes. Returns stats dict:
        {filename, chunks, pages, dirs}
    Reuses the exact same chunking + page-tracking + 3-index build as the
    batch ingestion, just scoped to the session directory.
    """
    dirs = session_dirs(session_root)

    doc    = _load_one(file_path, display_filename=display_filename)
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


def cleanup_sessions(
    sessions_root: str,
    ttl_hours: float = 24.0,
    max_sessions: int = 200,
) -> int:
    """
    Delete session directories older than ttl_hours (by mtime), and cap the
    total number of sessions (oldest evicted first). Returns how many were
    removed. Safe to call on every upload / at startup — no-op if root missing.

    NOTE: mtime refreshes on ingestion but not on query, so a session queried
    continuously for >ttl_hours could be reaped. Fine for typical short-lived
    upload sessions; raise ttl_hours if you need longer-lived ones.
    """
    if not os.path.isdir(sessions_root):
        return 0

    cutoff  = time.time() - ttl_hours * 3600
    entries = []
    for name in os.listdir(sessions_root):
        p = os.path.join(sessions_root, name)
        if not os.path.isdir(p):
            continue
        try:
            entries.append((os.path.getmtime(p), p))
        except OSError:
            continue

    removed = 0
    for mtime, p in entries:
        if mtime < cutoff:
            shutil.rmtree(p, ignore_errors=True)
            removed += 1

    survivors = sorted((m, p) for m, p in entries if os.path.exists(p))
    while len(survivors) > max_sessions:
        _, p = survivors.pop(0)   # oldest
        shutil.rmtree(p, ignore_errors=True)
        removed += 1

    return removed
