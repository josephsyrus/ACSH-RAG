import os
import re
import bisect
import tiktoken
from pathlib import Path
from typing import List, Dict, Any


# ─────────────────────────────────────────────
# File Loaders
# ─────────────────────────────────────────────

def load_pdf(file_path: str) -> List[str]:
    """Extract text from a PDF, one entry per page (page order preserved).
    Empty pages are kept as "" so list index i == page number (i + 1)."""
    import pypdf
    pages = []
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            pages.append(page.extract_text() or "")
    return pages


def load_markdown(file_path: str) -> str:
    """Read a Markdown file as plain text."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_documents(documents_dir: str) -> List[Dict[str, Any]]:
    documents = []
    docs_path = Path(documents_dir)

    if not docs_path.exists():
        raise FileNotFoundError(
            f"Documents directory '{documents_dir}' not found. "
            "Create it and add your PDF/Markdown files."
        )

    supported = {".pdf", ".md", ".markdown", ".txt"}

    for file_path in sorted(docs_path.rglob("*")):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext not in supported:
            continue

        try:
            print(f"  Loading: {file_path.name}")
            if ext == ".pdf":
                pages = load_pdf(str(file_path))
                text = "\n".join(pages)
                doc_type = "pdf"
            else:
                text = load_markdown(str(file_path))
                pages = None          # non-PDF: no page structure
                doc_type = "text"

            if not text.strip():
                print(f"    WARNING: '{file_path.name}' appears empty. Skipping.")
                continue

            documents.append({
                "text":     text,
                "pages":    pages,     # list[str] for PDFs (index i == page i+1), else None
                "source":   str(file_path.resolve()),
                "filename": file_path.name,
                "type":     doc_type,
            })
        except Exception as e:
            print(f"    ERROR loading '{file_path.name}': {e}. Skipping.")

    print(f"\nLoaded {len(documents)} document(s).")
    return documents


# ─────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = 250,
    overlap: int = 100,
    encoding_name: str = "cl100k_base",
    return_offsets: bool = False,
) -> List:
    """Token-window chunker. With return_offsets=True, yields (chunk_str,
    start_token_index) so callers can map each chunk back to a page."""

    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)

    chunks = []
    start = 0
    step = chunk_size - overlap  # how far to advance each iteration

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_str = encoding.decode(chunk_tokens).strip()

        # Only keep chunks with meaningful content (>50 chars)
        if len(chunk_str) > 50:
            chunks.append((chunk_str, start) if return_offsets else chunk_str)

        if end == len(tokens):
            break
        start += step

    return chunks


def chunk_documents(
    documents: List[Dict],
    chunk_size: int = 250,
    overlap: int = 50,
) -> List[Dict]:
    """

    Each chunk dict contains:
      chunk_id      — unique string ID (used by ChromaDB and BM25)
      text          — the actual text content
      source        — absolute path to source file
      filename      — just the filename
      doc_type      — "pdf" or "text"
      chunk_index   — which chunk number within the document
      total_chunks  — total chunks in that document
      page          — 1-indexed start page (PDFs only; None for text)
    """
    encoding   = tiktoken.get_encoding("cl100k_base")
    all_chunks = []

    for doc in documents:
        # For PDFs, compute the token offset where each page begins, in the same
        # token space chunk_text uses, so each chunk maps back to its start page.
        pages = doc.get("pages")
        page_token_starts = None
        if pages:
            page_token_starts = []
            cum     = 0
            sep_len = len(encoding.encode("\n"))   # "\n".join separator between pages
            for pidx, ptext in enumerate(pages):
                page_token_starts.append(cum)
                cum += len(encoding.encode(ptext))
                if pidx < len(pages) - 1:
                    cum += sep_len

        raw_chunks = chunk_text(
            doc["text"], chunk_size=chunk_size, overlap=overlap, return_offsets=True
        )
        print(f"  {doc['filename']}: {len(raw_chunks)} chunks")

        for i, (chunk_text_content, start_tok) in enumerate(raw_chunks):
            # Build a unique, stable ID from filename + chunk index
            safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", doc["filename"])
            chunk_id = f"{safe_name}_chunk_{i:05d}"

            page = None
            if page_token_starts is not None:
                # Page whose start offset is the greatest one <= this chunk's start.
                pi   = bisect.bisect_right(page_token_starts, start_tok) - 1
                page = max(0, pi) + 1   # 1-indexed page number

            all_chunks.append({
                "chunk_id":    chunk_id,
                "text":        chunk_text_content,
                "source":      doc["source"],
                "filename":    doc["filename"],
                "doc_type":    doc["type"],
                "chunk_index": i,
                "total_chunks": len(raw_chunks),
                "page":        page,
            })

    print(f"\nTotal chunks created: {len(all_chunks)}")
    return all_chunks
