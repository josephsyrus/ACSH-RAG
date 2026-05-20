import os
import re
import tiktoken
from pathlib import Path
from typing import List, Dict, Any


# ─────────────────────────────────────────────
# File Loaders
# ─────────────────────────────────────────────

def load_pdf(file_path: str) -> str:
    """Extract all text from a PDF file, page by page."""
    import pypdf
    text_parts = []
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text_parts.append(page_text)
    return "\n".join(text_parts)


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
                text = load_pdf(str(file_path))
                doc_type = "pdf"
            else:
                text = load_markdown(str(file_path))
                doc_type = "text"

            if not text.strip():
                print(f"    WARNING: '{file_path.name}' appears empty. Skipping.")
                continue

            documents.append({
                "text":     text,
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
) -> List[str]:

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
            chunks.append(chunk_str)

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
    """
    all_chunks = []

    for doc in documents:
        raw_chunks = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        print(f"  {doc['filename']}: {len(raw_chunks)} chunks")

        for i, chunk_text_content in enumerate(raw_chunks):
            # Build a unique, stable ID from filename + chunk index
            safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", doc["filename"])
            chunk_id = f"{safe_name}_chunk_{i:05d}"

            all_chunks.append({
                "chunk_id":    chunk_id,
                "text":        chunk_text_content,
                "source":      doc["source"],
                "filename":    doc["filename"],
                "doc_type":    doc["type"],
                "chunk_index": i,
                "total_chunks": len(raw_chunks),
            })

    print(f"\nTotal chunks created: {len(all_chunks)}")
    return all_chunks
