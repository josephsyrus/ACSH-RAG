import os
import re
import bisect
import tiktoken
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Tuple


# ─────────────────────────────────────────────
# File Loaders
# ─────────────────────────────────────────────

_HYPHENATED_PREFIXES = {
    "anti", "co", "cross", "e", "ex", "half", "high", "low", "mid", "multi",
    "non", "post", "pre", "pro", "re", "self", "semi", "short", "state", "well",
}


def normalize_extracted_text(text: str) -> str:
    """Repair PDF layout artifacts that otherwise break retrieval terms."""
    text = unicodedata.normalize("NFKC", text or "").replace("\r\n", "\n")

    def repair_hyphen(match: re.Match) -> str:
        left, right = match.group(1), match.group(2)
        separator = "-" if left.lower() in _HYPHENATED_PREFIXES else ""
        return f"{left}{separator}{right}"

    # PDF extraction commonly turns a line-wrapped word such as
    # "con-\nversations" into two retrieval tokens.
    text = re.sub(r"([A-Za-z]{2,})-\s*\n\s*([a-z][A-Za-z]{1,})", repair_hyphen, text)
    # OCR can produce letter-spaced all-caps headings ("T H E").
    text = re.sub(
        r"\b(?:[A-Z]\s+){2,}[A-Z]\b",
        lambda match: re.sub(r"\s+", "", match.group(0)),
        text,
    )
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _extraction_quality(text: str) -> float:
    """Prefer text extraction that retains real word boundaries."""
    words = re.findall(r"[A-Za-z]{2,}", text)
    spaced_pairs = len(re.findall(r"[A-Za-z]{2,}\s+[A-Za-z]{2,}", text))
    run_together = sum(1 for word in words if len(word) >= 18)
    return min(len(text), 4000) + 24 * spaced_pairs - 60 * run_together


def load_pdf(file_path: str, ocr_engine=None) -> Tuple[List[str], List[bool]]:
    """Extract text from a PDF, one entry per page (page order preserved).
    Empty pages are kept as "" so list index i == page number (i + 1).

    Pages carrying no embedded text are assumed to be scans and are rasterised
    and OCR'd when an engine is supplied. Detection is per page, not per file, so
    a PDF mixing born-digital text with a scanned appendix is handled correctly —
    that case used to lose the scanned pages with no warning at all.

    Returns (page_texts, page_was_ocred)."""
    import pypdf
    from .ocr import page_needs_ocr, file_sha1

    pages: List[str]   = []
    ocr_flags: List[bool] = []

    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            pages.append(normalize_extracted_text(page.extract_text() or ""))
            ocr_flags.append(False)

    try:
        import fitz
        with fitz.open(file_path) as pdf:
            fitz_pages = [normalize_extracted_text(page.get_text("text") or "") for page in pdf]
        if len(fitz_pages) == len(pages):
            pages = [
                alternative if _extraction_quality(alternative) > _extraction_quality(current) else current
                for current, alternative in zip(pages, fitz_pages)
            ]
    except (ImportError, RuntimeError, OSError):
        # PyMuPDF is optional for ordinary text PDFs; pypdf remains the fallback.
        pass

    if ocr_engine is None or not ocr_engine.available():
        return pages, ocr_flags

    scanned = [i for i, text in enumerate(pages) if page_needs_ocr(text)]
    if not scanned:
        return pages, ocr_flags

    print(f"    {len(scanned)} page(s) have no embedded text — running local OCR...")
    file_hash = file_sha1(file_path)
    recovered = 0
    for i in scanned:
        text, conf = ocr_engine.ocr_pdf_page(file_path, i, file_hash)
        if text.strip():
            pages[i]     = normalize_extracted_text(text)
            ocr_flags[i] = True
            recovered   += 1
    print(f"    OCR recovered text from {recovered}/{len(scanned)} page(s).")

    return pages, ocr_flags


def load_markdown(file_path: str) -> str:
    """Read a Markdown file as plain text."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


PDF_EXTENSIONS   = {".pdf"}
TEXT_EXTENSIONS  = {".md", ".markdown", ".txt"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def load_documents(documents_dir: str, enable_ocr: bool = True,
                   ocr_cache_dir: str = "./ocr_cache") -> List[Dict[str, Any]]:
    documents = []
    docs_path = Path(documents_dir)

    if not docs_path.exists():
        raise FileNotFoundError(
            f"Documents directory '{documents_dir}' not found. "
            "Create it and add your PDF/Markdown files."
        )

    supported = PDF_EXTENSIONS | TEXT_EXTENSIONS | (IMAGE_EXTENSIONS if enable_ocr else set())

    ocr_engine = None
    if enable_ocr:
        from .ocr import OcrEngine
        ocr_engine = OcrEngine(cache_dir=ocr_cache_dir)

    for file_path in sorted(docs_path.rglob("*")):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext not in supported:
            continue

        try:
            print(f"  Loading: {file_path.name}")
            if ext in PDF_EXTENSIONS:
                pages, page_ocr = load_pdf(str(file_path), ocr_engine=ocr_engine)
                text = "\n".join(pages)
                doc_type = "pdf"
            elif ext in IMAGE_EXTENSIONS:
                if ocr_engine is None or not ocr_engine.available():
                    print(f"    SKIP: '{file_path.name}' needs OCR, which is unavailable.")
                    continue
                text, conf = ocr_engine.ocr_image_file(str(file_path))
                pages, page_ocr = [text], [True]
                doc_type = "image"
                if text.strip():
                    print(f"    OCR extracted {len(text)} chars (confidence {conf:.2f}).")
            else:
                text = normalize_extracted_text(load_markdown(str(file_path)))
                pages, page_ocr = None, None   # non-paged source
                doc_type = "text"

            if not text.strip():
                print(f"    WARNING: '{file_path.name}' appears empty. Skipping.")
                continue

            documents.append({
                "text":     text,
                "pages":    pages,     # list[str] for PDFs (index i == page i+1), else None
                "page_ocr": page_ocr,  # list[bool] parallel to pages, else None
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

        # Keep normal chunks above the noise floor, but preserve a short
        # single-document result (for example, text recognized from a label or
        # caption in an image-only upload). Otherwise successful OCR can leave
        # the retriever with no chunks at all.
        is_only_chunk = start == 0 and end == len(tokens)
        if len(chunk_str) > 50 or (is_only_chunk and chunk_str):
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
      ocr           — True if the chunk's start page came from OCR (noisier text)
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

            page    = None
            is_ocr  = False
            if page_token_starts is not None:
                # Page whose start offset is the greatest one <= this chunk's start.
                pi   = bisect.bisect_right(page_token_starts, start_tok) - 1
                pi   = max(0, pi)
                page = pi + 1           # 1-indexed page number
                page_ocr = doc.get("page_ocr")
                if page_ocr and pi < len(page_ocr):
                    is_ocr = bool(page_ocr[pi])

            all_chunks.append({
                "chunk_id":    chunk_id,
                "text":        chunk_text_content,
                "source":      doc["source"],
                "filename":    doc["filename"],
                "doc_type":    doc["type"],
                "chunk_index": i,
                "total_chunks": len(raw_chunks),
                "page":        page,
                "ocr":         is_ocr,
            })

    print(f"\nTotal chunks created: {len(all_chunks)}")
    return all_chunks
