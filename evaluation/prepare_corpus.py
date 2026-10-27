"""Build the evaluation corpus and its own indexes, separate from the app's.

    python evaluation/prepare_corpus.py --text PAPER.pdf --scanned OTHER.pdf

--text     copies a PDF as-is (born-digital, has a text layer)
--scanned  re-renders a PDF as page images with no text layer, the way a scanner
           would produce it, so it can only be read through OCR

Each file is saved under evaluation/corpus/ with the name given after "=" (for
example --text paper.pdf=gnn_leak_detection.pdf); those names must match the
`source` column of evaluation/dataset.csv. Indexes go to evaluation/index/.
"""

import argparse
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# Windows consoles and pipes default to cp1252, which cannot print the "→" etc.
# used in the project's log lines.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EVAL_DIR = os.path.join(ROOT, "evaluation")
CORPUS_DIR = os.path.join(EVAL_DIR, "corpus")
INDEX_DIR = os.path.join(EVAL_DIR, "index")
INDEX_DIRS = {
    "chroma": os.path.join(INDEX_DIR, "chroma_db"),
    "bm25": os.path.join(INDEX_DIR, "bm25_index"),
    "graph": os.path.join(INDEX_DIR, "graph_db"),
}
SCAN_DPI = 150


def _split(spec: str):
    src, _, name = spec.partition("=")
    return src, name or os.path.basename(src)


def make_scanned_copy(src: str, dest: str) -> int:
    """Rasterise every page into an image-only PDF. Returns the page count."""
    import fitz
    with fitz.open(src) as doc, fitz.open() as out:
        for page in doc:
            pix = page.get_pixmap(dpi=SCAN_DPI, colorspace=fitz.csGRAY)
            new = out.new_page(width=page.rect.width, height=page.rect.height)
            new.insert_image(new.rect, stream=pix.tobytes("png"))
        out.save(dest, garbage=4, deflate=True)
        return out.page_count


def build_indexes() -> int:
    from src.ingestion import load_documents, chunk_documents
    from src.vector_store import VectorStore
    from src.bm25_retriever import BM25Retriever
    from src.graph_store import GraphStore

    shutil.rmtree(INDEX_DIR, ignore_errors=True)
    documents = load_documents(CORPUS_DIR, enable_ocr=True)
    chunks = chunk_documents(documents, chunk_size=250, overlap=50)
    VectorStore(persist_directory=INDEX_DIRS["chroma"]).add_chunks(chunks)
    BM25Retriever(index_path=INDEX_DIRS["bm25"]).build_index(chunks)
    graph = GraphStore(graph_db_dir=INDEX_DIRS["graph"])
    graph.build(chunks)
    graph.save()
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", action="append", default=[], metavar="PDF[=NAME]")
    parser.add_argument("--scanned", action="append", default=[], metavar="PDF[=NAME]")
    parser.add_argument("--index-only", action="store_true", help="rebuild indexes from the existing corpus")
    args = parser.parse_args()

    if not args.index_only:
        if not (args.text or args.scanned):
            parser.error("give at least one --text or --scanned file (or --index-only)")
        shutil.rmtree(CORPUS_DIR, ignore_errors=True)
        os.makedirs(CORPUS_DIR)
        for spec in args.text:
            src, name = _split(spec)
            shutil.copyfile(src, os.path.join(CORPUS_DIR, name))
            print(f"  text     {name}")
        for spec in args.scanned:
            src, name = _split(spec)
            pages = make_scanned_copy(src, os.path.join(CORPUS_DIR, name))
            print(f"  scanned  {name}  ({pages} image-only pages)")

    print("\nBuilding evaluation indexes (vector, BM25, graph; OCR for scanned pages)...")
    print(f"\nDone: {build_indexes()} chunks indexed in {INDEX_DIR}")


if __name__ == "__main__":
    main()
