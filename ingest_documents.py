import sys
import os
import argparse

# Make sure Python can find our src/ package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingestion import load_documents, chunk_documents
from src.vector_store import VectorStore
from src.bm25_retriever import BM25Retriever
from src.graph_store import GraphStore

# ── Configuration ─────────────────────────────
DOCUMENTS_DIR = "./Documents"
CHROMA_DIR    = "./chroma_db"
BM25_DIR      = "./bm25_index"
GRAPH_DIR     = "./graph_db"
CHUNK_SIZE    = 250   # tokens per chunk — must stay ≤256 to match all-MiniLM-L6-v2's
                      # hard sequence limit; exceeding it causes silent truncation
CHUNK_OVERLAP = 50    # overlap tokens between adjacent chunks
# ─────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into retrieval layer.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingestion even if data already exists.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" ACSH-RAG — Document Ingestion")
    print("=" * 60)

    # ── Guard: already ingested? ──────────────
    chroma_exists = os.path.exists(os.path.join(CHROMA_DIR, "chroma.sqlite3"))
    bm25_exists   = os.path.exists(os.path.join(BM25_DIR,   "bm25.pkl"))
    graph_exists  = os.path.exists(os.path.join(GRAPH_DIR,  "graph.json"))

    if chroma_exists and bm25_exists and graph_exists and not args.force:
        print(
            "\nIngestion already done. Indexes found at:\n"
            f"  ChromaDB : {CHROMA_DIR}/\n"
            f"  BM25     : {BM25_DIR}/\n"
            f"  GraphDB  : {GRAPH_DIR}/\n\n"
            "To re-ingest (e.g., after adding new documents), run:\n"
            "  python ingest_documents.py --force\n"
            "\nOr run: python query.py"
        )
        return

    if args.force:
        print("\n--force flag set. Re-ingesting from scratch...")
        import shutil
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
        if os.path.exists(BM25_DIR):
            shutil.rmtree(BM25_DIR)
        if os.path.exists(GRAPH_DIR):
            shutil.rmtree(GRAPH_DIR)

    # ── Step 1: Load documents ────────────────
    print(f"\n[Step 1/5] Loading documents from '{DOCUMENTS_DIR}/'...")
    documents = load_documents(DOCUMENTS_DIR)

    if not documents:
        print(
            f"\nERROR: No supported documents found in '{DOCUMENTS_DIR}/'.\n"
            "Supported formats: .pdf, .md, .markdown, .txt\n"
            "Add at least one file and try again."
        )
        sys.exit(1)

    # ── Step 2: Chunk ─────────────────────────
    print(f"\n[Step 2/5] Chunking documents...")
    print(f"  chunk_size={CHUNK_SIZE} tokens, overlap={CHUNK_OVERLAP} tokens")
    chunks = chunk_documents(documents, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    # ── Step 3: ChromaDB ──────────────────────
    print(f"\n[Step 3/5] Building vector store (ChromaDB)...")
    vector_store = VectorStore(persist_directory=CHROMA_DIR)
    vector_store.add_chunks(chunks)

    # ── Step 4: BM25 ─────────────────────────
    print(f"\n[Step 4/5] Building BM25 keyword index...")
    bm25 = BM25Retriever(index_path=BM25_DIR)
    bm25.build_index(chunks)

    # ── Step 5: Graph DB ──────────────────────
    print(f"\n[Step 5/5] Building graph database (entities + relationships)...")
    graph_store = GraphStore(graph_db_dir=GRAPH_DIR)
    graph_store.build(chunks)
    graph_store.save()

    # ── Summary ───────────────────────────────
    print("\n" + "=" * 60)
    print(" Ingestion Complete!")
    print("=" * 60)
    print(f"  Documents  : {len(documents)}")
    print(f"  Chunks     : {len(chunks)}")
    print(f"  ChromaDB   : {CHROMA_DIR}/")
    print(f"  BM25 index : {BM25_DIR}/")
    print(f"  GraphDB    : {GRAPH_DIR}/")
    print("\nNext step: python query.py")


if __name__ == "__main__":
    main()
