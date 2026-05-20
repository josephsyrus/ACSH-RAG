"""
retrieve_api.py
─────────────────────────────────────────────────────────────────────
PUBLIC API for the retrieval layer. This is what Person B and Person C import.

USAGE (for teammates):
    from retrieve_api import retrieve_chunks

    results = retrieve_chunks("what is the statute of limitations?", top_k=5)

    for chunk in results:
        print(chunk["text"])          # The actual text content
        print(chunk["filename"])      # Which file it came from
        print(chunk["rrf_score"])     # Relevance score (higher = more relevant)
        print(chunk["chunk_id"])      # Unique ID (use this for citations)

─────────────────────────────────────────────────────────────────────
RETURN FORMAT — each item in the list is a dict with these keys:

    chunk_id      (str)   — unique identifier, e.g. "contract_pdf_chunk_00003"
    text          (str)   — the actual chunk text (180–200 words / ~250 tokens)
    metadata      (dict)  — {filename, source, doc_type, chunk_index, total_chunks}
    filename      (str)   — shortcut to metadata["filename"]
    rrf_score     (float) — combined relevance score, range ~0.003–0.016, higher=better
    vector_score  (float) — semantic similarity, range 0–1
    bm25_score    (float) — normalized keyword score, range 0–1
    vector_rank   (int)   — rank in vector search results (1=best), None if not found
    bm25_rank     (int)   — rank in BM25 search results (1=best), None if not found
    found_in      (list)  — e.g. ["vector", "bm25"] or ["vector"] or ["bm25"]
─────────────────────────────────────────────────────────────────────
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.hybrid_retriever import HybridRetriever
from typing import List, Dict

# Lazily initialised so import is fast — retriever loads on first call
_retriever: HybridRetriever = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(
            chroma_persist_dir="./chroma_db",
            bm25_index_dir="./bm25_index",
        )
    return _retriever


def retrieve_chunks(
    query:          str,
    top_k:          int   = 5,
    fetch_k:        int   = 20,
    vector_weight:  float = 0.5,
    bm25_weight:    float = 0.5,
    vector_query:   str   = None,
) -> List[Dict]:

    return _get_retriever().retrieve(
        query=query,
        top_k=top_k,
        fetch_k=fetch_k,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        vector_query=vector_query,
    )


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_query = sys.argv[1] if len(sys.argv) > 1 else "test query"
    print(f"\nRunning self-test with query: '{test_query}'\n")
    results = retrieve_chunks(test_query, top_k=5)
    print(f"\nReturned {len(results)} chunk(s):\n")
    for i, r in enumerate(results, 1):
        filename = r.get('filename') or r.get('metadata', {}).get('filename', 'unknown')
        clean_text = ' '.join(r['text'].split())
        print(f"  [{i}] {filename} | rrf={r['rrf_score']:.5f} | found_in={r['found_in']}")
        print(f"       {clean_text}\n")
