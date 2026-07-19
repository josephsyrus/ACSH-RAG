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
    rrf_score     (float) — combined relevance score (3-way fusion), higher=better
    vector_score  (float) — semantic similarity score, range 0–1
    bm25_score    (float) — normalized BM25 keyword score, range 0–1
    graph_score   (float) — entity-overlap graph score, range 0–1
    vector_rank   (int)   — rank in vector search results (1=best), None if not found
    bm25_rank     (int)   — rank in BM25 search results (1=best), None if not found
    graph_rank    (int)   — rank in graph search results (1=best), None if not found
    found_in      (list)  — e.g. ["vector", "bm25", "graph"] or any subset
─────────────────────────────────────────────────────────────────────
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.hybrid_retriever import HybridRetriever
from typing import List, Dict, Optional

# The global (default) corpus directories.
_DEFAULT_DIRS = {"chroma": "./chroma_db", "bm25": "./bm25_index", "graph": "./graph_db"}

# One retriever per distinct index-dir set (global corpus + each uploaded
# session), built lazily and cached. Keeps per-session uploads isolated.
_retrievers: Dict[tuple, HybridRetriever] = {}


def _get_retriever(index_dirs: Optional[Dict[str, str]] = None) -> HybridRetriever:
    dirs = index_dirs or _DEFAULT_DIRS
    key  = (dirs["chroma"], dirs["bm25"], dirs["graph"])
    if key not in _retrievers:
        _retrievers[key] = HybridRetriever(
            chroma_persist_dir=dirs["chroma"],
            bm25_index_dir=dirs["bm25"],
            graph_db_dir=dirs["graph"],
        )
    return _retrievers[key]


def retrieve_chunks(
    query:          str,
    top_k:          int   = 5,
    fetch_k:        int   = 20,
    vector_weight:  float = 0.4,
    bm25_weight:    float = 0.3,
    graph_weight:   float = 0.3,
    vector_query:   str   = None,
    index_dirs:     Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """Retrieve from the global corpus, or from a per-session index set when
    index_dirs={'chroma':..., 'bm25':..., 'graph':...} is provided."""

    return _get_retriever(index_dirs).retrieve(
        query=query,
        top_k=top_k,
        fetch_k=fetch_k,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        graph_weight=graph_weight,
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
        print(f"       graph_score={r.get('graph_score', 0):.4f} | graph_rank={r.get('graph_rank')}")
        print(f"       {clean_text}\n")
