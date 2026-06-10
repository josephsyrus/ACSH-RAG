"""
graph_retrieve_api.py

Graph-only retrieval interface.
Sets vector_weight=0 and bm25_weight=0 so ONLY the graph contributes to ranking.

Use this to:
  1. Test graph retrieval in isolation
  2. Compare against vector+BM25 results
  3. Feed into the comparison tool

Usage:
    from graph_retrieve_api import retrieve_chunks_graph

    results = retrieve_chunks_graph("What is habit stacking?", top_k=5)
    for r in results:
        print(r["graph_score"], r["text"][:80])
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrieve_api import retrieve_chunks
from typing import List, Dict


def retrieve_chunks_graph(
    query:   str,
    top_k:   int = 5,
    fetch_k: int = 20,
) -> List[Dict]:
    """
    Graph-only retrieval.

    Passes graph_weight=1.0, vector_weight=0.0, bm25_weight=0.0.
    Results are ranked purely by graph entity/concept overlap score.

    Note: HyDE is NOT used here because HyDE was designed to help
    vector search (embedding a hypothesis paragraph). For graph
    retrieval, the raw query is better since graph matching works
    on exact entity strings extracted by spaCy.

    Args:
        query:   The raw user question.
        top_k:   How many chunks to return.
        fetch_k: How many candidates each retriever fetches before merge.

    Returns:
        List of chunk dicts, sorted by graph_score descending.
        Each chunk has: chunk_id, text, graph_score, graph_rank, found_in.
    """
    results = retrieve_chunks(
        query=query,
        top_k=top_k,
        fetch_k=fetch_k,
        vector_weight=0.0,   # disable vector
        bm25_weight=0.0,     # disable BM25
        graph_weight=1.0,    # only graph
        vector_query=None,   # no HyDE for graph retrieval
    )

    # Filter to only chunks that were actually found by graph
    # (some chunks may sneak in with rrf_score=0 due to merge logic)
    graph_results = [r for r in results if r.get("graph_score", 0) > 0]

    print(f"  [GraphRetrieve] Returned {len(graph_results)} graph-matched chunks.")
    return graph_results


def retrieve_chunks_vector_only(
    query:        str,
    top_k:        int = 5,
    fetch_k:      int = 20,
    vector_query: str = None,
) -> List[Dict]:
    """
    Vector + BM25 only retrieval (no graph).
    Use this as the baseline to compare against graph results.

    Args:
        query:        The raw user question (drives BM25).
        top_k:        How many chunks to return.
        fetch_k:      Candidates per retriever before merge.
        vector_query: Optional HyDE paragraph for vector search.
    """
    results = retrieve_chunks(
        query=query,
        top_k=top_k,
        fetch_k=fetch_k,
        vector_weight=0.5,   # split weight between vector and BM25
        bm25_weight=0.5,
        graph_weight=0.0,    # disable graph
        vector_query=vector_query,
    )
    print(f"  [VectorRetrieve] Returned {len(results)} vector+BM25 chunks.")
    return results


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    query = "What is habit stacking and how does it work?"

    print("\n=== GRAPH ONLY ===")
    g_results = retrieve_chunks_graph(query, top_k=3)
    for r in g_results:
        print(f"  graph_score={r['graph_score']:.4f} | {r['text'][:80]}...")

    print("\n=== VECTOR + BM25 ONLY ===")
    v_results = retrieve_chunks_vector_only(query, top_k=3)
    for r in v_results:
        print(f"  rrf_score={r['rrf_score']:.5f} | {r['text'][:80]}...")