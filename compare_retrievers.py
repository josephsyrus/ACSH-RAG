"""
compare_retrievers.py

Side-by-side comparison of three retrieval modes:
  1. Vector + BM25 only  (no graph)
  2. Graph only          (no vector, no BM25)
  3. 3-way hybrid        (vector + BM25 + graph, current default)

Usage:
    python compare_retrievers.py "What is habit stacking?"
    python compare_retrievers.py  ← interactive mode
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph_retrieve_api import retrieve_chunks_graph, retrieve_chunks_vector_only
from retrieve_api import retrieve_chunks


# ── Comparison Runner ─────────────────────────────────────────────────────────

def compare(query: str, top_k: int = 5) -> dict:
    """
    Run the same query through all three retrieval modes.

    Returns:
        dict with keys:
            vector_results  — chunks from vector+BM25 only
            graph_results   — chunks from graph only
            hybrid_results  — chunks from 3-way hybrid
            overlap         — chunk_ids that appear in both vector and graph
    """
    print(f"\n{'='*65}")
    print(f" Query: {query}")
    print(f"{'='*65}")

    print("\n[1/3] Running Vector + BM25 retrieval...")
    vector_results = retrieve_chunks_vector_only(query, top_k=top_k)

    print("\n[2/3] Running Graph-only retrieval...")
    graph_results = retrieve_chunks_graph(query, top_k=top_k)

    print("\n[3/3] Running 3-way Hybrid retrieval...")
    hybrid_results = retrieve_chunks(query, top_k=top_k)

    # Find overlap — same chunk appearing in both vector and graph results
    vector_ids = {r["chunk_id"] for r in vector_results}
    graph_ids  = {r["chunk_id"] for r in graph_results}
    overlap    = vector_ids & graph_ids

    return {
        "query":          query,
        "vector_results": vector_results,
        "graph_results":  graph_results,
        "hybrid_results": hybrid_results,
        "overlap":        overlap,
    }


def print_comparison(result: dict) -> None:
    """Pretty-print the comparison results."""
    query          = result["query"]
    vector_results = result["vector_results"]
    graph_results  = result["graph_results"]
    hybrid_results = result["hybrid_results"]
    overlap        = result["overlap"]

    def _preview(text: str, width: int = 70) -> str:
        clean = " ".join(text.split())
        return clean[:width] + "..." if len(clean) > width else clean

    # ── Vector + BM25 results ─────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f" MODE 1: Vector + BM25 only")
    print(f"{'─'*65}")
    if not vector_results:
        print("  No results.")
    for i, r in enumerate(vector_results, 1):
        flag = " ◄ also in graph" if r["chunk_id"] in overlap else ""
        print(f"  [{i}] rrf={r['rrf_score']:.5f} | vec={r['vector_score']:.3f} bm25={r['bm25_score']:.3f}{flag}")
        print(f"       {_preview(r['text'])}")

    # ── Graph-only results ────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f" MODE 2: Graph only")
    print(f"{'─'*65}")
    if not graph_results:
        print("  No results. (Query may have no extractable entities/noun phrases)")
    for i, r in enumerate(graph_results, 1):
        flag = " ◄ also in vector" if r["chunk_id"] in overlap else ""
        print(f"  [{i}] graph_score={r['graph_score']:.4f}{flag}")
        print(f"       {_preview(r['text'])}")

    # ── 3-way Hybrid results ──────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f" MODE 3: 3-way Hybrid (vector + BM25 + graph)")
    print(f"{'─'*65}")
    for i, r in enumerate(hybrid_results, 1):
        found = ", ".join(r.get("found_in", []))
        print(f"  [{i}] rrf={r['rrf_score']:.5f} | found_in=[{found}]")
        print(f"       vec={r['vector_score']:.3f} bm25={r['bm25_score']:.3f} graph={r['graph_score']:.3f}")
        print(f"       {_preview(r['text'])}")

    # ── Overlap summary ───────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f" OVERLAP SUMMARY")
    print(f"{'─'*65}")
    print(f"  Vector+BM25 returned : {len(vector_results)} chunks")
    print(f"  Graph returned       : {len(graph_results)} chunks")
    print(f"  Chunks in BOTH       : {len(overlap)}")
    if overlap:
        print(f"  Shared chunk IDs: {list(overlap)}")
    agreement = (len(overlap) / max(len(vector_results), len(graph_results), 1)) * 100
    print(f"  Agreement rate       : {agreement:.0f}%")
    print(f"\n  Insight:")
    if agreement >= 60:
        print("  High agreement — both methods found similar chunks.")
        print("  Graph adds entity-level precision on top of semantic search.")
    elif agreement >= 30:
        print("  Moderate agreement — methods are complementary.")
        print("  3-way hybrid likely gives best recall for this query type.")
    else:
        print("  Low agreement — methods found very different chunks.")
        print("  This query is where graph and vector diverge most.")
        print("  Manually inspect which results are actually more relevant.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        # Direct query mode
        query = " ".join(sys.argv[1:])
        result = compare(query)
        print_comparison(result)
    else:
        # Interactive mode
        print("\nComparison tool — interactive mode")
        print("Type 'exit' to quit\n")
        while True:
            try:
                query = input("Query> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye.")
                break
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                break
            result = compare(query)
            print_comparison(result)


if __name__ == "__main__":
    main()