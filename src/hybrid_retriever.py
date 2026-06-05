import os
from typing import List, Dict, Optional
from .vector_store import VectorStore
from .bm25_retriever import BM25Retriever
from .graph_store import GraphStore


# ─────────────────────────────────────────────
# RRF Merge Function
# ─────────────────────────────────────────────

def reciprocal_rank_fusion(
    vector_results: List[Dict],
    bm25_results:   List[Dict],
    graph_results:  Optional[List[Dict]] = None,
    k:              int   = 60,
    vector_weight:  float = 0.4,
    bm25_weight:    float = 0.3,
    graph_weight:   float = 0.3,
) -> List[Dict]:
    """
    3-way Reciprocal Rank Fusion over vector, BM25, and graph results.

    Args:
        vector_results: Ranked list from ChromaDB vector search (best first).
        bm25_results:   Ranked list from BM25 keyword search (best first).
        graph_results:  Ranked list from graph entity search (best first). Optional.
        k:              RRF smoothing constant (60 is the empirically best default).
        vector_weight:  Weight multiplier for vector ranks.
        bm25_weight:    Weight multiplier for BM25 ranks.
        graph_weight:   Weight multiplier for graph ranks.

    Returns:
        Merged list of unique chunks, sorted by combined RRF score (best first).
    """
    # accumulator: chunk_id -> { rrf_score, chunk_data, per-source metadata }
    scores: Dict[str, Dict] = {}

    def _ensure(cid, chunk):
        if cid not in scores:
            scores[cid] = {
                "chunk":        chunk,
                "rrf_score":    0.0,
                "found_in":     [],
                "vector_score": 0.0,
                "vector_rank":  None,
                "bm25_score":   0.0,
                "bm25_rank":    None,
                "graph_score":  0.0,
                "graph_rank":   None,
            }

    # ── score vector results ──────────────────
    for rank, chunk in enumerate(vector_results, start=1):
        cid = chunk["chunk_id"]
        _ensure(cid, chunk)
        scores[cid]["rrf_score"]   += vector_weight / (k + rank)
        scores[cid]["found_in"].append("vector")
        scores[cid]["vector_score"] = chunk.get("vector_score", 0.0)
        scores[cid]["vector_rank"]  = rank

    # ── score bm25 results ────────────────────
    for rank, chunk in enumerate(bm25_results, start=1):
        cid = chunk["chunk_id"]
        _ensure(cid, chunk)
        scores[cid]["rrf_score"]  += bm25_weight / (k + rank)
        scores[cid]["found_in"].append("bm25")
        scores[cid]["bm25_score"]  = chunk.get("bm25_score_normalized", 0.0)
        scores[cid]["bm25_rank"]   = rank

    # ── score graph results ───────────────────
    for rank, chunk in enumerate(graph_results or [], start=1):
        cid = chunk["chunk_id"]
        _ensure(cid, chunk)
        scores[cid]["rrf_score"]  += graph_weight / (k + rank)
        scores[cid]["found_in"].append("graph")
        scores[cid]["graph_score"] = chunk.get("graph_score", 0.0)
        scores[cid]["graph_rank"]  = rank

    # ── sort by combined rrf score ────────────
    ranked = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)

    # ── assemble output dicts ─────────────────
    output = []
    for item in ranked:
        chunk = dict(item["chunk"])
        chunk["rrf_score"]    = round(item["rrf_score"],    6)
        chunk["vector_score"] = round(item["vector_score"], 6)
        chunk["bm25_score"]   = round(item["bm25_score"],   6)
        chunk["graph_score"]  = round(item["graph_score"],  6)
        chunk["vector_rank"]  = item["vector_rank"]
        chunk["bm25_rank"]    = item["bm25_rank"]
        chunk["graph_rank"]   = item["graph_rank"]
        chunk["found_in"]     = list(dict.fromkeys(item["found_in"]))  # deduplicate, preserve order
        output.append(chunk)

    return output


# ─────────────────────────────────────────────
# HybridRetriever Class
# ─────────────────────────────────────────────

class HybridRetriever:
    def __init__(
        self,
        chroma_persist_dir: str = "./chroma_db",
        bm25_index_dir:     str = "./bm25_index",
        graph_db_dir:       str = "./graph_db",
        embedding_model:    str = "all-MiniLM-L6-v2",
    ):
        print("Initialising HybridRetriever...")
        self.vector_store = VectorStore(
            persist_directory=chroma_persist_dir,
            model_name=embedding_model,
        )
        self.bm25 = BM25Retriever(index_path=bm25_index_dir)
        self.bm25.load()

        self.graph_store = GraphStore(graph_db_dir=graph_db_dir)
        graph_loaded = self.graph_store.load()
        if not graph_loaded:
            print(
                "  WARNING: graph_db not found. Graph retrieval disabled.\n"
                "  Re-run: python ingest_documents.py --force"
            )

        print("HybridRetriever ready.\n")

    def retrieve(
        self,
        query:          str,
        top_k:          int   = 5,
        fetch_k:        int   = 20,
        vector_weight:  float = 0.4,
        bm25_weight:    float = 0.3,
        graph_weight:   float = 0.3,
        vector_query:   str   = None,
    ) -> List[Dict]:

        # Determine what each retriever actually searches
        _vector_query = vector_query if vector_query is not None else query

        print(f'Query (BM25/graph): "{query}"')
        if vector_query is not None:
            print(f'Query (vector):     "{_vector_query[:80]}{"..." if len(_vector_query) > 80 else ""}"')

        print("  [1/4] Vector search...")
        vector_results = self.vector_store.search(_vector_query, top_k=fetch_k)
        print(f"        → {len(vector_results)} candidates")

        print("  [2/4] BM25 keyword search...")
        bm25_results = self.bm25.search(query, top_k=fetch_k)
        print(f"        → {len(bm25_results)} candidates")

        print("  [3/4] Graph entity search...")
        graph_results = self.graph_store.search(query, top_k=fetch_k)
        print(f"        → {len(graph_results)} candidates")

        print("  [4/4] Merging with 3-way Reciprocal Rank Fusion...")
        merged = reciprocal_rank_fusion(
            vector_results,
            bm25_results,
            graph_results,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            graph_weight=graph_weight,
        )

        final = merged[:top_k]
        print(f"        → Returning top {len(final)} results")
        return final
