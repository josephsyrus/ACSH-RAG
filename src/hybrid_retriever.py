import os
from typing import List, Dict
from .vector_store import VectorStore
from .bm25_retriever import BM25Retriever


# ─────────────────────────────────────────────
# RRF Merge Function
# ─────────────────────────────────────────────

def reciprocal_rank_fusion(
    vector_results: List[Dict],
    bm25_results:   List[Dict],
    k:              int   = 60,
    vector_weight:  float = 0.5,
    bm25_weight:    float = 0.5,
) -> List[Dict]:
    """
    Args:
        vector_results: Ranked list from ChromaDB vector search (best first).
        bm25_results:   Ranked list from BM25 keyword search (best first).
        k:              RRF smoothing constant (60 is the empirically best default).
        vector_weight:  Weight multiplier for vector ranks (0–1).
        bm25_weight:    Weight multiplier for BM25 ranks (0–1).

    Returns:
        Merged list of unique chunks, sorted by combined RRF score (best first).
    """
    # accumulator: chunk_id -> { rrf_score, chunk_data, per-source metadata }
    scores: Dict[str, Dict] = {}

    # ── score vector results ──────────────────
    for rank, chunk in enumerate(vector_results, start=1):
        cid = chunk["chunk_id"]
        contribution = vector_weight / (k + rank)

        if cid not in scores:
            scores[cid] = {
                "chunk":        chunk,
                "rrf_score":    0.0,
                "found_in":     [],
                "vector_score": 0.0,
                "vector_rank":  None,
                "bm25_score":   0.0,
                "bm25_rank":    None,
            }
        scores[cid]["rrf_score"]   += contribution
        scores[cid]["found_in"].append("vector")
        scores[cid]["vector_score"] = chunk.get("vector_score", 0.0)
        scores[cid]["vector_rank"]  = rank

    # ── score bm25 results ────────────────────
    for rank, chunk in enumerate(bm25_results, start=1):
        cid = chunk["chunk_id"]
        contribution = bm25_weight / (k + rank)

        if cid not in scores:
            scores[cid] = {
                "chunk":        chunk,
                "rrf_score":    0.0,
                "found_in":     [],
                "vector_score": 0.0,
                "vector_rank":  None,
                "bm25_score":   0.0,
                "bm25_rank":    None,
            }
        scores[cid]["rrf_score"]  += contribution
        scores[cid]["found_in"].append("bm25")
        scores[cid]["bm25_score"]  = chunk.get("bm25_score_normalized", 0.0)
        scores[cid]["bm25_rank"]   = rank

    # ── sort by combined rrf score ────────────
    ranked = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)

    # ── assemble output dicts ─────────────────
    output = []
    for item in ranked:
        chunk = dict(item["chunk"])           # copy base chunk
        chunk["rrf_score"]    = round(item["rrf_score"],    6)
        chunk["vector_score"] = round(item["vector_score"], 6)
        chunk["bm25_score"]   = round(item["bm25_score"],   6)
        chunk["vector_rank"]  = item["vector_rank"]
        chunk["bm25_rank"]    = item["bm25_rank"]
        chunk["found_in"]     = list(set(item["found_in"]))  # deduplicate
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
        embedding_model:    str = "all-MiniLM-L6-v2",
    ):
        print("Initialising HybridRetriever...")
        self.vector_store = VectorStore(
            persist_directory=chroma_persist_dir,
            model_name=embedding_model,
        )
        self.bm25 = BM25Retriever(index_path=bm25_index_dir)
        self.bm25.load()
        print("HybridRetriever ready.\n")

    def retrieve(
        self,
        query:          str,
        top_k:          int   = 5,
        fetch_k:        int   = 20,
        vector_weight:  float = 0.5,
        bm25_weight:    float = 0.5,
        vector_query:   str   = None,
    ) -> List[Dict]:
        
        # Determine what each retriever actually searches
        _vector_query = vector_query if vector_query is not None else query

        print(f'Query (BM25):   "{query}"')
        if vector_query is not None:
            print(f'Query (vector): "{_vector_query[:80]}{"..." if len(_vector_query) > 80 else ""}"')

        print("  [1/3] Vector search...")
        vector_results = self.vector_store.search(_vector_query, top_k=fetch_k)
        print(f"        → {len(vector_results)} candidates")

        print("  [2/3] BM25 keyword search...")
        bm25_results = self.bm25.search(query, top_k=fetch_k)
        print(f"        → {len(bm25_results)} candidates")

        print("  [3/3] Merging with Reciprocal Rank Fusion...")
        merged = reciprocal_rank_fusion(
            vector_results,
            bm25_results,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )

        final = merged[:top_k]
        print(f"        → Returning top {len(final)} results")
        return final
