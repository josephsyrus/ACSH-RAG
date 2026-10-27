"""
pipeline_api.py

Clean public interface for the full ACSH-RAG pipeline.

Usage:
    from pipeline_api import run_pipeline

    result = run_pipeline("What is the penalty for late payment?")

    print(result["answer"])       # Final grounded answer
    print(result["citations"])    # List of chunk_ids cited
    print(result["route"])        # "simple" | "complex"
    print(result["confidence"])   # "pass" | "low_confidence" | "refused"
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.graph import get_pipeline
from typing import Dict, List


def shape_citations(state: Dict) -> List[Dict]:
    """Turn cited chunk_ids + reranked chunks into human-readable citations
    [{chunk_id, filename, page}]. Shared by the blocking and streaming APIs."""
    by_id = {c.get("chunk_id"): c for c in state.get("reranked_chunks", [])}
    citations = []
    for cid in state.get("cited_chunk_ids", []):
        c    = by_id.get(cid, {})
        meta = c.get("metadata") or {}
        citations.append({
            "chunk_id": cid,
            "filename": c.get("filename") or meta.get("filename") or "",
            "page":     c.get("page") or meta.get("page") or None,
        })
    return citations


def shape_details(state: Dict) -> Dict:
    """The numbers behind an answer's acceptance, for the 'advanced details' view:
    the confidence gate's score and thresholds, and the Self-RAG support check."""
    from pipeline.confidence_gate import MAX_RETRIES, PASS_THRESHOLD, RETRY_THRESHOLD

    chunks = state.get("reranked_chunks") or []
    score = chunks[0].get("rerank_score") if chunks else None
    if score is None:
        decision = "refuse"
    elif score >= PASS_THRESHOLD:
        decision = "pass"
    elif score >= RETRY_THRESHOLD:
        decision = "retry"
    else:
        decision = "refuse"

    critic = state.get("critic_result") or {}
    verification = None
    if critic:
        verification = {
            "checked":               critic.get("checked", True),
            "verdict":               critic.get("verdict"),
            "support_ratio":         critic.get("support_ratio"),
            "sentences_checked":     critic.get("sentences_checked"),
            "sentences_unsupported": critic.get("sentences_unsupported"),
            "threshold":             critic.get("support_threshold"),
            "explanation":           str(critic.get("explanation") or "")[:500],
        }

    return {
        "gate": {
            "score":           None if score is None else round(float(score), 3),
            "pass_threshold":  PASS_THRESHOLD,
            "retry_threshold": RETRY_THRESHOLD,
            "decision":        decision,
            "retries_used":    state.get("retry_count", 0),
            "max_retries":     MAX_RETRIES,
        },
        "verification": verification,
        # The gate passed, but the answer step said the passages don't answer it.
        "insufficient_context": state.get("draft_answer") == "INSUFFICIENT_CONTEXT",
    }


def run_pipeline(query: str, index_dirs: Dict = None) -> Dict:
    """
    Run the full ACSH-RAG pipeline for a query.

    Args:
        query (str): The raw user question.
        index_dirs (dict, optional): per-session index set
            {"chroma":..., "bm25":..., "graph":...} for an uploaded document.
            None = the global corpus.

    Returns:
        dict with keys:
            answer      (str)  — final answer text
            citations   (list) — {chunk_id, filename, page} cited in the answer
            route       (str)  — "simple" | "complex"
            confidence  (str)  — "pass" | "low_confidence" | "refused"
    """
    initial_state = {
        "original_query":  query,
        "index_dirs":      index_dirs,
        "route":           "",
        "active_query":    query,
        "sub_questions":   [],
        "hyde_text":       "",
        "raw_chunks":      [],
        "reranked_chunks": [],
        "gate_decision":   "",
        "retry_count":     0,
        "draft_answer":    "",
        "cited_chunk_ids": [],
        "critic_result":   {},
        "final_answer":    "",
        "confidence":      "",
    }

    pipeline = get_pipeline()
    result   = pipeline.invoke(initial_state)

    return {
        "answer":     result.get("final_answer",    "No answer generated."),
        "citations":  shape_citations(result),
        "route":      result.get("route",           "unknown"),
        "confidence": result.get("confidence",      "unknown"),
        "details":    shape_details(result),
        # The passages the answer was actually written from (after reranking),
        # so evaluation judges the answer against the same evidence.
        "contexts":   [
            {
                "text":     c.get("text", ""),
                "filename": c.get("filename") or (c.get("metadata") or {}).get("filename") or "",
                "page":     c.get("page") or (c.get("metadata") or {}).get("page") or None,
            }
            for c in result.get("reranked_chunks", [])
        ],
    }


# ── Quick self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "What is 2+2?",
        "What is the statute of limitations for breach of contract?",
    ]
    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print(f"{'='*60}")
        r = run_pipeline(q)
        print(f"\nRoute:      {r['route']}")
        print(f"Confidence: {r['confidence']}")
        print(f"Citations:  {r['citations']}")
        print(f"\nAnswer:\n{r['answer']}")
