"""
pipeline_api.py

Clean public interface for the full ACSH-RAG pipeline.

Usage:
    from pipeline_api import run_pipeline

    result = run_pipeline("What is the penalty for late payment?")

    print(result["answer"])       # Final grounded answer
    print(result["citations"])    # List of chunk_ids cited
    print(result["route"])        # "direct" | "simple" | "complex"
    print(result["confidence"])   # "pass" | "low_confidence" | "refused"
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.graph import get_pipeline
from typing import Dict


def run_pipeline(query: str) -> Dict:
    """
    Run the full ACSH-RAG pipeline for a query.

    Args:
        query (str): The raw user question.

    Returns:
        dict with keys:
            answer      (str)  — final answer text
            citations   (list) — chunk IDs cited in the answer
            route       (str)  — "direct" | "simple" | "complex"
            confidence  (str)  — "pass" | "low_confidence" | "refused"
    """
    initial_state = {
        "original_query":  query,
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
        "citations":  result.get("cited_chunk_ids", []),
        "route":      result.get("route",           "unknown"),
        "confidence": result.get("confidence",      "unknown"),
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
