"""
pipeline/reranker.py

Cross-encoder re-ranker: scores (query, chunk) pairs jointly.

Why is this better than Person A's retrieval scores?
Person A uses "bi-encoders" — the query and document are embedded
separately, then compared. This is fast but imprecise.

A cross-encoder reads the full "(query + chunk)" concatenated string
and outputs a single relevance score. It cannot be pre-indexed
(too slow for that), but dramatically improves precision when applied
after retrieval as a filter on a small candidate set.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - ~250MB download on first run (one time only)
  - Trained on MS MARCO passage ranking
  - No API key, fully local, no data leaves your machine
  - Scores range roughly from -10 (irrelevant) to +10 (very relevant)
"""

import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()


class LocalReranker:
    """
    Uses cross-encoder/ms-marco-MiniLM-L-6-v2 from sentence-transformers.
    Fully local — ideal for legal/medical closed-document use cases.
    """

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self):
        from sentence_transformers import CrossEncoder
        print(f"Loading cross-encoder model '{self.MODEL_NAME}'...")
        print("  (First run downloads ~250MB — takes 1-3 minutes.)")
        self.model = CrossEncoder(self.MODEL_NAME, max_length=512)
        print("  Cross-encoder loaded.")

    def rerank(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Re-score and re-sort chunks against the query.

        Args:
            query:  The original user query. Do NOT use the HyDE paragraph here —
                    you want the re-ranker to judge relevance to the actual question.
            chunks: List of chunk dicts from Person A's retrieve_chunks().
            top_k:  How many top chunks to return after re-ranking.

        Returns:
            Re-sorted list of chunk dicts, each with an added 'rerank_score' field.
        """
        if not chunks:
            return []

        # Build (query, text) pairs — the cross-encoder scores them jointly
        pairs = [(query, chunk["text"]) for chunk in chunks]

        # Returns a numpy array of floats — higher = more relevant
        scores = self.model.predict(pairs)

        # Attach score to each chunk
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
        print(f"  [Reranker] Top rerank score: {reranked[0]['rerank_score']:.4f}")

        return reranked[:top_k]


def get_reranker() -> LocalReranker:
    """
    Factory function. Returns a LocalReranker.
    Extend this later if you want to add a Cohere option.
    """
    return LocalReranker()
