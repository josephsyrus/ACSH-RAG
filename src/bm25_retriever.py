import os
import json
import pickle
from rank_bm25 import BM25Okapi
from typing import List, Dict


# ─────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    return text.lower().split()


# ─────────────────────────────────────────────
# BM25 Retriever Class
# ─────────────────────────────────────────────

class BM25Retriever:
    def __init__(self, index_path: str = "./bm25_index"):
        self.index_path = index_path
        self.bm25: BM25Okapi = None
        self.chunks: List[Dict] = []
        os.makedirs(index_path, exist_ok=True)

    # ─────────────────────────────────────────
    # Build & Save
    # ─────────────────────────────────────────

    def build_index(self, chunks: List[Dict]):
        print(f"Building BM25 index from {len(chunks)} chunks...")
        self.chunks = chunks

        # BM25 needs a list-of-lists (tokenized documents)
        tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        self._save()
        print("BM25 index built and saved to disk.")

    def _save(self):
        bm25_path   = os.path.join(self.index_path, "bm25.pkl")
        chunks_path = os.path.join(self.index_path, "chunks.json")

        with open(bm25_path, "wb") as f:
            pickle.dump(self.bm25, f)

        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        print(f"  Saved BM25 index to '{bm25_path}'")

    # ─────────────────────────────────────────
    # Load
    # ─────────────────────────────────────────

    def load(self):
        bm25_path   = os.path.join(self.index_path, "bm25.pkl")
        chunks_path = os.path.join(self.index_path, "chunks.json")

        if not os.path.exists(bm25_path):
            raise FileNotFoundError(
                f"BM25 index not found at '{bm25_path}'.\n"
                "Run `python ingest_documents.py` first."
            )

        with open(bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)

        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        print(f"BM25 index loaded: {len(self.chunks)} chunks.")

    # ─────────────────────────────────────────
    # Search
    # ─────────────────────────────────────────

    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        if self.bm25 is None:
            raise RuntimeError("BM25 not loaded. Call build_index() or load() first.")

        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)  # numpy array, one score per chunk

        # Sort indices by score, descending
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        # Normalise scores to [0, 1] relative to the best result
        best_score = scores[top_indices[0]] if scores[top_indices[0]] > 0 else 1.0

        results = []
        for idx in top_indices:
            raw_score = float(scores[idx])
            if raw_score <= 0:
                break  # remaining scores are 0, not useful

            chunk = dict(self.chunks[idx])  # copy to avoid mutating the index
            chunk["bm25_score"]            = raw_score
            chunk["bm25_score_normalized"] = raw_score / best_score
            results.append(chunk)

        return results
