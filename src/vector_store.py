import os
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict


EMBEDDING_MODEL = "all-MiniLM-L6-v2"



class VectorStore:
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        model_name: str = EMBEDDING_MODEL,
    ):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        print(f"Loading embedding model '{model_name}'...")
        print("  (First run downloads ~90MB — takes 1–2 minutes.)")
        self.model = SentenceTransformer(model_name)
        print("  Model loaded.")

        # PersistentClient saves everything to disk automatically
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
            # cosine similarity is standard for text — measures angle between
            # vectors, not their magnitude
        )
        count = self.collection.count()
        print(f"ChromaDB ready. Collection has {count} existing chunk(s).")

    # ─────────────────────────────────────────
    # Indexing
    # ─────────────────────────────────────────

    def is_empty(self) -> bool:
        return self.collection.count() == 0

    def add_chunks(self, chunks: List[Dict], batch_size: int = 64):
        total = len(chunks)
        print(f"Embedding and storing {total} chunks (batch size: {batch_size})...")

        for batch_start in range(0, total, batch_size):
            batch = chunks[batch_start : batch_start + batch_size]

            texts     = [c["text"]     for c in batch]
            ids       = [c["chunk_id"] for c in batch]
            metadatas = [
                {
                    "source":       c["source"],
                    "filename":     c["filename"],
                    "doc_type":     c["doc_type"],
                    "chunk_index":  c["chunk_index"],
                    "total_chunks": c["total_chunks"],
                }
                for c in batch
            ]

            # encode() returns a numpy array of shape (batch_size, 384)
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,  # unit-normalize for cosine similarity
            ).tolist()

            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            done = min(batch_start + batch_size, total)
            print(f"  [{done}/{total}] chunks stored in ChromaDB")

        print(f"Done. ChromaDB now contains {self.collection.count()} chunk(s).")

    # ─────────────────────────────────────────
    # Searching
    # ─────────────────────────────────────────

    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        count = self.collection.count()
        if count == 0:
            return []

        # Embed the query using the same model
        query_embedding = self.model.encode(
            [query], normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for i in range(len(results["ids"][0])):
            # ChromaDB returns cosine DISTANCE (0 = identical, 2 = opposite)
            # Convert to similarity score: 1 - distance (1 = identical, -1 = opposite)
            distance   = results["distances"][0][i]
            similarity = 1.0 - distance

            output.append({
                "chunk_id":     results["ids"][0][i],
                "text":         results["documents"][0][i],
                "metadata":     results["metadatas"][0][i],
                "vector_score": round(similarity, 6),
            })

        return output
