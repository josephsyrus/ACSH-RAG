# ACSH-RAG — Full Project Explanation

## What Problem Does This Solve?

When you have a large collection of documents (contracts, research papers, manuals, etc.) and want to ask questions about them, a plain LLM cannot help — it hasn't seen your documents, and hallucinating an answer is dangerous. A plain keyword search also fails when your question uses different words than the document does.

ACSH-RAG solves this by combining **three different ways to search your documents** and then using an LLM only to *read and summarise the retrieved text* — never to invent facts. Every claim in the final answer is traceable back to a specific chunk of your documents.

---

## The Two Phases: Setup vs. Query Time

### Phase 1 — Document Ingestion (run once)

Before any query can be answered, the system reads and indexes all your documents. This is done by running `ingest_documents.py`.

```
PDF / Markdown files
       │
       ▼
  [1] Load & extract text
       │
       ▼
  [2] Chunk into 250-token pieces (50-token overlap)
       │
       ├──▶ [3a] Embed chunks → store in ChromaDB (vector index)
       ├──▶ [3b] Build BM25 keyword index (saved to disk)
       └──▶ [3c] Extract entities/concepts → build NetworkX graph
```

**Why chunks?** An LLM context window is limited. Splitting documents into small, overlapping pieces lets the system surface only the most relevant few hundred tokens for a given question, rather than feeding thousands of pages to the LLM.

**Why overlap?** A 50-token overlap between adjacent chunks ensures that a sentence crossing a chunk boundary is not silently lost.

**Three separate indexes are built:**
- **ChromaDB** — stores vector (embedding) representations for semantic search.
- **BM25 index** — classic keyword-frequency index for exact-word matching.
- **Graph DB** — a NetworkX graph of entities and concepts linked by co-occurrence, for relationship-aware retrieval.

---

### Phase 2 — Query Pipeline (runs on every user question)

This is the main flow. It is built as a **LangGraph state machine**, meaning the query travels through a series of nodes and the path it takes depends on decisions made along the way.

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                        ROUTER                               │
│  Classifies the query into one of two routes:               │
│   • simple  — single-focus question about your documents    │
│   • complex — multi-part question needing several lookups   │
└─────────────────────────────────────────────────────────────┘
```

---

## The Retrieval Path (simple and complex queries)

### Step 1 — Route Decision

**Simple route** → one query, one retrieval pass.  
**Complex route** → the query is first *decomposed* into focused sub-questions.

**Query Decomposition (complex only):**  
A Gemini call breaks `"What are the differences between the termination and liability clauses, and what happens if both are triggered?"` into:
```json
[
  "What does the termination clause say?",
  "What does the liability clause say?",
  "What happens when both termination and liability are triggered simultaneously?"
]
```
Each sub-question is then handled independently through the retrieval step below, and the resulting chunks are merged and deduplicated before continuing.

---

### Step 2 — HyDE Generation (Hypothetical Document Embedding)

**This is the most important retrieval trick in the project.**

**The problem with naive vector search:**  
When you embed a short question like *"What is the penalty for late payment?"* and compare it against embedded document chunks, the match is often poor. The question and the answer look very different as vectors — a question is phrased as a question, while the document says *"In the event of delayed payment, a surcharge of 2% per month shall apply..."*

**What HyDE does:**  
Instead of embedding the question itself, the system first asks Gemini:  
*"Write a short paragraph that a document might contain that would answer this question."*

Gemini might produce:
> *"Late payment penalties are defined in Section 4.2. If payment is not received within 30 days of the invoice date, the defaulting party shall incur a 2% monthly surcharge on the outstanding balance, compounded monthly until full settlement."*

This hypothetical paragraph **looks like document text**, so its embedding lands much closer in vector space to the actual document chunks that contain the real answer.

**Key design point — split query usage:**
- The HyDE paragraph is used **only for vector (ChromaDB) search** — it is long and rich, perfect for semantic matching.
- The original short query is used **for BM25 keyword search** — BM25 works on exact term overlap, so feeding it the HyDE paragraph (which contains invented words) would hurt precision.

---

### Step 3 — Hybrid Retrieval (3-Way Fusion)

With the HyDE paragraph and the original query ready, three retrievers run in parallel:

```
HyDE paragraph ──▶  ChromaDB vector search  ──▶  top-20 chunks (by cosine similarity)
Original query ──▶  BM25 keyword search      ──▶  top-20 chunks (by BM25 score)
Original query ──▶  Graph retrieval          ──▶  chunks via entity/concept overlap
```

The three ranked lists are merged using **Reciprocal Rank Fusion (RRF)**:

```
RRF score for a chunk = Σ  weight / (k + rank_in_list)
                        over each list the chunk appears in
```

- `k = 60` (smoothing constant — dampens the advantage of rank 1 vs rank 2)
- A chunk that appears in **both** lists gets contributions from both, so it naturally floats to the top.
- Final output: a single merged, deduplicated list of the top 5 chunks.

**Why three methods?** Each has blind spots:
- Vector search finds *semantically similar* text but can miss exact terms.
- BM25 finds exact keywords but misses paraphrased or synonymous text.
- Graph search finds text connected by shared entities even if the phrasing differs entirely.
- Together, their weaknesses cancel out.

---

### Step 4 — Cross-Encoder Reranking

The 5 merged chunks are good candidates, but the RRF scores are indirect — they come from three separate ranking systems. A **cross-encoder** now reads each `(query, chunk)` pair *jointly* and outputs a single relevance score.

**Why is this better than the retrieval scores?**  
The embedding models (used in vector search) encode the query and each chunk *independently* and then compare them. This is fast but imprecise — nuance is lost.

A cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, runs locally) sees the full concatenated string `[query + chunk text]` at once and scores true relevance. This is too slow to run over thousands of chunks, but fast enough for 5 candidates.

Output: the 5 chunks re-sorted by their rerank score, highest first. The top 5 are passed forward.

---

### Step 5 — Confidence Gate

Before generating an answer, the system checks: *are these chunks actually relevant, or did the retriever just return the least-bad garbage?*

The **Confidence Gate** looks at the top chunk's rerank score:

| Score | Decision |
|-------|----------|
| ≥ 0.0 | **PASS** — chunks are good, proceed to answer generation |
| ≥ −3.0 | **RETRY** — chunks are weak, reformulate the query and try again |
| < −3.0 | **REFUSE** — nothing relevant found, give up |

**Corrective Loop (CRAG):**  
If the decision is RETRY, the system asks Gemini to rephrase the original query differently, regenerates a HyDE paragraph for the new query, and re-runs retrieval and reranking. This loop can happen up to **2 times** before refusing.

---

### Step 6 — Citation-Grounded Answer Generation

If chunks passed the gate, Gemini is asked to write an answer using **only the provided chunks**. Each chunk is labelled with a short ID (e.g., `[C42]`). The prompt explicitly instructs:
- Use only information from the provided chunks.
- Cite the chunk ID inline wherever you use information from it.
- If the chunks do not contain enough information, respond with `INSUFFICIENT_CONTEXT`.

Example output:
> *"The penalty for late payment is 2% per month on the outstanding balance [C42]. This surcharge compounds monthly until full settlement is received [C43]."*

The system then extracts all `[Cxx]` references to track which chunks were actually cited.

---

### Step 7 — Self-RAG Critic

A second Gemini call acts as a **fact-checker**. It receives the draft answer and the original chunks side-by-side and checks each sentence:

- Is this sentence actually supported by the provided chunks?
- Or did the LLM hallucinate or extrapolate beyond what the chunks say?

**Verdict options:**
- **pass** — every sentence is supported. Final answer = draft answer.
- **partial** — some sentences are unsupported. The critic identifies those sentences; they are deleted from the answer. Final answer = pruned answer.
- **fail** — the answer is fundamentally not grounded. The answer is withheld entirely.

---

### Final Output

The pipeline returns:
```python
{
    "final_answer":    "...",          # the verified, cited answer
    "confidence":      "pass",         # pass | low_confidence | refused
    "cited_chunk_ids": ["chunk_00042", "chunk_00043"],  # exact source chunks
    "critic_result":   { ... }         # full critic JSON for audit
}
```

---

## Complete Pipeline Flow (visual summary)

```
User Query
    │
    ▼
[Router]
    │
    ├── simple ──▶ [HyDE Generate]
    │                    │
    └── complex ──▶ [Decompose] ──▶ (for each sub-question: HyDE Generate)
                         │
                         ▼
                   [Retrieve]
                   ├── ChromaDB vector search  (uses HyDE paragraph)
                   ├── BM25 keyword search     (uses original query)
                   └── Graph retrieval         (uses original query)
                         │ Merge via RRF
                         ▼
                   [Rerank]
                   Cross-encoder scores each (query, chunk) pair
                         │
                         ▼
                   [Confidence Gate]
                   ├── PASS ──────────────────────────────────────────┐
                   ├── RETRY ──▶ [Reformulate] ──▶ [Retrieve] ──▶ ...│  (max 2x)
                   └── REFUSE ─────────────────────────────────────▶ [Refuse] ──▶ END
                                                                        │
                                                                        ▼
                                                              [Citation Generate]
                                                              Gemini writes answer
                                                              with inline [Cxx] citations
                                                                        │
                                                                        ▼
                                                              [Self-RAG Critic]
                                                              Checks every sentence
                                                              against source chunks
                                                                        │
                                                                        ▼
                                                                      END
                                                              final_answer returned
```

---

## File-by-File Reference

| File | What it does |
|------|-------------|
| `src/ingestion.py` | Loads PDFs/markdown, splits into 250-token overlapping chunks |
| `src/vector_store.py` | Wraps ChromaDB; embeds and stores/retrieves chunks by cosine similarity |
| `src/bm25_retriever.py` | BM25 keyword index; persisted to disk |
| `src/graph_store.py` | Builds and queries a NetworkX entity/concept graph |
| `src/hybrid_retriever.py` | Runs all three retrievers and merges results via RRF |
| `pipeline/router.py` | Classifies query as simple / complex using Gemini |
| `pipeline/hyde.py` | HyDE paragraph generation + complex query decomposition |
| `pipeline/reranker.py` | Cross-encoder reranking (fully local, no API) |
| `pipeline/confidence_gate.py` | Score-based pass/retry/refuse gate + query reformulation |
| `pipeline/citation_enforcer.py` | Grounded answer generation + Self-RAG critic fact-check |
| `pipeline/graph.py` | LangGraph state machine wiring all nodes together |
| `prompts/prompts.yaml` | All Gemini system prompts (router, HyDE, decomposer, critic, etc.) |
| `ingest_documents.py` | Entry point: build all three indexes from your documents |
| `pipeline_api.py` | Entry point: run the full pipeline on a query |
| `query.py` | Interactive CLI for querying |
| `retrieve_api.py` | Standalone hybrid retrieval (without the LLM pipeline) |
