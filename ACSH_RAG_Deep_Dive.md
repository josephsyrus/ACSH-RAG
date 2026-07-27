# ACSH-RAG — Complete Technical Deep-Dive

> **What is it?**  
> A production-quality, fully local Retrieval-Augmented Generation (RAG) system that answers questions over your own documents (PDF / Markdown / text) with citations, factual verification, and graceful fallback — all driven by Google Gemini and running entirely on your machine.

---

## Table of Contents

1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Phase 1 — Ingestion](#3-phase-1--ingestion)
4. [Phase 2 — Retrieval](#4-phase-2--retrieval)
5. [Phase 3 — The Adaptive Pipeline](#5-phase-3--the-adaptive-pipeline)
6. [Phase 4 — Answer Generation & Verification](#6-phase-4--answer-generation--verification)
7. [The Web App](#7-the-web-app)
8. [Evaluation (RAGAS)](#8-evaluation-ragas)
9. [What's Novel in This Project](#9-whats-novel-in-this-project)
10. [Data Flow: End-to-End Example](#10-data-flow-end-to-end-example)
11. [File Map](#11-file-map)
12. [Setup & Running](#12-setup--running)

---

## 1. What Problem Does This Solve?

Standard LLMs (GPT-4, Gemini, etc.) have two fundamental limits:

| Problem | Consequence |
|---|---|
| **Knowledge cutoff** | They don't know about your private documents, internal reports, or recent files |
| **Hallucination** | They confidently generate facts that don't exist in your documents |

**RAG** (Retrieval-Augmented Generation) fixes this by making the LLM *look things up* in your documents before answering. But basic RAG has its own issues:

- Simple keyword search misses semantic meaning
- Simple vector search misses exact keywords
- Neither surface relationships between concepts
- No way to detect when retrieved content is irrelevant
- No way to verify the LLM didn't make things up

**ACSH-RAG solves all of these** with a multi-signal retrieval layer, an adaptive query pipeline, and a Self-RAG critic that verifies every claim before responding.

---

## 2. High-Level Architecture

```
              ┌─────────────────────────────────────────────────┐
              │                USER QUESTION                     │
              └────────────────────┬────────────────────────────┘
                                   ▼
                          ┌────────────────┐
                          │ Adaptive Router │  ← Gemini classifies query
                          └───┬────┬───────┘
                 ┌────────────┘    └──────────────┐
                 ▼                                 ▼
           "direct"                   "simple" / "complex"
                 │                                 │
                 ▼                     ┌───────────▼──────────────┐
         General knowledge             │    HyDE Generation        │
         (no retrieval)                │  (hypothetical document)  │
                                       └───────────┬──────────────┘
                                                   │
                               ┌───────────────────▼─────────────────────┐
                               │          3-Way Hybrid Retrieval           │
                               │   Vector (ChromaDB) + BM25 + Graph NER   │
                               │   ──────── RRF Fusion ──────────────────  │
                               └───────────┬─────────────────────────────┘
                                           │
                                ┌──────────▼──────────┐
                                │  Cross-Encoder       │
                                │  Reranker            │
                                └──────────┬──────────┘
                                           │
                                ┌──────────▼──────────┐
                                │  Confidence Gate     │  → RETRY loop (CRAG)
                                └──────────┬──────────┘
                                           │ PASS
                                ┌──────────▼──────────┐
                                │  Citation Enforcer   │  ← Gemini with citations
                                └──────────┬──────────┘
                                           │
                                ┌──────────▼──────────┐
                                │  Self-RAG Critic     │  ← Gemini verifies claims
                                └──────────┬──────────┘
                                           │
                              ┌────────────▼──────────────┐
                              │     FINAL VERIFIED ANSWER  │
                              └───────────────────────────┘
```

---

## 3. Phase 1 — Ingestion

> **Files:** [`ingest_documents.py`](file:///d:/ACSH-RAG/ingest_documents.py), [`src/ingestion.py`](file:///d:/ACSH-RAG/src/ingestion.py), [`src/vector_store.py`](file:///d:/ACSH-RAG/src/vector_store.py), [`src/bm25_retriever.py`](file:///d:/ACSH-RAG/src/bm25_retriever.py), [`src/graph_store.py`](file:///d:/ACSH-RAG/src/graph_store.py)

Ingestion transforms raw documents into **three distinct indexes** that serve fundamentally different retrieval strategies. You run this once (or `--force` to rebuild):

```bash
python ingest_documents.py
```

### Step 1: Document Loading

Supports `.pdf`, `.md`, `.markdown`, `.txt`. For PDFs, each page is extracted individually (using `pypdf`) and the **page number is preserved** — so every chunk produced from a PDF knows what page it came from.

### Step 2: Token-Accurate Chunking

> **File:** [`src/ingestion.py → chunk_text()`](file:///d:/ACSH-RAG/src/ingestion.py)

This is not naive character splitting. The chunker uses **tiktoken** (`cl100k_base` encoding — the same tokenizer used by modern LLMs) to slice text into windows of exactly 250 tokens with 50-token overlap.

**Why token-based chunking matters:**
- Embedding models like `all-MiniLM-L6-v2` have a hard 256-token sequence limit. Exceed it and the model silently truncates text without any error.
- Character-based chunking can produce wildly different token counts for different languages/content types.
- Overlap ensures context is not lost at chunk boundaries — a sentence split across a boundary appears in both adjacent chunks.

For PDFs, the chunker also tracks the **token offset** at which each page starts, allowing it to compute an exact 1-indexed page number for every chunk using a `bisect` binary search.

Each chunk gets a stable, unique `chunk_id` like `google_terms_of_service_pdf_chunk_00042` derived from the filename and chunk index.

### Step 3: Three Parallel Indexes

All three indexes are built from the **same set of chunk dicts** — the data is consistent across all retrieval methods.

---

#### 3a. Vector Index — ChromaDB + `all-MiniLM-L6-v2`

> **File:** [`src/vector_store.py`](file:///d:/ACSH-RAG/src/vector_store.py)

Each chunk's text is converted to a **384-dimensional dense embedding vector** using the `all-MiniLM-L6-v2` sentence transformer. These vectors are stored in ChromaDB using **cosine similarity** as the distance metric (two passages with the same meaning have similar vectors, regardless of word choice).

**Key engineering detail:** The embedding model is cached at module level — all sessions (including per-user uploads in the web app) share **one loaded model instance** instead of each paying ~90MB of memory.

Queries are answered by embedding the query and returning the `top_k` chunks with the highest cosine similarity.

---

#### 3b. BM25 Keyword Index

> **File:** [`src/bm25_retriever.py`](file:///d:/ACSH-RAG/src/bm25_retriever.py)

**BM25 (Best Match 25)** is a classical information-retrieval algorithm that scores documents by term frequency and inverse document frequency (TF-IDF with saturation). It rewards chunks that contain exactly the words in the query, penalising very common words.

The BM25 index (`rank-bm25` library, `BM25Okapi` variant) is serialized to `bm25_index/bm25.pkl` via pickle. Scores are normalized to `[0, 1]` relative to the top result.

**Why BM25 alongside vectors?** Vector search excels at semantic similarity ("what is the penalty?" matches "the consequence for breach is...") but can fail on rare domain-specific terms, proper nouns, and exact phrases. BM25 excels precisely where vectors fail — and vice versa.

---

#### 3c. Graph Index — NetworkX + spaCy NER

> **File:** [`src/graph_store.py`](file:///d:/ACSH-RAG/src/graph_store.py)

This is the most architecturally novel index. A **knowledge graph** is constructed using NetworkX (`DiGraph`) with three types of nodes:

| Node Type | Example | Description |
|---|---|---|
| `chunk` | `contract_pdf_chunk_00003` | One per document chunk |
| `entity` | `ent_google llc` | Named entities extracted by spaCy NER |
| `concept` | `np_penalty clause` | Multi-word noun phrases from spaCy |

**Edges:**
- `chunk → entity/concept` (`CONTAINS`): a chunk mentions this named entity or concept
- `chunk → chunk` (`NEXT`): sequential adjacency within a document

**Building the graph:** spaCy processes all chunks in **batches of 64** (much faster than one-at-a-time) with the `lemmatizer` disabled but parser enabled (required for noun chunks). Text is pre-cleaned to strip PDF extraction noise like `13Combination` → `Combination`.

A key sub-phrase trick: a noun phrase `"combination cooking feature"` is also stored as `"combination cooking"` and `"cooking feature"`. This allows partial query phrases to match.

**Graph search:** At query time, spaCy extracts entities and noun phrases from the query, then looks up matching nodes in the graph. A raw n-gram fallback (all bigrams and trigrams from the query) catches cases where spaCy's parser fails. Chunks are scored with **IDF weighting** — rare signals (few chunks contain them) count more than common ones. The score is normalized by total possible IDF weight.

The graph is serialized to `graph_db/graph.json` via `networkx.readwrite.json_graph`.

---

## 4. Phase 2 — Retrieval

> **Files:** [`retrieve_api.py`](file:///d:/ACSH-RAG/retrieve_api.py), [`src/hybrid_retriever.py`](file:///d:/ACSH-RAG/src/hybrid_retriever.py)

### The Public API

`retrieve_api.py` is the **clean public interface** for the retrieval layer. Teammates import only `retrieve_chunks()` and get back a list of dicts with fields like `text`, `filename`, `rrf_score`, `vector_score`, `bm25_score`, `graph_score`, `found_in`, etc.

```python
from retrieve_api import retrieve_chunks

results = retrieve_chunks("what is the statute of limitations?", top_k=5)
```

### LRU Retriever Cache

The module maintains an **LRU (Least Recently Used) cache** of up to 12 `HybridRetriever` instances using `OrderedDict`. Each retriever holds BM25 and graph data in memory. When the web app creates per-user upload sessions, each session gets its own retriever that is evicted from cache after disuse. The **global corpus retriever is never evicted**.

### 3-Way Reciprocal Rank Fusion (RRF)

> **File:** [`src/hybrid_retriever.py → reciprocal_rank_fusion()`](file:///d:/ACSH-RAG/src/hybrid_retriever.py)

After all three retrievers return their top-20 candidates, the results are merged using **Reciprocal Rank Fusion** — a well-researched algorithm for combining ranked lists without needing to normalize raw scores across different scales.

The formula for each chunk's combined score is:

```
rrf_score = vector_weight / (k + rank_vector)
          + bm25_weight   / (k + rank_bm25)
          + graph_weight  / (k + rank_graph)
```

Default weights: `vector=0.4`, `bm25=0.3`, `graph=0.3`. The smoothing constant `k=60` is empirically the best default from the original RRF paper.

**Why RRF instead of score normalization?** Raw scores from three different systems (cosine similarity, BM25 Okapi, IDF-weighted graph overlap) are not comparable — normalizing them requires assumptions about their distributions. RRF only uses *rank position*, which is scale-free and robust.

The final output includes, for each chunk, which of the three systems found it (`found_in`), its rank in each system, and its individual score from each system — giving full transparency for debugging and analysis.

### The `vector_query` Split

A critical design decision: `retrieve_chunks` accepts a separate `vector_query` parameter. The pipeline passes the **HyDE paragraph** as `vector_query` while using the **original query** for BM25 and graph. This is intentional — using a HyDE paragraph for BM25 keyword matching would degrade its performance because HyDE text adds words not in the original query.

---

## 5. Phase 3 — The Adaptive Pipeline

> **Files:** [`pipeline/graph.py`](file:///d:/ACSH-RAG/pipeline/graph.py), [`pipeline/router.py`](file:///d:/ACSH-RAG/pipeline/router.py), [`pipeline/hyde.py`](file:///d:/ACSH-RAG/pipeline/hyde.py), [`pipeline/confidence_gate.py`](file:///d:/ACSH-RAG/pipeline/confidence_gate.py)

The pipeline is a **LangGraph state machine** — a directed graph of nodes where each node receives the full pipeline state, modifies some keys, and returns only what it changed.

### Pipeline State (`PipelineState`)

```python
class PipelineState(TypedDict):
    original_query:  str
    index_dirs:      Optional[Dict]   # None = global corpus
    emit:            Optional[object] # SSE streaming callback
    route:           str              # "direct" | "simple" | "complex"
    active_query:    str
    sub_questions:   List[str]
    hyde_text:       str
    raw_chunks:      List[Dict]
    reranked_chunks: List[Dict]
    gate_decision:   str              # "pass" | "retry" | "refuse"
    retry_count:     int
    draft_answer:    str
    cited_chunk_ids: List[str]
    critic_result:   Dict
    final_answer:    str
    confidence:      str
```

### Node 1: Adaptive Router

> **File:** [`pipeline/router.py → AdaptiveRouter`](file:///d:/ACSH-RAG/pipeline/router.py)

Gemini classifies the query into one of three routes:

| Route | When | What happens |
|---|---|---|
| `direct` | General knowledge question, no documents needed | Answer from Gemini directly, skip retrieval |
| `simple` | Single focused document question | HyDE → retrieve → rerank → gate → cite → critic |
| `complex` | Multi-part, comparative, or relational question | Decompose → retrieve per sub-question → merge → rerank → gate → cite → critic |

The router uses `temperature=0` and `thinking_budget=0` (disables Gemini's chain-of-thought reasoning, which would consume the 64-token output budget and return empty text). It expects JSON `{"route": "simple"}` but falls back to word-scan if parsing fails.

### Node 2: HyDE — Hypothetical Document Embedding

> **File:** [`pipeline/hyde.py → HyDEGenerator`](file:///d:/ACSH-RAG/pipeline/hyde.py)

**HyDE** is a technique from the paper *"Precise Zero-Shot Dense Retrieval without Relevance Labels"* (Gao et al., 2022). The insight is that **a hypothetical answer paragraph** is semantically much closer to real answer passages in embedding space than the raw question is.

Example:
- Query: `"What is the statute of limitations?"`
- HyDE output: `"The statute of limitations is a legal deadline after which a party may no longer bring a legal claim. For most civil matters, this period ranges from 2–6 years depending on jurisdiction and claim type..."`

This hypothetical text (even if factually imperfect) will have a much higher cosine similarity with the relevant document passage than the short question would.

### Node 3: Query Decomposer (for complex route)

> **File:** [`pipeline/hyde.py → QueryDecomposer`](file:///d:/ACSH-RAG/pipeline/hyde.py)

For complex queries, Gemini breaks the question into 2–4 independent sub-questions returned as a JSON array. Each sub-question then gets its own HyDE generation and its own retrieval pass. Results are merged and deduplicated by `chunk_id` before reranking.

### Node 4: Cross-Encoder Reranker

> **File:** [`pipeline/reranker.py → LocalReranker`](file:///d:/ACSH-RAG/pipeline/reranker.py)

The top-20 candidates from hybrid retrieval are reranked by a **cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`).

**Bi-encoder vs. cross-encoder:**

| | Bi-encoder (retrieval) | Cross-encoder (reranker) |
|---|---|---|
| How it works | Encode query and doc separately, compare | Read `[QUERY] + [DOC]` jointly |
| Speed | Fast — can pre-index all docs | Slow — cannot pre-index |
| Precision | Lower | Much higher |
| Use | First-pass retrieval over millions of docs | Second-pass reranking over top-N |

The cross-encoder scores range roughly −10 (irrelevant) to +10 (very relevant) and are **fully local** — no API calls, no data leaving the machine.

### Node 5: Confidence Gate + CRAG Loop

> **File:** [`pipeline/confidence_gate.py → ConfidenceGate`](file:///d:/ACSH-RAG/pipeline/confidence_gate.py)

The top reranker score determines whether retrieval was good enough:

| Score | Decision | Action |
|---|---|---|
| `≥ 0.0` | **PASS** | Proceed to answer generation |
| `≥ -3.0` | **RETRY** | Reformulate query and retrieve again |
| `< -3.0` | **REFUSE** | Inform user no relevant content was found |

**CRAG (Corrective RAG)** is implemented as a loop: on `RETRY`, Gemini rewrites the original query with different terminology. This retries up to `MAX_RETRIES=2` times before refusing.

A degenerate-reformulation guard prevents Gemini from returning a useless 1–2 word query, falling back to the original query instead.

---

## 6. Phase 4 — Answer Generation & Verification

> **File:** [`pipeline/citation_enforcer.py → CitationEnforcer`](file:///d:/ACSH-RAG/pipeline/citation_enforcer.py), [`prompts/prompts.yaml`](file:///d:/ACSH-RAG/prompts/prompts.yaml)

### Citation-Grounded Answer Generation

The top-5 reranked chunks are formatted with short citation labels like `[C42]`, `[C137]`, etc. (extracted from the numeric suffix of the `chunk_id`). The system prompt instructs Gemini to:

1. Answer **only** from the provided context chunks
2. Cite every factual claim with its exact bracket label
3. Return the single word `INSUFFICIENT_CONTEXT` if the context is inadequate

The citation IDs are then mapped back to full `chunk_id` strings, which are returned alongside the answer for downstream use.

**SSE token streaming:** When called from the web app's `/api/ask_stream` endpoint, the `on_token` callback is passed to `generate_content_stream()`, streaming tokens directly to the browser as they are generated.

### Self-RAG Critic

**Self-RAG** is a technique from the paper *"Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"* (Asai et al., 2023). Here it's implemented as a second Gemini call that acts as a **strict fact-checker**:

Given the draft answer and the context chunks, the critic evaluates every sentence and returns a JSON verdict:

```json
{
  "verdict": "partial",
  "unsupported_sentences": ["The fine is $1,000,000 per violation."],
  "explanation": "This amount is not stated in any context chunk.",
  "final_answer": "..."
}
```

| Verdict | Meaning | Action |
|---|---|---|
| `pass` | All sentences are supported | Return the answer with `confidence: pass` |
| `partial` | Some sentences unsupported | Remove unsupported sentences, return with `confidence: low_confidence` |
| `fail` | Most/all sentences unsupported | Withhold the answer entirely |

This is the safety net that prevents hallucinated content from reaching the user even if citation generation partially failed.

---

## 7. The Web App

> **File:** [`web_app.py`](file:///d:/ACSH-RAG/web_app.py)

A **FastAPI** application serving a chat UI at `http://localhost:8000` with three API endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | Liveness check |
| `/api/upload` | POST | Upload a document file; get back a `session_id` |
| `/api/ask` | POST | Ask a question (batch response) |
| `/api/ask_stream` | POST | Ask a question (SSE token streaming) |

### Per-Session Upload Isolation

When a user uploads a document via `/api/upload`:
1. The file is saved to `sessions/<session_id>/upload/`
2. `src/upload.ingest_upload()` runs the full ingestion pipeline (vector + BM25 + graph) into isolated per-session directories
3. The `session_id` is returned to the browser
4. Subsequent `/api/ask` calls with that `session_id` query **only that user's document**, completely isolated from the global corpus

Sessions are automatically reaped after 24 hours or when the session count exceeds 200.

### Streaming Architecture

`/api/ask_stream` uses a **producer-consumer pattern**:
- The pipeline runs in a **daemon thread** and pushes events onto a `queue.Queue`
- The request thread **drains the queue** and serialises each event as a Server-Sent Event (SSE) frame
- The browser receives stage labels ("Searching the documents…"), individual LLM tokens, and finally the complete answer — all in real time

---

## 8. Evaluation (RAGAS)

> **Files:** [`evaluation/eval.py`](file:///d:/ACSH-RAG/evaluation/eval.py), [`evaluation/ragas_config.yaml`](file:///d:/ACSH-RAG/evaluation/ragas_config.yaml)

The evaluation framework uses **RAGAS** (RAG Assessment) with three reference-free metrics that don't require a golden dataset:

| Metric | What it measures |
|---|---|
| **Faithfulness** | Does the answer only contain claims supported by the retrieved context? |
| **Answer Relevancy** | Does the answer actually address the question? |
| **Context Utilization** | Does the answer make good use of the retrieved context? |

The evaluation also implements a custom `_GeminiChat` wrapper — a minimal LangChain-compatible chat model using the project's own `google.genai` SDK, avoiding version conflicts with `langchain-google-genai`. Questions for evaluation are stored in `evaluation/questions.csv`.

---

## 9. What's Novel in This Project

Most RAG tutorials implement a basic vector search → LLM answer loop. ACSH-RAG stacks **seven innovations** on top of that baseline:

### 🔴 Novel #1: True 3-Signal Hybrid Retrieval

Most systems use vector + BM25 (2-way hybrid). Adding a **graph-based NER signal** as the third channel is uncommon in practice. The entity-overlap scoring with **IDF weighting** means rare, domain-specific terms (e.g., a specific company name or legal clause) are valued much more than generic phrases — exactly the right behaviour for legal/medical document search.

### 🔴 Novel #2: HyDE with Intentional Signal Splitting

HyDE is known in research but rarely implemented correctly. The key insight here — also novel in most implementations — is that **HyDE should only feed the vector search**, not BM25. Passing a hypothetical paragraph to BM25 would introduce false keywords and degrade keyword precision. The `vector_query` parameter in `retrieve_chunks()` was designed specifically for this split.

### 🔴 Novel #3: LangGraph State Machine Pipeline

Instead of a linear function chain, the pipeline is a **compiled LangGraph directed graph with conditional edges**. This enables:
- True branching based on runtime decisions (router → three different paths)
- Clean retry loops without recursion (CRAG corrective loop via `reformulate → retrieve` back-edge)
- Easy streaming of node-level progress to the UI

### 🔴 Novel #4: Adaptive Query Routing

The router's three-tier classification (`direct`, `simple`, `complex`) avoids doing expensive retrieval on questions that don't need it ("What is 2+2?") and applies query decomposition only when genuinely needed. This is a practical engineering decision rarely seen in demo-quality RAG systems.

### 🔴 Novel #5: CRAG — Corrective Retrieval Loop

When the cross-encoder reranker's top score is below the pass threshold, the system doesn't give up — it **reformulates the query** using Gemini and retries retrieval, up to 2 times. This CRAG loop means the system is resilient to poor initial query phrasing.

### 🔴 Novel #6: Self-RAG Critic for Hallucination Prevention

A **second LLM call** independently checks every sentence in the generated answer against the context. Unsupported sentences are **surgically removed** from the partial verdict. This creates a two-stage hallucination barrier: the citation enforcer prevents the model from making uncited claims, and the critic removes any that slip through.

### 🔴 Novel #7: Graph Sub-Phrase Indexing + N-Gram Fallback

The graph store indexes not only full noun phrases but all **sub-spans 2–3 words long**. At query time, if spaCy's parser fails to produce useful noun chunks, raw **n-gram windows** from the query tokens are used as a fallback — ensuring graph retrieval is robust to parser failures on short or unusual queries.

---

## 10. Data Flow: End-to-End Example

**Query:** `"What are the differences between the termination and liability clauses?"`

```
1. Router → Gemini: "complex" (multi-part comparison)

2. Decomposer → Gemini:
   ["What is the termination clause?",
    "What is the liability clause?",
    "How do the two clauses differ?"]

3. For each sub-question:
   - HyDE generates a hypothetical passage
   - retrieve_chunks() runs:
     - Vector: cosine similarity on HyDE paragraph
     - BM25: keyword match on sub-question text
     - Graph: entity/concept overlap on sub-question text
   - RRF merges 3×20 = 60 candidates into ranked list
   - Top 5 returned per sub-question

4. Merge & deduplicate all sub-question chunks by chunk_id
   (e.g. 12 unique chunks total)

5. Cross-encoder reranks 12 chunks → top 5

6. ConfidenceGate: top score = 3.4 → PASS

7. CitationEnforcer:
   Context shown as [C3], [C7], [C12], [C21], [C44]
   Gemini generates: "The termination clause [C3] specifies..."
   cited_ids = ["contract_pdf_chunk_00003", ...]

8. Self-RAG Critic:
   Checks each sentence against context
   verdict = "pass" → no removals
   final_answer = draft_answer
   confidence = "pass"

9. Response: {answer, citations, route="complex", confidence="pass"}
```

---

## 11. File Map

```
ACSH-RAG/
├── ingest_documents.py         Entry point: build all 3 indexes from Documents/
├── retrieve_api.py             Public retrieval API (3-way hybrid + RRF)
├── pipeline_api.py             Public pipeline API (full router→critic chain)
├── query.py                    Interactive CLI query tool
├── compare_retrievers.py       Debug: compare vector+BM25 vs graph vs hybrid
├── graph_retrieve_api.py       Standalone graph retrieval API
├── web_app.py                  FastAPI server + chat UI
│
├── src/
│   ├── ingestion.py            Document loader, tiktoken chunker, page mapping
│   ├── vector_store.py         ChromaDB + all-MiniLM-L6-v2 embedding
│   ├── bm25_retriever.py       BM25Okapi keyword index
│   ├── graph_store.py          NetworkX + spaCy NER knowledge graph
│   ├── hybrid_retriever.py     3-way RRF fusion + HybridRetriever class
│   └── upload.py               Per-session upload ingestion + session management
│
├── pipeline/
│   ├── graph.py                LangGraph state machine (all nodes + edges)
│   ├── router.py               AdaptiveRouter (Gemini → direct/simple/complex)
│   ├── hyde.py                 HyDEGenerator + QueryDecomposer
│   ├── reranker.py             Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
│   ├── confidence_gate.py      ConfidenceGate + CRAG reformulation loop
│   └── citation_enforcer.py    CitationEnforcer + Self-RAG critic
│
├── prompts/
│   └── prompts.yaml            All system prompts (router, HyDE, decomposer, etc.)
│
├── evaluation/
│   ├── eval.py                 RAGAS evaluation harness
│   ├── questions.csv           Test questions
│   └── ragas_config.yaml       Evaluation configuration
│
├── Documents/                  ← Put your PDFs/Markdown here
├── chroma_db/                  ← Auto-generated vector index
├── bm25_index/                 ← Auto-generated BM25 index
├── graph_db/                   ← Auto-generated graph database
└── sessions/                   ← Per-user upload sessions (web app)
```

---

## 12. Setup & Running

```bash
# 1. Clone and create virtual environment
python -m venv venv
.\venv\Scripts\activate          # Windows
# source venv/bin/activate       # Unix/Mac

# 2. Install dependencies
pip install -r requirements.txt -r requirements_b.txt
python -m spacy download en_core_web_sm

# 3. Configure API key
# Create .env in project root:
echo GEMINI_API_KEY=your_key_here > .env

# 4. Add your documents to Documents/ and ingest
python ingest_documents.py
# Add --force to rebuild existing indexes

# 5. Test retrieval only (no LLM)
python query.py "your question"

# 6. Run the full pipeline (router → critic chain)
python pipeline_api.py

# 7. Start the web app
pip install -r requirements_web.txt
python web_app.py
# Open http://localhost:8000

# 8. Run RAGAS evaluation
pip install -r requirements_c.txt
python evaluation/eval.py
```

### Key Dependency Groups

| File | What it installs |
|---|---|
| `requirements.txt` | Core: pypdf, sentence-transformers, chromadb, rank-bm25, tiktoken, networkx, spaCy |
| `requirements_b.txt` | Pipeline: langgraph, google-genai, fastapi, pydantic |
| `requirements_web.txt` | Web: uvicorn, python-multipart |
| `requirements_c.txt` | Evaluation: ragas, datasets, langchain-core |

---

> **Summary:** ACSH-RAG is a full-stack RAG system that goes from raw documents to verified, cited answers through a 7-stage adaptive pipeline. Its core novelty is the combination of 3-way hybrid retrieval (vector + BM25 + graph), HyDE with intentional signal splitting, and a Self-RAG critic — packaged in a clean, modular codebase with a streaming web UI and RAGAS evaluation.
