# ACSH-RAG — Full Project Explanation

## What Problem Does This Solve?

When you have a large collection of documents (contracts, research papers, manuals, scanned records) and want to ask questions about them, a plain LLM cannot help — it hasn't seen your documents, and hallucinating an answer is dangerous. A plain keyword search also fails when your question uses different words than the document does.

ACSH-RAG solves this by combining **three different ways to search your documents** and then using an LLM only to *read and summarise the retrieved text* — never to invent facts. Every claim in the final answer is traceable back to a specific page of your documents.

---

## What Changed in This Version

If you read the previous version of this document, these are the architectural changes:

| Area | Before | Now |
|------|--------|-----|
| **Routing** | Three routes: `direct` / `simple` / `complex`. `direct` answered from the LLM's own knowledge, skipping retrieval entirely. | Two routes: `simple` / `complex`. **The `direct` route is gone** — every query is grounded in retrieved documents. No ungrounded answers can escape the pipeline. |
| **Graph retrieval** | An inverted index in disguise: exact-string entity/noun-phrase nodes, 1-hop lookup, no normalisation. Redundant with BM25, so it contributed little to fusion. | A real knowledge graph: canonical lemma-normalised entities, typed relations from the dependency parse, and **multi-hop personalised PageRank** traversal. Surfaces chunks containing none of the query's words. |
| **Graph fusion** | Fixed 0.3 weight, always applied — even on queries with no recognisable entities. | **Confidence-scaled**: the graph arm's weight scales with how much of the query it actually resolved, and drops to zero below a floor. |
| **Scanned documents** | `pypdf` returned empty text; the whole file was skipped with a warning. Mixed documents lost their scanned pages *silently*. | **Local OCR** (RapidOCR + PyMuPDF, fully offline). Per-page detection, so mixed born-digital/scanned PDFs work. Standalone image files are ingestible too. |
| **BM25 tokenizer** | `text.lower().split()` | Adds character 4-grams for long words, so a damaged PDF text layer emitting `Prayertherapy` can still match a query for `prayer therapy`. |
| **Deployment** | CLI only. | FastAPI web app with a chat UI, per-session document upload with isolated indexes, and SSE token streaming. |
| **Evaluation** | Golden dataset required. | Reference-free RAGAS metrics + a GitHub Actions CI gate. |

---

## The Two Phases: Setup vs. Query Time

### Phase 1 — Document Ingestion (run once)

Before any query can be answered, the system reads and indexes all your documents. This is done by running `ingest_documents.py`.

```
PDF / Markdown / text / image files
       │
       ▼
  [1] Load & extract text
       │   PDFs: pypdf text layer, per page
       │   Pages with <120 chars → rasterise at 300 dpi → local OCR
       │   Images (.png/.jpg/.tiff/…) → local OCR
       ▼
  [2] Chunk into 250-token pieces (50-token overlap)
       │   Each chunk records its start page + whether that page was OCR'd
       │
       ├──▶ [3a] Embed chunks → store in ChromaDB (vector index)
       ├──▶ [3b] Build BM25 keyword index (saved to disk)
       └──▶ [3c] Extract entities + relations → build NetworkX knowledge graph
```

**Why chunks?** An LLM context window is limited. Splitting documents into small, overlapping pieces lets the system surface only the most relevant few hundred tokens for a given question, rather than feeding thousands of pages to the LLM.

**Why overlap?** A 50-token overlap between adjacent chunks ensures that a sentence crossing a chunk boundary is not silently lost.

**Why 250 tokens?** This must stay ≤ 256 to match `all-MiniLM-L6-v2`'s hard sequence limit. Exceeding it causes silent truncation — the tail of every chunk would simply not be embedded.

---

## OCR: Reading Scanned Documents

Previously, a scanned PDF produced no text at all and was dropped. Worse, a PDF that *mixed* born-digital pages with a scanned appendix passed the whole-document emptiness check and was ingested with the scanned pages missing and **no warning whatsoever**.

### How it works now

```
For each page of a PDF:
    text = pypdf.extract_text()
    │
    ├── len(text) ≥ 120 chars  ──▶  use it as-is (born-digital page)
    │
    └── len(text) <  120 chars  ──▶  page is assumed to be a scan
                                     │
                                     ▼
                          PyMuPDF rasterises the page at 300 dpi
                                     │
                                     ▼
                          RapidOCR (PP-OCRv4 on onnxruntime)
                                     │
                          ┌──────────┴──────────┐
                          │ text found?         │
                          │  yes → keep it      │
                          │  no  → retry with   │
                          │        contrast     │
                          │        normalisation│
                          │        + adaptive   │
                          │        thresholding │
                          └─────────────────────┘
                                     │
                                     ▼
                          Lines below 0.5 confidence dropped
                                     │
                                     ▼
                          Cached to ocr_cache/ by (file hash, page, dpi)
```

**Key design decisions:**

- **Detection is per page, not per file.** This is what fixes mixed documents.
- **Fully offline.** RapidOCR ships its models inside the pip wheel. Unlike Tesseract there is no system binary to install, and unlike a hosted vision model there is no API call and nothing leaves the machine. This was a hard requirement — OCR must never depend on the Gemini quota.
- **Two-pass OCR.** Low-contrast photographs and faded pages that produce nothing on the first pass get a second attempt after greyscale normalisation and adaptive Gaussian thresholding.
- **Disk cache keyed by content hash.** Re-ingesting a 300-page scan never re-pays the OCR cost. The key includes a cache version string, so improving the extraction logic invalidates stale empty results rather than serving them forever.
- **Provenance is tracked.** Every chunk carries an `ocr` boolean. Downstream, the graph discounts evidence from OCR'd chunks by 20%, because garbled OCR text produces noisy entities.

Failures degrade rather than crash: if the OCR dependencies are missing, `available()` returns `False`, a note is printed, and ingestion continues with text-only extraction.

---

## The Knowledge Graph

This is the largest change in the codebase, so it gets its own section.

### Why the old graph underperformed

The previous graph created a node per named entity and per multi-word noun phrase, linked each to the chunk it appeared in, and retrieved by exact string lookup. Five problems compounded:

1. **It wasn't really a graph.** Retrieval was strictly 1-hop: query signal → chunks containing it. That is what BM25 already does, but over a *subset* of tokens and with worse scoring. Two of the three retrieval arms were measuring the same thing, and the graph was the weaker copy. RRF had nothing to gain from it.
2. **Single-word concepts had no node.** A `>= 2 words` filter meant `oven` and `voltage` were unindexed, while spaCy's determiner-inclusive noun chunks indexed `the oven` instead.
3. **No normalisation.** The lemmatizer was disabled, so `oven` ≠ `ovens` ≠ `the Ovens`, and lookup was exact string match — one character of drift scored zero.
4. **Sub-phrase explosion.** `combination cooking feature` also indexed `combination cooking` and `cooking feature`, all pointing at the same chunk. One concept, three score contributions — and every phrase's document frequency was inflated, which deflated the IDF of genuinely rare terms. This is why the old graph had 17,614 concept nodes for only 784 chunks.
5. **The score was structurally tiny.** `graph_score` was `matched_idf / total_query_idf`, and the denominator included every 2- and 3-gram generated from the query as a fallback. More fallback n-grams meant a bigger denominator, so the score could never be high regardless of match quality.

### The new schema

```
Entity nodes  ──MENTIONS(count=tf)──▶  Chunk nodes
     │
     └──RELATED(labels=[...], weight)──▶  Entity nodes

Chunk nodes  ──NEXT──▶  Chunk nodes   (sequential, within one document)
```

- **Entity nodes** are keyed by a canonical string: lemmatised, lowercased, determiners and punctuation stripped. `the Ovens`, `ovens` and `oven` all collapse to one node. Each carries `df` (how many chunks mention it) and a proper `idf = log(1 + N/df)` — the old `1/log(1+degree)` separated a 1-chunk term from a 100-chunk term by only ~6×, far too flat for distinctive entities to outrank generic phrases.
- **Both the full noun phrase and its bare head noun are indexed**, so single-word concepts finally exist.
- **`RELATED` edges come from the dependency parse** — no LLM, entirely offline:
  - Subject–verb–object triples, labelled by the verb's lemma.
  - Copular `is a` relations (taxonomy).
  - Prepositional objects, labelled `verb_preposition`.
  - Passive constructions reversed (`X was built by Y` → `Y built X`).
  - Sentence-level co-occurrence, at reduced weight (0.3). Restricted to one sentence rather than the whole chunk — this keeps the edge count manageable and the association far more meaningful.

### Retrieval — how a query becomes chunks

```
Query
  │
  ▼
[1] Extract signals with spaCy
      primary  : noun phrases, their head nouns, named entities  (normalised)
      fallback : 1-, 2- and 3-grams over content tokens
  │
  ▼
[2] Resolve signals → seed entities
      exact node match                          → similarity 1.0
      else nearest neighbour over entity
           embeddings (MiniLM, cosine ≥ 0.80)   → similarity weighted, squared
      seed weight = idf × similarity²
  │
  ▼
[3] Traverse — personalised PageRank
      Build a bounded 2-hop neighbourhood around the seeds (≤ 8000 nodes),
      run PageRank with alpha=0.75 personalised on the seed weights.
      Relation edges also conduct backwards at half weight.
  │
  ▼
[4] Collect chunk evidence
      For the 300 heaviest entities:
        score[chunk] += pagerank_mass × idf × (1 + log(term_frequency))
      Then divide by sqrt(distinct entities in chunk)   ← length normalisation
      Then × 0.8 if the chunk came from OCR             ← noise discount
  │
  ▼
[5] Normalise to [0,1] by the top score, return top_k
```

**Step 3 is the whole point.** Because relevance mass travels two hops across relation edges, the graph surfaces chunks that contain *none* of the query's own terms but are relationally connected to them. Neither vector search nor BM25 can do that. This is what makes the graph arm genuinely orthogonal instead of a weaker duplicate of BM25.

**Step 2 is what fixed recall.** Embedding-based resolution turns the old all-or-nothing string lookup into graded matching.

### Confidence

`search_with_confidence()` returns a confidence score alongside the results:

```
confidence = (primary signals resolved / primary signals total) × mean similarity
```

Crucially, **only primary signals count towards the denominator**. The fallback n-grams still add recall, but folding them into the denominator is exactly what made the old score structurally tiny.

The retriever uses this to scale the graph's contribution to fusion:

| Confidence | Graph weight applied |
|-----------|----------------------|
| < 0.15 | **0** — the graph arm is dropped entirely for this query |
| ≥ 0.15 | `graph_weight × max(0.35, confidence)` |

So a query with no recognisable entities no longer injects 20 junk results at full weight, and a partial-but-real match still counts for something. The gate is skipped in graph-only mode (`graph_retrieve_api.py`) so the diagnostic tool always shows you what the graph found.

### Preserved fallbacks

The rewrite deliberately keeps every safety net the original had:

- **PDF digit-glue cleaning** — `13Combination Cooking` → `Combination Cooking`, now applied on the query side too (the original applied it only at build time).
- **Raw n-gram fallback** — spaCy sometimes produces no useful noun chunk for short imperative queries (`tell me about combination cooking` yields only `me`).
- **Graceful load failure** — a missing, corrupt, or schema-stale `graph_db/` prints a warning and disables graph retrieval rather than crashing. The schema version is checked explicitly, so an index built by the old code is refused with a message telling you to re-ingest.
- **PageRank non-convergence** — falls back to the seed weights alone, which still gives usable 1-hop results.

---

### Phase 2 — Query Pipeline (runs on every user question)

This is the main flow. It is built as a **LangGraph state machine**, meaning the query travels through a series of nodes and the path it takes depends on decisions made along the way.

```
┌─────────────────────────────────────────────────────────────┐
│                        ROUTER                               │
│  Classifies the query into one of two routes:               │
│   • simple  — single-focus question about your documents    │
│   • complex — multi-part question needing several lookups   │
└─────────────────────────────────────────────────────────────┘
```

There is no longer a `direct` route. Previously the router could decide a question was general knowledge and let Gemini answer from its own weights, bypassing retrieval and citations completely. That path is removed: **every answer is now grounded in retrieved document text**, or refused.

The router asks Gemini for JSON (`{"route": "simple"}`), retries up to 3 times on an empty response, falls back to scanning the raw text for a valid route word, and finally defaults to `simple`. Note that all Gemini calls in the pipeline set `thinking_budget=0` — `gemini-3.5-flash` is a thinking model, and without this the reasoning consumes the output token budget and the actual response comes back empty or truncated.

---

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
Each sub-question gets **its own HyDE paragraph** and its own full retrieval pass; the resulting chunks are merged and deduplicated by `chunk_id` before continuing. If the JSON fails to parse, it falls back to treating the original query as a single question.

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

This hypothetical paragraph **looks like document text**, so its embedding lands much closer in vector space to the actual chunks that contain the real answer.

**Key design point — split query usage:**
- The HyDE paragraph is used **only for vector (ChromaDB) search** — it is long and rich, perfect for semantic matching.
- The original short query is used **for BM25 and graph search** — both work on term and entity overlap, so feeding them a paragraph full of invented words would hurt precision.

If the HyDE call fails or returns empty, the original query is used for vector search too.

---

### Step 3 — Hybrid Retrieval (3-Way Fusion)

Three retrievers run over the corpus, each fetching 20 candidates:

```
HyDE paragraph ──▶  ChromaDB vector search   ──▶  top-20 (cosine similarity)
Original query ──▶  BM25 keyword search      ──▶  top-20 (BM25 score)
Original query ──▶  Graph traversal          ──▶  top-20 (+ a confidence score)
```

The three ranked lists are merged using **Reciprocal Rank Fusion (RRF)**:

```
RRF score for a chunk = Σ  weight / (k + rank_in_list)
                        over each list the chunk appears in
```

- `k = 60` (smoothing constant — dampens the advantage of rank 1 vs rank 2)
- Default weights: vector `0.4`, BM25 `0.3`, graph `0.3`
- The graph weight is then **scaled by the graph's own confidence**, as described above.
- A chunk appearing in multiple lists gets contributions from each, so it naturally floats to the top.
- Final output: a single merged, deduplicated list; the top 5 go forward.

**Why three methods?** Each has blind spots:
- Vector search finds *semantically similar* text but can miss exact terms.
- BM25 finds exact keywords but misses paraphrased or synonymous text.
- Graph traversal finds text connected by shared entities and relations even when the phrasing overlaps not at all.
- Together, their weaknesses cancel out.

**BM25 tokenizer note:** alongside normal word tokens, the tokenizer emits character 4-grams for words of 6+ characters. A damaged PDF text layer may collapse `prayer therapy` into `Prayertherapy`; the character grams give a narrow fallback match while ordinary tokens remain the primary signal.

---

### Step 4 — Cross-Encoder Reranking

The merged chunks are good candidates, but RRF scores are indirect — they come from three separate ranking systems. A **cross-encoder** now reads each `(query, chunk)` pair *jointly* and outputs a single relevance score.

**Why is this better than the retrieval scores?**
The embedding models used in vector search are *bi-encoders*: they encode the query and each chunk independently and then compare. This is fast but imprecise — nuance is lost.

A cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~250MB, runs locally, no API) sees the full concatenated `[query + chunk text]` at once and scores true relevance. Too slow for thousands of chunks, plenty fast for 20. Scores run roughly −10 (irrelevant) to +10 (very relevant).

The reranker is given the **original query, not the HyDE paragraph** — you want relevance judged against the real question.

Output: chunks re-sorted by rerank score, top 5 passed forward.

---

### Step 5 — Confidence Gate

Before generating an answer, the system checks: *are these chunks actually relevant, or did the retriever just return the least-bad garbage?*

The **Confidence Gate** looks at the top chunk's rerank score:

| Score | Decision |
|-------|----------|
| ≥ 0.0 | **PASS** — chunks are good, proceed to answer generation |
| ≥ −3.0 | **RETRY** — chunks are weak, reformulate the query and try again |
| < −3.0 | **REFUSE** — nothing relevant found, give up |

No chunks at all → REFUSE immediately.

**Corrective Loop (CRAG):**
If the decision is RETRY, the system asks Gemini to rephrase the original query differently, regenerates a HyDE paragraph for the new query, and re-runs retrieval and reranking. This loop can happen up to **2 times** before refusing.

The reformulation is sanitised — only the first non-empty line is kept, and quotes/fences are stripped. If the result is degenerate (fewer than 3 words), it falls back to the original query, since a one-word query retrieves garbage and causes a false refusal.

---

### Step 6 — Citation-Grounded Answer Generation

If chunks passed the gate, Gemini is asked to write an answer using **only the provided chunks**. Each chunk is labelled with a short ID derived from its chunk index (e.g. `chunk_00137` → `[C137]`). The prompt explicitly instructs:
- Answer only from the context. Do not use outside knowledge.
- Cite the exact bracket label after each claim — copy it verbatim, never invent one.
- If the context is insufficient, respond with exactly `INSUFFICIENT_CONTEXT`.

Example output:
> *"The penalty for late payment is 2% per month on the outstanding balance [C42]. This surcharge compounds monthly until full settlement is received [C43]."*

The system then extracts all `[Cxx]` references with a strict regex (so a bracketed phrase like `[NEW HABIT]` is not mistaken for a citation) and maps them back to real chunk IDs.

If `INSUFFICIENT_CONTEXT` comes back, the pipeline skips the critic and routes straight to refusal.

**Streaming:** when the pipeline is driven by the web app, this node streams tokens as they are generated via an `emit` callback placed in the pipeline state, so the UI shows the answer being written live.

---

### Step 7 — Self-RAG Critic

A second Gemini call acts as a **fact-checker**. It receives the draft answer and the original chunks side-by-side and checks each sentence:

- Is this sentence actually supported by the provided chunks?
- Or did the LLM hallucinate or extrapolate beyond what the chunks say?

**Verdict options:**
- **pass** — every sentence is supported. Final answer = draft answer.
- **partial** — some sentences are unsupported. The critic identifies them; they are deleted from the answer. Final answer = pruned answer.
- **fail** — the answer is fundamentally not grounded. The answer is withheld entirely.

If the critic's JSON fails to parse, the fallback is conservative: the answer is accepted as-is with a note, rather than being discarded on a formatting error.

---

### Final Output

The pipeline returns:
```python
{
    "answer":     "...",             # the verified, cited answer
    "citations":  [                  # resolved to human-readable sources
        {"chunk_id": "manual_pdf_chunk_00042", "filename": "manual.pdf", "page": 17}
    ],
    "route":      "simple",          # simple | complex
    "confidence": "pass",            # pass | low_confidence | refused
}
```

Citations carry **page numbers**, not just chunk IDs — each chunk records the page its first token came from, computed by mapping chunk token offsets back onto per-page token offsets during ingestion.

---

## Complete Pipeline Flow (visual summary)

```
User Query
    │
    ▼
[Router]  ── simple / complex only; no ungrounded "direct" path
    │
    ├── simple ──▶ [HyDE Generate]
    │                    │
    └── complex ──▶ [Decompose] ──▶ (per sub-question: HyDE + full retrieval)
                         │
                         ▼
                   [Retrieve]
                   ├── ChromaDB vector search  (HyDE paragraph)
                   ├── BM25 keyword search     (original query)
                   └── Graph traversal         (original query + confidence)
                         │ Merge via RRF, graph arm scaled by its confidence
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
                                                              (streams tokens to the UI)
                                                                        │
                                                          INSUFFICIENT_CONTEXT ──▶ [Refuse]
                                                                        │
                                                                        ▼
                                                              [Self-RAG Critic]
                                                              Checks every sentence
                                                              against source chunks
                                                                        │
                                                                        ▼
                                                                      END
```

---

## The Web Application

`web_app.py` serves a chat UI at `http://localhost:8000`.

| Endpoint | Purpose |
|----------|---------|
| `GET /api/health` | Liveness check, no LLM calls |
| `POST /api/upload` | Upload one document → isolated per-session index |
| `POST /api/ask` | Blocking: question in, full answer out |
| `POST /api/ask_stream` | Server-Sent Events: live pipeline stages + token streaming |

### Per-session uploads

Uploading a document builds a **completely isolated set of three indexes** under `sessions/<id>/`, so one user's upload never leaks into another user's answers. This ingestion is **fully local** — embeddings, BM25 and the spaCy graph only, no Gemini calls, so it never consumes API quota. OCR runs here too, so a scanned upload works.

Session hygiene:
- Sessions older than 24 hours are reaped, and the total is capped at 200 (oldest evicted first). Cleanup runs at startup and before each new upload.
- `retrieve_api.py` keeps an LRU cache of at most 12 retrievers (each holds a BM25 index and a graph in memory), so many uploads cannot exhaust memory. The global corpus retriever is never evicted.
- The embedding model is cached at module level and shared by every `VectorStore` instance, rather than each loading its own ~90MB copy.
- `session_id` is passed through `os.path.basename()` before use, guarding against path traversal.

### Streaming

`/api/ask_stream` runs the pipeline in a worker thread that pushes events onto a queue; the request thread drains it and serialises SSE frames:

- `{type: "stage", label: "..."}` — one per LangGraph node as it starts, giving the UI a live progress readout ("Searching the documents…", "Verifying each claim against the sources…").
- `{type: "token", text: "..."}` — answer tokens as Gemini produces them.
- `{type: "done", answer, route, confidence, citations}` — final payload.
- `{type: "error", detail}` — anything unexpected.

The browser renders streamed text live, then replaces it with rendered Markdown plus **clickable citation superscripts** that link each `[C21]` marker to a source card showing filename and page.

### Quota handling

Free-tier Gemini quota exhaustion is an expected operational state, not a bug. Both ask endpoints catch `429` / `RESOURCE_EXHAUSTED` and return a readable message instead of a 500.

---

## Evaluation

`evaluation/eval.py` runs **RAGAS** with reference-free metrics only — no golden dataset needed, so it works on any corpus or uploaded document:

| Metric | Threshold | What it catches |
|--------|-----------|-----------------|
| `faithfulness` | 0.70 | Hallucination — claims not supported by retrieved chunks |
| `answer_relevancy` | 0.70 | Answers that don't address the question asked |
| `context_utilization` | 0.65 | Retrieved chunks that weren't actually used |

`faithfulness` also acts as a **hard gate**: even if the average passes, faithfulness below 0.70 fails the run. `.github/workflows/eval.yml` runs this on every PR and push to `main` (20 samples in CI mode).

Thresholds and the judge model live in `evaluation/ragas_config.yaml`.

---

## File-by-File Reference

| File | What it does |
|------|-------------|
| `src/ingestion.py` | Loads PDFs/markdown/text/images, routes scanned pages to OCR, splits into 250-token overlapping chunks with page + OCR provenance |
| `src/ocr.py` | Local offline OCR — PyMuPDF rasterisation + RapidOCR, two-pass extraction, disk cache |
| `src/vector_store.py` | Wraps ChromaDB; shared embedding model; stores/retrieves chunks by cosine similarity |
| `src/bm25_retriever.py` | BM25 keyword index with character-gram fallback; persisted to disk |
| `src/graph_store.py` | Builds and queries the knowledge graph — canonical entities, dependency-parse relations, personalised-PageRank retrieval |
| `src/hybrid_retriever.py` | Runs all three retrievers and merges via RRF with confidence-scaled graph weight |
| `src/upload.py` | Per-session isolated ingestion for uploaded documents + session TTL cleanup |
| `pipeline/router.py` | Classifies query as simple / complex using Gemini |
| `pipeline/hyde.py` | HyDE paragraph generation + complex query decomposition |
| `pipeline/reranker.py` | Cross-encoder reranking (fully local, no API) |
| `pipeline/confidence_gate.py` | Score-based pass/retry/refuse gate + query reformulation |
| `pipeline/citation_enforcer.py` | Grounded answer generation (with token streaming) + Self-RAG critic |
| `pipeline/graph.py` | LangGraph state machine wiring all nodes together |
| `prompts/prompts.yaml` | All Gemini system prompts (router, HyDE, decomposer, answer, critic, reformulate) |
| `ingest_documents.py` | Entry point: build all three indexes from your documents |
| `pipeline_api.py` | Entry point: run the full pipeline on a query |
| `web_app.py` | FastAPI server — chat UI, uploads, SSE streaming |
| `static/index.html` | Chat front-end with live stages, streaming, and clickable citations |
| `query.py` | Interactive CLI for querying |
| `retrieve_api.py` | Standalone hybrid retrieval (without the LLM pipeline) |
| `graph_retrieve_api.py` | Graph-only retrieval, for isolating and debugging the graph arm |
| `compare_retrievers.py` | Side-by-side comparison: vector+BM25 vs graph-only vs hybrid |
| `evaluation/eval.py` | RAGAS reference-free evaluation harness |

---

## Configuration Reference

**Models**

| Role | Model | Local? |
|------|-------|--------|
| Embeddings | `all-MiniLM-L6-v2` (384-dim) | Yes |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Yes |
| NER + dependency parse | spaCy `en_core_web_sm` | Yes |
| OCR | RapidOCR (PP-OCRv4, onnxruntime) | Yes |
| Router / HyDE / decompose / answer / critic / reformulate | `gemini-3.5-flash` | No |
| RAGAS judge | `gemini-3.1-flash-lite` | No |

**Key constants**

| Setting | Value | Where |
|---------|-------|-------|
| Chunk size / overlap | 250 / 50 tokens | `ingest_documents.py` |
| Retrieval fetch_k / top_k | 20 / 5 | `pipeline/graph.py` |
| RRF k, weights | 60; 0.4 / 0.3 / 0.3 | `src/hybrid_retriever.py` |
| Graph confidence floor / min scale | 0.15 / 0.35 | `src/hybrid_retriever.py` |
| Fuzzy entity match threshold | cosine 0.80, top 3 | `src/graph_store.py` |
| PageRank alpha / hops / node cap | 0.75 / 2 / 8000 | `src/graph_store.py` |
| Gate pass / retry thresholds, max retries | 0.0 / −3.0 / 2 | `pipeline/confidence_gate.py` |
| OCR trigger / dpi / min line confidence | <120 chars / 300 / 0.5 | `src/ocr.py` |
| Session TTL / max sessions / retriever LRU | 24h / 200 / 12 | `src/upload.py`, `retrieve_api.py` |

**Environment**

`GEMINI_API_KEY` must be set in `.env`. It is required for the pipeline's LLM steps only — ingestion, OCR, embedding, BM25, the graph, and reranking are all fully local.

**Generated / git-ignored:** `chroma_db/`, `bm25_index/`, `graph_db/`, `ocr_cache/`, `sessions/`, `Documents/`, `venv/`.

---

## Commands

```bash
# Setup (Python 3.10 / 3.11)
python -m venv venv
.\venv\Scripts\activate                 # Windows (source venv/bin/activate on Unix)
pip install -r requirements.txt -r requirements_b.txt
python -m spacy download en_core_web_sm

# Ingest: build all three indexes from ./Documents  (add --force to rebuild)
python ingest_documents.py

# Retrieval only
python query.py "your question"                 # add --top_k N; no arg = interactive
python retrieve_api.py "your question"          # retrieval API self-test
python graph_retrieve_api.py "your question"    # graph arm in isolation
python compare_retrievers.py "your question"    # vector+BM25 vs graph-only vs hybrid

# Full pipeline
python pipeline_api.py                          # built-in self-test queries

# Web app
pip install -r requirements_web.txt
python web_app.py                               # → http://localhost:8000

# Evaluation (RAGAS)
pip install -r requirements_c.txt
python evaluation/eval.py
```

> **Note:** the graph schema version was bumped in this release. An index built by the previous code will be refused at load with a warning, and graph retrieval will stay disabled until you run `python ingest_documents.py --force`.
