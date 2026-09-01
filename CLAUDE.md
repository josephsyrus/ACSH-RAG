# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

# Project: ACSH-RAG

A hybrid Retrieval-Augmented Generation system over local documents (PDF / Markdown / text). Retrieval fuses three signals (vector + keyword + graph); an adaptive LangGraph pipeline adds query routing, HyDE, reranking, a confidence gate, citation enforcement, and a Self-RAG critic. LLM steps use Google Gemini.

## Architecture

**Ingestion** — `ingest_documents.py` → `src/ingestion.py`: load docs from `Documents/`, chunk (250 tokens, 50 overlap), then build three indexes:
- **Vector** — ChromaDB + `all-MiniLM-L6-v2` embeddings — `src/vector_store.py` → `chroma_db/`
- **BM25** — keyword index — `src/bm25_retriever.py` → `bm25_index/`
- **Graph** — NetworkX entity/concept graph via spaCy NER + noun phrases — `src/graph_store.py` → `graph_db/`

**Retrieval** — `retrieve_api.py::retrieve_chunks` → `src/hybrid_retriever.py`: runs all three retrievers and merges with 3-way Reciprocal Rank Fusion (default weights: vector 0.4 / bm25 0.3 / graph 0.3).

**Pipeline** — `pipeline_api.py::run_pipeline` → `pipeline/graph.py`: a LangGraph state machine.
- `router` classifies the query → `simple` | `complex` (`pipeline/router.py`)
- simple: `hyde → retrieve → rerank → confidence_gate → citation → self_rag_critic`
- complex: `decompose → retrieve_multi → …` (same tail from rerank)
- low confidence triggers a corrective reformulate-and-retry loop (CRAG), up to `MAX_RETRIES`
- components: `pipeline/hyde.py`, `pipeline/reranker.py` (cross-encoder `ms-marco-MiniLM-L-6-v2`), `pipeline/confidence_gate.py`, `pipeline/citation_enforcer.py`; prompts in `prompts/prompts.yaml`

## Commands

```bash
# Setup (Python 3.10 / 3.11)
python -m venv venv
.\venv\Scripts\activate                 # Windows  (use: source venv/bin/activate on Unix)
pip install -r requirements.txt -r requirements_b.txt
python -m spacy download en_core_web_sm

# Ingest: build all three indexes from ./Documents  (add --force to rebuild)
python ingest_documents.py

# Retrieval (hybrid retriever only)
python query.py "your question"                 # add --top_k N ; no arg = interactive
python retrieve_api.py "your question"          # retrieval API self-test
python compare_retrievers.py "your question"    # vector+BM25 vs graph-only vs hybrid

# Full pipeline (router → HyDE → retrieve → rerank → gate → cite → critic)
python pipeline_api.py                          # runs built-in self-test queries

# Web app (chat UI over the pipeline)
pip install -r requirements_web.txt
python web_app.py                               # → http://localhost:8000

# Evaluation (RAGAS — separate deps)
pip install -r requirements_c.txt
python evaluation/eval.py
```

## Configuration
- `GEMINI_API_KEY` must be set in `.env` (required for all pipeline LLM steps).
- Generated / git-ignored: `chroma_db/`, `bm25_index/`, `graph_db/`, `venv/`, `Documents/`, `graphify-out/cache/`.

## Codebase knowledge graph (graphify)
`graphify-out/graph.json` is a queryable graph of this codebase (functions, classes, call/import edges). Use it instead of re-reading files:
- `graphify query "how does retrieval merge results?"`
- `graphify explain HybridRetriever` · `graphify path "node_retrieve" "GraphStore"`
- Rebuild after refactors: `graphify update .` (offline, AST only) or `graphify extract .` (full; needs an LLM key).
