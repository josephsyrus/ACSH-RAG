# Graph Report - .  (2026-06-20)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 164 nodes · 264 edges · 13 communities
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 16 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]

## God Nodes (most connected - your core abstractions)
1. `PipelineState` - 23 edges
2. `GraphStore` - 16 edges
3. `BM25Retriever` - 11 edges
4. `HybridRetriever` - 11 edges
5. `CitationEnforcer` - 10 edges
6. `retrieve_chunks()` - 10 edges
7. `VectorStore` - 10 edges
8. `StateGraph` - 9 edges
9. `ConfidenceGate` - 8 edges
10. `HyDEGenerator` - 7 edges

## Surprising Connections (you probably didn't know these)
- `HybridRetriever` --uses--> `HybridRetriever`  [INFERRED]
  retrieve_api.py → src/hybrid_retriever.py
- `main()` --calls--> `GraphStore`  [EXTRACTED]
  ingest_documents.py → src/graph_store.py
- `main()` --calls--> `VectorStore`  [EXTRACTED]
  ingest_documents.py → src/vector_store.py
- `node_retrieve()` --calls--> `retrieve_chunks()`  [EXTRACTED]
  pipeline/graph.py → retrieve_api.py
- `node_retrieve_multi()` --calls--> `retrieve_chunks()`  [EXTRACTED]
  pipeline/graph.py → retrieve_api.py

## Import Cycles
- None detected.

## Communities (13 total, 0 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.10
Nodes (30): node_citation_generate(), node_confidence_gate(), node_decompose(), node_direct_answer(), node_hyde_generate(), node_reformulate(), node_refuse(), node_rerank() (+22 more)

### Community 1 - "Community 1"
Cohesion: 0.16
Nodes (12): Any, main(), BM25Retriever, tokenize(), chunk_documents(), chunk_text(), load_documents(), load_markdown() (+4 more)

### Community 2 - "Community 2"
Cohesion: 0.20
Nodes (15): compare(), main(), print_comparison(), compare_retrievers.py  Side-by-side comparison of three retrieval modes:   1., Run the same query through all three retrieval modes.      Returns:         d, Pretty-print the comparison results., graph_retrieve_api.py  Graph-only retrieval interface. Sets vector_weight=0 a, Graph-only retrieval.      Passes graph_weight=1.0, vector_weight=0.0, bm25_we (+7 more)

### Community 3 - "Community 3"
Cohesion: 0.16
Nodes (9): GraphStore, Graph database over document chunks using NetworkX.      Graph schema:, Persist the graph to graph_db/graph.json., Load the graph from disk.         Returns True on success, False if the file do, Find chunks relevant to query via named entity + noun phrase matching., Remove PDF section-number noise where digits are glued to words.         e.g. ', Return all consecutive sub-spans shorter than the full phrase,         with len, Return list of (normalised_text, label) for named entities in text. (+1 more)

### Community 4 - "Community 4"
Cohesion: 0.18
Nodes (6): main(), print_results(), HybridRetriever, 3-way Reciprocal Rank Fusion over vector, BM25, and graph results.      Args:, reciprocal_rank_fusion(), VectorStore

### Community 5 - "Community 5"
Cohesion: 0.21
Nodes (8): _build_context_string(), CitationEnforcer, _load_prompts(), pipeline/citation_enforcer.py  Citation Enforcement + Self-RAG Critic.  Upda, Direct answer without retrieval., Wrapper for all Gemini calls with 429 retry., Returns (context_string, id_map) where id_map maps short→real chunk_id., Generates grounded answers with citations and validates     them using a Self-R

### Community 6 - "Community 6"
Cohesion: 0.18
Nodes (8): HyDEGenerator, _load_prompt(), QueryDecomposer, pipeline/hyde.py  HyDE — Hypothetical Document Embeddings.  Uses the NEW Goo, Breaks a complex multi-part query into focused sub-questions., Decompose a complex query into sub-questions.          Args:             quer, Generates a hypothetical answer paragraph for a given query.     This paragraph, Generate a hypothetical document passage.          Args:             query: R

### Community 7 - "Community 7"
Cohesion: 0.21
Nodes (8): pipeline_api.py  Clean public interface for the full ACSH-RAG pipeline.  Usa, Run the full ACSH-RAG pipeline for a query.      Args:         query (str): T, run_pipeline(), build_pipeline(), get_pipeline(), AdaptiveRouter, _load_prompt(), StateGraph

### Community 8 - "Community 8"
Cohesion: 0.22
Nodes (6): ConfidenceGate, _load_prompt(), pipeline/confidence_gate.py  Confidence Quality Gate + Corrective Loop (CRAG)., Reformulate the query to improve retrieval.          Args:             origin, Quality gate for retrieved chunks.      Decides whether to:     - pass     -, Evaluate retrieval quality using reranker score.          Args:             c

### Community 9 - "Community 9"
Cohesion: 0.25
Nodes (6): get_reranker(), LocalReranker, pipeline/reranker.py  Cross-encoder re-ranker: scores (query, chunk) pairs joi, Uses cross-encoder/ms-marco-MiniLM-L-6-v2 from sentence-transformers.     Fully, Re-score and re-sort chunks against the query.          Args:             que, Factory function. Returns a LocalReranker.     Extend this later if you want to

## Knowledge Gaps
- **1 isolated node(s):** `Any`
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GraphStore` connect `Community 3` to `Community 1`, `Community 4`?**
  _High betweenness centrality (0.219) - this node is a cross-community bridge._
- **Why does `HybridRetriever` connect `Community 4` to `Community 1`, `Community 2`, `Community 3`?**
  _High betweenness centrality (0.213) - this node is a cross-community bridge._
- **Why does `retrieve_chunks()` connect `Community 2` to `Community 0`?**
  _High betweenness centrality (0.096) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `PipelineState` (e.g. with `CitationEnforcer` and `ConfidenceGate`) actually correct?**
  _`PipelineState` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `HybridRetriever` (e.g. with `HybridRetriever` and `BM25Retriever`) actually correct?**
  _`HybridRetriever` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `CitationEnforcer` (e.g. with `PipelineState` and `StateGraph`) actually correct?**
  _`CitationEnforcer` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `compare_retrievers.py  Side-by-side comparison of three retrieval modes:   1.`, `Run the same query through all three retrieval modes.      Returns:         d`, `Pretty-print the comparison results.` to the rest of the system?**
  _53 weakly-connected nodes found - possible documentation gaps or missing edges._