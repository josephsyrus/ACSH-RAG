"""
pipeline/graph.py

LangGraph orchestration for the full ACSH-RAG pipeline.

State flows through these nodes depending on the route:

DIRECT route:
  [start] → router → direct_answer → [end]

SIMPLE route:
  [start] → router → hyde_generate → retrieve → rerank
          → confidence_gate
              → (PASS)  citation_generate → self_rag_critic → [end]
              → (RETRY, up to 2x) reformulate → retrieve → rerank → ...
              → (REFUSE) refuse → [end]

COMPLEX route:
  [start] → router → decompose → retrieve_multi → rerank
          → confidence_gate → ... (same as simple from rerank onwards)
"""

import os
import sys
from typing import TypedDict, Literal, List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from retrieve_api import retrieve_chunks

from pipeline.router            import AdaptiveRouter
from pipeline.hyde              import HyDEGenerator, QueryDecomposer
from pipeline.reranker          import get_reranker
from pipeline.confidence_gate   import ConfidenceGate, MAX_RETRIES
from pipeline.citation_enforcer import CitationEnforcer

from dotenv import load_dotenv
load_dotenv()


# ─────────────────────────────────────────────
# 1. Pipeline State
# Every node reads from and writes to this dict.
# Only return the keys you actually changed in each node.
# ─────────────────────────────────────────────

class PipelineState(TypedDict):
    # Input
    original_query:  str

    # Router
    route:           str          # "direct" | "simple" | "complex"

    # Query manipulation
    active_query:    str          # current working query (may be reformulated)
    sub_questions:   List[str]    # for complex route
    hyde_text:       str          # HyDE paragraph

    # Retrieval
    raw_chunks:      List[Dict]   # from Person A's retrieve_chunks()
    reranked_chunks: List[Dict]   # after cross-encoder reranker

    # Confidence gate
    gate_decision:   str          # "pass" | "retry" | "refuse"
    retry_count:     int          # how many reformulation retries so far

    # Answer
    draft_answer:    str          # LLM's first answer attempt
    cited_chunk_ids: List[str]    # chunk IDs cited in the draft
    critic_result:   Dict         # full Self-RAG critic output
    final_answer:    str          # what gets returned to the user
    confidence:      str          # "pass" | "low_confidence" | "refused"


# ─────────────────────────────────────────────
# 2. Initialise components (loaded once at module level)
# ─────────────────────────────────────────────

_router    = AdaptiveRouter()
_hyde      = HyDEGenerator()
_decompose = QueryDecomposer()
_reranker  = get_reranker()
_gate      = ConfidenceGate()
_citation  = CitationEnforcer()


# ─────────────────────────────────────────────
# 3. Node Functions
# Each function: takes PipelineState, returns dict of only the keys it changed.
# ─────────────────────────────────────────────

def node_router(state: PipelineState) -> dict:
    """Classify query as direct / simple / complex."""
    query = state["original_query"]
    route = _router.classify(query)
    return {
        "route":        route,
        "active_query": query,
        "retry_count":  0,
    }


def node_direct_answer(state: PipelineState) -> dict:
    """Answer from Gemini's general knowledge — no retrieval."""
    answer = _citation.generate_direct_answer(state["original_query"])
    return {
        "final_answer":    answer,
        "confidence":      "pass",
        "cited_chunk_ids": [],
    }


def node_decompose(state: PipelineState) -> dict:
    """For complex queries: break into focused sub-questions."""
    sub_qs = _decompose.decompose(state["original_query"])
    return {"sub_questions": sub_qs}


def node_hyde_generate(state: PipelineState) -> dict:
    """Generate HyDE paragraph for the active query."""
    hyde_text = _hyde.generate(state["active_query"])
    return {"hyde_text": hyde_text}


def node_retrieve(state: PipelineState) -> dict:
    """
    Call Person A's retrieve_chunks().

    Key design decision:
      query        = active_query  → BM25 uses the short focused query
      vector_query = hyde_text     → vector search uses the HyDE paragraph

    This split is exactly what Person A's vector_query parameter was
    designed for. Using the HyDE paragraph for BM25 would degrade it.
    """
    chunks = retrieve_chunks(
        query=state["active_query"],
        top_k=5,
        fetch_k=20,
        vector_query=state.get("hyde_text") or None,
    )
    print(f"  [Retrieve] Got {len(chunks)} chunks.")
    return {"raw_chunks": chunks}


def node_retrieve_multi(state: PipelineState) -> dict:
    """
    For complex queries: retrieve chunks for EACH sub-question independently,
    then merge and deduplicate. Each sub-question gets its own HyDE paragraph.
    """
    all_chunks: Dict[str, Dict] = {}   # chunk_id → chunk (deduplication key)

    for sub_q in state["sub_questions"]:
        hyde_text  = _hyde.generate(sub_q)
        sub_chunks = retrieve_chunks(
            query=sub_q,
            top_k=5,
            fetch_k=20,
            vector_query=hyde_text,
        )
        for chunk in sub_chunks:
            all_chunks[chunk["chunk_id"]] = chunk   # later entries overwrite — fine

    merged = list(all_chunks.values())
    print(f"  [RetrieveMulti] Merged unique chunks: {len(merged)}")
    return {"raw_chunks": merged}


def node_rerank(state: PipelineState) -> dict:
    """Re-rank retrieved chunks with the local cross-encoder."""
    reranked = _reranker.rerank(
        query=state["active_query"],   # use original query, not HyDE
        chunks=state["raw_chunks"],
        top_k=5,
    )
    return {"reranked_chunks": reranked}


def node_confidence_gate(state: PipelineState) -> dict:
    """Check chunk quality and decide whether to proceed, retry, or refuse."""
    decision = _gate.check(state["reranked_chunks"])
    return {"gate_decision": decision}


def node_reformulate(state: PipelineState) -> dict:
    """Reformulate the query for another retrieval attempt (CRAG corrective loop)."""
    attempt   = state["retry_count"] + 1
    new_query = _gate.reformulate_query(state["original_query"], attempt)
    new_hyde  = _hyde.generate(new_query)
    return {
        "active_query": new_query,
        "hyde_text":    new_hyde,
        "retry_count":  attempt,
    }


def node_citation_generate(state: PipelineState) -> dict:
    """Generate grounded answer with inline citations using Gemini Pro."""
    answer, cited_ids = _citation.generate_grounded_answer(
        query=state["active_query"],
        chunks=state["reranked_chunks"],
    )

    if answer == "INSUFFICIENT_CONTEXT":
        return {
            "draft_answer":    "INSUFFICIENT_CONTEXT",
            "cited_chunk_ids": [],
            "gate_decision":   "refuse",   # signal to skip critic, go to refuse
        }

    return {
        "draft_answer":    answer,
        "cited_chunk_ids": cited_ids,
    }


def node_self_rag_critic(state: PipelineState) -> dict:
    """Run Self-RAG critic and produce the final verified answer."""
    critic = _citation.critic_check(
        answer=state["draft_answer"],
        chunks=state["reranked_chunks"],
    )

    verdict = critic.get("verdict", "pass")

    if verdict == "fail":
        final      = (
            "The generated answer could not be verified against the source "
            "documents and has been withheld. Please rephrase your question."
        )
        confidence = "refused"
    elif verdict == "partial":
        final      = critic.get("final_answer", state["draft_answer"])
        confidence = "low_confidence"
    else:
        final      = critic.get("final_answer", state["draft_answer"])
        confidence = "pass"

    return {
        "final_answer":  final,
        "confidence":    confidence,
        "critic_result": critic,
    }


def node_refuse(state: PipelineState) -> dict:
    """Terminal refuse node — retrieval gave up after max retries."""
    return {
        "final_answer": (
            "After multiple retrieval attempts, I was unable to find relevant "
            "content in the document store for this query. The documents may "
            "not contain information on this topic."
        ),
        "confidence":    "refused",
        "critic_result": {},
    }


# ─────────────────────────────────────────────
# 4. Edge Routing Functions
# ─────────────────────────────────────────────

def route_after_router(
    state: PipelineState,
) -> Literal["direct_answer", "hyde_generate", "decompose"]:
    route = state["route"]
    if route == "direct":
        return "direct_answer"
    elif route == "complex":
        return "decompose"
    else:
        return "hyde_generate"   # default: "simple"


def route_after_gate(
    state: PipelineState,
) -> Literal["citation_generate", "reformulate", "refuse"]:
    decision    = state["gate_decision"]
    retry_count = state.get("retry_count", 0)

    if decision == "pass":
        return "citation_generate"
    elif decision == "retry" and retry_count < MAX_RETRIES:
        return "reformulate"
    else:
        return "refuse"


def route_after_citation(
    state: PipelineState,
) -> Literal["self_rag_critic", "refuse"]:
    # If CitationEnforcer flagged INSUFFICIENT_CONTEXT, skip critic
    if state.get("gate_decision") == "refuse":
        return "refuse"
    return "self_rag_critic"


# ─────────────────────────────────────────────
# 5. Build the Graph
# ─────────────────────────────────────────────

def build_pipeline() -> StateGraph:
    builder = StateGraph(PipelineState)

    # Register all nodes
    builder.add_node("router",            node_router)
    builder.add_node("direct_answer",     node_direct_answer)
    builder.add_node("decompose",         node_decompose)
    builder.add_node("hyde_generate",     node_hyde_generate)
    builder.add_node("retrieve",          node_retrieve)
    builder.add_node("retrieve_multi",    node_retrieve_multi)
    builder.add_node("rerank",            node_rerank)
    builder.add_node("confidence_gate",   node_confidence_gate)
    builder.add_node("reformulate",       node_reformulate)
    builder.add_node("citation_generate", node_citation_generate)
    builder.add_node("self_rag_critic",   node_self_rag_critic)
    builder.add_node("refuse",            node_refuse)

    # Entry point
    builder.set_entry_point("router")

    # After router: branch to direct_answer, decompose, or hyde_generate
    builder.add_conditional_edges("router", route_after_router)

    # Direct path — ends immediately
    builder.add_edge("direct_answer", END)

    # Complex path: decompose → retrieve_multi → rerank
    builder.add_edge("decompose",      "retrieve_multi")
    builder.add_edge("retrieve_multi", "rerank")

    # Simple path: hyde_generate → retrieve → rerank
    builder.add_edge("hyde_generate", "retrieve")
    builder.add_edge("retrieve",      "rerank")

    # Common path from rerank onward
    builder.add_edge("rerank", "confidence_gate")

    # After gate: pass → citation, retry → reformulate, refuse → refuse
    builder.add_conditional_edges("confidence_gate", route_after_gate)

    # Corrective loop: reformulate loops back to retrieve
    builder.add_edge("reformulate", "retrieve")

    # After citation: to critic or refuse
    builder.add_conditional_edges("citation_generate", route_after_citation)

    # Terminal nodes
    builder.add_edge("self_rag_critic", END)
    builder.add_edge("refuse",          END)

    return builder.compile()


# Compile once at module level
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline()
    return _pipeline
