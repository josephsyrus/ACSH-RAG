"""
web_app.py

Web interface for the ACSH-RAG pipeline.

Serves a chat UI at http://localhost:8000 and exposes:
    GET  /api/health  — liveness check (no LLM calls)
    POST /api/ask     — {"question": "..."} → full pipeline answer

Run:
    python web_app.py
    # or: uvicorn web_app:app --host 127.0.0.1 --port 8000
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import uuid
import shutil
import json
import queue
import threading

app = FastAPI(title="ACSH-RAG", docs_url="/api/docs", openapi_url="/api/openapi.json")

SESSIONS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions")
SUPPORTED_EXT = {".pdf", ".md", ".markdown", ".txt"}

# Reap stale upload sessions at startup (24h TTL / 200 max — see src.upload).
try:
    from src.upload import cleanup_sessions as _cleanup_sessions
    _cleanup_sessions(SESSIONS_ROOT)
except Exception:
    pass

# Friendly labels streamed to the UI as each LangGraph node runs.
_STAGE_LABELS = {
    "router":            "Classifying your question…",
    "direct_answer":     "Answering from general knowledge…",
    "decompose":         "Breaking it into sub-questions…",
    "hyde_generate":     "Expanding the query…",
    "retrieve":          "Searching the documents…",
    "retrieve_multi":    "Searching the documents…",
    "rerank":            "Reranking the best passages…",
    "confidence_gate":   "Checking retrieval confidence…",
    "reformulate":       "Refining and retrying…",
    "citation_generate": "Writing a grounded answer…",
    "self_rag_critic":   "Verifying each claim against the sources…",
    "refuse":            "Preparing response…",
}


def _sse(obj: dict) -> str:
    return "data: " + json.dumps(obj) + "\n\n"


class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = None   # when set, answer only from that upload


def _resolve_session_dirs(session_id: str) -> dict:
    """Validate a session_id (guarding against path traversal) and return its
    isolated index directories."""
    from src.upload import session_dirs
    safe = os.path.basename((session_id or "").strip())   # strip any path parts
    sroot = os.path.join(SESSIONS_ROOT, safe)
    if not safe or not os.path.exists(os.path.join(sroot, "graph_db", "graph.json")):
        raise HTTPException(status_code=404, detail="Session not found. Upload a document first.")
    return session_dirs(sroot)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in SUPPORTED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'. Use PDF, Markdown, or text.")

    _cleanup_sessions(SESSIONS_ROOT)   # reap stale sessions before adding a new one
    session_id = uuid.uuid4().hex[:12]
    updir = os.path.join(SESSIONS_ROOT, session_id, "upload")
    os.makedirs(updir, exist_ok=True)
    dest = os.path.join(updir, os.path.basename(file.filename))
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # Local ingestion (embeddings + BM25 + spaCy graph) — no Gemini/quota.
        from src.upload import ingest_upload
        stats = ingest_upload(dest, os.path.join(SESSIONS_ROOT, session_id))
    except Exception as e:
        shutil.rmtree(os.path.join(SESSIONS_ROOT, session_id), ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Could not process document: {e}")

    return {
        "session_id": session_id,
        "filename":   stats["filename"],
        "chunks":     stats["chunks"],
        "pages":      stats["pages"],
    }


@app.post("/api/ask")
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    index_dirs = _resolve_session_dirs(req.session_id) if req.session_id else None

    try:
        # Imported lazily so the server starts fast; first question pays the
        # one-time model-loading cost (embedder + cross-encoder, ~1-2 min).
        from pipeline_api import run_pipeline
        result = run_pipeline(question, index_dirs=index_dirs)
        return {
            "answer":     result["answer"],
            "citations":  result["citations"],
            "route":      result["route"],
            "confidence": result["confidence"],
        }
    except Exception as e:
        msg = str(e)
        if any(tok in msg for tok in ("429", "RESOURCE_EXHAUSTED", "Max retries")):
            # Quota exhaustion is an expected operational state on free tier —
            # surface it as a readable answer instead of a 500.
            return {
                "answer": (
                    "The Gemini API quota is currently exhausted. "
                    "Please try again later (free-tier quotas reset daily at midnight Pacific)."
                ),
                "citations":  [],
                "route":      "error",
                "confidence": "quota_exhausted",
            }
        raise HTTPException(status_code=500, detail=msg)


@app.post("/api/ask_stream")
def ask_stream(req: AskRequest):
    """Streaming /api/ask — emits Server-Sent Events: one {type:'stage'} per
    pipeline node as it runs, then a final {type:'done'} with the answer."""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")
    index_dirs = _resolve_session_dirs(req.session_id) if req.session_id else None

    # The pipeline runs in a worker thread and pushes events onto this queue:
    # ('stage', label) between nodes, ('token', text) from inside the answer
    # node, then ('done', payload) / ('error', msg). The request thread below
    # drains the queue and serialises each as an SSE frame.
    q: "queue.Queue" = queue.Queue()

    def worker():
        try:
            from pipeline.graph import get_pipeline
            from pipeline_api import shape_citations
            state = {
                "original_query":  question,
                "index_dirs":      index_dirs,
                "emit":            lambda kind, payload: q.put((kind, payload)),
                "route":           "",
                "active_query":    question,
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
            final: dict = {}
            for update in get_pipeline().stream(state):
                for node, delta in (update or {}).items():
                    if delta:
                        final.update(delta)
                    label = _STAGE_LABELS.get(node)
                    if label:
                        q.put(("stage", label))
            q.put(("done", {
                "answer":     final.get("final_answer", "No answer generated."),
                "route":      final.get("route", "unknown"),
                "confidence": final.get("confidence", "unknown"),
                "citations":  shape_citations(final),
            }))
        except Exception as e:
            msg = str(e)
            if any(tok in msg for tok in ("429", "RESOURCE_EXHAUSTED", "Max retries")):
                q.put(("done", {
                    "answer": ("The Gemini API quota is currently exhausted. "
                               "Please try again later (free-tier quotas reset daily at midnight Pacific)."),
                    "route": "error", "confidence": "quota_exhausted", "citations": [],
                }))
            else:
                q.put(("error", msg))
        finally:
            q.put(("__end__", None))

    def gen():
        threading.Thread(target=worker, daemon=True).start()
        while True:
            kind, payload = q.get()
            if kind == "__end__":
                break
            if kind == "stage":
                yield _sse({"type": "stage", "label": payload})
            elif kind == "token":
                yield _sse({"type": "token", "text": payload})
            elif kind == "done":
                yield _sse({"type": "done", **payload})
            elif kind == "error":
                yield _sse({"type": "error", "detail": payload})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Mounted last so /api/* routes above take precedence.
app.mount(
    "/",
    StaticFiles(directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "static"), html=True),
    name="static",
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
