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
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import uuid
import shutil

app = FastAPI(title="ACSH-RAG", docs_url="/api/docs", openapi_url="/api/openapi.json")

SESSIONS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions")
SUPPORTED_EXT = {".pdf", ".md", ".markdown", ".txt"}


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


# Mounted last so /api/* routes above take precedence.
app.mount(
    "/",
    StaticFiles(directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "static"), html=True),
    name="static",
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
