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

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="ACSH-RAG", docs_url="/api/docs", openapi_url="/api/openapi.json")


class AskRequest(BaseModel):
    question: str


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/ask")
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    try:
        # Imported lazily so the server starts fast; first question pays the
        # one-time model-loading cost (embedder + cross-encoder, ~1-2 min).
        from pipeline_api import run_pipeline
        result = run_pipeline(question)
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
