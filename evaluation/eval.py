"""ACSH-RAG evaluation: run the pipeline over evaluation/dataset.csv and score it.

    python evaluation/eval.py                    # full run: pipeline + RAGAS judge
    python evaluation/eval.py --sample 6         # quick run on a stratified sample
    python evaluation/eval.py --retrieval-only   # free: retrieval hit rate only, no API calls

Uses its own corpus and indexes, built with evaluation/prepare_corpus.py.
Writes evaluation/results/run_<time>.json, latest.json and dashboard.html, and
exits 1 if any measured metric is below its threshold in ragas_config.yaml.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import sys
import time
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv()

from evaluation.prepare_corpus import INDEX_DIRS, INDEX_DIR   # noqa: E402
from evaluation import dashboard                               # noqa: E402

EVAL_DIR = os.path.join(ROOT, "evaluation")
DATASET = os.path.join(EVAL_DIR, "dataset.csv")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_recall", "factual_correctness"]


def load_config() -> dict:
    with open(os.path.join(EVAL_DIR, "ragas_config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── dataset ──────────────────────────────────────────────────────────────────

def load_dataset(sample: Optional[int], seed: int) -> pd.DataFrame:
    df = pd.read_csv(DATASET, dtype=str).fillna("")
    if sample and sample < len(df):
        # Keep every question type represented, in proportion.
        parts = []
        for _, group in df.groupby("type", sort=False):
            n = max(1, round(sample * len(group) / len(df)))
            parts.append(group.sample(n=min(n, len(group)), random_state=seed))
        df = pd.concat(parts)
    return df.reset_index(drop=True)


def dataset_id() -> str:
    with open(DATASET, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:10]


def _norm(text: str) -> str:
    # Letters and digits only: OCR often drops spaces ("abletocapture") and dashes,
    # which must not count as a miss.
    return re.sub(r"[^a-z0-9]", "", text.lower())


def retrieval_hit(row, contexts: List[dict]) -> bool:
    """True if a passage containing the answer's evidence reached the answer step.
    Matching the page alone is too lenient: a different passage from the same page
    would count."""
    evidence = [_norm(e) for e in row["evidence"].split("|") if e.strip()]
    return any(
        c.get("filename") == row["source"] and any(e in _norm(c.get("text", "")) for e in evidence)
        for c in contexts
    )


# ── running the system ───────────────────────────────────────────────────────

def _context(c: dict) -> dict:
    meta = c.get("metadata") or {}
    return {"text": c.get("text", ""), "filename": c.get("filename") or meta.get("filename") or "",
            "page": c.get("page") or meta.get("page") or None}


def run_questions(df: pd.DataFrame, retrieval_only: bool, pause: float) -> List[dict]:
    from retrieve_api import retrieve_chunks
    if not retrieval_only:
        from pipeline_api import run_pipeline

    rows, quota_errors = [], 0
    for i, r in df.iterrows():
        if i and pause:
            time.sleep(pause)   # spreads requests out to stay under per-minute API limits
        rec = {k: r[k] for k in ("id", "type", "source", "pages", "question", "reference")}
        t0 = time.time()
        try:
            if retrieval_only:
                chunks = retrieve_chunks(r["question"], top_k=5, index_dirs=INDEX_DIRS)
                rec.update(answer="", refused=None, route="", confidence="", contexts=[_context(c) for c in chunks])
            else:
                out = run_pipeline(r["question"], index_dirs=INDEX_DIRS)
                rec.update(
                    answer=out["answer"], route=out["route"], confidence=out["confidence"],
                    # "refused" covers every way of declining: the confidence gate,
                    # INSUFFICIENT_CONTEXT, and the critic withholding the answer.
                    refused=out["confidence"] == "refused",
                    contexts=out["contexts"],
                )
            rec["retrieval_hit"] = retrieval_hit(r, rec["contexts"]) if r["type"] != "out_of_scope" else None
            quota_errors = 0
        except Exception as e:
            rec["error"] = str(e)[:300]
            # Quota, rate-limit and network failures say nothing about answer quality;
            # if they repeat, every remaining question would fail the same way.
            outage = ("429", "RESOURCE_EXHAUSTED", "Max retries", "UNAVAILABLE", "503", "getaddrinfo", "SSL")
            quota_errors = quota_errors + 1 if any(t in rec["error"] for t in outage) else 0
        rec["seconds"] = round(time.time() - t0, 1)
        rows.append(rec)

        status = ("ERROR " + rec["error"][:60]) if "error" in rec else (
            "retrieved" if retrieval_only else ("refused" if rec["refused"] else f"answered ({rec['route']})"))
        hit = "" if rec.get("retrieval_hit") is None else ("  hit" if rec["retrieval_hit"] else "  MISS")
        print(f"  [{i + 1}/{len(df)}] {rec['id']:<4} {status}{hit}  {rec['seconds']}s", flush=True)

        if quota_errors >= 2:
            print("\n  Stopping early: the Gemini API is refusing requests (quota, rate limit or network).\n"
                  "  Wait for the quota to reset or check your connection, then run again.")
            break
    return rows


# ── RAGAS judge ──────────────────────────────────────────────────────────────

def _judge(cfg: dict):
    from google import genai
    from google.genai import types
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models.chat_models import SimpleChatModel
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY is not set (needed for the judge). Use --retrieval-only for a free run.")
    client = genai.Client(api_key=api_key)

    class GeminiChat(SimpleChatModel):
        """LangChain chat model over the google-genai SDK the project already uses."""
        model_name: str

        def _call(self, messages, stop=None, run_manager=None, **kwargs) -> str:
            text = "\n".join(str(getattr(m, "content", m)) for m in messages)
            cfg_ = types.GenerateContentConfig(
                temperature=0, max_output_tokens=4096,
                # Thinking models spend the output budget on reasoning otherwise,
                # which truncates the JSON the judge must return.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
            resp = client.models.generate_content(model=self.model_name, contents=text, config=cfg_)
            return (resp.text or "").strip()

        @property
        def _llm_type(self) -> str:
            return "gemini-genai"

    class GeminiEmbeddings(Embeddings):
        def embed_documents(self, texts):
            resp = client.models.embed_content(model=cfg["embedding_model"], contents=list(texts))
            return [list(e.values) for e in resp.embeddings]

        def embed_query(self, text):
            return self.embed_documents([text])[0]

    return LangchainLLMWrapper(GeminiChat(model_name=cfg["judge_model"])), LangchainEmbeddingsWrapper(GeminiEmbeddings())


def ragas_score(rows: List[dict], cfg: dict) -> None:
    """Adds RAGAS scores to the rows that produced an answer, in place."""
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import FactualCorrectness, Faithfulness, LLMContextRecall, ResponseRelevancy

    scored = [r for r in rows if r["type"] != "out_of_scope" and r.get("refused") is False and r.get("contexts")]
    if not scored:
        print("  Nothing to grade: no answerable question was answered.")
        return
    samples = [SingleTurnSample(
        user_input=r["question"], response=r["answer"], reference=r["reference"],
        retrieved_contexts=[c["text"] for c in r["contexts"]],
    ) for r in scored]

    llm, emb = _judge(cfg)
    print(f"\n  Grading {len(samples)} answers with {cfg['judge_model']}...")
    result = evaluate(
        EvaluationDataset(samples=samples),
        metrics=[Faithfulness(), ResponseRelevancy(), LLMContextRecall(), FactualCorrectness(mode="recall")],
        llm=llm, embeddings=emb, raise_exceptions=False, show_progress=True,
        run_config=RunConfig(max_workers=cfg.get("max_workers", 4), timeout=240, max_retries=8, max_wait=60),
    )
    table = result.to_pandas()
    for name in RAGAS_METRICS:
        col = next((c for c in table.columns if c.startswith(name)), None)
        for r, value in zip(scored, table[col] if col else [None] * len(scored)):
            r[name] = None if value is None or (isinstance(value, float) and math.isnan(value)) else round(float(value), 4)


# ── summary ──────────────────────────────────────────────────────────────────

def _mean(values) -> Optional[float]:
    values = [float(v) for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def _metrics(rows: List[dict], retrieval_only: bool) -> Dict[str, Optional[float]]:
    ok = [r for r in rows if "error" not in r]
    answerable = [r for r in ok if r["type"] != "out_of_scope"]
    out_of_scope = [r for r in ok if r["type"] == "out_of_scope"]
    m = {"retrieval_hit_rate": _mean(r["retrieval_hit"] for r in answerable)}
    if not retrieval_only:
        m["answer_rate"] = _mean(not r["refused"] for r in answerable)
        m["refusal_accuracy"] = _mean(r["refused"] for r in out_of_scope)
        for name in RAGAS_METRICS:
            m[name] = _mean(r.get(name) for r in answerable)
    return m


def summarize(rows: List[dict], cfg: dict, mode: str, retrieval_only: bool) -> dict:
    metrics = _metrics(rows, retrieval_only)
    groups = {t: _metrics([r for r in rows if r["type"] == t], retrieval_only)
              for t in ("text", "scanned") if any(r["type"] == t for r in rows)}
    thresholds = {k: v["threshold"] for k, v in cfg["metrics"].items()}
    failed = [k for k, v in metrics.items() if v is not None and k in thresholds and v < thresholds[k]]
    errors = [r for r in rows if "error" in r]
    for r in rows:   # keep result files small: passage location plus the first 400 characters
        r["contexts"] = [{**c, "text": c["text"][:400]} for c in r.get("contexts", [])]
    return {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "dataset_id": dataset_id(),
        "judge_model": None if retrieval_only else cfg["judge_model"],
        "questions_run": len(rows),
        "errors": len(errors),
        "metrics": metrics,
        "groups": groups,
        "thresholds": thresholds,
        "failed_metrics": failed,
        "passed": not failed and not errors,
        "questions": rows,
    }


def print_report(report: dict) -> None:
    print("\n" + "=" * 66 + f"\n  ACSH-RAG EVALUATION  ·  {report['mode']}  ·  {report['questions_run']} questions\n" + "=" * 66)
    for name, value in report["metrics"].items():
        thr = report["thresholds"].get(name)
        if value is None:
            print(f"  {name:<22} n/a")
            continue
        verdict = "PASS" if thr is None or value >= thr else "FAIL"
        print(f"  {name:<22} {value:6.3f}   threshold {thr:.2f}   {verdict}")
    for group, m in report["groups"].items():
        shown = "  ".join(f"{k}={v:.2f}" for k, v in m.items() if v is not None)
        print(f"\n  {group:<8} {shown}")
    if report["errors"]:
        print(f"\n  {report['errors']} question(s) errored and were not scored.")
    print("\n  RESULT: " + ("PASSED" if report["passed"] else "FAILED") + "\n" + "=" * 66)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ACSH-RAG evaluation")
    parser.add_argument("--sample", type=int, help="evaluate a stratified sample of N questions")
    parser.add_argument("--ci", action="store_true", help="use ci_sample_size from the config")
    parser.add_argument("--retrieval-only", action="store_true", help="retrieval hit rate only; makes no API calls")
    parser.add_argument("--pause", type=float, default=0, help="seconds to wait between questions (for rate limits)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(os.path.join(INDEX_DIR, "graph_db", "graph.json")):
        sys.exit("No evaluation index found. Build it first:\n"
                 "  python evaluation/prepare_corpus.py --text A.pdf=name.pdf --scanned B.pdf=name.pdf")

    cfg = load_config()
    sample = cfg.get("ci_sample_size") if args.ci else args.sample
    mode = ("retrieval-only" if args.retrieval_only else "full") + (f" · sample {sample}" if sample else "")
    df = load_dataset(sample, args.seed)
    print(f"\nRunning {len(df)} questions ({mode})...\n")

    rows = run_questions(df, args.retrieval_only, 0 if args.retrieval_only else args.pause)
    if not args.retrieval_only:
        ragas_score(rows, cfg)

    report = summarize(rows, cfg, mode, args.retrieval_only)
    print_report(report)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for name in (f"run_{stamp}.json", "latest.json"):
        with open(os.path.join(RESULTS_DIR, name), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    path = dashboard.build(RESULTS_DIR)
    print(f"\n  Saved results/run_{stamp}.json and {os.path.relpath(path, ROOT)}")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
