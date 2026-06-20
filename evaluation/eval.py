
import os
import sys
import json
import argparse
import datetime
import random

# Reconfigure stdout to UTF-8 to handle Unicode characters (emojis, etc.) on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv()


# ── Config ────────────────────────────────────────────────────────────────────

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "ragas_config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── RAGAS imports (loaded once, lazily) ───────────────────────────────────────

def _load_ragas():
    """Load RAGAS and wrap the Gemini LLM for use as the judge."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )
        return {
            "Dataset":           Dataset,
            "evaluate":          evaluate,
            "faithfulness":      faithfulness,
            "answer_relevancy":  answer_relevancy,
            "context_recall":    context_recall,
            "context_precision": context_precision,
        }
    except ImportError as e:
        print(f"\nERROR: Missing dependency — {e}")
        print("Run: pip install 'ragas>=0.2.0,<0.3' langchain-google-genai datasets")
        sys.exit(1)


# ── Custom LLM + Embeddings using google.genai (no langchain-google-genai) ────

from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.embeddings import Embeddings
from pydantic import Field
from typing import List, Optional


class _GeminiChat(SimpleChatModel):
    """
    Minimal LangChain-compatible chat model using the project's
    existing google.genai SDK. Avoids langchain-google-genai entirely
    so there are no version conflicts.
    """
    model_name: str = Field(default="gemini-1.5-flash")
    api_key:    str = Field(default="")

    def _call(
        self,
        messages,
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> str:
        from google import genai
        from google.genai import types

        client   = genai.Client(api_key=self.api_key)
        contents = "\n".join(
            str(getattr(m, "content", m))
            for m in (messages if isinstance(messages, list) else [messages])
        )
        resp = client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=2000,
            ),
        )
        return resp.text.strip() if resp.text else ""

    @property
    def _llm_type(self) -> str:
        return "gemini-direct"


class _GeminiEmbeddings(Embeddings):
    """
    Minimal LangChain-compatible embeddings using the project's
    existing google.genai SDK.
    """
    def __init__(self, api_key: str, model: str = "models/embedding-001"):
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model  = model

    def embed_query(self, text: str) -> List[float]:
        resp = self._client.models.embed_content(
            model=self._model,
            contents=text,
        )
        return list(resp.embeddings[0].values)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(t) for t in texts]


def _make_judge_llm(cfg):
    """Build the RAGAS judge LLM and embeddings using google.genai directly."""
    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError as e:
        print(f"\nERROR: {e}\nRun: pip install 'ragas>=0.2.0,<0.3'")
        sys.exit(1)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\nERROR: GEMINI_API_KEY not found in .env")
        sys.exit(1)

    judge_model = cfg.get("judge_model", "gemini-1.5-flash")
    embed_model = cfg.get("embedding_model", "models/embedding-001")

    print(f"  Judge LLM  : {judge_model} (via google.genai)")
    print(f"  Embeddings : {embed_model}")

    llm        = _GeminiChat(model_name=judge_model, api_key=api_key)
    embeddings = _GeminiEmbeddings(api_key=api_key, model=embed_model)

    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)


# ── Golden dataset ────────────────────────────────────────────────────────────

def load_golden_dataset(sample_size=None, seed=42):
    csv_path = os.path.join(os.path.dirname(__file__), "golden_dataset.csv")
    if not os.path.exists(csv_path):
        print(f"\nERROR: {csv_path} not found.")
        print("Create evaluation/golden_dataset.csv with columns: question, ground_truth, source_file")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Validate
    for col in ["question", "ground_truth"]:
        if col not in df.columns:
            print(f"\nERROR: golden_dataset.csv is missing column '{col}'")
            sys.exit(1)

    before = len(df)
    df = df.dropna(subset=["question", "ground_truth"])
    df = df[df["question"].str.strip() != ""]
    df = df[df["ground_truth"].str.strip() != ""]
    dropped = before - len(df)
    if dropped:
        print(f"  Warning: dropped {dropped} rows with empty question or ground_truth.")

    if len(df) == 0:
        print("\nERROR: No valid rows in golden_dataset.csv.")
        sys.exit(1)

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
        print(f"  Sampled {sample_size} of {before} rows.")

    print(f"  Loaded {len(df)} golden dataset rows.")
    return df


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_pipeline_on_dataset(df, retrieval_only=False):
    """
    Run every question through the pipeline (or retrieval only).
    Returns (answers, contexts) — two parallel lists for RAGAS.
    """
    answers  = []
    contexts = []
    skipped  = 0

    mode_label = "retrieval-only" if retrieval_only else "full pipeline"
    print(f"\n  Running {mode_label} on {len(df)} questions...")
    print("  Each question makes LLM calls — this may take a few minutes.\n")

    # ── Import the right API ──────────────────────────────────────
    from retrieve_api import retrieve_chunks

    if not retrieval_only:
        try:
            from pipeline_api import run_pipeline
        except ImportError:
            print(
                "  WARNING: pipeline_api.py could not be imported.\n"
                "  Falling back to retrieval-only mode.\n"
            )
            retrieval_only = True

    # ── Process each row ──────────────────────────────────────────
    for i, row in df.iterrows():
        question = str(row["question"]).strip()
        pos      = list(df.index).index(i) + 1
        total    = len(df)
        print(f"  [{pos}/{total}] {question[:72]}{'...' if len(question) > 72 else ''}")

        try:
            if retrieval_only:
                # ── Retrieval only (no LLM answer) ────────────────────
                chunks  = retrieve_chunks(question, top_k=5)
                context = [c["text"] for c in chunks if c.get("text")]
                if not context:
                    print(f"         → No chunks returned. Skipping.")
                    skipped += 1
                    continue
                # For retrieval-only testing, synthesise a fake answer
                # from the top 2 chunks. This tests the eval setup works
                # before the pipeline is ready.
                answer = " ".join(context[:2])

            else:
                # ── Full pipeline ─────────────────────────────────────
                result     = run_pipeline(question)
                answer     = result.get("answer", "")
                confidence = result.get("confidence", "unknown")
                cited_ids  = result.get("citations", [])
                route      = result.get("route", "?")

                if confidence == "refused" or not answer:
                    # Pipeline explicitly declined — still need context for metrics
                    chunks  = retrieve_chunks(question, top_k=5)
                    context = [c["text"] for c in chunks if c.get("text")]
                    answer  = "I do not have sufficient information to answer this question."
                    print(f"         → REFUSED by pipeline. Using decline message.")
                else:
                    # Get context chunks (prefer cited ones, fall back to top-5)
                    all_chunks = retrieve_chunks(question, top_k=10)
                    if cited_ids:
                        cited = [
                            c for c in all_chunks
                            if c.get("chunk_id") in cited_ids
                        ]
                        context = [c["text"] for c in cited] if cited else [c["text"] for c in all_chunks[:5]]
                    else:
                        context = [c["text"] for c in all_chunks[:5]]
                    print(f"         → route={route}  confidence={confidence}  citations={len(cited_ids)}")

                if not context:
                    print(f"         → No context. Skipping.")
                    skipped += 1
                    continue

            answers.append(answer)
            contexts.append(context)

        except Exception as e:
            print(f"         → ERROR: {e}. Skipping.")
            skipped += 1
            continue

    print(f"\n  Done. {len(answers)} evaluated, {skipped} skipped.\n")
    return answers, contexts


# ── RAGAS evaluation ──────────────────────────────────────────────────────────

def run_ragas_evaluation(df, answers, contexts, cfg):
    """Build RAGAS dataset and run all four metrics."""
    R = _load_ragas()
    wrapped_llm, wrapped_emb = _make_judge_llm(cfg)

    valid_count = len(answers)
    dataset = R["Dataset"].from_dict({
        "question":     list(df["question"])[:valid_count],
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": list(df["ground_truth"])[:valid_count],
    })

    metrics = [
        R["faithfulness"],
        R["answer_relevancy"],
        R["context_recall"],
        R["context_precision"],
    ]

    judge = cfg.get("judge_model", "gemini-1.5-flash")
    print(f"  Running RAGAS on {len(dataset)} samples using {judge} as judge...")
    print("  (Takes 2–8 minutes depending on dataset size)\n")

    try:
        result = R["evaluate"](
            dataset,
            metrics=metrics,
            llm=wrapped_llm,
            embeddings=wrapped_emb,
            raise_exceptions=False,
        )
    except Exception as e:
        print(f"\nERROR during RAGAS evaluation: {e}")
        print("Check your GEMINI_API_KEY and internet connection.")
        raise

    return result


# ── Report generation ─────────────────────────────────────────────────────────

def generate_report(result, cfg, mode, sample_count, output_dir):
    import math

    def _safe(val):
        """Return float or None if the metric failed (NaN)."""
        try:
            f = float(val)
            return None if math.isnan(f) else f
        except (TypeError, ValueError):
            return None

    scores = {
        "faithfulness":      _safe(result["faithfulness"]),
        "answer_relevancy":  _safe(result["answer_relevancy"]),
        "context_recall":    _safe(result["context_recall"]),
        "context_precision": _safe(result["context_precision"]),
    }

    # Check if any metric failed entirely
    failed_metrics = [k for k, v in scores.items() if v is None]
    if failed_metrics:
        print(f"\n  WARNING: These metrics returned NaN (LLM call failed): {failed_metrics}")
        print("  Scores that are None will be treated as 0.0 for pass/fail.\n")
        scores = {k: (v if v is not None else 0.0) for k, v in scores.items()}

    overall = sum(scores.values()) / len(scores)

    thresholds = {k: cfg["metrics"][k]["threshold"] for k in scores}
    faith_hard  = cfg.get("faithfulness_hard_threshold", 0.70)
    overall_thr = cfg.get("overall_threshold", 0.70)

    print("\n" + "=" * 62)
    print("  ACSH-RAG EVALUATION RESULTS")
    print("=" * 62)
    print(f"  Mode      : {mode}")
    print(f"  Samples   : {sample_count}")
    print(f"  Timestamp : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)

    labels = {
        "faithfulness":      "Faithfulness      (anti-hallucination) ",
        "answer_relevancy":  "Answer Relevancy  (on-topic)           ",
        "context_recall":    "Context Recall    (retrieval coverage) ",
        "context_precision": "Context Precision (retrieval relevance)",
    }

    for key, label in labels.items():
        score  = scores[key]
        thr    = thresholds[key]
        status = "✅ PASS" if score >= thr else "❌ FAIL"
        filled = int(score * 20)
        bar    = "█" * filled + "░" * (20 - filled)
        print(f"\n  {label}")
        print(f"  [{bar}]  {score:.4f}  (threshold: {thr:.2f})  {status}")

    print("\n" + "-" * 62)
    print(f"  Overall average : {overall:.4f}  (threshold: {overall_thr:.2f})")

    faith_hard_pass = scores["faithfulness"] >= faith_hard
    overall_pass    = overall >= overall_thr and faith_hard_pass

    print("\n" + "=" * 62)
    if overall_pass:
        print("  ✅  EVALUATION PASSED — System is trustworthy")
    else:
        print("  ❌  EVALUATION FAILED")
        if not faith_hard_pass:
            print(f"      Faithfulness {scores['faithfulness']:.4f} < hard threshold {faith_hard:.2f}")
            print("      The system is hallucinating. Do not demo this version.")
        else:
            print(f"      Overall {overall:.4f} < threshold {overall_thr:.2f}")
    print("=" * 62)

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp":                  datetime.datetime.now().isoformat(),
        "mode":                       mode,
        "samples_evaluated":          sample_count,
        "scores":                     scores,
        "overall_average":            overall,
        "thresholds":                 thresholds,
        "faithfulness_hard_threshold":faith_hard,
        "overall_threshold":          overall_thr,
        "passed":                     overall_pass,
        "faithfulness_hard_pass":     faith_hard_pass,
    }
    for path in [
        os.path.join(output_dir, f"eval_{timestamp}.json"),
        os.path.join(output_dir, "latest.json"),
    ]:
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

    print(f"\n  Report saved → evaluation/results/eval_{timestamp}.json\n")

    try:
        detail_df   = result.to_pandas()
        detail_path = os.path.join(output_dir, f"detail_{timestamp}.csv")
        detail_df.to_csv(detail_path, index=False)
    except Exception:
        pass

    return overall_pass, scores


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ACSH-RAG Evaluation (Person C)")
    parser.add_argument("--ci",             action="store_true",
                        help="CI mode: use ci_sample_size from config")
    parser.add_argument("--sample",         type=int, default=None,
                        help="Evaluate only N rows (quick testing)")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Use retrieval only, no LLM pipeline — for early testing")
    parser.add_argument("--seed",           type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    args = parser.parse_args()

    print("\n" + "=" * 62)
    print("  ACSH-RAG — Evaluation Layer (Person C)")
    print("=" * 62)

    cfg = load_config()

    if args.ci:
        sample_size = cfg.get("ci_sample_size", 20)
        mode = "CI"
    elif args.sample:
        sample_size = args.sample
        mode = f"local-{args.sample}-rows"
    else:
        sample_size = None
        mode = "full"

    print(f"\n  Mode: {mode}")

    print("\n[Step 1/4] Loading golden dataset...")
    df = load_golden_dataset(sample_size=sample_size, seed=args.seed)

    print("\n[Step 2/4] Running pipeline on each question...")
    answers, contexts = run_pipeline_on_dataset(df, retrieval_only=args.retrieval_only)

    if not answers:
        print("\nERROR: No answers generated. Check your pipeline and retrieval setup.")
        sys.exit(1)

    df = df.iloc[:len(answers)].reset_index(drop=True)

    print("\n[Step 3/4] Running RAGAS evaluation...")
    result = run_ragas_evaluation(df, answers, contexts, cfg)

    print("\n[Step 4/4] Generating report...")
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    passed, scores = generate_report(result, cfg, mode, len(answers), output_dir)

    # Exit code: 0 = pass (CI build passes), 1 = fail (CI build fails)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()