"""
pipeline/citation_enforcer.py

Citation Enforcement + Self-RAG Critic.

Updated to:
1. NEW Google GenAI SDK pattern

Uses:
- genai.Client(...)
- client.models.generate_content(...)
- types.GenerateContentConfig(...)
"""

import os
import re
import json
import yaml
import time

from typing import List, Dict, Tuple
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()


# ─────────────────────────────────────────────
# Gemini Client
# ─────────────────────────────────────────────

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# ─────────────────────────────────────────────
# Load prompts
# ─────────────────────────────────────────────

def _load_prompts() -> dict:

    prompts_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "prompts",
        "prompts.yaml"
    )

    with open(prompts_path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# Build context string
# ─────────────────────────────────────────────

def _build_context_string(chunks: list) -> tuple:
    """Returns (context_string, id_map) where id_map maps short→real chunk_id."""
    parts  = []
    id_map = {}

    for i, chunk in enumerate(chunks):
        real_id  = chunk.get("chunk_id", f"chunk_{i}")
        # Extract just the number at the end: chunk_00137 → [C137]
        import re
        match    = re.search(r'chunk_(\d+)$', real_id)
        short_id = f"C{int(match.group(1))}" if match else f"C{i+1}"

        id_map[short_id] = real_id
        parts.append(f"[{short_id}]\n{chunk['text']}")

    return "\n\n---\n\n".join(parts), id_map

# ─────────────────────────────────────────────
# Citation Enforcer
# ─────────────────────────────────────────────

class CitationEnforcer:
    """
    Generates grounded answers with citations and validates
    them using a Self-RAG critic pass.
    """

    def __init__(self):

        self.prompts = _load_prompts()

        # KEEPING requested models
        self._answer_model_name = "gemini-3.5-flash"

        self._fast_model_name = "gemini-3.5-flash"

    # ─────────────────────────────────────────
    # Step 1: Grounded Answer Generation
    # ─────────────────────────────────────────

    def generate_grounded_answer(self, query, chunks):

        context, id_map = _build_context_string(chunks)

        system_prompt = self.prompts["answer_grounded"]["system"].format(
            context=context
        )

        response = self._call_with_retry(

            model=self._answer_model_name,

            config=types.GenerateContentConfig(

                system_instruction=system_prompt,
                temperature=0.1,
                max_output_tokens=1200,
            ),

            contents=query,
        )

        answer = response.text.strip() if response.text else ""

        if "INSUFFICIENT_CONTEXT" in answer:
            return ("INSUFFICIENT_CONTEXT", [])

        # Extract short IDs like [C137]
        # Avoids matching things like [NEW HABIT]

        import re

        found_short = re.findall(r'\[C(\d+)\]', answer)

        cited_ids = list(set([

            id_map[f"C{n}"]

            for n in found_short

            if f"C{n}" in id_map
        ]))

        print(f"  [CitationEnforcer] Citations found: {cited_ids}")

        return (answer, cited_ids)

    # ─────────────────────────────────────────
    # Step 2: Self-RAG Critic
    # ─────────────────────────────────────────

    def critic_check(
        self,
        answer: str,
        chunks: List[Dict],
    ) -> Dict:

        context = _build_context_string(chunks)

        system_prompt = (
            self.prompts["self_rag_critic"]["system"]
            .format(
                context=context,
                answer=answer,
            )
        )

        response = self._call_with_retry(

            model=self._answer_model_name,

            contents="Check the answer above.",

            config=types.GenerateContentConfig(

                system_instruction=system_prompt,

                temperature=0,

                max_output_tokens=1500,
            )
        )

        raw = response.text.strip()

        try:

            clean = (
                raw
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

            result = json.loads(clean)

        except json.JSONDecodeError:

            print(
                f"[SelfRAG] JSON parse error. "
                f"Raw: {raw[:100]}"
            )

            # Conservative fallback
            return {
                "verdict": "pass",
                "unsupported_sentences": [],
                "explanation":
                    "Critic parse failed; answer accepted as-is.",
                "final_answer": answer,
            }

        verdict = result.get("verdict", "pass")

        unsupported = result.get(
            "unsupported_sentences",
            []
        )

        print(
            f"[SelfRAG] Verdict: {verdict} | "
            f"Unsupported: {len(unsupported)} sentences"
        )

        # Remove unsupported sentences if partial
        final_answer = answer

        if verdict == "partial" and unsupported:

            for bad_sentence in unsupported:

                final_answer = (
                    final_answer
                    .replace(bad_sentence, "")
                    .strip()
                )

            # Cleanup whitespace
            final_answer = " ".join(
                final_answer.split()
            )

        result["final_answer"] = (
            final_answer
            if verdict != "fail"
            else None
        )

        return result

    # ─────────────────────────────────────────
    # Step 3: Direct Answer
    # ─────────────────────────────────────────

    def generate_direct_answer(
        self,
        query: str
    ) -> str:
        """
        Direct answer without retrieval.
        """

        response = self._call_with_retry(

            model=self._fast_model_name,

            contents=query,

            config=types.GenerateContentConfig(

                system_instruction=(
                    self.prompts["answer_direct"]["system"]
                ),

                temperature=0.3,

                max_output_tokens=400,
            )
        )

        return response.text.strip()
    
    def _call_with_retry(self, **kwargs):
        """Wrapper for all Gemini calls with 429 retry."""
        for attempt in range(3):
            try:
                return client.models.generate_content(**kwargs)
            except Exception as e:
                if "429" in str(e):
                    wait = 20 * (attempt + 1)
                    print(f"  [CitationEnforcer] Rate limit. Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Max retries exceeded in CitationEnforcer.")