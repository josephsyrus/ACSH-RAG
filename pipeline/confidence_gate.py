"""
pipeline/confidence_gate.py

Confidence Quality Gate + Corrective Loop (CRAG).

Uses the NEW Google GenAI SDK pattern:
- genai.Client(...)
- client.models.generate_content(...)
- types.GenerateContentConfig(...)

Model:
- gemini-2.5-flash
"""

import os
import yaml

from typing import List, Dict
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
# Thresholds
# ─────────────────────────────────────────────

# Tune after RAGAS evaluation

PASS_THRESHOLD = 0.0

RETRY_THRESHOLD = -3.0

MAX_RETRIES = 2


# ─────────────────────────────────────────────
# Load prompts from YAML
# ─────────────────────────────────────────────

def _load_prompt(key: str) -> str:

    prompts_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "prompts",
        "prompts.yaml"
    )

    with open(prompts_path, "r") as f:
        prompts = yaml.safe_load(f)

    return prompts[key]["system"]


# ─────────────────────────────────────────────
# Confidence Gate
# ─────────────────────────────────────────────

class ConfidenceGate:
    """
    Quality gate for retrieved chunks.

    Decides whether to:
    - pass
    - retry retrieval with reformulated query
    - refuse response
    """

    def __init__(self):

        self.model_name = "gemini-3.5-flash"

        self.system_prompt = _load_prompt("reformulate")

    def check(self, chunks: List[Dict]) -> str:
        """
        Evaluate retrieval quality using reranker score.

        Args:
            chunks: Retrieved chunks with rerank scores

        Returns:
            str:
                "pass"
                "retry"
                "refuse"
        """

        if not chunks:

            print("[ConfidenceGate] No chunks returned → REFUSE")

            return "refuse"

        top_score = chunks[0].get("rerank_score", 0.0)

        print(
            f"[ConfidenceGate] Top rerank score: {top_score:.4f}"
        )

        if top_score >= PASS_THRESHOLD:

            print("[ConfidenceGate] → PASS")

            return "pass"

        elif top_score >= RETRY_THRESHOLD:

            print("[ConfidenceGate] → RETRY")

            return "retry"

        else:

            print("[ConfidenceGate] → REFUSE")

            return "refuse"

    def reformulate_query(
        self,
        original_query: str,
        attempt: int
    ) -> str:
        """
        Reformulate the query to improve retrieval.

        Args:
            original_query: Failed retrieval query
            attempt: Retry attempt number

        Returns:
            str: Reformulated query
        """

        response = client.models.generate_content(

            model=self.model_name,

            contents=f"Original query: {original_query}",

            config=types.GenerateContentConfig(

                system_instruction=self.system_prompt,

                temperature=0.4,

                max_output_tokens=100,
            )
        )

        new_query = response.text.strip()

        print(
            f"[ConfidenceGate] Reformulated query "
            f"(attempt {attempt}): {new_query}"
        )

        return new_query