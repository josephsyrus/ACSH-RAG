"""
pipeline/hyde.py

HyDE — Hypothetical Document Embeddings.

Uses the NEW Google GenAI SDK pattern:
- genai.Client(...)
- client.models.generate_content(...)
- types.GenerateContentConfig(...)

Model:
- gemini-2.5-flash
"""

import os
import json
import yaml
import time

from dotenv import load_dotenv
from typing import List

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
# HyDE Generator
# ─────────────────────────────────────────────

class HyDEGenerator:
    """
    Generates a hypothetical answer paragraph for a given query.
    This paragraph is used as the vector search query instead of
    the raw user question.
    """

    def __init__(self):

        self.model_name = "gemini-3.5-flash"

        self.system_prompt = _load_prompt("hyde")

    def generate(self, query: str) -> str:
        for attempt in range(3):

            try:

                response = client.models.generate_content(

                    model=self.model_name,

                    contents=query,

                    config=types.GenerateContentConfig(

                        system_instruction=self.system_prompt,

                        temperature=0.3,

                        max_output_tokens=400,
                    )
                )

                hypothesis = response.text.strip()

                print(
                    f"[HyDE] Error: {e}"
                )

                print(
                    f"[HyDE] Retry after "
                    f"{wait}s..."
                )

                return hypothesis

            except Exception as e:

                if any(code in str(e) for code in ["429", "503"]):

                    wait = 10 * (attempt + 1)

                    print(
                        f"[HyDE] Retry after "
                        f"{wait}s..."
                    )

                    time.sleep(wait)

                else:
                    raise

        print(
            "[HyDE] Failed after retries. "
            "Using original query."
        )

        return query


# ─────────────────────────────────────────────
# Query Decomposer
# ─────────────────────────────────────────────

class QueryDecomposer:
    """
    Breaks a complex multi-part query into focused sub-questions.
    """

    def __init__(self):

        self.model_name = "gemini-3.5-flash"

        self.system_prompt = _load_prompt("decomposer")

    def decompose(self, query: str) -> List[str]:
        """
        Decompose a complex query into sub-questions.

        Args:
            query: Complex user question

        Returns:
            List[str]: List of sub-questions
        """

        response = client.models.generate_content(

            model=self.model_name,

            contents=query,

            config=types.GenerateContentConfig(

                system_instruction=self.system_prompt,

                temperature=0,

                max_output_tokens=400,
            )
        )

        raw = response.text.strip()

        try:

            # Remove markdown code fences
            clean = (
                raw
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

            sub_questions = json.loads(clean)

            if not isinstance(sub_questions, list):
                raise ValueError("Expected a JSON array")

            print(
                f"[Decomposer] {len(sub_questions)} sub-questions: "
                f"{sub_questions}"
            )

            return sub_questions

        except (json.JSONDecodeError, ValueError) as e:

            print(
                f"[Decomposer] JSON parse error: {e}. "
                f"Falling back to single query."
            )

            return [query]


# ─────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────

if __name__ == "__main__":

    hyde = HyDEGenerator()

    decomposer = QueryDecomposer()

    q_simple = "What is the penalty for late payment?"

    q_complex = (
        "What are the differences between the termination and liability clauses, "
        "and what happens if both are triggered simultaneously?"
    )

    print("=== HyDE ===")

    print(hyde.generate(q_simple))

    print("\n=== Decomposer ===")

    print(decomposer.decompose(q_complex))