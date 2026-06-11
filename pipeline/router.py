import os
import yaml
from dotenv import load_dotenv
import time
import json

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
# Router
# ─────────────────────────────────────────────

class AdaptiveRouter:

    VALID_ROUTES = {
        "direct",
        "simple",
        "complex"
    }

    def __init__(self):

        self.system_prompt = _load_prompt("router")

        self.model_name = "gemini-3.5-flash"

    def classify(self, query: str) -> str:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=self.model_name,   # make sure this is "gemini-2.0-flash"
                    contents=query,
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_prompt,
                        temperature=0,
                        max_output_tokens=20,
                    ),
                )

                # Safer extraction than response.text shortcut
                raw = ""

                try:

                    if (
                        response.candidates
                        and response.candidates[0].content.parts
                    ):
                        raw = (
                            response.candidates[0]
                            .content.parts[0]
                            .text
                            .strip()
                            .lower()
                        )

                except Exception:
                    raw = ""

                raw = raw.rstrip(".,!? \n")

                print(f"  [Router] Raw: '{raw}'")

                if raw not in self.VALID_ROUTES:
                    print(f"  [Router] '{raw}' not valid, defaulting to 'simple'")
                    return "simple"

                print(f"  [Router] Route: {raw}")
                return raw

            except Exception as e:

                if any(code in str(e) for code in ["429", "503"]):

                    wait = 10 * (2 ** attempt)

                    print(
                        f"  [Router] Gemini unavailable. "
                        f"Retrying in {wait}s..."
                    )

                    time.sleep(wait)

                else:

                    print(
                        f"  [Router] Error: {e}. "
                        f"Defaulting to 'simple'"
                    )

                    return "simple"

        print("  [Router] All retries failed. Defaulting to 'simple'")
        return "simple"

# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":

    router = AdaptiveRouter()

    tests = [

        "What is 2+2?",

        "What is the penalty clause in Section 5?",

        (
            "What are the differences between the "
            "termination clause and the liability cap, "
            "and how do they interact with each other?"
        ),
    ]

    for q in tests:

        print(f"\nQ: {q}")

        result = router.classify(q)

        print(f"Route → {result}")