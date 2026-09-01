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
                    model=self.model_name,
                    contents=query,
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_prompt,
                        temperature=0,
                        max_output_tokens=64,
                        # gemini-3.5-flash is a thinking model: without this,
                        # reasoning consumes max_output_tokens and the text
                        # part comes back EMPTY (raw='').
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )

                # Safer extraction than response.text shortcut — join ALL parts
                raw = ""
                if response.candidates and response.candidates[0].content.parts:
                    raw = " ".join(
                        p.text for p in response.candidates[0].content.parts
                        if getattr(p, "text", None)
                    ).strip().lower()

                print(f"  [Router] Raw: '{raw}'")

                if not raw:
                    # Empty output is transient (truncation/thinking) — retry
                    print("  [Router] Empty response, retrying...")
                    continue

                # The prompt asks for JSON: {"route": "simple"}
                route = None
                try:
                    clean = raw.replace("```json", "").replace("```", "").strip()
                    route = json.loads(clean).get("route", "")
                except (json.JSONDecodeError, AttributeError):
                    # Fallback: find a valid route word anywhere in the text
                    for candidate in self.VALID_ROUTES:
                        if candidate in raw:
                            route = candidate
                            break

                if route not in self.VALID_ROUTES:
                    print(f"  [Router] '{raw}' not valid, defaulting to 'simple'")
                    return "simple"

                print(f"  [Router] Route: {route}")
                return route

            except Exception as e:
                if "429" in str(e):
                    wait = 15 * (attempt + 1)
                    print(f"  [Router] Rate limit. Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  [Router] Error: {e}. Defaulting to 'simple'")
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
