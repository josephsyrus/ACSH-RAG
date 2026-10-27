"""Print evaluation/results/latest.json as a Markdown table (for the CI job summary)."""

import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "latest.json")
if not os.path.exists(path):
    print("No evaluation results were produced.")
    sys.exit(0)

with open(path, encoding="utf-8") as f:
    r = json.load(f)

print(f"## {'✅ Passed' if r['passed'] else '❌ Failed'}: ACSH-RAG evaluation ({r['mode']}, {r['questions_run']} questions)\n")
print("| Metric | Score | Threshold | |")
print("|---|---|---|---|")
for name, value in r["metrics"].items():
    thr = r["thresholds"].get(name)
    if value is None:
        print(f"| {name} | n/a | {thr} | |")
    else:
        print(f"| {name} | {value:.3f} | {thr:.2f} | {'✅' if thr is None or value >= thr else '❌'} |")
for group, m in r.get("groups", {}).items():
    shown = ", ".join(f"{k} {v:.2f}" for k, v in m.items() if v is not None)
    print(f"\n**{group}:** {shown}")
missed = [q["id"] for q in r["questions"] if q.get("retrieval_hit") is False]
if missed:
    print(f"\nAnswer passage not retrieved for: {', '.join(missed)}")
if r.get("errors"):
    print(f"\n{r['errors']} question(s) errored.")
print("\nThe full dashboard is in the uploaded artifact (dashboard.html).")
