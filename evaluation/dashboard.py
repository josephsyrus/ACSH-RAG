"""Builds evaluation/results/dashboard.html from the run_*.json reports.

Only runs on the same question set as the latest one are compared, so a changed
dataset never produces a misleading trend line.
"""

import glob
import html
import json
import os

LABELS = {
    "retrieval_hit_rate": ("Retrieval hit rate", "Passage with the answer reached the answer step"),
    "answer_rate": ("Answer rate", "Answerable questions that were answered"),
    "refusal_accuracy": ("Refusal accuracy", "Unanswerable questions correctly refused"),
    "faithfulness": ("Faithfulness", "Claims supported by the passages used"),
    "answer_relevancy": ("Answer relevancy", "Answer addresses the question"),
    "context_recall": ("Context recall", "Passages contain the reference facts"),
    "factual_correctness": ("Factual correctness", "Reference facts stated correctly"),
}
TREND = ["retrieval_hit_rate", "faithfulness", "factual_correctness", "answer_rate"]
TREND_COLORS = ["#0C969C", "#6BA3BE", "#E3C46E", "#5CC9A7"]
esc = lambda s: html.escape(str(s if s is not None else ""))


def _pct(v):
    return "—" if v is None else f"{v * 100:.0f}%"


def _runs(results_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "run_*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                runs.append(json.load(f))
        except (OSError, ValueError):
            continue
    return runs


def _cards(report):
    out = []
    for key, value in report["metrics"].items():
        if key not in LABELS:
            continue
        title, sub = LABELS[key]
        thr = report["thresholds"].get(key)
        state = "na" if value is None else ("ok" if thr is None or value >= thr else "bad")
        width = 0 if value is None else max(2, value * 100)
        mark = f'<i style="left:{thr * 100:.0f}%" title="threshold"></i>' if thr is not None else ""
        out.append(f"""<div class="card {state}"><div class="c-top"><span>{esc(title)}</span>
          <b>{'n/a' if value is None else ('PASS' if state == 'ok' else 'FAIL')}</b></div>
          <div class="c-val">{_pct(value)}</div><div class="c-bar"><s style="width:{width:.0f}%"></s>{mark}</div>
          <div class="c-sub">{esc(sub)} · threshold {_pct(thr)}</div></div>""")
    return "".join(out)


def _groups(report):
    groups = report.get("groups") or {}
    if not groups:
        return ""
    keys = [k for k in LABELS if any(g.get(k) is not None for g in groups.values())]
    head = "".join(f"<th>{esc(LABELS[k][0])}</th>" for k in keys)
    body = "".join(
        f"<tr><td><b>{esc('Scanned (via OCR)' if name == 'scanned' else 'Born-digital')}</b></td>"
        + "".join(f"<td>{_pct(g.get(k))}</td>" for k in keys) + "</tr>"
        for name, g in groups.items())
    return f'<h2>Born-digital vs scanned documents</h2><div class="scroll"><table><tr><th>Document type</th>{head}</tr>{body}</table></div>'


def _trend(runs):
    if len(runs) < 2:
        return '<p class="muted">The trend appears once there are two or more runs on this question set.</p>'
    w, h, pad = 760, 220, 34
    n = len(runs)
    x = lambda i: pad + (w - 2 * pad) * (i / (n - 1))
    y = lambda v: h - pad - (h - 2 * pad) * v
    grid = "".join(f'<line x1="{pad}" x2="{w - pad}" y1="{y(t):.1f}" y2="{y(t):.1f}" class="g"/>'
                   f'<text x="{pad - 6}" y="{y(t) + 4:.1f}" class="t" text-anchor="end">{int(t * 100)}</text>'
                   for t in (0, .25, .5, .75, 1))
    lines, legend = [], []
    for key, color in zip(TREND, TREND_COLORS):
        pts = [(x(i), y(r["metrics"][key])) for i, r in enumerate(runs) if r["metrics"].get(key) is not None]
        if not pts:
            continue
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="'
                     + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + '"/>'
                     + "".join(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="3.5" fill="{color}"/>' for a, b in pts))
        legend.append(f'<span><i style="background:{color}"></i>{esc(LABELS[key][0])}</span>')
    labels = "".join(f'<text x="{x(i):.1f}" y="{h - 10}" class="t" text-anchor="middle">{esc(r["timestamp"][5:16].replace("T", " "))}</text>'
                     for i, r in enumerate(runs) if n <= 8 or i in (0, n - 1))
    return (f'<div class="scroll"><svg viewBox="0 0 {w} {h}" role="img" aria-label="Metric trend across runs">{grid}{"".join(lines)}{labels}</svg></div>'
            f'<div class="legend">{"".join(legend)}</div>')


def _questions(report):
    cols = [k for k in ("faithfulness", "answer_relevancy", "context_recall", "factual_correctness")
            if any(q.get(k) is not None for q in report["questions"])]
    head = "".join(f"<th>{esc(LABELS[k][0])}</th>" for k in cols)
    rows = []
    for q in report["questions"]:
        if "error" in q:
            outcome, cls = "error", "bad"
        elif q.get("refused") is None:
            outcome, cls = "retrieved", ""
        elif q["type"] == "out_of_scope":
            outcome, cls = ("refused (correct)", "ok") if q["refused"] else ("answered (should refuse)", "bad")
        else:
            outcome, cls = ("refused", "bad") if q["refused"] else (f"answered · {q.get('route', '')}", "ok")
        hit = "" if q.get("retrieval_hit") is None else ("✓" if q["retrieval_hit"] else "✗")
        hit_cls = "" if not hit else ("ok" if q["retrieval_hit"] else "bad")
        cells = "".join(f'<td class="num">{"—" if q.get(k) is None else f"{q[k]:.2f}"}</td>' for k in cols)
        detail = esc(q.get("error") or q.get("answer") or "")
        rows.append(f"""<tr><td>{esc(q['id'])}</td><td>{esc(q['type'].replace('_', ' '))}</td>
          <td class="q"><details><summary>{esc(q['question'])}</summary><div class="ans">{detail}</div>
          {f'<div class="ref"><b>Reference:</b> {esc(q["reference"])}</div>' if q.get("reference") else ''}</details></td>
          <td class="{cls}">{esc(outcome)}</td><td class="{hit_cls} num">{hit}</td>{cells}<td class="num">{esc(q.get('seconds'))}s</td></tr>""")
    return (f'<div class="scroll"><table class="qs"><tr><th>ID</th><th>Type</th><th>Question</th><th>Outcome</th>'
            f'<th>Hit</th>{head}<th>Time</th></tr>{"".join(rows)}</table></div>')


def build(results_dir: str) -> str:
    with open(os.path.join(results_dir, "latest.json"), encoding="utf-8") as f:
        latest = json.load(f)
    runs = [r for r in _runs(results_dir) if r.get("dataset_id") == latest.get("dataset_id")]
    verdict = "Passed" if latest["passed"] else "Failed"
    failed = ", ".join(LABELS.get(k, (k,))[0] for k in latest.get("failed_metrics", []))
    note = f" · below threshold: {esc(failed)}" if failed else ""
    if latest.get("errors"):
        note += f" · {latest['errors']} question(s) errored"
    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ACSH-RAG Evaluation</title>
<style>
:root{{--bg:#031716;--s:#032F30;--b:rgba(107,163,190,.18);--t:#E6F2F2;--t2:#A8C7CE;--t3:#7298A3;--a:#0C969C;--ok:#5CC9A7;--bad:#E8907F;--warn:#E3C46E}}
*{{box-sizing:border-box;margin:0;padding:0}}body{{background:var(--bg);color:var(--t);font:14.5px/1.55 "Segoe UI",system-ui,sans-serif;padding:32px 24px}}
main{{max-width:1180px;margin:0 auto}}h1{{font-size:26px;letter-spacing:-.01em}}h2{{font-size:17px;margin:34px 0 12px}}
.muted{{color:var(--t3)}}.meta{{color:var(--t2);margin-top:4px}}
.banner{{margin-top:18px;padding:14px 18px;border-radius:12px;border:1px solid;font-weight:600}}
.banner.ok{{border-color:rgba(92,201,167,.4);background:rgba(92,201,167,.08);color:var(--ok)}}
.banner.bad{{border-color:rgba(232,144,127,.4);background:rgba(232,144,127,.08);color:var(--bad)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px;margin-top:20px}}
.card{{background:var(--s);border:1px solid var(--b);border-radius:14px;padding:16px}}
.c-top{{display:flex;justify-content:space-between;color:var(--t2);font-size:13px}}.c-top b{{font-size:11px;letter-spacing:.06em}}
.card.ok .c-top b{{color:var(--ok)}}.card.bad .c-top b{{color:var(--bad)}}.card.na .c-top b{{color:var(--t3)}}
.c-val{{font-size:30px;font-weight:700;margin:4px 0 8px}}
.c-bar{{position:relative;height:7px;border-radius:9px;background:rgba(107,163,190,.18)}}
.c-bar s{{position:absolute;inset:0 auto 0 0;border-radius:9px;background:var(--a)}}.card.bad .c-bar s{{background:var(--bad)}}
.c-bar i{{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--t)}}
.c-sub{{color:var(--t3);font-size:12.5px;margin-top:8px}}
.scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;background:var(--s);border:1px solid var(--b);border-radius:12px}}
th,td{{padding:9px 12px;border-bottom:1px solid var(--b);text-align:left;vertical-align:top}}th{{color:var(--t3);font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.05em}}
td.num{{font-variant-numeric:tabular-nums;white-space:nowrap}}td.ok{{color:var(--ok)}}td.bad{{color:var(--bad)}}
td.q{{min-width:320px}}summary{{cursor:pointer}}.ans,.ref{{color:var(--t2);font-size:13px;margin-top:8px;white-space:pre-wrap}}
svg{{width:100%;min-width:560px;height:auto}}.g{{stroke:rgba(107,163,190,.16)}}.t{{fill:var(--t3);font-size:11px}}
.legend{{display:flex;flex-wrap:wrap;gap:16px;color:var(--t2);font-size:13px;margin-top:6px}}.legend i{{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px}}
</style></head><body><main>
<h1>ACSH-RAG evaluation</h1>
<div class="meta">{esc(latest['timestamp'].replace('T', ' '))} · {esc(latest['mode'])} · {latest['questions_run']} questions
{f"· judge {esc(latest['judge_model'])}" if latest.get('judge_model') else ''} · question set {esc(latest.get('dataset_id'))}</div>
<div class="banner {'ok' if latest['passed'] else 'bad'}">{verdict}{note}</div>
<div class="cards">{_cards(latest)}</div>
{_groups(latest)}
<h2>Trend ({len(runs)} run{'s' if len(runs) != 1 else ''} on this question set)</h2>{_trend(runs)}
<h2>Questions</h2><p class="muted" style="margin-bottom:10px">Click a question to see the answer and the reference.</p>{_questions(latest)}
</main></body></html>"""
    path = os.path.join(results_dir, "dashboard.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path
