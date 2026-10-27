"""The Self-RAG support threshold and the 'advanced details' payload. No API calls.

Run with:  python -m pytest tests/test_answer_details.py
"""

import pytest

from pipeline.citation_enforcer import SUPPORT_THRESHOLD, apply_support_threshold
from pipeline.confidence_gate import MAX_RETRIES, PASS_THRESHOLD, RETRY_THRESHOLD

FIVE = ("Leaks waste water in every city. Sensors measure pressure at junctions. "
        "Graph networks model the pipes well. The method beats the baseline model. "
        "Results hold on the L-town network.")


def test_threshold_is_seventy_percent():
    assert SUPPORT_THRESHOLD == 0.70


@pytest.mark.parametrize("model_verdict,unsupported,verdict,ratio", [
    ("pass",    [],                                   "pass",    1.0),
    ("partial", ["Sensors measure pressure at junctions."], "partial", 0.8),
    ("partial", ["a.", "b."],                         "fail",    0.6),   # 3/5 = 60% < 70%: withheld
    ("fail",    [],                                   "fail",    1.0),   # critic's overall 'fail' is kept
    ("partial", [],                                   "pass",    1.0),   # nothing listed: nothing to remove
    ("pass",    ["x."] * 9,                           "fail",    0.0),   # more listed than exist: capped
])
def test_verdict_follows_support_ratio(model_verdict, unsupported, verdict, ratio):
    out = apply_support_threshold(FIVE, model_verdict, unsupported)
    assert (out["verdict"], out["support_ratio"], out["sentences_checked"]) == (verdict, ratio, 5)
    assert out["support_threshold"] == SUPPORT_THRESHOLD and out["checked"] is True


def test_bullets_count_as_sentences_but_headings_do_not():
    answer = "Key points:\n- Leaks waste treated water [C3].\n- Sensors are placed at 33 junctions [C9].\n1. Make good habits easy to repeat."
    assert apply_support_threshold(answer, "pass", [])["sentences_checked"] == 3


def test_exactly_at_threshold_is_accepted():
    ten = " ".join(f"Sentence number {i} is a claim." for i in range(10))
    assert apply_support_threshold(ten, "partial", ["s."] * 3)["verdict"] == "partial"   # 7/10 = 70%


# ── details payload ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def shape_details():
    from pipeline_api import shape_details
    return shape_details


def _state(score=None, retries=0, critic=None, draft="An answer."):
    return {
        "reranked_chunks": [] if score is None else [{"rerank_score": score}, {"rerank_score": score - 1}],
        "retry_count": retries, "critic_result": critic or {}, "draft_answer": draft,
    }


def test_details_report_gate_thresholds_and_decision(shape_details):
    d = shape_details(_state(score=4.94321, retries=1, critic=apply_support_threshold(FIVE, "pass", [])))
    assert d["gate"] == {"score": 4.943, "pass_threshold": PASS_THRESHOLD, "retry_threshold": RETRY_THRESHOLD,
                         "decision": "pass", "retries_used": 1, "max_retries": MAX_RETRIES}
    v = d["verification"]
    assert (v["verdict"], v["support_ratio"], v["sentences_checked"], v["threshold"]) == ("pass", 1.0, 5, 0.70)


@pytest.mark.parametrize("score,decision", [(0.0, "pass"), (-0.01, "retry"), (-3.0, "retry"), (-3.01, "refuse"), (None, "refuse")])
def test_gate_decision_boundaries(shape_details, score, decision):
    assert shape_details(_state(score=score))["gate"]["decision"] == decision


def test_refusals_have_no_verification(shape_details):
    gate_refused = shape_details(_state(score=-6.0, retries=MAX_RETRIES))
    assert gate_refused["verification"] is None and gate_refused["insufficient_context"] is False
    no_answer = shape_details(_state(score=2.0, draft="INSUFFICIENT_CONTEXT"))
    assert no_answer["verification"] is None and no_answer["insufficient_context"] is True
