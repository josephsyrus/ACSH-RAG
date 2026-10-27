"""Accounts: the daily allowance, Google sign-in handling, and endpoint protection.

Run with:  python -m pytest tests/test_accounts.py
Google and the language-model pipeline are faked, so no network or API key is needed.
"""

import itertools
import json
import os
import sys
import threading
import types
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from authlib.integrations.base_client import OAuthError

from accounts import db as accounts_db
from accounts.db import (
    SessionLocal, User, consume_message, refund_message, upsert_google_user, usage_snapshot,
)

_ids = itertools.count(1)
# Unique per run, so a persistent test database never hands back a user from an earlier run.
RUN = uuid.uuid4().hex[:8]


@pytest.fixture(scope="session", autouse=True)
def tables():
    accounts_db.init_db()


@pytest.fixture
def user():
    n = next(_ids)
    with SessionLocal() as s:
        yield upsert_google_user(s, f"{RUN}-sub-{n}", f"{RUN}-user{n}@example.com", f"User {n}", "")


def _consume(user, now=None):
    with SessionLocal() as s:
        return consume_message(s, user, now)


def _used(user, now=None):
    with SessionLocal() as s:
        return usage_snapshot(s, user.id, now)["used"]


# ── the allowance ───────────────────────────────────────────────────────────────

def test_demo_user_gets_exactly_three_messages_a_day(user):
    assert [_consume(user) for _ in range(5)] == [True, True, True, False, False]


def test_allowance_resets_on_the_next_utc_day(user):
    today = datetime(2026, 11, 3, 23, 0, tzinfo=timezone.utc)
    assert all(_consume(user, today) for _ in range(3))
    assert not _consume(user, today)
    tomorrow = today + timedelta(hours=2)
    assert _used(user, tomorrow) == 0
    assert _consume(user, tomorrow)
    assert _used(user, tomorrow) == 1


def test_refund_gives_one_message_back_and_never_goes_below_zero(user):
    assert _consume(user)
    with SessionLocal() as s:
        refund_message(s, user)
        refund_message(s, user)
    assert _used(user) == 0


def test_pro_plan_has_a_higher_limit(user):
    with SessionLocal() as s:
        s.get(User, user.id).plan = "pro"
        s.commit()
    with SessionLocal() as s:
        pro = s.get(User, user.id)
        assert sum(consume_message(s, pro) for _ in range(8)) == 5


def test_simultaneous_requests_cannot_overspend(user):
    results, barrier = [], threading.Barrier(20)

    def worker():
        barrier.wait()
        results.append(_consume(user))

    threads = [threading.Thread(target=worker) for _ in range(20)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sum(results) == 3


def test_snapshot_reports_remaining_and_reset_time(user):
    now = datetime(2026, 11, 3, 10, 0, tzinfo=timezone.utc)
    _consume(user, now)
    with SessionLocal() as s:
        snap = usage_snapshot(s, user.id, now)
    assert (snap["limit"], snap["used"], snap["remaining"]) == (3, 1, 2)
    assert snap["resets_at"] == "2026-11-04T00:00:00Z"


# ── the web app ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def web_app_module():
    import src.upload
    original = src.upload.cleanup_sessions
    src.upload.cleanup_sessions = lambda *a, **k: 0   # importing must not reap real sessions
    try:
        import web_app
    finally:
        src.upload.cleanup_sessions = original
    return web_app


@pytest.fixture
def app(web_app_module):
    return web_app_module.app


def _google_returns(monkeypatch, info=None, error=None):
    from accounts.auth import oauth

    async def fake(request):
        if error:
            raise error
        return {"userinfo": info}

    monkeypatch.setattr(oauth.google, "authorize_access_token", fake)


def _sign_in(app, monkeypatch, label):
    """A TestClient holding a real session cookie for a fresh Google user."""
    from fastapi.testclient import TestClient
    client = TestClient(app, follow_redirects=False)
    _google_returns(monkeypatch, {
        "sub": f"g-{RUN}-{label}", "email": f"{RUN}-{label}@example.com", "email_verified": True, "name": label,
    })
    assert client.get("/auth/callback").status_code == 303
    return client


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Stands in for the real pipeline so no model or API key is needed."""
    state = {"fail": False}

    class Pipeline:
        def stream(self, _state):
            if state["fail"]:
                raise RuntimeError("boom")
            yield {"router": {"route": "simple"}}
            yield {"self_rag_critic": {"final_answer": "An answer.", "confidence": "pass"}}

    graph = types.ModuleType("pipeline.graph")
    graph.get_pipeline = lambda: Pipeline()
    api = types.ModuleType("pipeline_api")
    api.shape_citations = lambda final: []
    api.run_pipeline = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")) if state["fail"] else {
        "answer": "An answer.", "citations": [], "route": "simple", "confidence": "pass"}
    pkg = types.ModuleType("pipeline")
    pkg.__path__ = []
    for name, mod in (("pipeline", pkg), ("pipeline.graph", graph), ("pipeline_api", api)):
        monkeypatch.setitem(sys.modules, name, mod)
    return state


def _events(response):
    return [json.loads(line[5:]) for line in response.text.splitlines() if line.startswith("data:")]


def test_endpoints_reject_visitors_who_are_not_signed_in(app):
    from fastapi.testclient import TestClient
    client = TestClient(app)
    assert client.get("/api/me").json()["authenticated"] is False
    assert client.post("/api/ask", json={"question": "hi"}).status_code == 401
    assert client.post("/api/ask_stream", json={"question": "hi"}).status_code == 401
    assert client.post("/api/upload", files={"file": ("a.txt", b"hello")}).status_code == 401
    assert client.get("/api/health").status_code == 200


def test_google_callback_creates_the_user_and_signs_them_in(app, monkeypatch):
    client = _sign_in(app, monkeypatch, "alice")
    me = client.get("/api/me").json()
    assert me["authenticated"] and me["email"] == f"{RUN}-alice@example.com"
    assert me["usage"]["limit"] == 3 and me["usage"]["plan"] == "demo"
    client.post("/auth/logout")
    assert client.get("/api/me").json()["authenticated"] is False


@pytest.mark.parametrize("info,error,expected", [
    ({"sub": "x", "email": "x@example.com", "email_verified": False}, None, "unverified_email"),
    ({"email": "x@example.com", "email_verified": True}, None, "unverified_email"),
    (None, OAuthError(error="access_denied"), "sign_in_failed"),
])
def test_callback_refuses_unverified_or_failed_sign_ins(app, monkeypatch, info, error, expected):
    from fastapi.testclient import TestClient
    client = TestClient(app, follow_redirects=False)
    _google_returns(monkeypatch, info, error)
    response = client.get("/auth/callback")
    assert response.headers["location"] == f"/?auth_error={expected}"
    assert client.get("/api/me").json()["authenticated"] is False


def test_disabled_account_is_locked_out_immediately(app, monkeypatch, fake_pipeline):
    client = _sign_in(app, monkeypatch, "mallory")
    with SessionLocal() as s:
        s.query(User).filter(User.email == f"{RUN}-mallory@example.com").one().is_disabled = True
        s.commit()
    assert client.get("/api/me").json()["authenticated"] is False
    assert client.post("/api/ask", json={"question": "hi"}).status_code == 403
    _google_returns(monkeypatch, {"sub": f"g-{RUN}-mallory", "email": f"{RUN}-mallory@example.com", "email_verified": True})
    assert client.get("/auth/callback").headers["location"] == "/?auth_error=disabled"


def test_fourth_message_of_the_day_is_refused_with_a_reset_time(app, monkeypatch, fake_pipeline):
    client = _sign_in(app, monkeypatch, "bob")
    remaining = []
    for _ in range(3):
        done = [e for e in _events(client.post("/api/ask_stream", json={"question": "q"})) if e["type"] == "done"][0]
        remaining.append(done["usage"]["remaining"])
    assert remaining == [2, 1, 0]

    blocked = client.post("/api/ask_stream", json={"question": "q"})
    assert blocked.status_code == 429
    body = blocked.json()
    assert body["code"] == "limit_reached" and body["resets_at"].endswith("Z")
    assert client.post("/api/ask", json={"question": "q"}).status_code == 429


def test_failed_questions_do_not_use_up_the_allowance(app, monkeypatch, fake_pipeline):
    client = _sign_in(app, monkeypatch, "carol")
    fake_pipeline["fail"] = True
    for _ in range(5):
        assert [e["type"] for e in _events(client.post("/api/ask_stream", json={"question": "q"}))][-1] == "error"
        assert client.post("/api/ask", json={"question": "q"}).status_code == 500
    assert client.get("/api/me").json()["usage"]["remaining"] == 3


def test_empty_question_is_not_charged(app, monkeypatch, fake_pipeline):
    client = _sign_in(app, monkeypatch, "dave")
    assert client.post("/api/ask", json={"question": "   "}).status_code == 400
    assert client.get("/api/me").json()["usage"]["remaining"] == 3


def test_uploaded_documents_belong_to_their_owner(app, web_app_module, monkeypatch, tmp_path, fake_pipeline):
    monkeypatch.setattr(web_app_module, "SESSIONS_ROOT", str(tmp_path))
    owner = _sign_in(app, monkeypatch, "erin")
    stranger = _sign_in(app, monkeypatch, "frank")

    def fake_ingest(path, session_root, display_filename=None):
        os.makedirs(os.path.join(session_root, "graph_db"), exist_ok=True)
        open(os.path.join(session_root, "graph_db", "graph.json"), "w").write("{}")
        return {"filename": display_filename, "chunks": 1, "pages": 1, "dirs": {}}

    monkeypatch.setattr("src.upload.ingest_upload", fake_ingest)
    uploaded = owner.post("/api/upload", files={"file": ("notes.txt", b"hello")})
    assert uploaded.status_code == 200
    session_id = uploaded.json()["session_id"]

    ask = lambda c: c.post("/api/ask", json={"question": "q", "session_id": session_id}).status_code
    assert ask(owner) == 200
    assert ask(stranger) == 404            # looks the same as a session that does not exist
    assert stranger.get("/api/me").json()["usage"]["remaining"] == 3   # and costs nothing

    os.remove(os.path.join(tmp_path, session_id, "owner"))
    assert ask(owner) == 404               # no recorded owner: nobody can use it
