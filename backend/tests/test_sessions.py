"""Per-visitor session isolation and cleanup, through the real session
manager and HTTP API. Each workspace uses simulated embeddings and a
FakeLLM, so no network, key or cost is involved."""

import glob
import os

import pytest

from app.api import routes
from app.config import settings
from app.memdb import InMemoryDatabase, WorkspaceClosed
from tests.conftest import new_session, upload
from tests.pdf_utils import make_pdf

MANUAL_PDF = make_pdf([
    "Orion Lab Manual. Experiment 1: Measure the resistance with the Orion R7 meter.",
    "Experiment 2: The Orion R7 meter must be calibrated every 90 days by the lab technician.",
])
QUESTION = "How often must the Orion R7 meter be calibrated?"


def ask(client, headers, query=QUESTION, **kwargs):
    res = client.post(
        "/api/rag/query",
        json={"query": query, "min_relevance": 0.15, **kwargs},
        headers=headers,
    )
    assert res.status_code == 200, res.text
    return res.json()


@pytest.fixture
def visitor_a(sessions):
    client = sessions.client
    a = new_session(client)
    res = upload(client, "orion-manual.pdf", MANUAL_PDF, headers=a)
    assert res.status_code == 202, res.text
    doc_id = res.json()["document"]["document_id"]
    assert client.get(f"/api/documents/{doc_id}", headers=a).json()["status"] == "ready"
    body = ask(client, a)
    assert body["outcome"] == "answered"
    assert "90 days" in body["answer"]
    return {"headers": a, "doc_id": doc_id, "answer": body}


# ------------------------------------------------------- 1-3: A and B


def test_visitor_a_gets_a_grounded_cited_answer(sessions, visitor_a):
    body = visitor_a["answer"]
    assert body["citations"][0]["filename"] == "orion-manual.pdf"
    assert body["citations"][0]["page_number"] == 2
    assert body["usage"]["estimated_cost"] > 0
    assert body["routing"]["selected_model"]


def test_visitor_b_starts_with_an_empty_workspace(sessions, visitor_a):
    client = sessions.client
    b = new_session(client)
    assert client.get("/api/documents", headers=b).json() == []
    assert client.get("/api/collections", headers=b).json() == []
    assert client.get("/api/metrics/requests", headers=b).json() == []
    assert client.get("/api/metrics/summary", headers=b).json()["total_requests"] == 0


def test_visitor_b_cannot_reach_visitor_a_data_through_the_api(sessions, visitor_a):
    client = sessions.client
    b = new_session(client)
    doc_id = visitor_a["doc_id"]
    assert client.get(f"/api/documents/{doc_id}", headers=b).status_code == 404
    assert client.delete(f"/api/documents/{doc_id}", headers=b).status_code == 404
    assert client.post(f"/api/documents/{doc_id}/reindex", headers=b).status_code == 404
    assert client.post(
        "/api/rag/query", json={"query": QUESTION, "document_ids": [doc_id]}, headers=b
    ).status_code == 404

    body = ask(client, b)
    assert body["outcome"] == "no_documents"
    assert body["cache"]["hit"] is False
    assert "90 days" not in body["answer"]
    assert body["citations"] == []

    # A's document is untouched by B's attempts.
    a = visitor_a["headers"]
    assert client.get(f"/api/documents/{doc_id}", headers=a).json()["status"] == "ready"


def test_requests_without_a_valid_session_are_rejected(sessions, visitor_a):
    client = sessions.client
    forged = {"X-Session-Token": "A" * 43}
    for headers in ({}, forged, {"X-Session-Token": "short"}, {"X-Session-Token": "x" * 500}):
        assert client.get("/api/documents", headers=headers).status_code == 401
        assert client.post("/api/rag/query", json={"query": QUESTION}, headers=headers).status_code == 401
        assert client.get("/api/metrics/requests", headers=headers).status_code == 401
        assert upload(client, "x.txt", b"some text here", headers=headers).status_code == 401
    # A guessed token that looks valid still cannot delete A's session.
    assert client.delete("/api/session", headers=forged).status_code == 401
    assert client.get("/api/documents", headers=visitor_a["headers"]).status_code == 200


def test_tokens_are_server_issued_unique_and_not_stored(sessions):
    client = sessions.client
    tokens = {new_session(client)["X-Session-Token"] for _ in range(5)}
    assert len(tokens) == 5
    stored = set(sessions.manager._sessions)
    assert not (tokens & stored)  # only hashes are kept
    # GET /api/session does not echo the token back.
    headers = {"X-Session-Token": next(iter(tokens))}
    assert client.get("/api/session", headers=headers).json()["session_token"] is None


def test_health_reports_no_visitor_data(sessions, visitor_a):
    health = sessions.client.get("/health").json()
    assert health["document_count"] is None
    assert health["active_sessions"] == 1


# ----------------------------------------------------- 4: refresh = new


def test_a_new_session_after_refresh_is_empty(sessions, visitor_a):
    # The frontend keeps the token only in memory, so a refresh starts a
    # new session; the old one stays private until it expires.
    client = sessions.client
    fresh = new_session(client)
    assert client.get("/api/documents", headers=fresh).json() == []
    assert ask(client, fresh)["outcome"] == "no_documents"


# ------------------------------------------------------------ 5: reset


def test_reset_deletes_the_session_and_its_data(sessions, visitor_a):
    client = sessions.client
    a = visitor_a["headers"]
    workspace = sessions.manager.get(a["X-Session-Token"])
    assert client.delete("/api/session", headers=a).status_code == 200
    assert client.get("/api/documents", headers=a).status_code == 401
    assert sessions.manager.active_count == 0
    assert workspace.db.closed
    with pytest.raises(WorkspaceClosed):
        workspace.rag.store.list(None)


def test_beacon_close_ends_the_session(sessions, visitor_a):
    client = sessions.client
    a = visitor_a["headers"]
    res = client.post("/api/session/close", content=a["X-Session-Token"],
                      headers={"Content-Type": "text/plain"})
    assert res.status_code == 204
    assert client.get("/api/documents", headers=a).status_code == 401
    # Unknown tokens get the same response.
    assert client.post("/api/session/close", content="nope").status_code == 204


def test_idle_and_old_sessions_expire_and_are_freed(sessions, visitor_a):
    client = sessions.client
    a = visitor_a["headers"]
    workspace = sessions.manager.get(a["X-Session-Token"])
    sessions.now += 1800  # idle TTL
    assert client.get("/api/documents", headers=a).status_code == 401
    assert workspace.db.closed

    b = new_session(client)
    for _ in range(13):  # stay active, but exceed the 6 h maximum age
        sessions.now += 1700
        client.get("/api/session", headers=b)
    assert client.get("/api/documents", headers=b).status_code == 401


def test_session_count_is_bounded(sessions):
    sessions.manager.max_active = 3
    client = sessions.client
    first = new_session(client)
    for _ in range(3):
        new_session(client)
    assert sessions.manager.active_count == 3
    assert client.get("/api/documents", headers=first).status_code == 401


def test_per_session_upload_quota(sessions, monkeypatch):
    monkeypatch.setattr(settings, "SESSION_MAX_DOCUMENTS", 2)
    client = sessions.client
    a = new_session(client)
    assert upload(client, "one.txt", b"first document text", headers=a).status_code == 202
    assert upload(client, "two.txt", b"second document text", headers=a).status_code == 202
    assert upload(client, "three.txt", b"third document text", headers=a).status_code == 409
    # Replacing an existing file is still allowed.
    assert upload(client, "two.txt", b"second document, revised", headers=a).status_code == 202
    # Another session has its own quota.
    b = new_session(client)
    assert upload(client, "three.txt", b"third document text", headers=b).status_code == 202


def test_processing_after_reset_does_not_crash(sessions):
    client = sessions.client
    a = new_session(client)
    workspace = sessions.manager.get(a["X-Session-Token"])
    sessions.manager.close(a["X-Session-Token"])
    # The background task would run after the session ended.
    routes._process_upload(workspace.rag.ingestion, "doc_missing", 1, b"text")


# ---------------------------------------------------------- 6: restart


def test_restart_clears_all_session_data(sessions, visitor_a):
    client = sessions.client
    a = visitor_a["headers"]
    old_manager = sessions.manager
    # A restart builds a fresh manager; nothing is reloaded from anywhere.
    old_manager.close_all()
    from tests.conftest import SessionHarness

    restarted = SessionHarness()
    from app.main import app

    app.dependency_overrides[routes.get_session_manager] = lambda: restarted.manager
    assert client.get("/api/documents", headers=a).status_code == 401
    b = new_session(client)
    assert client.get("/api/documents", headers=b).json() == []
    restarted.manager.close_all()


def test_session_data_never_touches_disk(sessions, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = set(glob.glob(os.path.join(os.path.dirname(settings.DATABASE_PATH), "*.db*")))
    client = sessions.client
    a = new_session(client)
    upload(client, "orion-manual.pdf", MANUAL_PDF, headers=a)
    ask(client, a)
    after = set(glob.glob(os.path.join(os.path.dirname(settings.DATABASE_PATH), "*.db*")))
    assert after == before
    assert list(tmp_path.iterdir()) == []
    workspace = sessions.manager.get(a["X-Session-Token"])
    assert isinstance(workspace.db, InMemoryDatabase)


# ---------------------------------------------------- 7: cache isolation


def test_cache_hits_never_cross_sessions(sessions, visitor_a):
    client = sessions.client
    a = visitor_a["headers"]
    # A's repeat is served from A's own cache.
    assert ask(client, a)["cache"]["hit"] is True

    # B uploads the very same file and asks the very same question: B must
    # get a freshly generated answer from B's own workspace, not A's entry.
    b = new_session(client)
    upload(client, "orion-manual.pdf", MANUAL_PDF, headers=b)
    body = ask(client, b)
    assert body["cache"]["hit"] is False
    b_fake = sessions.fakes[-1]
    assert len(b_fake.calls) == 1

    # General (non-document) mode is per session too.
    for headers in (a, b):
        client.post("/api/query", json={"query": "What is retrieval augmented generation?"}, headers=headers)
    general_b = client.get("/api/metrics/requests", params={"mode": "general"}, headers=b).json()
    assert len(general_b) == 1 and general_b[0]["cache_hit"] is False


def test_metrics_are_per_session(sessions, visitor_a):
    client = sessions.client
    a = visitor_a["headers"]
    b = new_session(client)
    ask(client, b)
    a_records = client.get("/api/metrics/requests", headers=a).json()
    b_records = client.get("/api/metrics/requests", headers=b).json()
    assert [r["outcome"] for r in a_records] == ["answered"]
    assert [r["outcome"] for r in b_records] == ["no_documents"]
    assert QUESTION in a_records[0]["query"] or not settings.TELEMETRY_STORE_QUERY_TEXT


# ------------------------------------------------- 8: features still work


def test_rag_citations_routing_and_costs_still_work_in_a_session(sessions, visitor_a):
    client = sessions.client
    a = visitor_a["headers"]
    compare = ask(client, a, "Compare the purpose of experiment 1 and experiment 2 in the Orion manual, and explain why calibration matters.")
    assert compare["routing"]["selected_model"] == sessions.shared.router.complex.model_id
    assert compare["citations"]
    summary = client.get("/api/metrics/summary", params={"mode": "rag"}, headers=a).json()
    assert summary["total_requests"] == 2
    assert summary["total_estimated_spend"] > 0
    assert client.delete(f"/api/documents/{visitor_a['doc_id']}", headers=a).status_code == 200
    assert client.get("/api/documents", headers=a).json() == []


def test_in_memory_database_rolls_back_unfinished_work():
    db = InMemoryDatabase()
    with db.connection() as conn:
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.commit()
    with db.connection() as conn:
        conn.execute("INSERT INTO t VALUES (1)")  # never committed
    with db.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 0
    db.close()
    with pytest.raises(WorkspaceClosed):
        with db.connection():
            pass
