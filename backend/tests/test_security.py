"""Security and reliability controls around the API."""

import sqlite3

from app.config import settings
from tests.conftest import upload


def test_reset_is_open_in_development(api):
    assert api.post("/api/metrics/reset").status_code == 200


def test_reset_is_disabled_in_production_without_token(api, monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "ADMIN_TOKEN", "")
    assert api.post("/api/metrics/reset").status_code == 403


def test_reset_requires_matching_token_in_production(api, monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "ADMIN_TOKEN", "s3cret-token")
    assert api.post("/api/metrics/reset", headers={"X-Admin-Token": "wrong"}).status_code == 403
    assert api.post("/api/metrics/reset", headers={"X-Admin-Token": "s3cret-token"}).status_code == 200


def test_client_request_id_is_echoed_only_when_safe(api):
    ok = api.get("/health", headers={"X-Request-ID": "trace-123"})
    assert ok.headers["X-Request-ID"] == "trace-123"
    bad = api.get("/health", headers={"X-Request-ID": "x\" injected=1 " + "a" * 100})
    assert bad.headers["X-Request-ID"] != "x\" injected=1 " + "a" * 100
    assert bad.headers["X-Content-Type-Options"] == "nosniff"


def test_cors_allows_configured_origin_only(api):
    allowed = settings.CORS_ORIGINS[0]
    pre = {"Access-Control-Request-Method": "POST"}
    res = api.options("/api/rag/query", headers={"Origin": allowed, **pre})
    assert res.headers.get("access-control-allow-origin") == allowed
    res = api.options("/api/rag/query", headers={"Origin": "https://evil.example", **pre})
    assert "access-control-allow-origin" not in res.headers


def test_telemetry_can_redact_query_text(api, monkeypatch, tmp_db_path):
    monkeypatch.setattr(settings, "TELEMETRY_STORE_QUERY_TEXT", False)
    upload(api, "notes.txt", b"Project codename Bluebird ships in March.")
    api.post("/api/rag/query", json={"query": "When does Project Bluebird ship?", "min_relevance": 0.1})
    conn = sqlite3.connect(tmp_db_path)
    try:
        query, preview = conn.execute(
            "SELECT query, response_preview FROM request_records"
        ).fetchone()
    finally:
        conn.close()
    assert "Bluebird" not in query and query.startswith("[redacted sha256:")
    assert "Bluebird" not in preview


def test_error_telemetry_does_not_store_provider_message(api):
    def boom(query, model_id):
        raise RuntimeError("provider said: user prompt was 'secret plans'")

    api.qs._generate_response = boom
    assert api.post("/api/query", json={"query": "secret plans"}).status_code == 502
    record = api.get("/api/metrics/requests").json()[0]
    assert "secret plans" not in record["error_message"]
    assert record["error_category"] == "provider_error"


def test_config_endpoint_exposes_no_secrets(api, monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gsk_should_never_leak")
    monkeypatch.setattr(settings, "ADMIN_TOKEN", "admin_should_never_leak")
    body = api.get("/api/config").text + api.get("/health").text
    assert "should_never_leak" not in body


def test_upload_rate_limit(api):
    statuses = [
        upload(api, f"f{i}.txt", f"document number {i} body text".encode()).status_code
        for i in range(11)
    ]
    assert statuses[:10] == [202] * 10
    assert statuses[10] == 429
