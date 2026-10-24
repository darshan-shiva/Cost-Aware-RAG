import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.api.routes import get_query_service
from app.services.query_service import QueryService
from app.cache.semantic import SemanticCache
from app.cache.embeddings import (
    SentenceTransformersEmbeddingProvider,
    SimulatedEmbeddingProvider,
)
from app.llm import LLMGateway
from app.metrics.storage import SQLiteMetricsStore
import tempfile
import os
from app.rate_limit import limiter


def _make_client(provider):
    # Isolated test database; the Groq call is mocked so no API key or network
    # access is needed and no usage is billed.
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    test_store = SQLiteMetricsStore(db_path=path)
    test_cache = SemanticCache(
        embedding_provider=provider,
        default_threshold=0.80,
        db_path=path,
    )
    test_service = QueryService(
        cache=test_cache,
        metrics_store=test_store,
        gateway=LLMGateway(client=None, max_completion_tokens=512),
    )
    test_service._generate_response = lambda query, model_id: (
        f"Mocked answer from {model_id}",
        42,
        17,
    )

    app.dependency_overrides[get_query_service] = lambda: test_service
    limiter.reset()
    return path


@pytest.fixture
def client():
    path = _make_client(SimulatedEmbeddingProvider(dimension=64))
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    try:
        if os.path.exists(path):
            os.remove(path)
    except PermissionError:
        pass


@pytest.fixture
def real_embedding_client():
    path = _make_client(SentenceTransformersEmbeddingProvider("all-MiniLM-L6-v2"))
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    try:
        os.remove(path)
    except OSError:
        pass


def test_health_check(api):
    res = api.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "version" in data
    # The test fixture uses the hashing embedder, which must be reported.
    assert data["simulated_mode"] is True
    assert data["llm_configured"] is False
    assert data["database_ok"] is True

def test_query_flow_cache_miss_then_hit(client):
    query_payload = {"query": "What is semantic caching in large language models?"}
    
    # 1. First query is a cache miss
    res1 = client.post("/api/query", json=query_payload)
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["cache_hit"] is False
    assert data1["request_id"] is not None
    assert data1["is_cache_simulated"] is True
    assert data1["estimated_cost"] > 0.0
    assert data1["mode"] == "general"
    assert data1["latency_ms"] > 0.0

    # 2. Check metrics summary
    metrics_res = client.get("/api/metrics/summary")
    assert metrics_res.status_code == 200
    summary = metrics_res.json()
    assert summary["total_requests"] == 1
    assert summary["cache_hits"] == 0
    assert summary["cache_misses"] == 1
    assert summary["is_simulated"] is True

    # 3. Second identical query is a cache hit
    res2 = client.post("/api/query", json=query_payload)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["cache_hit"] is True
    assert data2["cached_similarity"] >= 0.99
    assert data2["cache_match_type"] == "exact"
    assert data2["estimated_cost"] == 0.0  # Zero spend on cache hit!
    assert "Cache Hit" in data2["routing_reason"]

    # The avoided cost is the original generation's estimated cost.
    summary = client.get("/api/metrics/summary").json()
    assert summary["total_estimated_cost_avoided"] == pytest.approx(data1["estimated_cost"])

@pytest.mark.requires_embedding_model
def test_semantic_paraphrase_hit_via_api(real_embedding_client):
    """Verify that a paraphrased query hits the cache over the HTTP API based on actual embeddings and threshold."""
    client = real_embedding_client
    base_query = {"query": "What is semantic caching in large language models?"}
    res1 = client.post("/api/query", json=base_query)
    assert res1.status_code == 200
    assert res1.json()["cache_hit"] is False

    # Paraphrased query with 0.65 threshold
    paraphrase = {
        "query": "What is semantic caching in LLMs?",
        "similarity_threshold": 0.65,
    }
    res2 = client.post("/api/query", json=paraphrase)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["cache_hit"] is True
    assert data2["cached_similarity"] is not None
    assert data2["cached_similarity"] >= 0.65
    assert data2["estimated_cost"] == 0.0


def test_cache_bypass_via_force_refresh(client):
    """Verify force_refresh=True bypasses the semantic cache and performs a fresh generation."""
    query = {"query": "Explain model routing algorithms"}
    
    # Prime cache
    res1 = client.post("/api/query", json=query)
    assert res1.status_code == 200
    assert res1.json()["cache_hit"] is False

    # Regular repeat hits cache
    res2 = client.post("/api/query", json=query)
    assert res2.status_code == 200
    assert res2.json()["cache_hit"] is True

    # Force refresh bypasses cache
    res3 = client.post("/api/query", json={"query": "Explain model routing algorithms", "force_refresh": True})
    assert res3.status_code == 200
    assert res3.json()["cache_hit"] is False
    assert res3.json()["estimated_cost"] > 0.0

def test_llm_failure_is_recorded_in_metrics(client):
    """A failed LLM generation increments error count and error rate."""
    service = app.dependency_overrides[get_query_service]()

    def raise_llm_error(query, model_id):
        raise RuntimeError("Mocked provider failure")

    service._generate_response = raise_llm_error

    response = client.post(
        "/api/query",
        json={"query": "Trigger a controlled provider failure"},
    )
    # The API should return an error, not a successful answer.
    assert response.status_code >= 400

    # The failed request must be recorded in SQLite.
    summary_response = client.get("/api/metrics/summary")
    assert summary_response.status_code == 200

    summary = summary_response.json()
    assert summary["total_requests"] == 1
    assert summary["cache_misses"] == 1
    assert summary["error_count"] == 1
    assert summary["error_rate"] == 1.0
def test_query_rejects_oversized_input(client):
    response = client.post(
        "/api/query",
        json={"query": "a" * 8001},
    )

    assert response.status_code == 422

def test_empty_provider_response_is_handled_safely(client):
    from types import SimpleNamespace

    service = app.dependency_overrides[get_query_service]()

    def empty_response(query, model_id):
        raise RuntimeError("The model provider returned no completion choices.")

    service._generate_response = empty_response

    response = client.post(
        "/api/query",
        json={"query": "Test an empty provider response"},
    )

    assert response.status_code == 502
    assert "provider failed" in response.json()["detail"].lower()

    summary_response = client.get("/api/metrics/summary")
    assert summary_response.status_code == 200
    assert summary_response.json()["error_count"] == 1


def test_provider_failure_returns_safe_502(client):
    service = app.dependency_overrides[get_query_service]()

    def simulate_timeout(query, model_id):
        raise TimeoutError("Internal provider timeout details")

    service._generate_response = simulate_timeout

    response = client.post(
        "/api/query",
        json={"query": "Test provider timeout handling"},
    )

    assert response.status_code == 502
    assert "Internal provider timeout details" not in response.text
    assert "provider failed" in response.json()["detail"].lower()

    summary = client.get("/api/metrics/summary").json()
    assert summary["error_count"] == 1


def test_query_rejects_oversized_metadata(client):
    response = client.post(
        "/api/query",
        json={
            "query": "Test metadata validation",
            "metadata": {"payload": "a" * 5000},
        },
    )

    assert response.status_code == 422
def test_query_rate_limit_returns_429(client):
    from app.rate_limit import limiter

    service = app.dependency_overrides[get_query_service]()

    # Avoid real Groq calls during this test.
    service._generate_response = lambda query, model_id: (
        "Mocked response",
        10,
        5,
    )

    limiter.reset()

    try:
        statuses = []
        for _ in range(31):
            response = client.post(
                "/api/query",
                json={"query": "Rate limit verification test"},
            )
            statuses.append(response.status_code)

        assert statuses[:30] == [200] * 30
        assert statuses[30] == 429
    finally:
        limiter.reset()

def test_provider_timeout_returns_safe_502(client):
    import httpx

    service = app.dependency_overrides[get_query_service]()

    def simulate_timeout(query, model_id):
        raise httpx.TimeoutException("Internal timeout details")

    service._generate_response = simulate_timeout

    response = client.post(
        "/api/query",
        json={"query": "Test provider timeout handling"},
    )

    assert response.status_code == 502
    assert "Internal timeout details" not in response.text
    assert "provider failed" in response.json()["detail"].lower()

    summary = client.get("/api/metrics/summary").json()
    assert summary["error_count"] == 1