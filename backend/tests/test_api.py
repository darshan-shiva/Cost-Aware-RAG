import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.api.routes import get_query_service
from app.services.query_service import QueryService
from app.cache.semantic import SemanticCache
from app.cache.embeddings import SentenceTransformersEmbeddingProvider
from app.metrics.storage import SQLiteMetricsStore
import tempfile
import os

@pytest.fixture
def client():
    # Setup isolated test database and service with real local embedding model
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    test_store = SQLiteMetricsStore(db_path=path)
    test_provider = SentenceTransformersEmbeddingProvider("all-MiniLM-L6-v2")
    test_cache = SemanticCache(
        embedding_provider=test_provider,
        default_threshold=0.80,
        db_path=path,
    )
    test_service = QueryService(
        cache=test_cache,
        metrics_store=test_store,
    )

    app.dependency_overrides[get_query_service] = lambda: test_service
    
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
        
    app.dependency_overrides.clear()
    try:
        if os.path.exists(path):
            os.remove(path)
    except PermissionError:
        pass

def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "version" in data
    # Genuine embeddings mean simulated_mode is False
    assert data["simulated_mode"] is False

def test_query_flow_cache_miss_then_hit(client):
    query_payload = {"query": "What is semantic caching in large language models?"}
    
    # 1. First query is a cache miss
    res1 = client.post("/api/query", json=query_payload)
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["cache_hit"] is False
    assert data1["request_id"] is not None
    assert data1["is_cache_simulated"] is False
    assert data1["estimated_cost"] > 0.0
    assert data1["latency_ms"] > 0.0

    # 2. Check metrics summary
    metrics_res = client.get("/api/metrics/summary")
    assert metrics_res.status_code == 200
    summary = metrics_res.json()
    assert summary["total_requests"] == 1
    assert summary["cache_hits"] == 0
    assert summary["cache_misses"] == 1
    assert summary["is_simulated"] is False

    # 3. Second identical query is a cache hit
    res2 = client.post("/api/query", json=query_payload)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["cache_hit"] is True
    assert data2["cached_similarity"] >= 0.99
    assert data2["is_cache_simulated"] is False
    assert data2["estimated_cost"] == 0.0  # Zero spend on cache hit!
    assert "Cache Hit" in data2["routing_reason"]

def test_semantic_paraphrase_hit_via_api(client):
    """Verify that a paraphrased query hits the cache over the HTTP API based on actual embeddings and threshold."""
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
