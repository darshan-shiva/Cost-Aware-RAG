import os
import tempfile
import pytest
from app.metrics.storage import SQLiteMetricsStore
from app.schemas import RequestRecord

@pytest.fixture
def temp_store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = SQLiteMetricsStore(db_path=path)
    yield store
    try:
        if os.path.exists(path):
            os.remove(path)
    except PermissionError:
        pass


def test_metrics_empty_store_summary(temp_store):
    summary = temp_store.get_summary()
    assert summary.total_requests == 0
    assert summary.cache_hits == 0
    assert summary.cache_misses == 0
    assert summary.cache_hit_rate == 0.0
    assert summary.total_estimated_spend == 0.0
    assert summary.p50_latency_ms == 0.0
    assert summary.p95_latency_ms == 0.0

def test_metrics_record_and_percentiles(temp_store):
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    
    for i, lat in enumerate(latencies):
        rec = RequestRecord(
            request_id=f"req-{i}",
            timestamp="2026-10-20T12:00:00Z",
            query=f"query {i}",
            response_preview=f"response preview {i}",
            cache_hit=(i % 2 == 0),
            cached_similarity=0.95 if (i % 2 == 0) else None,
            is_cache_simulated=True,
            selected_model="llama-3.1-8b" if i < 7 else "gpt-4o",
            routing_reason="test routing",
            latency_ms=lat,
            estimated_input_tokens=10,
            estimated_output_tokens=20,
            estimated_cost=0.001 * i,
            actual_cost=None,
            has_error=False,
        )
        temp_store.record_request(rec)

    records = temp_store.get_recent_requests(limit=5)
    assert len(records) == 5
    assert records[0].request_id == "req-9"

    summary = temp_store.get_summary()
    assert summary.total_requests == 10
    assert summary.cache_hits == 5
    assert summary.cache_misses == 5
    assert summary.cache_hit_rate == 0.5
    assert summary.p50_latency_ms == 55.0  # (50 + 60)/2
    assert summary.p95_latency_ms == 100.0
    assert summary.routing_breakdown["llama-3.1-8b"] == 7
    assert summary.routing_breakdown["gpt-4o"] == 3
