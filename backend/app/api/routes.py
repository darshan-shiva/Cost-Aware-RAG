from fastapi import APIRouter, Depends, Query
from datetime import datetime, timezone
from typing import List
from fastapi import APIRouter, Depends, Query, Request
from ..config import settings
from ..schemas import HealthResponse, QueryRequest, QueryResponse, RequestRecord, MetricsSummary
from ..services.query_service import QueryService
from ..rate_limit import limiter

router = APIRouter()

# Singleton or dependency-injected query service
_service_instance: QueryService | None = None

def get_query_service() -> QueryService:
    global _service_instance
    if _service_instance is None:
        _service_instance = QueryService()
    return _service_instance

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(
    service: QueryService = Depends(get_query_service),
):
    """Application health and status check."""
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        simulated_mode=service.cache.embedding_provider.is_simulated,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

@router.post("/api/query", response_model=QueryResponse, tags=["Query"])
@limiter.limit("30/minute")
def process_query(
    request: Request,
    query_request: QueryRequest,
    service: QueryService = Depends(get_query_service),
):
    """Process a query with rate limiting and request ID tracking."""
    return service.process_query(
        query_request,
        request_id=getattr(request.state, "request_id", None),
    )

@router.get("/api/metrics/summary", response_model=MetricsSummary, tags=["Metrics"])
def get_metrics_summary(
    service: QueryService = Depends(get_query_service),
):
    """Retrieve aggregate telemetry, hit rates, p50/p95 latencies, and estimated spend."""
    summary = service.metrics_store.get_summary()
    summary.is_simulated = service.cache.embedding_provider.is_simulated
    return summary


@router.get("/api/metrics/requests", response_model=List[RequestRecord], tags=["Metrics"])
def get_recent_requests(
    limit: int = Query(default=50, ge=1, le=200, description="Max records to return"),
    service: QueryService = Depends(get_query_service),
):
    """Retrieve the most recent request execution records from SQLite."""
    return service.metrics_store.get_recent_requests(limit=limit)

@router.post("/api/metrics/reset", tags=["Metrics"])
def reset_metrics_and_cache(
    service: QueryService = Depends(get_query_service),
):
    """Utility endpoint to clear local metrics and in-memory cache for a clean state."""
    service.cache.clear()
    service.metrics_store.clear()
    return {"message": "Cache and metrics reset successfully."}
