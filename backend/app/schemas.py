from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    simulated_mode: bool
    timestamp: str

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User prompt or question to process")
    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override default similarity threshold for semantic cache matching"
    )
    force_refresh: bool = Field(
        default=False,
        description="Bypass cache lookup and generate a fresh response"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional client metadata (e.g. user ID, session ID)"
    )

class QueryResponse(BaseModel):
    request_id: str
    query: str
    response: str
    cache_hit: bool
    cached_similarity: Optional[float] = None
    is_cache_simulated: bool = True
    selected_model: str
    routing_reason: str
    latency_ms: float
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost: float
    actual_cost: Optional[float] = Field(
        default=None,
        description="Provider-reported cost. None in mock/development mode as no live billing APIs are called."
    )
    timestamp: str

class RequestRecord(BaseModel):
    request_id: str
    timestamp: str
    query: str
    response_preview: str
    cache_hit: bool
    cached_similarity: Optional[float] = None
    is_cache_simulated: bool = True
    selected_model: str
    routing_reason: str
    latency_ms: float
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost: float
    actual_cost: Optional[float] = None
    has_error: bool = False
    error_message: Optional[str] = None

class MetricsSummary(BaseModel):
    total_requests: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    total_estimated_spend: float
    p50_latency_ms: float
    p95_latency_ms: float
    avg_latency_ms: float
    routing_breakdown: Dict[str, int]
    cache_breakdown: Dict[str, int]
    is_simulated: bool = True
