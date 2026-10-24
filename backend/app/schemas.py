from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from pydantic import field_validator

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    simulated_mode: bool
    timestamp: str
    embedding_provider: Optional[str] = None
    llm_configured: Optional[bool] = None
    database_ok: Optional[bool] = None
    document_count: Optional[int] = None
    environment: Optional[str] = None


class QueryRequest(BaseModel):
    query: str = Field(
        ..., min_length=1,max_length=8000,
        description="User prompt or question to process"
    )
    similarity_threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
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
    @field_validator("metadata")
    @classmethod
    def validate_metadata_size(cls, value):
        if value is not None and len(str(value)) > 4000:
            raise ValueError("Metadata must not exceed 4000 characters.")
        return value
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
        description="Provider-reported cost. None when actual billing data is unavailable."
    )
    timestamp: str
    mode: str = Field(
        default="general",
        description="'general' = model knowledge only, no document retrieval.",
    )
    cache_match_type: Optional[str] = None


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
    # Fields added for the RAG pipeline. Older rows leave them unset.
    mode: str = "general"
    collection: Optional[str] = None
    outcome: Optional[str] = None
    cache_match_type: Optional[str] = None
    cache_lookup_ms: Optional[float] = None
    embedding_ms: Optional[float] = None
    retrieval_ms: Optional[float] = None
    generation_ms: Optional[float] = None
    retrieved_chunk_count: Optional[int] = None
    top_retrieval_score: Optional[float] = None
    citation_count: Optional[int] = None
    error_category: Optional[str] = None
    retry_count: int = 0
    fallback_model: Optional[str] = None
    estimated_cost_avoided: float = 0.0
    token_source: Optional[str] = None


class ModelStats(BaseModel):
    requests: int
    avg_latency_ms: float
    total_estimated_cost: float
    input_tokens: int
    output_tokens: int


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
    error_count: int = 0
    error_rate: float = 0.0
    total_estimated_cost_avoided: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    mode_breakdown: Dict[str, int] = Field(default_factory=dict)
    outcome_breakdown: Dict[str, int] = Field(default_factory=dict)
    match_type_breakdown: Dict[str, int] = Field(default_factory=dict)
    error_category_breakdown: Dict[str, int] = Field(default_factory=dict)
    model_stats: Dict[str, ModelStats] = Field(default_factory=dict)
    # Averages over rows where the stage ran; None when no row measured it.
    avg_cache_lookup_ms: Optional[float] = None
    avg_embedding_ms: Optional[float] = None
    avg_retrieval_ms: Optional[float] = None
    avg_generation_ms: Optional[float] = None
    avg_latency_cache_hit_ms: Optional[float] = None
    avg_latency_cache_miss_ms: Optional[float] = None
    fallback_count: int = 0
    pricing_note: str = (
        "Costs are estimates from configured per-token prices, not provider billing. "
        "Cost avoided = estimated cost of the original generation that a cache hit reused."
    )


# ---------------------------------------------------------------- documents


class DocumentOut(BaseModel):
    document_id: str
    collection: str
    filename: str
    file_type: str
    size_bytes: int
    content_sha256: str
    version: int
    status: str
    error_message: Optional[str] = None
    page_count: int
    chunk_count: int
    char_count: int
    embedding_model: Optional[str] = None
    needs_reindex: bool = False
    created_at: str
    updated_at: str


class UploadResponse(BaseModel):
    document: DocumentOut
    outcome: str = Field(description="'created', 'replaced' or 'duplicate'")
    message: str


class CollectionOut(BaseModel):
    name: str
    document_count: int
    ready_count: int
    chunk_count: int


# ---------------------------------------------------------------------- RAG


class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    collection: str = Field(default="default", min_length=1, max_length=64)
    document_ids: Optional[List[str]] = Field(
        default=None, max_length=50,
        description="Restrict retrieval to these documents within the collection.",
    )
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    min_relevance: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    force_refresh: bool = False

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Query must not be blank.")
        return value

    @field_validator("document_ids")
    @classmethod
    def validate_ids(cls, value):
        if value is not None:
            value = sorted(set(v.strip() for v in value if v and v.strip()))
            if any(len(v) > 64 for v in value):
                raise ValueError("Document IDs must be at most 64 characters.")
        return value or None


class Citation(BaseModel):
    label: str
    document_id: str
    filename: str
    page_number: int
    chunk_id: str
    chunk_index: int
    document_version: int
    relevance_score: float
    excerpt: str


class RetrievedSource(BaseModel):
    label: Optional[str] = None
    document_id: str
    filename: str
    page_number: int
    chunk_id: str
    relevance_score: float
    in_prompt: bool
    cited: bool


class RetrievalInfo(BaseModel):
    top_k: int
    min_relevance: float
    retrieved_count: int
    relevant_count: int
    top_score: Optional[float] = None
    sources: List[RetrievedSource] = Field(default_factory=list)
    suspicious_passages: int = 0
    performed: bool = True


class CacheInfo(BaseModel):
    hit: bool
    match_type: Optional[str] = None
    similarity: Optional[float] = None
    lookup_ms: float = 0.0
    rejection_reason: Optional[str] = None
    original_request_id: Optional[str] = None
    original_timestamp: Optional[str] = None
    bypassed: bool = False


class RoutingInfo(BaseModel):
    selected_model: Optional[str] = None
    tier: Optional[str] = None
    reason: str
    policy: str
    factors: List[str] = Field(default_factory=list)
    fallback_used: bool = False
    retry_count: int = 0
    attempts: List[Dict[str, Any]] = Field(default_factory=list)


class UsageInfo(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    token_source: str = Field(
        default="none",
        description="'provider' = reported by the API; 'none' = no model call was made.",
    )
    estimated_cost: float = 0.0
    estimated_cost_avoided: float = 0.0
    actual_cost: Optional[float] = None


class LatencyInfo(BaseModel):
    total_ms: float
    embedding_ms: Optional[float] = None
    cache_lookup_ms: Optional[float] = None
    retrieval_ms: Optional[float] = None
    generation_ms: Optional[float] = None


class RAGQueryResponse(BaseModel):
    request_id: str
    mode: str = "rag"
    query: str
    collection: str
    answer: str
    outcome: str = Field(
        description="'answered', 'insufficient_evidence', 'uncited' or 'no_documents'"
    )
    citations: List[Citation]
    retrieval: RetrievalInfo
    cache: CacheInfo
    routing: RoutingInfo
    usage: UsageInfo
    latency: LatencyInfo
    timestamp: str
