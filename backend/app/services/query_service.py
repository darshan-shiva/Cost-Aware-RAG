import time
import uuid
from datetime import datetime, timezone
from typing import Optional
import logging
logger = logging.getLogger("cost_aware_rag.service")
from groq import Groq
from ..config import settings
from ..schemas import QueryRequest, QueryResponse, RequestRecord
from ..cache import SemanticCache, CacheLookupResult, get_embedding_provider
from ..routing import CostAwareRouter
from ..metrics import SQLiteMetricsStore
from ..metrics.privacy import telemetry_text
from ..llm import GenerationFailed, LLMGateway
from ..llm.gateway import NOT_CONFIGURED
from fastapi import HTTPException

GENERAL_SYSTEM_PROMPT = (
    "Answer the user's question accurately and clearly. "
    "For complex questions, structure the answer logically. "
    "If you are uncertain, say so."
)


def build_groq_client() -> Optional[Groq]:
    if settings.LLM_PROVIDER.lower() != "groq":
        raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")
    if not settings.GROQ_API_KEY:
        return None
    return Groq(
        api_key=settings.GROQ_API_KEY,
        timeout=settings.LLM_TIMEOUT_SECONDS,
        max_retries=settings.LLM_MAX_RETRIES,
    )


def build_embedding_provider():
    if settings.SIMULATED_CACHE:
        from ..cache.embeddings import SimulatedEmbeddingProvider
        return SimulatedEmbeddingProvider(settings.SIMULATED_EMBEDDING_DIM)
    return get_embedding_provider(
        provider_type=settings.EMBEDDING_PROVIDER,
        model_name=settings.EMBEDDING_MODEL_NAME,
        dimension=settings.SIMULATED_EMBEDDING_DIM,
    )


def build_llm_gateway() -> LLMGateway:
    return LLMGateway(
        client=build_groq_client(),
        max_completion_tokens=settings.LLM_MAX_COMPLETION_TOKENS,
        fallback_enabled=settings.LLM_FALLBACK_ENABLED,
        availability_ttl_seconds=settings.MODEL_AVAILABILITY_TTL_SECONDS,
    )


def provider_error_response(exc: GenerationFailed) -> HTTPException:
    """Map a generation failure to a safe HTTP error (no provider details)."""
    if exc.category == NOT_CONFIGURED:
        return HTTPException(
            status_code=503,
            detail="No language model provider is configured on the server (RAG_GROQ_API_KEY).",
        )
    return HTTPException(
        status_code=502,
        detail="The language model provider failed to process your request. Please try again.",
    )


class QueryService:
    """General-knowledge mode: semantic caching, Groq generation, routing and
    SQLite telemetry, without document retrieval. Also owns the components
    shared with the RAG pipeline."""

    def __init__(
        self,
        cache: Optional[SemanticCache] = None,
        router: Optional[CostAwareRouter] = None,
        metrics_store: Optional[SQLiteMetricsStore] = None,
        gateway: Optional[LLMGateway] = None,
    ):
        if cache is None:
            provider = build_embedding_provider()
            cache = SemanticCache(
                embedding_provider=provider,
                default_threshold=settings.SIMILARITY_THRESHOLD,
                max_entries=settings.CACHE_MAX_ENTRIES,
                db_path=settings.DATABASE_PATH,
            )

        self.cache = cache
        self.router = router or CostAwareRouter()
        self.metrics_store = metrics_store or SQLiteMetricsStore(
            settings.DATABASE_PATH
        )
        # A missing API key no longer stops the app from starting: document
        # management and telemetry still work, and generation returns 503.
        self.gateway = gateway or build_llm_gateway()
        provider = self.cache.embedding_provider
        logger.debug(
            f"Embedding provider: {provider.provider_name} | "
            f"Simulated: {provider.is_simulated} | "
            f"Dimension: {provider.dimension}"
        )

    @property
    def client(self):
        return self.gateway.client

    def _generate_response(self, query: str, model_id: str):
        """Return generated text and provider-reported token counts."""
        text, input_tokens, output_tokens, _ = self.gateway.complete(
            [
                {"role": "system", "content": GENERAL_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            model_id,
        )
        return text, input_tokens, output_tokens

    def process_query(
            self,
            req: QueryRequest,
            request_id: Optional[str] = None,
        ) -> QueryResponse:
        start_time = time.perf_counter()
        request_id = request_id or str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()

        cache_result = CacheLookupResult(is_hit=False)

        if not req.force_refresh:
            cache_result = self.cache.lookup(
                req.query,
                req.similarity_threshold,
            )

        # Cache hit: do not call the LLM.
        if cache_result.is_hit and cache_result.matched_entry is not None:
            cached = cache_result.matched_entry
            response_text = cached.response
            model_used = cached.metadata.get("model_id", "cache-hit")
            cost_avoided = float(cached.metadata.get("estimated_cost", 0.0) or 0.0)

            total_latency_ms = round(
                (time.perf_counter() - start_time) * 1000.0, 2
            )

            routing_reason = (
                f"Cache Hit ({cache_result.match_type}): matched existing query "
                f"'{cached.query[:40]}...' "
                f"(cosine similarity: {cache_result.similarity_score:.4f}). "
                "LLM inference bypassed."
            )

            record = RequestRecord(
                request_id=request_id,
                timestamp=now_iso,
                query=telemetry_text(req.query),
                response_preview=telemetry_text(response_text, 120),
                cache_hit=True,
                cached_similarity=cache_result.similarity_score,
                is_cache_simulated=cache_result.is_simulated,
                selected_model=model_used,
                routing_reason=routing_reason,
                latency_ms=total_latency_ms,
                # No tokens are consumed on a cache hit.
                estimated_input_tokens=0,
                estimated_output_tokens=0,
                estimated_cost=0.0,
                actual_cost=None,
                has_error=False,
                mode="general",
                outcome="answered",
                cache_match_type=cache_result.match_type,
                cache_lookup_ms=cache_result.lookup_latency_ms,
                estimated_cost_avoided=cost_avoided,
                token_source="none",
            )
            self.metrics_store.record_request(record)
            logger.info(
                "query_completed",
                extra={
                    "request_id": request_id,
                    "cache_hit": True,
                    "selected_model": model_used,
                    "latency_ms": total_latency_ms,
                },
            )

            return QueryResponse(
                request_id=request_id,
                query=req.query,
                response=response_text,
                cache_hit=True,
                cached_similarity=cache_result.similarity_score,
                is_cache_simulated=cache_result.is_simulated,
                selected_model=model_used,
                routing_reason=routing_reason,
                latency_ms=total_latency_ms,
                estimated_input_tokens=0,
                estimated_output_tokens=0,
                estimated_cost=0.0,
                actual_cost=None,
                timestamp=now_iso,
                cache_match_type=cache_result.match_type,
            )

        # Cache miss: route the query to a real model.
        decision = self.router.route(req.query)
        generation_start = time.perf_counter()

        try:
            result = self.gateway.generate(
                [],
                decision,
                generate_fn=lambda _messages, model_id: (
                    *self._generate_response(req.query, model_id),
                    None,
                ),
            )
        except GenerationFailed as exc:
            # Record the failure, but never cache it as a successful answer.
            total_latency_ms = round(
                (time.perf_counter() - start_time) * 1000.0, 2
            )
            error_record = RequestRecord(
                request_id=request_id,
                timestamp=now_iso,
                query=telemetry_text(req.query),
                response_preview="",
                cache_hit=False,
                cached_similarity=cache_result.similarity_score,
                is_cache_simulated=cache_result.is_simulated,
                selected_model=decision.selected_model.model_id,
                routing_reason=decision.routing_reason,
                latency_ms=total_latency_ms,
                estimated_input_tokens=0,
                estimated_output_tokens=0,
                estimated_cost=0.0,
                actual_cost=None,
                has_error=True,
                # The category is stored rather than raw provider text, which
                # can echo request content.
                error_message=f"{exc.category}: {type(exc.cause).__name__}",
                mode="general",
                outcome="error",
                cache_lookup_ms=cache_result.lookup_latency_ms,
                error_category=exc.category,
                retry_count=max(0, len(exc.attempts) - 1),
                token_source="none",
            )
            self.metrics_store.record_request(error_record)

            logger.error(
                "query_failed",
                extra={
                    "request_id": request_id,
                    "cache_hit": False,
                    "selected_model": decision.selected_model.model_id,
                    "latency_ms": total_latency_ms,
                    "error_type": exc.category,
                },
            )

            raise provider_error_response(exc) from exc

        generation_ms = round((time.perf_counter() - generation_start) * 1000.0, 2)
        total_latency_ms = round(
            (time.perf_counter() - start_time) * 1000.0, 2
        )
        selected_model = result.model
        response_text, in_tokens, out_tokens = result.text, result.input_tokens, result.output_tokens
        routing_reason = decision.routing_reason
        if result.fallback_used:
            routing_reason += (
                f" Fallback: {decision.selected_model.model_id} failed"
                f" ({result.attempts[0]['error_category']}); answered by {selected_model.model_id}."
            )

        # Provider-reported token counts; prices are configured estimates.
        estimated_cost = self.router.estimate_cost(
            selected_model,
            in_tokens,
            out_tokens,
        )

        # Cache only a successful, non-empty response.
        self.cache.store(
            query=req.query,
            response=response_text,
            metadata={
                "model_id": selected_model.model_id,
                "tier": selected_model.tier,
                "complexity": decision.estimated_complexity,
                "estimated_cost": estimated_cost,
                "request_id": request_id,
            },
        )

        record = RequestRecord(
            request_id=request_id,
            timestamp=now_iso,
            query=telemetry_text(req.query),
            response_preview=telemetry_text(response_text, 120),
            cache_hit=False,
            cached_similarity=cache_result.similarity_score,
            is_cache_simulated=cache_result.is_simulated,
            selected_model=selected_model.model_id,
            routing_reason=routing_reason,
            latency_ms=total_latency_ms,
            estimated_input_tokens=in_tokens,
            estimated_output_tokens=out_tokens,
            estimated_cost=estimated_cost,
            actual_cost=None,
            has_error=False,
            mode="general",
            outcome="answered",
            cache_lookup_ms=cache_result.lookup_latency_ms,
            generation_ms=generation_ms,
            retry_count=result.retry_count,
            fallback_model=selected_model.model_id if result.fallback_used else None,
            token_source="provider",
        )
        self.metrics_store.record_request(record)

        return QueryResponse(
            request_id=request_id,
            query=req.query,
            response=response_text,
            cache_hit=False,
            cached_similarity=cache_result.similarity_score,
            is_cache_simulated=cache_result.is_simulated,
            selected_model=selected_model.model_id,
            routing_reason=routing_reason,
            latency_ms=total_latency_ms,
            estimated_input_tokens=in_tokens,
            estimated_output_tokens=out_tokens,
            estimated_cost=estimated_cost,
            actual_cost=None,
            timestamp=now_iso,
        )
