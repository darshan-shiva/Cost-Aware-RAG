
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from groq import Groq

from ..config import settings
from ..schemas import QueryRequest, QueryResponse, RequestRecord
from ..cache import SemanticCache, CacheLookupResult, get_embedding_provider
from ..routing import CostAwareRouter
from ..metrics import SQLiteMetricsStore


class QueryService:
    """Semantic caching, real Groq generation, routing, and SQLite telemetry."""

    def __init__(
        self,
        cache: Optional[SemanticCache] = None,
        router: Optional[CostAwareRouter] = None,
        metrics_store: Optional[SQLiteMetricsStore] = None,
    ):
        if cache is None:
            if settings.SIMULATED_CACHE:
                from ..cache.embeddings import SimulatedEmbeddingProvider
                provider = SimulatedEmbeddingProvider()
            else:
                provider = get_embedding_provider(
                    provider_type=settings.EMBEDDING_PROVIDER,
                    model_name=settings.EMBEDDING_MODEL_NAME,
                )

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

        self.client = None
        if settings.LLM_PROVIDER.lower() == "groq":
            if settings.GROQ_API_KEY:
                self.client = Groq(
                    api_key=settings.GROQ_API_KEY,
                    timeout=settings.LLM_TIMEOUT_SECONDS,
                    max_retries=1,
                )
            else:
                raise RuntimeError(
                    "Groq is selected but RAG_GROQ_API_KEY is not configured."
                )
        else:
            raise ValueError(
                f"Unsupported LLM provider: {settings.LLM_PROVIDER}"
            )

        provider = self.cache.embedding_provider
        print(
            f"Embedding provider: {provider.provider_name} | "
            f"Simulated: {provider.is_simulated} | "
            f"Dimension: {provider.dimension}",
            flush=True,
        )

    def _generate_response(self, query: str, model_id: str):
        """Return generated text and provider-reported token counts."""
        result = self.client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer the user's question accurately and clearly. "
                        "For complex questions, structure the answer logically. "
                        "If you are uncertain, say so."
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_completion_tokens=settings.LLM_MAX_COMPLETION_TOKENS,
        )

        choice = result.choices[0]
        response_text = (choice.message.content or "").strip()

        if not response_text:
            raise RuntimeError(
                "The model returned no visible text. "
                f"Finish reason: {choice.finish_reason!r}. "
                "Try increasing RAG_LLM_MAX_COMPLETION_TOKENS."
            )

        usage = result.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        return response_text, input_tokens, output_tokens

    def process_query(self, req: QueryRequest) -> QueryResponse:
        start_time = time.perf_counter()
        request_id = str(uuid.uuid4())
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

            total_latency_ms = round(
                (time.perf_counter() - start_time) * 1000.0, 2
            )
            in_tokens = max(1, int(len(req.query.split()) * 1.3))
            out_tokens = max(1, int(len(response_text.split()) * 1.3))

            routing_reason = (
                f"Cache Hit: Semantically matched existing query "
                f"'{cached.query[:40]}...' "
                f"(cosine similarity: {cache_result.similarity_score:.4f}). "
                "LLM inference bypassed."
            )

            record = RequestRecord(
                request_id=request_id,
                timestamp=now_iso,
                query=req.query,
                response_preview=response_text[:120],
                cache_hit=True,
                cached_similarity=cache_result.similarity_score,
                is_cache_simulated=cache_result.is_simulated,
                selected_model=model_used,
                routing_reason=routing_reason,
                latency_ms=total_latency_ms,
                estimated_input_tokens=in_tokens,
                estimated_output_tokens=out_tokens,
                estimated_cost=0.0,
                actual_cost=None,
                has_error=False,
            )
            self.metrics_store.record_request(record)

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
                estimated_input_tokens=in_tokens,
                estimated_output_tokens=out_tokens,
                estimated_cost=0.0,
                actual_cost=None,
                timestamp=now_iso,
            )

        # Cache miss: route the query to a real model.
        decision = self.router.route(req.query)
        selected_model = decision.selected_model

        try:
            response_text, in_tokens, out_tokens = self._generate_response(
                req.query,
                selected_model.model_id,
            )
        except Exception as exc:
            # Record the failure, but never cache it as a successful answer.
            total_latency_ms = round(
                (time.perf_counter() - start_time) * 1000.0, 2
            )
            fallback_in_tokens = max(
                1, int(len(req.query.split()) * 1.3)
            )

            error_record = RequestRecord(
                request_id=request_id,
                timestamp=now_iso,
                query=req.query,
                response_preview="",
                cache_hit=False,
                cached_similarity=cache_result.similarity_score,
                is_cache_simulated=cache_result.is_simulated,
                selected_model=selected_model.model_id,
                routing_reason=decision.routing_reason,
                latency_ms=total_latency_ms,
                estimated_input_tokens=fallback_in_tokens,
                estimated_output_tokens=0,
                estimated_cost=0.0,
                actual_cost=None,
                has_error=True,
                error_message=str(exc)[:500],
            )
            self.metrics_store.record_request(error_record)
            raise

        total_latency_ms = round(
            (time.perf_counter() - start_time) * 1000.0, 2
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
            },
        )

        record = RequestRecord(
            request_id=request_id,
            timestamp=now_iso,
            query=req.query,
            response_preview=response_text[:120],
            cache_hit=False,
            cached_similarity=cache_result.similarity_score,
            is_cache_simulated=cache_result.is_simulated,
            selected_model=selected_model.model_id,
            routing_reason=decision.routing_reason,
            latency_ms=total_latency_ms,
            estimated_input_tokens=in_tokens,
            estimated_output_tokens=out_tokens,
            estimated_cost=estimated_cost,
            actual_cost=None,
            has_error=False,
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
            routing_reason=decision.routing_reason,
            latency_ms=total_latency_ms,
            estimated_input_tokens=in_tokens,
            estimated_output_tokens=out_tokens,
            estimated_cost=estimated_cost,
            actual_cost=None,
            timestamp=now_iso,
        )
