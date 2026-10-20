import time
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from ..config import settings
from ..schemas import QueryRequest, QueryResponse, RequestRecord
from ..cache import (
    SemanticCache,
    CacheLookupResult,
    get_embedding_provider,
)
from ..routing import CostAwareRouter, RoutingDecision, ModelProfile
from ..metrics import SQLiteMetricsStore

class QueryService:
    """
    Coordinates semantic cache lookup, cost-aware model routing,
    mock response synthesis, and SQLite metrics telemetry.
    """

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
        self.metrics_store = metrics_store or SQLiteMetricsStore(settings.DATABASE_PATH)


    def _generate_mock_response(self, query: str, decision: RoutingDecision) -> str:
        model = decision.selected_model
        if decision.estimated_complexity == "high":
            return (
                f"[Frontier Response ({model.model_id})]\n"
                f"Analysis for: '{query}'\n\n"
                f"1. Key Architectural Considerations: Evaluated multi-dimensional trade-offs with respect to computational overhead, latency guarantees, and semantic accuracy.\n"
                f"2. Synthetic Deductions: High-complexity query requires structured decomposition and contextual synthesis.\n"
                f"3. Recommendation: Deploy layered caching strategies paired with dynamic threshold routing to optimize inference expenditure."
            )
        else:
            return (
                f"[Fast Tier Response ({model.model_id})]\n"
                f"Direct response to '{query}': Processed efficiently via lightweight reasoning tier."
            )

    def process_query(self, req: QueryRequest) -> QueryResponse:
        start_time = time.perf_counter()
        request_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()

        # Step 1: Semantic Cache Lookup (unless force_refresh is requested)
        cache_result: CacheLookupResult = CacheLookupResult(is_hit=False)
        if not req.force_refresh:
            cache_result = self.cache.lookup(req.query, req.similarity_threshold)

        # Case A: Cache Hit
        if cache_result.is_hit and cache_result.matched_entry is not None:
            total_latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            cached = cache_result.matched_entry
            
            # Cache hits save LLM generation cost (incur $0.00 model inference spend)
            response_text = cached.response
            model_used = cached.metadata.get("model_id", "cache-hit")
            routing_reason = (
                f"Cache Hit: Semantically matched existing query '{cached.query[:40]}...' "
                f"(cosine similarity: {cache_result.similarity_score:.4f} >= threshold). LLM inference bypassed."
            )
            
            in_tokens = max(1, int(len(req.query.split()) * 1.3))
            out_tokens = max(1, int(len(response_text.split()) * 1.3))
            estimated_cost = 0.0  # Zero spend on cache hit!

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
                estimated_cost=estimated_cost,
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
                estimated_cost=estimated_cost,
                actual_cost=None,
                timestamp=now_iso,
            )

        # Case B: Cache Miss -> Cost-aware Model Routing
        decision = self.router.route(req.query)
        selected_model = decision.selected_model

        # Simulate tier-appropriate latency (lightweight vs frontier)
        # We add a brief sleep to mimic realistic generation timing
        simulated_delay = 0.06 if decision.estimated_complexity == "low" else 0.18
        time.sleep(simulated_delay)

        response_text = self._generate_mock_response(req.query, decision)
        total_latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        in_tokens = max(1, int(len(req.query.split()) * 1.3))
        out_tokens = max(1, int(len(response_text.split()) * 1.3))
        estimated_cost = self.router.estimate_cost(selected_model, in_tokens, out_tokens)

        # Store into semantic cache for future queries
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
