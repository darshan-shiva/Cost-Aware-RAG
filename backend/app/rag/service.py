"""Document-grounded question answering.

Pipeline per request:
  1. embed the question once (reused for cache lookup and retrieval)
  2. cache lookup scoped to the collection / document filter and the current
     corpus fingerprint; exact matches are reused directly, semantic matches
     only if the new question retrieves overlapping evidence
  3. retrieve top-k chunks and drop those below the relevance threshold
  4. no relevant chunks -> insufficient-evidence answer, no model call
  5. build a grounded prompt, route, generate (one fallback at most)
  6. map [S#] citations back to real chunks, discard invalid ones
  7. cache answered / insufficient results; record telemetry
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from fastapi import HTTPException

from ..cache import CacheEntry, CacheLookupResult
from ..cache.guards import contrastive_terms_conflict, uncovered_terms_conflict
from ..config import settings
from ..documents import (
    DocumentStore,
    DocumentValidationError,
    IngestionService,
    RetrievedChunk,
    VectorIndex,
    validate_collection,
)
from ..llm import GenerationFailed
from ..metrics.privacy import telemetry_text
from ..schemas import (
    CacheInfo,
    Citation,
    LatencyInfo,
    RAGQueryRequest,
    RAGQueryResponse,
    RequestRecord,
    RetrievalInfo,
    RetrievedSource,
    RoutingInfo,
    UsageInfo,
)
from ..services.query_service import QueryService, provider_error_response
from .prompt import (
    build_messages,
    estimate_tokens,
    is_insufficient,
    looks_like_injection,
    parse_citations,
    strip_invalid_citations,
)

logger = logging.getLogger("cost_aware_rag.rag")

# Bump when the prompt or citation format changes so old cached answers,
# produced under different instructions, stop matching.
PROMPT_VERSION = "rag-v1"

INSUFFICIENT_MESSAGE = (
    "I could not find enough support for this question in the selected documents, "
    "so I am not answering it from them."
)
NO_DOCUMENTS_MESSAGE = (
    "There are no indexed documents in this collection yet. Upload a PDF or TXT "
    "file and wait for it to finish processing."
)
UNCITED_NOTE = (
    "The model's answer did not cite any of the retrieved sources, so it could not "
    "be verified against the documents."
)
EXCERPT_CHARS = 1200


def rag_scope(collection: str, document_ids: Optional[Sequence[str]]) -> str:
    docs = ",".join(sorted(document_ids)) if document_ids else "*"
    return f"rag:{collection}:{docs}"


class RAGService:
    def __init__(
        self,
        query_service: QueryService,
        document_store: DocumentStore,
        vector_index: VectorIndex,
        ingestion: IngestionService,
    ):
        self.qs = query_service
        self.store = document_store
        self.index = vector_index
        self.ingestion = ingestion
        self.ingestion.add_listener(self.on_collection_changed)

    # Convenience accessors for shared components.
    @property
    def cache(self):
        return self.qs.cache

    @property
    def router(self):
        return self.qs.router

    @property
    def embedding_provider(self):
        return self.qs.cache.embedding_provider

    def on_collection_changed(self, collection: str) -> None:
        """Drop the collection's vector matrix and every cached answer scoped
        to it. Fingerprint checks would already reject stale entries; deleting
        them also stops deleted documents' text living on in the cache."""
        self.index.invalidate(collection)
        removed = self.cache.invalidate_scope_prefix(f"rag:{collection}:")
        if removed:
            logger.info("rag_cache_invalidated", extra={"path": collection})

    def _generate(self, messages: List[Dict[str, str]], model_id: str) -> tuple:
        """One provider call; patched in tests."""
        return self.qs.gateway.complete(messages, model_id)

    def _fingerprint(
        self, collection: str, document_ids: Optional[Sequence[str]], top_k: int, min_rel: float
    ) -> Tuple[str, int]:
        parts = self.store.collection_fingerprint_parts(collection)
        if document_ids:
            wanted = set(document_ids)
            parts = [p for p in parts if p[0] in wanted]
        payload = json.dumps(
            {
                "docs": parts,
                "embedding": self.embedding_provider.provider_name,
                "top_k": top_k,
                "min_relevance": round(min_rel, 4),
                "prompt": PROMPT_VERSION,
                "chunking": [settings.CHUNK_SIZE_CHARS, settings.CHUNK_OVERLAP_CHARS],
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24], len(parts)

    @staticmethod
    def _citation(label_index: int, chunk: RetrievedChunk) -> Citation:
        c = chunk.chunk
        return Citation(
            label=f"S{label_index}",
            document_id=c.document_id,
            filename=c.filename,
            page_number=c.page_number,
            chunk_id=c.chunk_id,
            chunk_index=c.chunk_index,
            document_version=c.document_version,
            relevance_score=chunk.score,
            excerpt=c.text[:EXCERPT_CHARS],
        )

    def answer(self, req: RAGQueryRequest, request_id: Optional[str] = None) -> RAGQueryResponse:
        start = time.perf_counter()
        request_id = request_id or str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            collection = validate_collection(req.collection)
        except DocumentValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        top_k = min(req.top_k or settings.RETRIEVAL_TOP_K, settings.RETRIEVAL_MAX_TOP_K)
        min_rel = (
            req.min_relevance if req.min_relevance is not None else settings.RETRIEVAL_MIN_RELEVANCE
        )
        if req.document_ids:
            for doc_id in req.document_ids:
                doc = self.store.get(doc_id)
                if doc is None or doc.collection != collection:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Document {doc_id} was not found in collection '{collection}'.",
                    )

        fingerprint, ready_docs = self._fingerprint(collection, req.document_ids, top_k, min_rel)
        scope = rag_scope(collection, req.document_ids)
        provider = self.embedding_provider

        ctx = _RequestContext(
            request_id=request_id,
            timestamp=now_iso,
            query=req.query,
            collection=collection,
            top_k=top_k,
            min_relevance=min_rel,
            is_simulated=provider.is_simulated,
        )

        if ready_docs == 0:
            return self._finish(
                ctx, start, answer=NO_DOCUMENTS_MESSAGE, outcome="no_documents",
                retrieval=RetrievalInfo(
                    top_k=top_k, min_relevance=min_rel, retrieved_count=0,
                    relevant_count=0, performed=False,
                ),
                cache=CacheInfo(hit=False, bypassed=req.force_refresh),
                routing=RoutingInfo(reason="No model call: the collection has no indexed documents.",
                                    policy=self.router.policy),
            )

        t = time.perf_counter()
        query_embedding = provider.embed_text(req.query)
        ctx.embedding_ms = round((time.perf_counter() - t) * 1000, 2)

        retrieved: Optional[List[RetrievedChunk]] = None
        retrieval_ms = 0.0

        def retrieve() -> List[RetrievedChunk]:
            nonlocal retrieved, retrieval_ms
            if retrieved is None:
                t0 = time.perf_counter()
                retrieved = self.index.search(
                    collection, query_embedding, top_k, min_score=-1.0,
                    document_ids=req.document_ids,
                )
                retrieval_ms = round((time.perf_counter() - t0) * 1000, 2)
            return retrieved

        def evidence_validator(entry: CacheEntry, score: float) -> Optional[str]:
            # Compare the cached answer's evidence with the same number of
            # top-ranked passages for the new question. Comparing against the
            # whole relevant set would be too lenient: lookalike questions
            # often both clear the relevance bar for each other's passages.
            cached_ids = list(dict.fromkeys(entry.metadata.get("evidence_chunk_ids") or []))
            relevant_now = [r.chunk.chunk_id for r in retrieve() if r.score >= min_rel]
            if not cached_ids:
                return None if not relevant_now else "cached answer had no evidence but new query does"
            top_now = set(relevant_now[: len(cached_ids)])
            overlap = len(set(cached_ids) & top_now) / len(cached_ids)
            if overlap < settings.CACHE_MIN_EVIDENCE_OVERLAP:
                return f"different evidence retrieved (overlap {overlap:.2f})"
            evidence_text = " ".join(
                c.get("excerpt", "") for c in entry.metadata.get("citations") or []
            )
            return contrastive_terms_conflict(
                req.query, entry.query, evidence_text
            ) or uncovered_terms_conflict(
                req.query, entry.query, evidence_text, entry.response
            )

        cache_result = CacheLookupResult(is_hit=False)
        if not req.force_refresh:
            cache_result = self.cache.lookup(
                req.query,
                threshold=req.similarity_threshold,
                scope=scope,
                fingerprint=fingerprint,
                query_embedding=query_embedding,
                validator=evidence_validator,
            )
        ctx.cache_lookup_ms = cache_result.lookup_latency_ms

        if cache_result.is_hit and cache_result.matched_entry is not None:
            ctx.retrieval_ms = retrieval_ms if retrieved is not None else None
            try:
                return self._from_cache(ctx, start, cache_result, retrieved)
            except Exception as exc:
                # Unusable entry (stale citation, corrupt metadata): drop it and
                # answer fresh instead of failing the request.
                logger.warning("rag_cache_entry_unusable", extra={"error_type": type(exc).__name__})
                key = cache_result.matched_entry.key
                self.cache.invalidate(lambda e: e.key == key)
                cache_result = CacheLookupResult(
                    is_hit=False,
                    similarity_score=cache_result.similarity_score,
                    lookup_latency_ms=cache_result.lookup_latency_ms,
                    rejection_reason="cached entry was unusable and was discarded",
                )

        chunks = retrieve()
        ctx.retrieval_ms = retrieval_ms
        relevant = [r for r in chunks if r.score >= min_rel]
        cache_info = CacheInfo(
            hit=False,
            similarity=cache_result.similarity_score,
            lookup_ms=cache_result.lookup_latency_ms,
            rejection_reason=cache_result.rejection_reason,
            bypassed=req.force_refresh,
        )

        if not relevant:
            return self._finish(
                ctx, start, answer=INSUFFICIENT_MESSAGE, outcome="insufficient_evidence",
                retrieval=self._retrieval_info(ctx, chunks, [], set()),
                cache=cache_info,
                routing=RoutingInfo(
                    reason=f"No model call: no passage scored at or above the relevance threshold ({min_rel}).",
                    policy=self.router.policy,
                ),
            )

        budget = min(m.max_context_tokens for m in self.router.profiles) - settings.LLM_MAX_COMPLETION_TOKENS
        messages, included = build_messages(req.query, relevant, max(budget, 500))
        context_tokens = sum(estimate_tokens(m["content"]) for m in messages)
        source_docs = len({r.chunk.document_id for r in included})
        unavailable = self.qs.gateway.unavailable_models([m.model_id for m in self.router.profiles])
        decision = self.router.route(
            req.query,
            context_tokens=context_tokens,
            source_document_count=source_docs,
            unavailable_models=unavailable,
        )

        t = time.perf_counter()
        try:
            result = self.qs.gateway.generate(messages, decision, generate_fn=self._generate)
        except GenerationFailed as exc:
            ctx.generation_ms = round((time.perf_counter() - t) * 1000, 2)
            self._record_error(ctx, start, decision, exc, chunks, cache_info)
            raise provider_error_response(exc) from exc
        ctx.generation_ms = round((time.perf_counter() - t) * 1000, 2)

        raw = result.text
        if is_insufficient(raw):
            outcome, answer, cited_labels = "insufficient_evidence", INSUFFICIENT_MESSAGE, []
        else:
            valid, invalid = parse_citations(raw, len(included))
            answer = strip_invalid_citations(raw, invalid).strip()
            cited_labels = valid
            outcome = "answered" if valid else "uncited"
            if invalid:
                logger.warning("invalid_citations_removed", extra={"request_id": request_id})

        citations = [self._citation(n, included[n - 1]) for n in cited_labels]
        cited_ids = {c.chunk_id for c in citations}
        cost = self.router.estimate_cost(result.model, result.input_tokens, result.output_tokens)

        routing_reason = decision.routing_reason
        if result.fallback_used:
            routing_reason += (
                f" Fallback: {decision.selected_model.model_id} failed"
                f" ({result.attempts[0]['error_category']}); answered by {result.model.model_id}."
            )
        routing = RoutingInfo(
            selected_model=result.model.model_id,
            tier=result.model.tier,
            reason=routing_reason,
            policy=decision.policy,
            factors=decision.factors,
            fallback_used=result.fallback_used,
            retry_count=result.retry_count,
            attempts=result.attempts,
        )
        usage = UsageInfo(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            token_source="provider",
            estimated_cost=cost,
        )
        retrieval = self._retrieval_info(ctx, chunks, included, cited_ids)

        if outcome in ("answered", "insufficient_evidence"):
            evidence = (
                [c.chunk_id for c in citations]
                if outcome == "answered"
                else [r.chunk.chunk_id for r in included]
            )
            self.cache.store(
                query=req.query,
                response=answer,
                embedding=query_embedding,
                metadata={
                    "scope": scope,
                    "fingerprint": fingerprint,
                    "collection": collection,
                    "outcome": outcome,
                    "evidence_chunk_ids": evidence,
                    "citations": [c.model_dump() for c in citations],
                    "retrieval": retrieval.model_dump(),
                    "model_id": result.model.model_id,
                    "tier": result.model.tier,
                    "estimated_cost": cost,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "request_id": request_id,
                    "timestamp": now_iso,
                },
            )

        if outcome == "uncited":
            answer = f"{answer}\n\n[{UNCITED_NOTE}]"

        return self._finish(
            ctx, start, answer=answer, outcome=outcome, retrieval=retrieval,
            cache=cache_info, routing=routing, usage=usage, citations=citations,
        )

    # ----------------------------------------------------------------- parts

    def _retrieval_info(
        self,
        ctx: "_RequestContext",
        chunks: List[RetrievedChunk],
        included: List[RetrievedChunk],
        cited_ids: set,
    ) -> RetrievalInfo:
        labels = {r.chunk.chunk_id: f"S{i + 1}" for i, r in enumerate(included)}
        sources = [
            RetrievedSource(
                label=labels.get(r.chunk.chunk_id),
                document_id=r.chunk.document_id,
                filename=r.chunk.filename,
                page_number=r.chunk.page_number,
                chunk_id=r.chunk.chunk_id,
                relevance_score=r.score,
                in_prompt=r.chunk.chunk_id in labels,
                cited=r.chunk.chunk_id in cited_ids,
            )
            for r in chunks
        ]
        return RetrievalInfo(
            top_k=ctx.top_k,
            min_relevance=ctx.min_relevance,
            retrieved_count=len(chunks),
            relevant_count=sum(1 for r in chunks if r.score >= ctx.min_relevance),
            top_score=chunks[0].score if chunks else None,
            sources=sources,
            suspicious_passages=sum(1 for r in included if looks_like_injection(r.chunk.text)),
        )

    def _from_cache(
        self,
        ctx: "_RequestContext",
        start: float,
        cache_result: CacheLookupResult,
        retrieved: Optional[List[RetrievedChunk]],
    ) -> RAGQueryResponse:
        entry = cache_result.matched_entry
        meta = entry.metadata
        citations = [Citation(**c) for c in meta.get("citations", [])]
        # Citations must still resolve to stored chunks. The fingerprint makes
        # this true by construction; checking keeps a corrupted entry from
        # ever surfacing a citation to a chunk that no longer exists.
        if citations:
            live = self.store.get_chunks([c.chunk_id for c in citations])
            if len(live) != len(citations):
                raise _StaleCacheEntry()
        cached_retrieval = meta.get("retrieval")
        retrieval = (
            RetrievalInfo(**cached_retrieval)
            if cached_retrieval
            else RetrievalInfo(top_k=ctx.top_k, min_relevance=ctx.min_relevance,
                               retrieved_count=0, relevant_count=0, performed=False)
        )
        retrieval.performed = retrieved is not None
        return self._finish(
            ctx,
            start,
            answer=entry.response,
            outcome=meta.get("outcome", "answered"),
            retrieval=retrieval,
            cache=CacheInfo(
                hit=True,
                match_type=cache_result.match_type,
                similarity=cache_result.similarity_score,
                lookup_ms=cache_result.lookup_latency_ms,
                original_request_id=meta.get("request_id"),
                original_timestamp=meta.get("timestamp"),
            ),
            routing=RoutingInfo(
                selected_model=meta.get("model_id"),
                tier=meta.get("tier"),
                reason=(
                    f"Cache hit ({cache_result.match_type}, similarity "
                    f"{cache_result.similarity_score:.4f}); no model call. Answer originally "
                    f"generated by {meta.get('model_id')}."
                ),
                policy=self.router.policy,
            ),
            usage=UsageInfo(
                token_source="none",
                estimated_cost=0.0,
                estimated_cost_avoided=float(meta.get("estimated_cost", 0.0) or 0.0),
            ),
            citations=citations,
        )

    def _finish(
        self,
        ctx: "_RequestContext",
        start: float,
        *,
        answer: str,
        outcome: str,
        retrieval: RetrievalInfo,
        cache: CacheInfo,
        routing: RoutingInfo,
        usage: Optional[UsageInfo] = None,
        citations: Optional[List[Citation]] = None,
    ) -> RAGQueryResponse:
        usage = usage or UsageInfo()
        citations = citations or []
        total_ms = round((time.perf_counter() - start) * 1000, 2)
        latency = LatencyInfo(
            total_ms=total_ms,
            embedding_ms=ctx.embedding_ms,
            cache_lookup_ms=ctx.cache_lookup_ms,
            retrieval_ms=ctx.retrieval_ms,
            generation_ms=ctx.generation_ms,
        )
        self.qs.metrics_store.record_request(
            RequestRecord(
                request_id=ctx.request_id,
                timestamp=ctx.timestamp,
                query=telemetry_text(ctx.query),
                response_preview=telemetry_text(answer, 120),
                cache_hit=cache.hit,
                cached_similarity=cache.similarity,
                is_cache_simulated=ctx.is_simulated,
                selected_model=routing.selected_model or "none",
                routing_reason=routing.reason,
                latency_ms=total_ms,
                estimated_input_tokens=usage.input_tokens,
                estimated_output_tokens=usage.output_tokens,
                estimated_cost=usage.estimated_cost,
                actual_cost=None,
                has_error=False,
                mode="rag",
                collection=ctx.collection,
                outcome=outcome,
                cache_match_type=cache.match_type,
                cache_lookup_ms=ctx.cache_lookup_ms,
                embedding_ms=ctx.embedding_ms,
                retrieval_ms=ctx.retrieval_ms,
                generation_ms=ctx.generation_ms,
                retrieved_chunk_count=retrieval.retrieved_count if retrieval.performed else None,
                top_retrieval_score=retrieval.top_score,
                citation_count=len(citations),
                retry_count=routing.retry_count,
                fallback_model=routing.selected_model if routing.fallback_used else None,
                estimated_cost_avoided=usage.estimated_cost_avoided,
                token_source=usage.token_source,
            )
        )
        logger.info(
            "rag_query_completed",
            extra={
                "request_id": ctx.request_id,
                "cache_hit": cache.hit,
                "selected_model": routing.selected_model,
                "latency_ms": total_ms,
            },
        )
        return RAGQueryResponse(
            request_id=ctx.request_id,
            query=ctx.query,
            collection=ctx.collection,
            answer=answer,
            outcome=outcome,
            citations=citations,
            retrieval=retrieval,
            cache=cache,
            routing=routing,
            usage=usage,
            latency=latency,
            timestamp=ctx.timestamp,
        )

    def _record_error(self, ctx, start, decision, exc: GenerationFailed, chunks, cache_info) -> None:
        total_ms = round((time.perf_counter() - start) * 1000, 2)
        self.qs.metrics_store.record_request(
            RequestRecord(
                request_id=ctx.request_id,
                timestamp=ctx.timestamp,
                query=telemetry_text(ctx.query),
                response_preview="",
                cache_hit=False,
                cached_similarity=cache_info.similarity,
                is_cache_simulated=ctx.is_simulated,
                selected_model=decision.selected_model.model_id,
                routing_reason=decision.routing_reason,
                latency_ms=total_ms,
                estimated_input_tokens=0,
                estimated_output_tokens=0,
                estimated_cost=0.0,
                actual_cost=None,
                has_error=True,
                error_message=f"{exc.category}: {type(exc.cause).__name__}",
                mode="rag",
                collection=ctx.collection,
                outcome="error",
                cache_lookup_ms=ctx.cache_lookup_ms,
                embedding_ms=ctx.embedding_ms,
                retrieval_ms=ctx.retrieval_ms,
                generation_ms=ctx.generation_ms,
                retrieved_chunk_count=len(chunks),
                top_retrieval_score=chunks[0].score if chunks else None,
                error_category=exc.category,
                retry_count=max(0, len(exc.attempts) - 1),
                token_source="none",
            )
        )
        logger.error(
            "rag_query_failed",
            extra={
                "request_id": ctx.request_id,
                "selected_model": decision.selected_model.model_id,
                "latency_ms": total_ms,
                "error_type": exc.category,
            },
        )


class _StaleCacheEntry(Exception):
    pass


class _RequestContext:
    def __init__(self, **kwargs):
        self.embedding_ms: Optional[float] = None
        self.cache_lookup_ms: Optional[float] = None
        self.retrieval_ms: Optional[float] = None
        self.generation_ms: Optional[float] = None
        self.__dict__.update(kwargs)
