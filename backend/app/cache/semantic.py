import time
import math
import uuid
from typing import Optional, Dict, Any, List
from .base import BaseCache, CacheEntry, CacheLookupResult
from .embeddings import BaseEmbeddingProvider, SimulatedEmbeddingProvider

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if len(v1) != len(v2) or not v1:
        return 0.0
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    # Clamp to [-1.0, 1.0] to avoid floating point precision artifacts
    return max(-1.0, min(1.0, dot_product / (norm_a * norm_b)))


class SemanticCache(BaseCache):
    """
    Semantic cache storing query-response pairs alongside embedding vectors.
    Evaluates similarity thresholds to detect semantic cache hits.
    """

    def __init__(
        self,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        default_threshold: float = 0.82,
        max_entries: int = 1000,
    ):
        self.embedding_provider = embedding_provider or SimulatedEmbeddingProvider()
        self.default_threshold = default_threshold
        self.max_entries = max_entries
        self._entries: Dict[str, CacheEntry] = {}

    def lookup(self, query: str, threshold: Optional[float] = None) -> CacheLookupResult:
        start_time = time.perf_counter()
        effective_threshold = threshold if threshold is not None else self.default_threshold

        if not self._entries:
            lookup_latency = (time.perf_counter() - start_time) * 1000.0
            return CacheLookupResult(
                is_hit=False,
                similarity_score=None,
                matched_entry=None,
                is_simulated=self.embedding_provider.is_simulated,
                lookup_latency_ms=round(lookup_latency, 2),
            )

        query_embedding = self.embedding_provider.embed_text(query)

        best_score = -1.0
        best_entry: Optional[CacheEntry] = None

        for entry in self._entries.values():
            score = cosine_similarity(query_embedding, entry.embedding)
            if score > best_score:
                best_score = score
                best_entry = entry

        lookup_latency = (time.perf_counter() - start_time) * 1000.0

        if best_entry is not None and best_score >= effective_threshold:
            best_entry.hit_count += 1
            best_entry.last_accessed_at = time.time()
            return CacheLookupResult(
                is_hit=True,
                similarity_score=round(best_score, 4),
                matched_entry=best_entry,
                is_simulated=self.embedding_provider.is_simulated,
                lookup_latency_ms=round(lookup_latency, 2),
            )

        return CacheLookupResult(
            is_hit=False,
            similarity_score=round(best_score, 4) if best_score > -1.0 else None,
            matched_entry=None,
            is_simulated=self.embedding_provider.is_simulated,
            lookup_latency_ms=round(lookup_latency, 2),
        )

    def store(
        self, query: str, response: str, metadata: Optional[Dict[str, Any]] = None
    ) -> CacheEntry:
        # Enforce max entries with LRU eviction
        if len(self._entries) >= self.max_entries:
            oldest_key = min(self._entries.keys(), key=lambda k: self._entries[k].last_accessed_at)
            del self._entries[oldest_key]

        embedding = self.embedding_provider.embed_text(query)
        entry_id = str(uuid.uuid4())
        entry = CacheEntry(
            key=entry_id,
            query=query,
            response=response,
            embedding=embedding,
            created_at=time.time(),
            last_accessed_at=time.time(),
            hit_count=0,
            metadata=metadata or {},
        )
        self._entries[entry_id] = entry
        return entry

    def clear(self) -> None:
        self._entries.clear()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_entries": len(self._entries),
            "max_entries": self.max_entries,
            "default_threshold": self.default_threshold,
            "embedding_provider": self.embedding_provider.provider_name,
            "is_simulated": self.embedding_provider.is_simulated,
        }
