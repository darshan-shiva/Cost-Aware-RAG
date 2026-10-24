import time
import math
import uuid
import json
import sqlite3
import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Generator, Callable, Union
from ..memdb import InMemoryDatabase
from .base import BaseCache, CacheEntry, CacheLookupResult
from .embeddings import BaseEmbeddingProvider, SimulatedEmbeddingProvider
from .guards import normalize_query, salient_terms_conflict

GENERAL_SCOPE = "general"
# Semantic candidates above the threshold that are validated before giving up.
MAX_SEMANTIC_CANDIDATES = 3

# A validator inspects a semantic candidate and returns a rejection reason,
# or None to accept it.
CandidateValidator = Callable[[CacheEntry, float], Optional[str]]

logger = logging.getLogger(__name__)

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Compute cosine similarity between two float vectors.
    Safely handles empty vectors, dimension mismatches, NaN/Inf values, and type errors.
    """
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    try:
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm_a = math.sqrt(sum(a * a for a in v1))
        norm_b = math.sqrt(sum(b * b for b in v2))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        if (
            math.isnan(dot_product) or math.isnan(norm_a) or math.isnan(norm_b)
            or math.isinf(norm_a) or math.isinf(norm_b) or math.isinf(dot_product)
        ):
            return 0.0
        # Clamp to [-1.0, 1.0] to avoid floating point precision artifacts
        return max(-1.0, min(1.0, dot_product / (norm_a * norm_b)))
    except (TypeError, ValueError, OverflowError):
        return 0.0



class SemanticCache(BaseCache):
    """
    Semantic cache storing query-response pairs alongside dense embedding vectors.
    Evaluates similarity thresholds to detect semantic cache hits.
    Optionally persists entries to SQLite to survive application restarts.
    """

    def __init__(
        self,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        default_threshold: float = 0.82,
        max_entries: int = 1000,
        db_path: Optional[Union[str, InMemoryDatabase]] = None,
    ):
        self.embedding_provider = embedding_provider or SimulatedEmbeddingProvider()
        self.default_threshold = default_threshold
        self.max_entries = max_entries
        self.db_path = db_path
        self._entries: Dict[str, CacheEntry] = {}

        if self.db_path:
            self._init_db()
            self._load_from_db()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        if not self.db_path:
            raise RuntimeError("Database path not configured for persistent cache.")
        if isinstance(self.db_path, InMemoryDatabase):
            with self.db_path.connection() as conn:
                yield conn
            return
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        if not self.db_path:
            return
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_accessed_at REAL NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT
                );
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_cache_last_accessed ON cache_entries(last_accessed_at ASC);"
            )
            conn.commit()

    def _load_from_db(self) -> None:
        if not self.db_path:
            return
        try:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT key, query, response, embedding_json,
                           created_at, last_accessed_at, hit_count, metadata_json
                    FROM cache_entries
                    ORDER BY last_accessed_at DESC
                    LIMIT ?;
                    """,
                    (self.max_entries,),
                )
                rows = cursor.fetchall()
                for row in rows:
                    entry = CacheEntry(
                        key=row["key"],
                        query=row["query"],
                        response=row["response"],
                        embedding=json.loads(row["embedding_json"]),
                        created_at=row["created_at"],
                        last_accessed_at=row["last_accessed_at"],
                        hit_count=row["hit_count"],
                        metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
                    )
                    self._entries[entry.key] = entry
                logger.info(f"Loaded {len(self._entries)} persistent cache entries from SQLite.")
        except Exception as e:
            logger.error(f"Failed to load cache entries from database: {e}")

    @staticmethod
    def _entry_scope(entry: CacheEntry) -> str:
        return entry.metadata.get("scope") or GENERAL_SCOPE

    def _miss(self, start_time: float, **kwargs: Any) -> CacheLookupResult:
        return CacheLookupResult(
            is_hit=False,
            matched_entry=None,
            is_simulated=self.embedding_provider.is_simulated,
            lookup_latency_ms=round((time.perf_counter() - start_time) * 1000.0, 2),
            **kwargs,
        )

    def _record_hit(self, entry: CacheEntry) -> None:
        now = time.time()
        entry.hit_count += 1
        entry.last_accessed_at = now
        # Sync hit count to DB if persistent
        if self.db_path:
            try:
                with self._connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE cache_entries SET hit_count = hit_count + 1, last_accessed_at = ? WHERE key = ?;",
                        (now, entry.key),
                    )
                    conn.commit()
            except Exception as e:
                logger.warning(f"Failed to update cache hit in database: {e}")

    def lookup(
        self,
        query: str,
        threshold: Optional[float] = None,
        scope: Optional[str] = None,
        fingerprint: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        validator: Optional[CandidateValidator] = None,
    ) -> CacheLookupResult:
        """Find a reusable cached response.

        Only entries in the same ``scope`` (and, when given, with the same
        corpus ``fingerprint``) are considered. A normalized exact text match
        is reused directly. Otherwise the most similar entries at or above
        the threshold are checked, best first, against the salient-term guard
        and the optional ``validator``; the first that passes is a hit.
        """
        start_time = time.perf_counter()
        effective_threshold = threshold if threshold is not None else self.default_threshold
        scope = scope or GENERAL_SCOPE

        cleaned_query = query.strip() if query else ""
        candidates = [
            e for e in self._entries.values()
            if self._entry_scope(e) == scope
            and (fingerprint is None or e.metadata.get("fingerprint") == fingerprint)
        ]
        if not cleaned_query or not candidates:
            return self._miss(start_time)

        normalized = normalize_query(cleaned_query)
        for entry in candidates:
            if normalize_query(entry.query) == normalized:
                self._record_hit(entry)
                return CacheLookupResult(
                    is_hit=True,
                    similarity_score=1.0,
                    matched_entry=entry,
                    is_simulated=self.embedding_provider.is_simulated,
                    lookup_latency_ms=round((time.perf_counter() - start_time) * 1000.0, 2),
                    match_type="exact",
                    candidates_checked=1,
                )

        if query_embedding is None:
            try:
                query_embedding = self.embedding_provider.embed_text(cleaned_query)
            except Exception as e:
                logger.error(f"Embedding generation failed during cache lookup: {e}")
                return self._miss(start_time)

        scored = sorted(
            ((cosine_similarity(query_embedding, e.embedding), e) for e in candidates),
            key=lambda pair: pair[0],
            reverse=True,
        )
        best_score = scored[0][0]
        rejection: Optional[str] = None
        checked = 0
        for score, entry in scored[:MAX_SEMANTIC_CANDIDATES]:
            if score < effective_threshold:
                break
            checked += 1
            reason = salient_terms_conflict(cleaned_query, entry.query)
            if reason is None and validator is not None:
                try:
                    reason = validator(entry, score)
                except Exception as e:
                    logger.warning(f"Cache candidate validation failed: {e}")
                    reason = "validation error"
            if reason is None:
                self._record_hit(entry)
                return CacheLookupResult(
                    is_hit=True,
                    similarity_score=round(score, 4),
                    matched_entry=entry,
                    is_simulated=self.embedding_provider.is_simulated,
                    lookup_latency_ms=round((time.perf_counter() - start_time) * 1000.0, 2),
                    match_type="semantic",
                    candidates_checked=checked,
                )
            rejection = rejection or reason

        return self._miss(
            start_time,
            similarity_score=round(best_score, 4),
            rejection_reason=rejection,
            candidates_checked=checked,
        )

    def store(
        self,
        query: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> Optional[CacheEntry]:
        cleaned_query = query.strip() if query else ""
        cleaned_response = response.strip() if response else ""

        # Do not cache empty queries or responses
        if not cleaned_query or not cleaned_response:
            logger.warning("Refusing to cache empty query or response.")
            return None

        # Enforce max entries with LRU eviction
        if len(self._entries) >= self.max_entries:
            oldest_key = min(self._entries.keys(), key=lambda k: self._entries[k].last_accessed_at)
            del self._entries[oldest_key]
            if self.db_path:
                try:
                    with self._connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("DELETE FROM cache_entries WHERE key = ?;", (oldest_key,))
                        conn.commit()
                except Exception as e:
                    logger.warning(f"Failed to evict entry from database: {e}")

        if embedding is None:
            try:
                embedding = self.embedding_provider.embed_text(cleaned_query)
            except Exception as e:
                logger.error(f"Failed to generate embedding for cache storage: {e}")
                return None

        entry_id = str(uuid.uuid4())
        now = time.time()
        entry = CacheEntry(
            key=entry_id,
            query=cleaned_query,
            response=cleaned_response,
            embedding=embedding,
            created_at=now,
            last_accessed_at=now,
            hit_count=0,
            metadata=metadata or {},
        )
        self._entries[entry_id] = entry

        # Persist to SQLite
        if self.db_path:
            try:
                with self._connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO cache_entries (
                            key, query, response, embedding_json,
                            created_at, last_accessed_at, hit_count, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            entry.key,
                            entry.query,
                            entry.response,
                            json.dumps(entry.embedding),
                            entry.created_at,
                            entry.last_accessed_at,
                            entry.hit_count,
                            json.dumps(entry.metadata),
                        ),
                    )
                    conn.commit()
            except Exception as e:
                logger.error(f"Failed to persist cache entry to database: {e}")

        return entry

    def invalidate(self, predicate: Callable[[CacheEntry], bool]) -> int:
        """Delete every entry for which ``predicate`` is true. Returns count."""
        doomed = [k for k, e in self._entries.items() if predicate(e)]
        for key in doomed:
            del self._entries[key]
        if doomed and self.db_path:
            try:
                with self._connection() as conn:
                    conn.executemany(
                        "DELETE FROM cache_entries WHERE key = ?;",
                        [(k,) for k in doomed],
                    )
                    conn.commit()
            except Exception as e:
                logger.error(f"Failed to delete invalidated cache entries: {e}")
        return len(doomed)

    def invalidate_scope_prefix(self, prefix: str) -> int:
        return self.invalidate(lambda e: self._entry_scope(e).startswith(prefix))

    def clear(self) -> None:
        self._entries.clear()
        if self.db_path:
            try:
                with self._connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM cache_entries;")
                    conn.commit()
            except Exception as e:
                logger.error(f"Failed to clear cache table in database: {e}")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_entries": len(self._entries),
            "max_entries": self.max_entries,
            "default_threshold": self.default_threshold,
            "embedding_provider": self.embedding_provider.provider_name,
            "is_simulated": self.embedding_provider.is_simulated,
            "persistent_storage": bool(self.db_path),
            "entries_by_scope": self._count_by_scope(),
        }

    def _count_by_scope(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for entry in self._entries.values():
            scope = self._entry_scope(entry)
            counts[scope] = counts.get(scope, 0) + 1
        return counts
