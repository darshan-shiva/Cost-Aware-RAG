import sqlite3
import math
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Generator, Union
from ..memdb import InMemoryDatabase
from ..schemas import RequestRecord, MetricsSummary, ModelStats

# Columns added after the original schema, with their SQLite types. Existing
# databases are migrated in place with ALTER TABLE on startup.
_ADDED_COLUMNS = {
    "mode": "TEXT NOT NULL DEFAULT 'general'",
    "collection": "TEXT",
    "outcome": "TEXT",
    "cache_match_type": "TEXT",
    "cache_lookup_ms": "REAL",
    "embedding_ms": "REAL",
    "retrieval_ms": "REAL",
    "generation_ms": "REAL",
    "retrieved_chunk_count": "INTEGER",
    "top_retrieval_score": "REAL",
    "citation_count": "INTEGER",
    "error_category": "TEXT",
    "retry_count": "INTEGER NOT NULL DEFAULT 0",
    "fallback_model": "TEXT",
    "estimated_cost_avoided": "REAL NOT NULL DEFAULT 0",
    "token_source": "TEXT",
}

_BASE_COLUMNS = [
    "request_id", "timestamp", "query", "response_preview", "cache_hit",
    "cached_similarity", "is_cache_simulated", "selected_model",
    "routing_reason", "latency_ms", "estimated_input_tokens",
    "estimated_output_tokens", "estimated_cost", "actual_cost", "has_error",
    "error_message",
]
_ALL_COLUMNS = _BASE_COLUMNS + list(_ADDED_COLUMNS)
_BOOL_COLUMNS = {"cache_hit", "is_cache_simulated", "has_error"}


def _percentile_nearest_rank(sorted_values: List[float], pct: float) -> float:
    n = len(sorted_values)
    index = min(n - 1, int(math.ceil(pct * n)) - 1)
    return sorted_values[max(0, index)]


class SQLiteMetricsStore:
    """
    SQLite persistence layer for request telemetry, latency percentiles,
    and routing metrics.
    """

    def __init__(self, db_path: Union[str, InMemoryDatabase]):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        if isinstance(self.db_path, InMemoryDatabase):
            with self.db_path.connection() as conn:
                yield conn
            return
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS request_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response_preview TEXT NOT NULL,
                    cache_hit INTEGER NOT NULL,
                    cached_similarity REAL,
                    is_cache_simulated INTEGER NOT NULL,
                    selected_model TEXT NOT NULL,
                    routing_reason TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    estimated_input_tokens INTEGER NOT NULL,
                    estimated_output_tokens INTEGER NOT NULL,
                    estimated_cost REAL NOT NULL,
                    actual_cost REAL,
                    has_error INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT
                );
                """
            )
            existing = {
                row["name"] for row in cursor.execute("PRAGMA table_info(request_records);")
            }
            for column, ddl in _ADDED_COLUMNS.items():
                if column not in existing:
                    cursor.execute(f"ALTER TABLE request_records ADD COLUMN {column} {ddl};")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON request_records(timestamp DESC);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_requests_mode ON request_records(mode);"
            )
            conn.commit()


    def record_request(self, record: RequestRecord) -> None:
        values = []
        for column in _ALL_COLUMNS:
            value = getattr(record, column)
            if column in _BOOL_COLUMNS:
                value = 1 if value else 0
            values.append(value)
        placeholders = ", ".join("?" for _ in _ALL_COLUMNS)
        with self._connection() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO request_records ({', '.join(_ALL_COLUMNS)})"
                f" VALUES ({placeholders});",
                values,
            )
            conn.commit()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> RequestRecord:
        data: Dict[str, Any] = {}
        for column in _ALL_COLUMNS:
            value = row[column]
            if column in _BOOL_COLUMNS:
                value = bool(value)
            if value is None and column in ("mode", "retry_count", "estimated_cost_avoided"):
                continue
            data[column] = value
        return RequestRecord(**data)

    def get_recent_requests(
        self, limit: int = 50, mode: Optional[str] = None
    ) -> List[RequestRecord]:
        with self._connection() as conn:
            if mode:
                rows = conn.execute(
                    f"SELECT {', '.join(_ALL_COLUMNS)} FROM request_records"
                    " WHERE mode = ? ORDER BY id DESC LIMIT ?;",
                    (mode, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT {', '.join(_ALL_COLUMNS)} FROM request_records"
                    " ORDER BY id DESC LIMIT ?;",
                    (limit,),
                ).fetchall()
            return [self._row_to_record(row) for row in rows]

    def get_summary(self, mode: Optional[str] = None) -> MetricsSummary:
        where = "WHERE mode = ?" if mode else ""
        params: tuple = (mode,) if mode else ()
        with self._connection() as conn:
            cursor = conn.cursor()

            # Aggregate counts and spend
            cursor.execute(
                f"""
                SELECT
                    COUNT(*) AS total_requests,
                    SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) AS cache_hits,
                    SUM(CASE WHEN cache_hit = 0 THEN 1 ELSE 0 END) AS cache_misses,
                    SUM(CASE WHEN has_error = 1 THEN 1 ELSE 0 END) AS error_count,
                    COALESCE(SUM(estimated_cost), 0.0) AS total_estimated_spend,
                    COALESCE(SUM(estimated_cost_avoided), 0.0) AS total_avoided,
                    COALESCE(SUM(estimated_input_tokens), 0) AS input_tokens,
                    COALESCE(SUM(estimated_output_tokens), 0) AS output_tokens,
                    COALESCE(AVG(latency_ms), 0.0) AS avg_latency_ms,
                    AVG(cache_lookup_ms) AS avg_cache_lookup_ms,
                    AVG(embedding_ms) AS avg_embedding_ms,
                    AVG(retrieval_ms) AS avg_retrieval_ms,
                    AVG(generation_ms) AS avg_generation_ms,
                    AVG(CASE WHEN cache_hit = 1 THEN latency_ms END) AS avg_hit_latency,
                    AVG(CASE WHEN cache_hit = 0 AND has_error = 0 THEN latency_ms END) AS avg_miss_latency,
                    SUM(CASE WHEN fallback_model IS NOT NULL THEN 1 ELSE 0 END) AS fallback_count,
                    MAX(is_cache_simulated) AS any_simulated
                FROM request_records {where};
                """,
                params,
            )
            agg = cursor.fetchone()
            total_requests = agg["total_requests"] or 0
            cache_hits = agg["cache_hits"] or 0
            cache_misses = agg["cache_misses"] or 0
            error_count = agg["error_count"] or 0
            error_rate = round(
                error_count / total_requests, 4
            ) if total_requests > 0 else 0.0
            total_estimated_spend = round(agg["total_estimated_spend"] or 0.0, 6)
            avg_latency_ms = round(agg["avg_latency_ms"] or 0.0, 2)
            cache_hit_rate = round(cache_hits / total_requests, 4) if total_requests > 0 else 0.0

            def _opt(value: Optional[float]) -> Optional[float]:
                return round(value, 2) if value is not None else None

            # Calculate actual percentiles from sorted latencies
            p50_latency_ms = 0.0
            p95_latency_ms = 0.0

            if total_requests > 0:
                cursor.execute(
                    f"SELECT latency_ms FROM request_records {where} ORDER BY latency_ms ASC;",
                    params,
                )
                latencies = [r["latency_ms"] for r in cursor.fetchall()]
                n = len(latencies)

                # p50 calculation
                mid = n // 2
                if n % 2 == 1:
                    p50_latency_ms = latencies[mid]
                else:
                    p50_latency_ms = (latencies[mid - 1] + latencies[mid]) / 2.0

                # p95 calculation (nearest rank method)
                p95_latency_ms = _percentile_nearest_rank(latencies, 0.95)

            def _breakdown(column: str) -> Dict[str, int]:
                cursor.execute(
                    f"SELECT {column} AS k, COUNT(*) AS c FROM request_records {where}"
                    f" {'AND' if where else 'WHERE'} {column} IS NOT NULL GROUP BY {column};",
                    params,
                )
                return {row["k"]: row["c"] for row in cursor.fetchall()}

            # Model routing distribution
            routing_breakdown = _breakdown("selected_model")

            cursor.execute(
                f"""
                SELECT selected_model, COUNT(*) AS requests,
                       AVG(latency_ms) AS avg_latency,
                       COALESCE(SUM(estimated_cost), 0) AS cost,
                       COALESCE(SUM(estimated_input_tokens), 0) AS tin,
                       COALESCE(SUM(estimated_output_tokens), 0) AS tout
                FROM request_records {where}
                {'AND' if where else 'WHERE'} cache_hit = 0
                GROUP BY selected_model;
                """,
                params,
            )
            model_stats = {
                row["selected_model"]: ModelStats(
                    requests=row["requests"],
                    avg_latency_ms=round(row["avg_latency"] or 0.0, 2),
                    total_estimated_cost=round(row["cost"], 6),
                    input_tokens=row["tin"],
                    output_tokens=row["tout"],
                )
                for row in cursor.fetchall()
            }

            return MetricsSummary(
                total_requests=total_requests,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                cache_hit_rate=cache_hit_rate,
                total_estimated_spend=total_estimated_spend,
                p50_latency_ms=round(p50_latency_ms, 2),
                p95_latency_ms=round(p95_latency_ms, 2),
                avg_latency_ms=avg_latency_ms,
                routing_breakdown=routing_breakdown,
                error_count=error_count,
                error_rate=error_rate,
                cache_breakdown={"hits": cache_hits, "misses": cache_misses},
                is_simulated=bool(agg["any_simulated"]),
                total_estimated_cost_avoided=round(agg["total_avoided"] or 0.0, 6),
                total_input_tokens=int(agg["input_tokens"] or 0),
                total_output_tokens=int(agg["output_tokens"] or 0),
                mode_breakdown=_breakdown("mode"),
                outcome_breakdown=_breakdown("outcome"),
                match_type_breakdown=_breakdown("cache_match_type"),
                error_category_breakdown=_breakdown("error_category"),
                model_stats=model_stats,
                avg_cache_lookup_ms=_opt(agg["avg_cache_lookup_ms"]),
                avg_embedding_ms=_opt(agg["avg_embedding_ms"]),
                avg_retrieval_ms=_opt(agg["avg_retrieval_ms"]),
                avg_generation_ms=_opt(agg["avg_generation_ms"]),
                avg_latency_cache_hit_ms=_opt(agg["avg_hit_latency"]),
                avg_latency_cache_miss_ms=_opt(agg["avg_miss_latency"]),
                fallback_count=int(agg["fallback_count"] or 0),
            )

    def clear(self) -> None:
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM request_records;")
            conn.commit()
