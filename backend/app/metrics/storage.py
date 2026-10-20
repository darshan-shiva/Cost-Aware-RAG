import sqlite3
import math
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Generator
from ..schemas import RequestRecord, MetricsSummary

class SQLiteMetricsStore:
    """
    SQLite persistence layer for request telemetry, latency percentiles,
    and routing metrics.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
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
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON request_records(timestamp DESC);"
            )
            conn.commit()


    def record_request(self, record: RequestRecord) -> None:
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO request_records (
                    request_id, timestamp, query, response_preview,
                    cache_hit, cached_similarity, is_cache_simulated,
                    selected_model, routing_reason, latency_ms,
                    estimated_input_tokens, estimated_output_tokens,
                    estimated_cost, actual_cost, has_error, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    record.request_id,
                    record.timestamp,
                    record.query,
                    record.response_preview,
                    1 if record.cache_hit else 0,
                    record.cached_similarity,
                    1 if record.is_cache_simulated else 0,
                    record.selected_model,
                    record.routing_reason,
                    record.latency_ms,
                    record.estimated_input_tokens,
                    record.estimated_output_tokens,
                    record.estimated_cost,
                    record.actual_cost,
                    1 if record.has_error else 0,
                    record.error_message,
                ),
            )
            conn.commit()

    def get_recent_requests(self, limit: int = 50) -> List[RequestRecord]:
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT request_id, timestamp, query, response_preview,
                       cache_hit, cached_similarity, is_cache_simulated,
                       selected_model, routing_reason, latency_ms,
                       estimated_input_tokens, estimated_output_tokens,
                       estimated_cost, actual_cost, has_error, error_message
                FROM request_records
                ORDER BY id DESC
                LIMIT ?;
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            return [
                RequestRecord(
                    request_id=row["request_id"],
                    timestamp=row["timestamp"],
                    query=row["query"],
                    response_preview=row["response_preview"],
                    cache_hit=bool(row["cache_hit"]),
                    cached_similarity=row["cached_similarity"],
                    is_cache_simulated=bool(row["is_cache_simulated"]),
                    selected_model=row["selected_model"],
                    routing_reason=row["routing_reason"],
                    latency_ms=row["latency_ms"],
                    estimated_input_tokens=row["estimated_input_tokens"],
                    estimated_output_tokens=row["estimated_output_tokens"],
                    estimated_cost=row["estimated_cost"],
                    actual_cost=row["actual_cost"],
                    has_error=bool(row["has_error"]),
                    error_message=row["error_message"],
                )
                for row in rows
            ]

    def get_summary(self) -> MetricsSummary:
        with self._connection() as conn:
            cursor = conn.cursor()

            # Aggregate counts and spend
            cursor.execute(
                """
                SELECT 
                    COUNT(*) as total_requests,
                    SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) as cache_hits,
                    SUM(CASE WHEN cache_hit = 0 THEN 1 ELSE 0 END) as cache_misses,
                    COALESCE(SUM(estimated_cost), 0.0) as total_estimated_spend,
                    COALESCE(AVG(latency_ms), 0.0) as avg_latency_ms
                FROM request_records;
                """
            )
            agg = cursor.fetchone()
            total_requests = agg["total_requests"] or 0
            cache_hits = agg["cache_hits"] or 0
            cache_misses = agg["cache_misses"] or 0
            total_estimated_spend = round(agg["total_estimated_spend"] or 0.0, 6)
            avg_latency_ms = round(agg["avg_latency_ms"] or 0.0, 2)
            cache_hit_rate = round(cache_hits / total_requests, 4) if total_requests > 0 else 0.0

            # Calculate actual percentiles from sorted latencies
            p50_latency_ms = 0.0
            p95_latency_ms = 0.0

            if total_requests > 0:
                cursor.execute("SELECT latency_ms FROM request_records ORDER BY latency_ms ASC;")
                latencies = [r["latency_ms"] for r in cursor.fetchall()]
                n = len(latencies)
                
                # p50 calculation
                mid = n // 2
                if n % 2 == 1:
                    p50_latency_ms = latencies[mid]
                else:
                    p50_latency_ms = (latencies[mid - 1] + latencies[mid]) / 2.0
                
                # p95 calculation (nearest rank method)
                p95_index = min(n - 1, int(math.ceil(0.95 * n)) - 1)
                p95_latency_ms = latencies[p95_index]

            # Model routing distribution
            cursor.execute(
                """
                SELECT selected_model, COUNT(*) as count 
                FROM request_records 
                GROUP BY selected_model;
                """
            )
            routing_breakdown = {row["selected_model"]: row["count"] for row in cursor.fetchall()}

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
                cache_breakdown={"hits": cache_hits, "misses": cache_misses},
                is_simulated=True,
            )

    def clear(self) -> None:
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM request_records;")
            conn.commit()
