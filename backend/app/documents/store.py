"""SQLite persistence for documents, page text, chunks and chunk embeddings."""

import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Generator, Iterable, List, Optional, Sequence, Union

import numpy as np

from ..memdb import InMemoryDatabase


DOCUMENT_STATUSES = ("pending", "processing", "ready", "failed")


@dataclass
class DocumentRecord:
    document_id: str
    collection: str
    filename: str
    file_type: str
    size_bytes: int
    content_sha256: str
    version: int
    status: str
    error_message: Optional[str]
    page_count: int
    chunk_count: int
    char_count: int
    embedding_model: Optional[str]
    created_at: float
    updated_at: float


@dataclass
class ChunkRecord:
    chunk_id: str
    document_id: str
    collection: str
    document_version: int
    chunk_index: int
    page_number: int
    start_char: int
    end_char: int
    text: str
    filename: str = ""
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


class DocumentStore:
    """Stores documents and their chunks.

    Embeddings are stored as float32 blobs next to their chunk so that a
    document and everything derived from it are deleted in one transaction.
    """

    def __init__(self, db_path: Union[str, InMemoryDatabase]):
        self.db_path = db_path
        self._write_lock = threading.Lock()
        self._init_db()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        if isinstance(self.db_path, InMemoryDatabase):
            with self.db_path.connection() as conn:
                yield conn
            return
        conn = _connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connection() as conn:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    collection TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    page_count INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    char_count INTEGER NOT NULL DEFAULT 0,
                    embedding_model TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE (collection, filename)
                );
                CREATE INDEX IF NOT EXISTS idx_documents_collection
                    ON documents(collection);
                CREATE INDEX IF NOT EXISTS idx_documents_sha
                    ON documents(collection, content_sha256);

                CREATE TABLE IF NOT EXISTS document_pages (
                    document_id TEXT NOT NULL
                        REFERENCES documents(document_id) ON DELETE CASCADE,
                    page_number INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    PRIMARY KEY (document_id, page_number)
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL
                        REFERENCES documents(document_id) ON DELETE CASCADE,
                    collection TEXT NOT NULL,
                    document_version INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    page_number INTEGER NOT NULL,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_collection
                    ON chunks(collection);
                CREATE INDEX IF NOT EXISTS idx_chunks_document
                    ON chunks(document_id);
                """
            )
            conn.commit()

    # ------------------------------------------------------------------ rows

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(**{k: row[k] for k in row.keys()})

    def get(self, document_id: str) -> Optional[DocumentRecord]:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?;", (document_id,)
            ).fetchone()
        return self._row_to_document(row) if row else None

    def find_by_filename(
        self, collection: str, filename: str
    ) -> Optional[DocumentRecord]:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE collection = ? AND filename = ?;",
                (collection, filename),
            ).fetchone()
        return self._row_to_document(row) if row else None

    def find_by_sha(
        self, collection: str, sha256: str
    ) -> Optional[DocumentRecord]:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE collection = ? AND content_sha256 = ?"
                " ORDER BY created_at ASC LIMIT 1;",
                (collection, sha256),
            ).fetchone()
        return self._row_to_document(row) if row else None

    def list(self, collection: Optional[str] = None) -> List[DocumentRecord]:
        with self._connection() as conn:
            if collection is None:
                rows = conn.execute(
                    "SELECT * FROM documents ORDER BY updated_at DESC;"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM documents WHERE collection = ?"
                    " ORDER BY updated_at DESC;",
                    (collection,),
                ).fetchall()
        return [self._row_to_document(r) for r in rows]

    def list_collections(self) -> List[Dict[str, object]]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT collection,
                       COUNT(*) AS document_count,
                       SUM(CASE WHEN status = 'ready' THEN 1 ELSE 0 END) AS ready_count,
                       COALESCE(SUM(chunk_count), 0) AS chunk_count
                FROM documents GROUP BY collection ORDER BY collection;
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def create(self, record: DocumentRecord) -> None:
        with self._write_lock, self._connection() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    document_id, collection, filename, file_type, size_bytes,
                    content_sha256, version, status, error_message, page_count,
                    chunk_count, char_count, embedding_model, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    record.document_id, record.collection, record.filename,
                    record.file_type, record.size_bytes, record.content_sha256,
                    record.version, record.status, record.error_message,
                    record.page_count, record.chunk_count, record.char_count,
                    record.embedding_model, record.created_at, record.updated_at,
                ),
            )
            conn.commit()

    def begin_new_version(
        self,
        document_id: str,
        file_type: str,
        size_bytes: int,
        sha256: str,
    ) -> int:
        """Replace a document's content: bump version, drop old pages/chunks."""
        with self._write_lock, self._connection() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?;", (document_id,))
            conn.execute(
                "DELETE FROM document_pages WHERE document_id = ?;", (document_id,)
            )
            conn.execute(
                """
                UPDATE documents SET version = version + 1, status = 'pending',
                    error_message = NULL, file_type = ?, size_bytes = ?,
                    content_sha256 = ?, page_count = 0, chunk_count = 0,
                    char_count = 0, updated_at = ?
                WHERE document_id = ?;
                """,
                (file_type, size_bytes, sha256, time.time(), document_id),
            )
            conn.commit()
            row = conn.execute(
                "SELECT version FROM documents WHERE document_id = ?;",
                (document_id,),
            ).fetchone()
        return int(row["version"])

    def bump_version(self, document_id: str) -> int:
        """Start a re-index: new version, chunks dropped, page text kept."""
        with self._write_lock, self._connection() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?;", (document_id,))
            conn.execute(
                "UPDATE documents SET version = version + 1, status = 'pending',"
                " error_message = NULL, chunk_count = 0, updated_at = ?"
                " WHERE document_id = ?;",
                (time.time(), document_id),
            )
            conn.commit()
            row = conn.execute(
                "SELECT version FROM documents WHERE document_id = ?;",
                (document_id,),
            ).fetchone()
        return int(row["version"])

    def set_status(
        self, document_id: str, status: str, error_message: Optional[str] = None
    ) -> None:
        if status not in DOCUMENT_STATUSES:
            raise ValueError(f"Unknown document status: {status}")
        with self._write_lock, self._connection() as conn:
            conn.execute(
                "UPDATE documents SET status = ?, error_message = ?, updated_at = ?"
                " WHERE document_id = ?;",
                (status, error_message, time.time(), document_id),
            )
            conn.commit()

    def save_pages(self, document_id: str, pages: Iterable[tuple]) -> None:
        with self._write_lock, self._connection() as conn:
            conn.execute(
                "DELETE FROM document_pages WHERE document_id = ?;", (document_id,)
            )
            conn.executemany(
                "INSERT INTO document_pages (document_id, page_number, text)"
                " VALUES (?, ?, ?);",
                [(document_id, p, t) for p, t in pages],
            )
            conn.commit()

    def get_pages(self, document_id: str) -> List[tuple]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT page_number, text FROM document_pages"
                " WHERE document_id = ? ORDER BY page_number;",
                (document_id,),
            ).fetchall()
        return [(r["page_number"], r["text"]) for r in rows]

    def save_chunks(
        self,
        document_id: str,
        expected_version: int,
        chunks: Sequence[ChunkRecord],
        page_count: int,
        char_count: int,
        embedding_model: str,
    ) -> bool:
        """Atomically store chunks and mark the document ready.

        Returns False (and writes nothing) if the document was deleted or a
        newer version started while this one was being processed.
        """
        with self._write_lock, self._connection() as conn:
            row = conn.execute(
                "SELECT version FROM documents WHERE document_id = ?;",
                (document_id,),
            ).fetchone()
            if row is None or int(row["version"]) != expected_version:
                return False
            conn.execute("DELETE FROM chunks WHERE document_id = ?;", (document_id,))
            conn.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, document_id, collection, document_version,
                    chunk_index, page_number, start_char, end_char, text,
                    embedding, embedding_dim
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                [
                    (
                        c.chunk_id, c.document_id, c.collection,
                        c.document_version, c.chunk_index, c.page_number,
                        c.start_char, c.end_char, c.text,
                        np.asarray(c.embedding, dtype=np.float32).tobytes(),
                        int(np.asarray(c.embedding).shape[0]),
                    )
                    for c in chunks
                ],
            )
            conn.execute(
                """
                UPDATE documents SET status = 'ready', error_message = NULL,
                    chunk_count = ?, page_count = ?, char_count = ?,
                    embedding_model = ?, updated_at = ?
                WHERE document_id = ?;
                """,
                (len(chunks), page_count, char_count, embedding_model,
                 time.time(), document_id),
            )
            conn.commit()
        return True

    def delete(self, document_id: str) -> bool:
        with self._write_lock, self._connection() as conn:
            cur = conn.execute(
                "DELETE FROM documents WHERE document_id = ?;", (document_id,)
            )
            conn.commit()
            return cur.rowcount > 0

    def load_chunks(
        self, collection: str, embedding_model: Optional[str] = None
    ) -> List[ChunkRecord]:
        """Load all chunks of ready documents in a collection, with vectors."""
        sql = """
            SELECT c.*, d.filename AS filename
            FROM chunks c JOIN documents d ON d.document_id = c.document_id
            WHERE c.collection = ? AND d.status = 'ready'
              AND c.document_version = d.version
        """
        params: list = [collection]
        if embedding_model is not None:
            sql += " AND d.embedding_model = ?"
            params.append(embedding_model)
        sql += " ORDER BY c.document_id, c.chunk_index;"
        with self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            ChunkRecord(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                collection=r["collection"],
                document_version=r["document_version"],
                chunk_index=r["chunk_index"],
                page_number=r["page_number"],
                start_char=r["start_char"],
                end_char=r["end_char"],
                text=r["text"],
                filename=r["filename"],
                embedding=np.frombuffer(r["embedding"], dtype=np.float32),
            )
            for r in rows
        ]

    def get_chunks(self, chunk_ids: Sequence[str]) -> Dict[str, ChunkRecord]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._connection() as conn:
            rows = conn.execute(
                f"""
                SELECT c.chunk_id, c.document_id, c.collection, c.document_version,
                       c.chunk_index, c.page_number, c.start_char, c.end_char,
                       c.text, d.filename AS filename
                FROM chunks c JOIN documents d ON d.document_id = c.document_id
                WHERE c.chunk_id IN ({placeholders});
                """,
                list(chunk_ids),
            ).fetchall()
        return {
            r["chunk_id"]: ChunkRecord(**{k: r[k] for k in r.keys()})
            for r in rows
        }

    def collection_fingerprint_parts(self, collection: str) -> List[tuple]:
        """(document_id, version, sha256, embedding_model) for ready documents."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT document_id, version, content_sha256, embedding_model"
                " FROM documents WHERE collection = ? AND status = 'ready'"
                " ORDER BY document_id;",
                (collection,),
            ).fetchall()
        return [tuple(r) for r in rows]

    def mark_interrupted(self) -> int:
        """Fail documents left mid-processing by a previous process."""
        with self._write_lock, self._connection() as conn:
            cur = conn.execute(
                "UPDATE documents SET status = 'failed', error_message = ?,"
                " updated_at = ? WHERE status IN ('pending', 'processing');",
                ("Processing was interrupted by a server restart. Re-index to retry.",
                 time.time()),
            )
            conn.commit()
            return cur.rowcount

    def clear(self) -> None:
        with self._write_lock, self._connection() as conn:
            conn.execute("DELETE FROM documents;")
            conn.commit()
