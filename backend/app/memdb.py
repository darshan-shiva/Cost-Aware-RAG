"""A private, in-memory SQLite database for one visitor session.

The document store, semantic cache and telemetry store normally open a new
connection to a database file for every operation. An in-memory SQLite
database exists only inside the connection that created it, so a session
instead owns one connection and every store borrows it under a lock.
Nothing is written to disk: closing the session (or stopping the process)
frees the documents, chunks, embeddings, cached answers and telemetry.
"""

import sqlite3
import threading
from contextlib import contextmanager
from typing import Generator


class WorkspaceClosed(RuntimeError):
    """Raised when a store is used after its session has been closed."""


class InMemoryDatabase:
    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON;")
        # Re-entrant: a store method may call another store method that also
        # borrows the connection on the same thread.
        self._lock = threading.RLock()
        self._depth = 0
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        with self._lock:
            if self._closed:
                raise WorkspaceClosed("This session has been closed.")
            self._depth += 1
            try:
                yield self._conn
            finally:
                self._depth -= 1
                # The file-backed stores relied on close() discarding any
                # uncommitted work; keep that behaviour for the shared
                # connection once the outermost borrower is done.
                if self._depth == 0 and not self._closed and self._conn.in_transaction:
                    self._conn.rollback()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._conn.close()
