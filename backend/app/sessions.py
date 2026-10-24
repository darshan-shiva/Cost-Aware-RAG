"""Per-visitor sessions with private, in-memory workspaces.

Every browser session gets its own workspace: a document store, vector
index, semantic cache and telemetry store on a private in-memory SQLite
database (see memdb.py). Nothing is shared between workspaces except
stateless components (embedding model, router, LLM gateway), so one visitor's
documents, answers, cached answers and request history cannot be reached
from another session.

Session tokens are minted by the server with 256 bits of randomness and are
the only way to reach a workspace. Clients cannot choose a token, and only a
SHA-256 hash of each token is kept, so the session table never holds a
usable credential. Unknown, expired or malformed tokens are rejected.
"""

import hashlib
import logging
import re
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

from .memdb import InMemoryDatabase

logger = logging.getLogger("cost_aware_rag.sessions")

# token_urlsafe(32) yields 43 URL-safe characters.
_TOKEN_FORMAT = re.compile(r"^[A-Za-z0-9_-]{43}$")
_SWEEP_INTERVAL_SECONDS = 30.0


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


@dataclass
class Workspace:
    db: InMemoryDatabase
    query_service: object
    rag: object
    created_at: float
    last_seen: float
    # Short, non-reversible label for logs; never the token itself.
    label: str = ""
    upload_lock: threading.Lock = field(default_factory=threading.Lock)

    def close(self) -> None:
        self.db.close()


WorkspaceFactory = Callable[[InMemoryDatabase], Tuple[object, object]]


class SessionManager:
    def __init__(
        self,
        factory: WorkspaceFactory,
        idle_ttl_seconds: float,
        max_age_seconds: float,
        max_active: int,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._factory = factory
        self.idle_ttl_seconds = idle_ttl_seconds
        self.max_age_seconds = max_age_seconds
        self.max_active = max(1, max_active)
        self._clock = clock
        # Least recently used first.
        self._sessions: "OrderedDict[str, Workspace]" = OrderedDict()
        self._lock = threading.Lock()
        self._last_sweep = clock()

    # ---------------------------------------------------------------- public

    def create(self) -> Tuple[str, Workspace]:
        token = secrets.token_urlsafe(32)
        key = hash_token(token)
        db = InMemoryDatabase()
        try:
            query_service, rag = self._factory(db)
        except Exception:
            db.close()
            raise
        now = self._clock()
        workspace = Workspace(
            db=db,
            query_service=query_service,
            rag=rag,
            created_at=now,
            last_seen=now,
            label=key[:8],
        )
        evicted = []
        with self._lock:
            self._sweep_locked(now, force=True, closed=evicted)
            while len(self._sessions) >= self.max_active:
                _, oldest = self._sessions.popitem(last=False)
                evicted.append(oldest)
            self._sessions[key] = workspace
        for ws in evicted:
            ws.close()
        if evicted:
            logger.info("sessions_closed", extra={"count": len(evicted)})
        logger.info("session_created", extra={"session": workspace.label})
        return token, workspace

    def get(self, token: Optional[str]) -> Optional[Workspace]:
        if not token or not _TOKEN_FORMAT.match(token):
            return None
        key = hash_token(token)
        now = self._clock()
        expired = []
        with self._lock:
            self._sweep_locked(now, force=False, closed=expired)
            workspace = self._sessions.get(key)
            if workspace is not None and self._is_expired(workspace, now):
                del self._sessions[key]
                expired.append(workspace)
                workspace = None
            if workspace is not None:
                workspace.last_seen = now
                self._sessions.move_to_end(key)
        for ws in expired:
            ws.close()
        return workspace

    def close(self, token: Optional[str]) -> bool:
        if not token or not _TOKEN_FORMAT.match(token):
            return False
        with self._lock:
            workspace = self._sessions.pop(hash_token(token), None)
        if workspace is None:
            return False
        workspace.close()
        logger.info("session_closed", extra={"session": workspace.label})
        return True

    def expires_in(self, workspace: Workspace) -> float:
        now = self._clock()
        return max(
            0.0,
            min(
                workspace.last_seen + self.idle_ttl_seconds,
                workspace.created_at + self.max_age_seconds,
            )
            - now,
        )

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def sweep(self) -> int:
        closed = []
        with self._lock:
            self._sweep_locked(self._clock(), force=True, closed=closed)
        for ws in closed:
            ws.close()
        return len(closed)

    def close_all(self) -> None:
        with self._lock:
            workspaces = list(self._sessions.values())
            self._sessions.clear()
        for ws in workspaces:
            ws.close()

    # -------------------------------------------------------------- internal

    def _is_expired(self, workspace: Workspace, now: float) -> bool:
        return (
            now - workspace.last_seen >= self.idle_ttl_seconds
            or now - workspace.created_at >= self.max_age_seconds
        )

    def _sweep_locked(self, now: float, force: bool, closed: list) -> None:
        if not force and now - self._last_sweep < _SWEEP_INTERVAL_SECONDS:
            return
        self._last_sweep = now
        for key in [k for k, ws in self._sessions.items() if self._is_expired(ws, now)]:
            closed.append(self._sessions.pop(key))
