import os
import re
import tempfile
from typing import Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from app.api.routes import (
    SharedComponents,
    build_rag_service,
    build_workspace_services,
    get_query_service,
    get_rag_service,
    get_session_manager,
    get_shared,
)
from app.cache.embeddings import SimulatedEmbeddingProvider
from app.cache.semantic import SemanticCache
from app.llm import LLMGateway
from app.main import app
from app.metrics.storage import SQLiteMetricsStore
from app.rate_limit import limiter
from app.routing import CostAwareRouter
from app.services.query_service import QueryService
from app.sessions import SessionManager


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_embedding_model: needs the all-MiniLM-L6-v2 weights "
        "(downloaded from Hugging Face or already cached locally)",
    )


_MODEL_AVAILABLE: Optional[bool] = None


def embedding_model_available() -> bool:
    global _MODEL_AVAILABLE
    if _MODEL_AVAILABLE is None:
        try:
            from app.cache.embeddings import SentenceTransformersEmbeddingProvider

            SentenceTransformersEmbeddingProvider("all-MiniLM-L6-v2")
            _MODEL_AVAILABLE = True
        except Exception:
            _MODEL_AVAILABLE = False
    return _MODEL_AVAILABLE


def pytest_collection_modifyitems(config, items):
    needs_model = [i for i in items if "requires_embedding_model" in i.keywords]
    if needs_model and not embedding_model_available():
        skip = pytest.mark.skip(
            reason="all-MiniLM-L6-v2 weights unavailable (no network access to Hugging Face and no local cache)"
        )
        for item in needs_model:
            item.add_marker(skip)


class FakeLLM:
    """Deterministic stand-in for the Groq chat API.

    By default it answers with the first sentence of source S1 and cites it.
    Tests can set ``reply`` to a fixed string or a callable, or ``error`` to
    an exception raised for listed model IDs.
    """

    def __init__(self):
        self.calls: List[Dict[str, object]] = []
        self.reply: Optional[object] = None
        self.errors: Dict[str, Exception] = {}
        self.input_tokens = 120
        self.output_tokens = 30

    @staticmethod
    def sources(messages) -> Dict[str, str]:
        user = messages[-1]["content"]
        return {
            m.group(1): m.group(2).strip()
            for m in re.finditer(r'<source id="(S\d+)"[^>]*>\n(.*?)\n</source>', user, re.S)
        }

    def __call__(self, messages, model_id):
        self.calls.append({"messages": messages, "model": model_id})
        if model_id in self.errors:
            raise self.errors[model_id]
        if callable(self.reply):
            text = self.reply(messages, model_id)
        elif self.reply is not None:
            text = self.reply
        else:
            srcs = self.sources(messages)
            first = srcs.get("S1", "")
            sentence = re.split(r"(?<=[.!?])\s", first)[0] if first else "No sources."
            text = f"{sentence} [S1]"
        return text, self.input_tokens, self.output_tokens, "stop"


@pytest.fixture
def tmp_db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(path + suffix)
        except OSError:
            pass


@pytest.fixture
def services(tmp_db_path):
    """QueryService + RAGService on an isolated DB with simulated embeddings
    and no real LLM client."""
    provider = SimulatedEmbeddingProvider(dimension=256)
    cache = SemanticCache(embedding_provider=provider, default_threshold=0.80, db_path=tmp_db_path)
    qs = QueryService(
        cache=cache,
        metrics_store=SQLiteMetricsStore(tmp_db_path),
        gateway=LLMGateway(client=None, max_completion_tokens=512),
    )
    rag = build_rag_service(qs, tmp_db_path)
    fake = FakeLLM()
    rag._generate = fake
    rag.fake_llm = fake
    return qs, rag


@pytest.fixture
def api(services):
    qs, rag = services
    app.dependency_overrides[get_query_service] = lambda: qs
    app.dependency_overrides[get_rag_service] = lambda: rag
    app.dependency_overrides[get_shared] = lambda: SharedComponents(
        qs.cache.embedding_provider, qs.router, qs.gateway
    )
    limiter.reset()
    with TestClient(app, raise_server_exceptions=False) as client:
        client.qs = qs
        client.rag = rag
        yield client
    app.dependency_overrides.clear()
    limiter.reset()


class SessionHarness:
    """The real session manager wired to simulated embeddings and a FakeLLM
    per workspace, so tests exercise the production isolation path."""

    def __init__(self):
        self.now = 1000.0
        self.shared = SharedComponents(
            embedding_provider=SimulatedEmbeddingProvider(dimension=256),
            router=CostAwareRouter(),
            gateway=LLMGateway(client=None, max_completion_tokens=512),
        )
        self.fakes: List[FakeLLM] = []
        self.manager = SessionManager(
            factory=self._factory,
            idle_ttl_seconds=1800,
            max_age_seconds=6 * 3600,
            max_active=50,
            clock=lambda: self.now,
        )

    def _factory(self, db):
        qs, rag = build_workspace_services(self.shared, db)
        fake = FakeLLM()
        rag._generate = fake
        rag.fake_llm = fake
        self.fakes.append(fake)
        return qs, rag


@pytest.fixture
def sessions():
    harness = SessionHarness()
    app.dependency_overrides[get_session_manager] = lambda: harness.manager
    app.dependency_overrides[get_shared] = lambda: harness.shared
    limiter.reset()
    with TestClient(app, raise_server_exceptions=False) as client:
        harness.client = client
        yield harness
    harness.manager.close_all()
    app.dependency_overrides.clear()
    limiter.reset()


def new_session(client) -> Dict[str, str]:
    res = client.post("/api/session")
    assert res.status_code == 201, res.text
    return {"X-Session-Token": res.json()["session_token"]}


def upload(client, filename: str, data: bytes, collection: str = "default", headers=None):
    return client.post(
        "/api/documents",
        files={"file": (filename, data, "application/octet-stream")},
        data={"collection": collection},
        headers=headers,
    )
