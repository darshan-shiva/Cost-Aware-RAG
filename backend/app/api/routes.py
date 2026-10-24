import hmac
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Union
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from ..config import settings
from ..schemas import (
    CollectionOut,
    DocumentOut,
    HealthResponse,
    MetricsSummary,
    QueryRequest,
    QueryResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    RequestRecord,
    SessionOut,
    UploadResponse,
)
from ..cache import SemanticCache
from ..cache.embeddings import BaseEmbeddingProvider
from ..llm import LLMGateway
from ..memdb import InMemoryDatabase, WorkspaceClosed
from ..metrics import SQLiteMetricsStore
from ..routing import CostAwareRouter
from ..services.query_service import (
    QueryService,
    build_embedding_provider,
    build_llm_gateway,
)
from ..sessions import SessionManager, Workspace
from ..documents import (
    DocumentRecord,
    DocumentStore,
    DocumentValidationError,
    IngestionService,
    VectorIndex,
    sanitize_filename,
)
from ..rag import RAGService
from ..rate_limit import limiter

logger = logging.getLogger("cost_aware_rag.api")
router = APIRouter()

SESSION_HEADER = "X-Session-Token"


@dataclass
class SharedComponents:
    """Stateless pieces every session uses: they hold no visitor data."""

    embedding_provider: BaseEmbeddingProvider
    router: CostAwareRouter
    gateway: LLMGateway


_shared: Optional[SharedComponents] = None
_session_manager: Optional[SessionManager] = None


def get_shared() -> SharedComponents:
    global _shared
    if _shared is None:
        gateway = build_llm_gateway()
        if not gateway.is_configured:
            logger.warning("llm_not_configured")
        _shared = SharedComponents(
            embedding_provider=build_embedding_provider(),
            router=CostAwareRouter(),
            gateway=gateway,
        )
    return _shared


def build_workspace_services(shared: SharedComponents, db: InMemoryDatabase):
    """A session's private cache, telemetry, document store and RAG service."""
    cache = SemanticCache(
        embedding_provider=shared.embedding_provider,
        default_threshold=settings.SIMILARITY_THRESHOLD,
        max_entries=settings.CACHE_MAX_ENTRIES,
        db_path=db,
    )
    query_service = QueryService(
        cache=cache,
        router=shared.router,
        metrics_store=SQLiteMetricsStore(db),
        gateway=shared.gateway,
    )
    return query_service, build_rag_service(query_service, db)


def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager(
            factory=lambda db: build_workspace_services(get_shared(), db),
            idle_ttl_seconds=settings.SESSION_IDLE_TTL_MINUTES * 60,
            max_age_seconds=settings.SESSION_MAX_AGE_HOURS * 3600,
            max_active=settings.SESSION_MAX_ACTIVE,
        )
    return _session_manager


def get_workspace(
    x_session_token: Optional[str] = Header(default=None, alias=SESSION_HEADER),
    manager: SessionManager = Depends(get_session_manager),
) -> Workspace:
    """Resolve the caller's private workspace from its session token."""
    workspace = manager.get(x_session_token)
    if workspace is None:
        raise HTTPException(
            status_code=401,
            detail="Your session is missing or has expired. Reload the page to start a new one.",
        )
    return workspace


def get_query_service(workspace: Workspace = Depends(get_workspace)) -> QueryService:
    return workspace.query_service


def get_rag_service(workspace: Workspace = Depends(get_workspace)) -> RAGService:
    return workspace.rag


def build_rag_service(
    query_service: QueryService, db_path: Union[str, InMemoryDatabase]
) -> RAGService:
    """Wire the document store, vector index and ingestion around the shared
    embedding provider, cache, router, gateway and telemetry store."""
    store = DocumentStore(db_path)
    store.mark_interrupted()
    provider = query_service.cache.embedding_provider
    index = VectorIndex(store, provider.provider_name)
    ingestion = IngestionService(
        store=store,
        embedding_provider=provider,
        chunk_size=settings.CHUNK_SIZE_CHARS,
        chunk_overlap=settings.CHUNK_OVERLAP_CHARS,
        min_chunk_chars=settings.MIN_CHUNK_CHARS,
        max_upload_bytes=int(settings.MAX_UPLOAD_MB * 1024 * 1024),
        max_pdf_pages=settings.MAX_PDF_PAGES,
        max_document_chars=settings.MAX_DOCUMENT_CHARS,
        embedding_batch_size=settings.EMBEDDING_BATCH_SIZE,
    )
    return RAGService(query_service, store, index, ingestion)


def _process_upload(ingestion: IngestionService, document_id: str, version: int, data) -> None:
    try:
        ingestion.process(document_id, version, data)
    except WorkspaceClosed:
        # The visitor reset or abandoned the session mid-processing.
        logger.info("ingestion_skipped_session_closed")


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _document_out(doc: DocumentRecord, current_model: str) -> DocumentOut:
    return DocumentOut(
        document_id=doc.document_id,
        collection=doc.collection,
        filename=doc.filename,
        file_type=doc.file_type,
        size_bytes=doc.size_bytes,
        content_sha256=doc.content_sha256,
        version=doc.version,
        status=doc.status,
        error_message=doc.error_message,
        page_count=doc.page_count,
        chunk_count=doc.chunk_count,
        char_count=doc.char_count,
        embedding_model=doc.embedding_model,
        needs_reindex=doc.status == "ready" and doc.embedding_model != current_model,
        created_at=_iso(doc.created_at),
        updated_at=_iso(doc.updated_at),
    )


def require_admin(x_admin_token: Optional[str] = Header(default=None)) -> None:
    """Gate destructive endpoints. Open in development; in production they
    need RAG_ADMIN_TOKEN, and are disabled entirely if it is unset."""
    if not settings.is_production:
        return
    if not settings.ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="This endpoint is disabled in production.")
    if not x_admin_token or not hmac.compare_digest(x_admin_token, settings.ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="A valid X-Admin-Token header is required.")


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(
    shared: SharedComponents = Depends(get_shared),
    manager: SessionManager = Depends(get_session_manager),
):
    """Application health and status check. Reports no visitor data."""
    provider = shared.embedding_provider
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        simulated_mode=provider.is_simulated,
        timestamp=datetime.now(timezone.utc).isoformat(),
        embedding_provider=provider.provider_name,
        llm_configured=shared.gateway.is_configured,
        database_ok=True,
        document_count=None,
        active_sessions=manager.active_count,
        environment=settings.ENVIRONMENT,
    )


@router.get("/api/config", tags=["System"])
def public_config(shared: SharedComponents = Depends(get_shared)):
    """Non-secret runtime configuration the dashboard uses for labels and defaults."""
    r = shared.router
    return {
        "embedding_provider": shared.embedding_provider.provider_name,
        "embedding_simulated": shared.embedding_provider.is_simulated,
        "llm_configured": shared.gateway.is_configured,
        "routing_policy": r.policy,
        "models": [
            {
                "model_id": m.model_id,
                "tier": m.tier,
                "cost_input_per_1k": m.cost_input_per_1k,
                "cost_output_per_1k": m.cost_output_per_1k,
                "max_context_tokens": m.max_context_tokens,
            }
            for m in r.profiles
        ],
        "pricing_is_estimate": True,
        "cache_similarity_threshold": settings.SIMILARITY_THRESHOLD,
        "retrieval_top_k": settings.RETRIEVAL_TOP_K,
        "retrieval_max_top_k": settings.RETRIEVAL_MAX_TOP_K,
        "retrieval_min_relevance": settings.RETRIEVAL_MIN_RELEVANCE,
        "rag_cache_min_evidence_overlap": settings.CACHE_MIN_EVIDENCE_OVERLAP,
        "max_upload_mb": settings.MAX_UPLOAD_MB,
        "allowed_file_types": [".pdf", ".txt"],
        "chunk_size_chars": settings.CHUNK_SIZE_CHARS,
        "chunk_overlap_chars": settings.CHUNK_OVERLAP_CHARS,
        "telemetry_stores_query_text": settings.TELEMETRY_STORE_QUERY_TEXT,
        "session_idle_ttl_minutes": settings.SESSION_IDLE_TTL_MINUTES,
        "session_max_documents": settings.SESSION_MAX_DOCUMENTS,
        "session_max_upload_mb": settings.SESSION_MAX_UPLOAD_MB,
    }


# ------------------------------------------------------------------ sessions


def _session_out(manager: SessionManager, workspace: Workspace, token: Optional[str] = None) -> SessionOut:
    return SessionOut(
        session_token=token,
        expires_in_seconds=round(manager.expires_in(workspace)),
        idle_ttl_seconds=round(manager.idle_ttl_seconds),
        max_documents=settings.SESSION_MAX_DOCUMENTS,
        max_upload_mb=settings.SESSION_MAX_UPLOAD_MB,
    )


@router.post("/api/session", response_model=SessionOut, status_code=201, tags=["Session"])
@limiter.limit(settings.SESSION_RATE_LIMIT)
def create_session(request: Request, manager: SessionManager = Depends(get_session_manager)):
    """Start a private, empty workspace. Send the returned token in the
    X-Session-Token header; it is the only key to this workspace."""
    token, workspace = manager.create()
    return _session_out(manager, workspace, token)


@router.get("/api/session", response_model=SessionOut, tags=["Session"])
def get_session_info(
    workspace: Workspace = Depends(get_workspace),
    manager: SessionManager = Depends(get_session_manager),
):
    return _session_out(manager, workspace)


@router.delete("/api/session", tags=["Session"])
def end_session(
    x_session_token: Optional[str] = Header(default=None, alias=SESSION_HEADER),
    manager: SessionManager = Depends(get_session_manager),
):
    """Delete this session's documents, chunks, embeddings, cached answers
    and telemetry. Only the session's own token can do this."""
    if not manager.close(x_session_token):
        raise HTTPException(status_code=401, detail="Session not found or already ended.")
    return {"message": "Session ended; its documents, answers and telemetry were deleted."}


@router.post("/api/session/close", status_code=204, tags=["Session"])
async def end_session_beacon(request: Request, manager: SessionManager = Depends(get_session_manager)):
    """Same as DELETE /api/session, for navigator.sendBeacon when the tab
    closes: the token is the plain-text body, because beacons cannot set
    headers. Always returns 204 so it reveals nothing about the token."""
    body = await request.body()
    if len(body) <= 128:
        manager.close(body.decode("ascii", errors="ignore").strip())
    return Response(status_code=204)


@router.post("/api/query", response_model=QueryResponse, tags=["Query"])
@limiter.limit(settings.QUERY_RATE_LIMIT)
def process_query(
    request: Request,
    query_request: QueryRequest,
    service: QueryService = Depends(get_query_service),
):
    """General-knowledge mode: answer from the model alone, without documents.

    Kept for backward compatibility. Answers are not grounded or cited."""
    return service.process_query(
        query_request,
        request_id=getattr(request.state, "request_id", None),
    )


@router.post("/api/rag/query", response_model=RAGQueryResponse, tags=["RAG"])
@limiter.limit(settings.QUERY_RATE_LIMIT)
def rag_query(
    request: Request,
    rag_request: RAGQueryRequest,
    rag: RAGService = Depends(get_rag_service),
):
    """Answer a question from the selected collection's documents, with citations."""
    return rag.answer(rag_request, request_id=getattr(request.state, "request_id", None))


# ----------------------------------------------------------------- documents


@router.get("/api/collections", response_model=List[CollectionOut], tags=["Documents"])
def list_collections(rag: RAGService = Depends(get_rag_service)):
    return [
        CollectionOut(
            name=c["collection"],
            document_count=c["document_count"],
            ready_count=c["ready_count"] or 0,
            chunk_count=c["chunk_count"] or 0,
        )
        for c in rag.store.list_collections()
    ]


@router.get("/api/documents", response_model=List[DocumentOut], tags=["Documents"])
def list_documents(
    collection: Optional[str] = Query(default=None, max_length=64),
    rag: RAGService = Depends(get_rag_service),
):
    model = rag.ingestion.embedding_model
    return [_document_out(d, model) for d in rag.store.list(collection)]


@router.get("/api/documents/{document_id}", response_model=DocumentOut, tags=["Documents"])
def get_document(document_id: str, rag: RAGService = Depends(get_rag_service)):
    doc = rag.store.get(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return _document_out(doc, rag.ingestion.embedding_model)


@router.post("/api/documents", response_model=UploadResponse, status_code=202, tags=["Documents"])
@limiter.limit(settings.UPLOAD_RATE_LIMIT)
async def upload_document(
    request: Request,
    background: BackgroundTasks,
    file: UploadFile = File(...),
    collection: str = Form(default="default"),
    rag: RAGService = Depends(get_rag_service),
):
    """Upload a PDF or TXT file into this session's workspace. Processing
    (extraction, chunking, embedding) runs in the background; poll
    GET /api/documents/{id} for its status."""
    limit = rag.ingestion.max_upload_bytes
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit + 64 * 1024:
        raise HTTPException(status_code=413, detail=f"The file exceeds the {settings.MAX_UPLOAD_MB:g} MB upload limit.")
    data = await file.read(limit + 1)
    await file.close()
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"The file exceeds the {settings.MAX_UPLOAD_MB:g} MB upload limit.")
    _check_session_quota(rag, collection, file.filename, len(data))
    try:
        outcome = rag.ingestion.register_upload(collection, file.filename, data)
    except DocumentValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    doc = outcome.document
    if outcome.outcome == "duplicate":
        message = f"Identical content is already indexed as '{doc.filename}'; nothing was re-processed."
    else:
        background.add_task(_process_upload, rag.ingestion, doc.document_id, doc.version, data)
        message = (
            f"Replacing '{doc.filename}' with version {doc.version}; cached answers for this collection were invalidated."
            if outcome.outcome == "replaced"
            else f"'{doc.filename}' was accepted and is being processed."
        )
    return UploadResponse(
        document=_document_out(doc, rag.ingestion.embedding_model),
        outcome=outcome.outcome,
        message=message,
    )


def _check_session_quota(rag: RAGService, collection: str, filename: Optional[str], size: int) -> None:
    """Bound the memory one session can use. A same-name upload replaces
    the earlier file, so it does not count twice."""
    existing = rag.store.list(None)
    try:
        name = sanitize_filename(filename)
    except DocumentValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    kept = [d for d in existing if not (d.collection == collection and d.filename == name)]
    if len(kept) >= settings.SESSION_MAX_DOCUMENTS:
        raise HTTPException(
            status_code=409,
            detail=f"This session already has {settings.SESSION_MAX_DOCUMENTS} documents. Delete one first.",
        )
    total = sum(d.size_bytes for d in kept) + size
    if total > settings.SESSION_MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"This session's uploads would exceed {settings.SESSION_MAX_UPLOAD_MB:g} MB in total. Delete a document first.",
        )


@router.post("/api/documents/{document_id}/reindex", response_model=DocumentOut, status_code=202, tags=["Documents"])
@limiter.limit(settings.UPLOAD_RATE_LIMIT)
def reindex_document(
    request: Request,
    document_id: str,
    background: BackgroundTasks,
    rag: RAGService = Depends(get_rag_service),
):
    """Re-chunk and re-embed a document from its stored text (new version)."""
    try:
        doc = rag.ingestion.start_reindex(document_id)
    except DocumentValidationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    background.add_task(_process_upload, rag.ingestion, doc.document_id, doc.version, None)
    return _document_out(doc, rag.ingestion.embedding_model)


@router.delete("/api/documents/{document_id}", tags=["Documents"])
def delete_document(document_id: str, rag: RAGService = Depends(get_rag_service)):
    """Delete a document with its pages, chunks, embeddings and the cached
    answers of its collection."""
    doc = rag.ingestion.delete(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"message": f"Deleted '{doc.filename}' and its chunks, embeddings and cached answers.",
            "document_id": document_id}


# ------------------------------------------------------------------- metrics


@router.get("/api/metrics/summary", response_model=MetricsSummary, tags=["Metrics"])
def get_metrics_summary(
    mode: Optional[str] = Query(default=None, pattern="^(general|rag)$"),
    service: QueryService = Depends(get_query_service),
):
    """Retrieve aggregate telemetry, hit rates, p50/p95 latencies, and estimated spend."""
    summary = service.metrics_store.get_summary(mode=mode)
    summary.is_simulated = service.cache.embedding_provider.is_simulated
    return summary


@router.get("/api/metrics/requests", response_model=List[RequestRecord], tags=["Metrics"])
def get_recent_requests(
    limit: int = Query(default=50, ge=1, le=200, description="Max records to return"),
    mode: Optional[str] = Query(default=None, pattern="^(general|rag)$"),
    service: QueryService = Depends(get_query_service),
):
    """Retrieve the most recent request execution records from SQLite."""
    return service.metrics_store.get_recent_requests(limit=limit, mode=mode)

@router.post("/api/metrics/reset", tags=["Metrics"], dependencies=[Depends(require_admin)])
def reset_metrics_and_cache(
    service: QueryService = Depends(get_query_service),
):
    """Clear this session's telemetry and semantic cache (documents are kept).

    Open in development; requires X-Admin-Token in production. Visitors use
    DELETE /api/session to clear their own workspace."""
    service.cache.clear()
    service.metrics_store.clear()
    return {"message": "Cache and metrics reset successfully."}
