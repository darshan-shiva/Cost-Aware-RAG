"""Document upload validation, processing, re-indexing and deletion."""

import hashlib
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Callable, List, Optional

from ..cache.embeddings import BaseEmbeddingProvider
from .chunking import chunk_pages
from .extraction import (
    DocumentValidationError,
    PageText,
    extract_pdf,
    extract_txt,
)
from .store import ChunkRecord, DocumentRecord, DocumentStore

logger = logging.getLogger("cost_aware_rag.ingestion")

ALLOWED_EXTENSIONS = {".pdf": "pdf", ".txt": "txt"}
COLLECTION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._ ()-]+")


def validate_collection(name: str) -> str:
    name = (name or "").strip()
    if not COLLECTION_PATTERN.match(name):
        raise DocumentValidationError(
            "Collection names must be 1-64 characters: letters, digits, '-' or '_'."
        )
    return name


def sanitize_filename(raw: Optional[str]) -> str:
    """Reduce a client-supplied filename to a safe display name.

    The name is only ever stored as metadata (files are not written to disk
    under it), but directory parts, control characters and unusual symbols
    are still removed so it cannot be confused with a path when displayed or
    logged.
    """
    name = (raw or "").replace("\\", "/").split("/")[-1]
    name = _UNSAFE_FILENAME_CHARS.sub("_", name).strip(" .")
    name = re.sub(r"_+", "_", name)
    if not name:
        raise DocumentValidationError("The file name is missing or invalid.")
    stem, ext = os.path.splitext(name)
    return stem[:120] + ext.lower()[:10]


@dataclass
class UploadOutcome:
    document: DocumentRecord
    outcome: str  # "created" | "replaced" | "duplicate"


class IngestionService:
    def __init__(
        self,
        store: DocumentStore,
        embedding_provider: BaseEmbeddingProvider,
        chunk_size: int,
        chunk_overlap: int,
        min_chunk_chars: int,
        max_upload_bytes: int,
        max_pdf_pages: int,
        max_document_chars: int,
        embedding_batch_size: int = 32,
        on_collection_changed: Optional[Callable[[str], None]] = None,
    ):
        self.store = store
        self.embedding_provider = embedding_provider
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_chars = min_chunk_chars
        self.max_upload_bytes = max_upload_bytes
        self.max_pdf_pages = max_pdf_pages
        self.max_document_chars = max_document_chars
        self.embedding_batch_size = max(1, embedding_batch_size)
        self._listeners: List[Callable[[str], None]] = []
        if on_collection_changed:
            self._listeners.append(on_collection_changed)

    @property
    def embedding_model(self) -> str:
        return self.embedding_provider.provider_name

    def add_listener(self, callback: Callable[[str], None]) -> None:
        self._listeners.append(callback)

    def _notify(self, collection: str) -> None:
        for callback in self._listeners:
            try:
                callback(collection)
            except Exception:
                logger.exception("collection_change_listener_failed")

    # ------------------------------------------------------------ validation

    def validate_upload(self, filename: Optional[str], data: bytes) -> tuple:
        safe_name = sanitize_filename(filename)
        ext = os.path.splitext(safe_name)[1].lower()
        file_type = ALLOWED_EXTENSIONS.get(ext)
        if file_type is None:
            raise DocumentValidationError("Only .pdf and .txt files are supported.")
        if not data:
            raise DocumentValidationError("The file is empty.")
        if len(data) > self.max_upload_bytes:
            limit_mb = self.max_upload_bytes / (1024 * 1024)
            raise DocumentValidationError(
                f"The file exceeds the {limit_mb:g} MB upload limit."
            )
        if file_type == "pdf" and not data.startswith(b"%PDF-"):
            raise DocumentValidationError(
                "The file has a .pdf extension but is not a PDF."
            )
        if file_type == "txt" and data.startswith(b"%PDF-"):
            raise DocumentValidationError(
                "The file has a .txt extension but contains a PDF."
            )
        return safe_name, file_type

    # ---------------------------------------------------------------- upload

    def register_upload(
        self, collection: str, filename: Optional[str], data: bytes
    ) -> UploadOutcome:
        """Validate and record an upload. Call ``process`` afterwards unless
        the outcome is ``duplicate``."""
        collection = validate_collection(collection)
        safe_name, file_type = self.validate_upload(filename, data)
        sha = hashlib.sha256(data).hexdigest()

        duplicate = self.store.find_by_sha(collection, sha)
        if duplicate is not None and duplicate.status != "failed":
            return UploadOutcome(document=duplicate, outcome="duplicate")

        existing = self.store.find_by_filename(collection, safe_name)
        if existing is not None:
            # Same name, new content: replace in place under a stable ID.
            self.store.begin_new_version(
                existing.document_id, file_type, len(data), sha
            )
            self._notify(collection)
            return UploadOutcome(
                document=self.store.get(existing.document_id), outcome="replaced"
            )

        now = time.time()
        record = DocumentRecord(
            document_id=f"doc_{uuid.uuid4().hex[:16]}",
            collection=collection,
            filename=safe_name,
            file_type=file_type,
            size_bytes=len(data),
            content_sha256=sha,
            version=1,
            status="pending",
            error_message=None,
            page_count=0,
            chunk_count=0,
            char_count=0,
            embedding_model=None,
            created_at=now,
            updated_at=now,
        )
        self.store.create(record)
        return UploadOutcome(document=record, outcome="created")

    # ------------------------------------------------------------ processing

    def _extract(self, file_type: str, data: bytes) -> List[PageText]:
        if file_type == "pdf":
            return extract_pdf(data, self.max_pdf_pages)
        return extract_txt(data)

    def process(self, document_id: str, version: int, data: Optional[bytes] = None) -> None:
        """Extract (or reuse stored page text), chunk, embed and index.

        Safe to run in a background thread. Failures are recorded on the
        document with a user-safe message; they never raise.
        """
        doc = self.store.get(document_id)
        if doc is None or doc.version != version:
            return
        started = time.perf_counter()
        self.store.set_status(document_id, "processing")
        try:
            if data is not None:
                pages = self._extract(doc.file_type, data)
                total_chars = sum(len(p.text) for p in pages)
                if total_chars > self.max_document_chars:
                    raise DocumentValidationError(
                        f"The document has {total_chars:,} characters of text;"
                        f" the limit is {self.max_document_chars:,}."
                    )
                self.store.save_pages(
                    document_id, [(p.page_number, p.text) for p in pages]
                )
            else:
                pages = [
                    PageText(page_number=n, text=t)
                    for n, t in self.store.get_pages(document_id)
                ]

            total_chars = sum(len(p.text) for p in pages)
            if total_chars == 0:
                raise DocumentValidationError(
                    "No extractable text was found. Scanned PDFs need OCR, "
                    "which this app does not perform."
                )

            pieces = chunk_pages(
                pages, self.chunk_size, self.chunk_overlap, self.min_chunk_chars
            )
            vectors: List[List[float]] = []
            for i in range(0, len(pieces), self.embedding_batch_size):
                batch = pieces[i : i + self.embedding_batch_size]
                vectors.extend(self.embedding_provider.embed_batch([c.text for c in batch]))

            chunks = [
                ChunkRecord(
                    chunk_id=f"{document_id}:v{version}:c{piece.chunk_index}",
                    document_id=document_id,
                    collection=doc.collection,
                    document_version=version,
                    chunk_index=piece.chunk_index,
                    page_number=piece.page_number,
                    start_char=piece.start_char,
                    end_char=piece.end_char,
                    text=piece.text,
                    embedding=vec,
                )
                for piece, vec in zip(pieces, vectors)
            ]
            saved = self.store.save_chunks(
                document_id,
                version,
                chunks,
                page_count=len(pages),
                char_count=total_chars,
                embedding_model=self.embedding_model,
            )
            if saved:
                self._notify(doc.collection)
            logger.info(
                "document_indexed" if saved else "document_index_superseded",
                extra={
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
        except DocumentValidationError as exc:
            self._fail(document_id, version, str(exc))
        except Exception as exc:
            logger.exception("document_processing_failed")
            self._fail(
                document_id,
                version,
                f"Processing failed ({type(exc).__name__}). Re-index to retry.",
            )

    def _fail(self, document_id: str, version: int, message: str) -> None:
        doc = self.store.get(document_id)
        if doc is not None and doc.version == version:
            self.store.set_status(document_id, "failed", message)

    # ---------------------------------------------------- reindex and delete

    def start_reindex(self, document_id: str) -> Optional[DocumentRecord]:
        doc = self.store.get(document_id)
        if doc is None:
            return None
        if not self.store.get_pages(document_id):
            raise DocumentValidationError(
                "This document has no stored text to re-index. Upload it again."
            )
        self.store.bump_version(document_id)
        self._notify(doc.collection)
        return self.store.get(document_id)

    def delete(self, document_id: str) -> Optional[DocumentRecord]:
        doc = self.store.get(document_id)
        if doc is None:
            return None
        self.store.delete(document_id)
        self._notify(doc.collection)
        return doc
