"""In-memory exact cosine search over chunk embeddings persisted in SQLite.

SQLite is the source of truth. Each collection's vectors are loaded into a
contiguous float32 matrix on first use and rebuilt after any write to that
collection. Exact search is a single matrix-vector product, which is fast
enough for the portfolio-scale corpora this app targets (tens of thousands of
chunks); an ANN index such as FAISS would be the next step beyond that.
"""

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .store import ChunkRecord, DocumentStore


@dataclass
class RetrievedChunk:
    chunk: ChunkRecord
    score: float
    rank: int


class _CollectionIndex:
    def __init__(self, chunks: List[ChunkRecord]):
        self.chunks = chunks
        if chunks:
            matrix = np.vstack([c.embedding for c in chunks]).astype(np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.matrix = matrix / norms
        else:
            self.matrix = np.zeros((0, 0), dtype=np.float32)
        self.doc_ids = np.array([c.document_id for c in chunks])


class VectorIndex:
    def __init__(self, store: DocumentStore, embedding_model: str):
        self.store = store
        self.embedding_model = embedding_model
        self._indexes: Dict[str, _CollectionIndex] = {}
        self._lock = threading.Lock()

    def invalidate(self, collection: Optional[str] = None) -> None:
        with self._lock:
            if collection is None:
                self._indexes.clear()
            else:
                self._indexes.pop(collection, None)

    def _get(self, collection: str) -> _CollectionIndex:
        with self._lock:
            index = self._indexes.get(collection)
        if index is None:
            index = _CollectionIndex(
                self.store.load_chunks(collection, self.embedding_model)
            )
            with self._lock:
                self._indexes[collection] = index
        return index

    def chunk_count(self, collection: str) -> int:
        return len(self._get(collection).chunks)

    def search(
        self,
        collection: str,
        query_embedding: Sequence[float],
        top_k: int,
        min_score: float = -1.0,
        document_ids: Optional[Sequence[str]] = None,
    ) -> List[RetrievedChunk]:
        index = self._get(collection)
        if not index.chunks or top_k <= 0:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        if q.shape[0] != index.matrix.shape[1]:
            # Embedding dimension mismatch: vectors from another provider.
            return []
        norm = float(np.linalg.norm(q))
        if norm == 0.0:
            return []
        scores = index.matrix @ (q / norm)
        if document_ids:
            mask = np.isin(index.doc_ids, list(document_ids))
            scores = np.where(mask, scores, -np.inf)
        k = min(top_k, len(index.chunks))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        results: List[RetrievedChunk] = []
        for i in top:
            score = float(scores[i])
            if not np.isfinite(score) or score < min_score:
                continue
            results.append(
                RetrievedChunk(chunk=index.chunks[i], score=round(score, 4), rank=len(results) + 1)
            )
        return results
