
"""Embedding providers for the Cost-Aware RAG semantic cache."""

from __future__ import annotations

import hashlib
import threading
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class BaseEmbeddingProvider(ABC):
    """Common interface for embedding providers."""

    provider_name: str
    dimension: int
    is_simulated: bool

    @abstractmethod
    def embed_text(self, text: str | None) -> list[float]:
        """Generate an embedding for one text."""
        raise NotImplementedError

    @abstractmethod
    def embed_batch(
        self, texts: list[str]
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        raise NotImplementedError


class SentenceTransformersEmbeddingProvider(BaseEmbeddingProvider):
    """Generate real dense embeddings using a local Sentence Transformers model."""

    _model_cache: dict[str, SentenceTransformer] = {}
    _model_cache_lock = threading.Lock()

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")

        self.model_name = model_name.strip()
        self.provider_name = (
            f"sentence-transformers/{self.model_name}"
        )
        self.is_simulated = False

        # Load each configured model once per Python process.
        with self._model_cache_lock:
            if self.model_name not in self._model_cache:
                try:
                    self._model_cache[self.model_name] = (
                        SentenceTransformer(self.model_name)
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "Could not initialize local embedding model "
                        f"'{self.model_name}': {exc}"
                    ) from exc

            self._model = self._model_cache[self.model_name]

        self.dimension = int(
            self._model.get_sentence_embedding_dimension()
        )

        if self.dimension <= 0:
            raise RuntimeError(
                "Embedding model returned an invalid dimension"
            )

    def embed_text(self, text: str | None) -> list[float]:
        """Return a normalized embedding, or a zero vector for empty input."""

        if text is None:
            return [0.0] * self.dimension

        if not isinstance(text, str):
            raise TypeError("Expected string input")

        if not text.strip():
            return [0.0] * self.dimension

        return self.embed_batch([text])[0]

    def embed_batch(
        self, texts: list[str]
    ) -> list[list[float]]:
        """Encode texts in a batch and return normalized Python float lists."""

        if not isinstance(texts, list) or any(
            not isinstance(text, str) for text in texts
        ):
            raise TypeError("Expected list of strings")

        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        nonempty_texts: list[str] = []
        nonempty_indices: list[int] = []

        for index, text in enumerate(texts):
            if text.strip():
                nonempty_texts.append(text)
                nonempty_indices.append(index)
            else:
                results[index] = [0.0] * self.dimension

        if nonempty_texts:
            try:
                vectors = self._model.encode(
                    nonempty_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                vectors = np.asarray(vectors, dtype=np.float32)

                if vectors.ndim == 1:
                    vectors = vectors.reshape(1, -1)

                if (
                    vectors.ndim != 2
                    or vectors.shape[0] != len(nonempty_texts)
                    or vectors.shape[1] != self.dimension
                ):
                    raise ValueError(
                        "Embedding model returned an unexpected shape"
                    )

                if not np.isfinite(vectors).all():
                    raise ValueError(
                        "Embedding model returned non-finite values"
                    )

                for index, vector in zip(
                    nonempty_indices, vectors
                ):
                    results[index] = [
                        float(value) for value in vector
                    ]

            except Exception as exc:
                raise RuntimeError(
                    f"Embedding generation failed: {exc}"
                ) from exc

        return [
            vector if vector is not None
            else [0.0] * self.dimension
            for vector in results
        ]


class SimulatedEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic feature-hashing vectors for explicit mock/testing mode.

    These are NOT genuine semantic embeddings.
    """

    def __init__(self, dimension: int = 384):
        if dimension <= 0:
            raise ValueError("dimension must be positive")

        self.dimension = dimension
        self.provider_name = "simulated-feature-hashing"
        self.is_simulated = True

    def embed_text(self, text: str | None) -> list[float]:
        if text is None:
            return [0.0] * self.dimension

        if not isinstance(text, str):
            raise TypeError("Expected string input")

        if not text.strip():
            return [0.0] * self.dimension

        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = text.lower().split()

        for token in tokens:
            digest = hashlib.sha256(
                token.encode("utf-8")
            ).digest()

            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm

        return [float(value) for value in vector]

    def embed_batch(
        self, texts: list[str]
    ) -> list[list[float]]:
        if not isinstance(texts, list) or any(
            not isinstance(text, str) for text in texts
        ):
            raise TypeError("Expected list of strings")

        return [self.embed_text(text) for text in texts]


def get_embedding_provider(
    provider_type: str = "sentence-transformers",
    model_name: str = "all-MiniLM-L6-v2",
) -> BaseEmbeddingProvider:
    """Construct the configured embedding provider."""

    normalized_type = provider_type.strip().lower()

    if normalized_type == "sentence-transformers":
        return SentenceTransformersEmbeddingProvider(model_name)

    if normalized_type == "simulated":
        return SimulatedEmbeddingProvider()

    raise ValueError(
        f"Unknown embedding provider type: {provider_type}"
    )
