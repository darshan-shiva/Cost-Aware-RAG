from abc import ABC, abstractmethod
import math
import hashlib
from typing import List

class BaseEmbeddingProvider(ABC):
    """Abstract interface for embedding generation."""
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @property
    @abstractmethod
    def is_simulated(self) -> bool:
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Convert a single text into a normalized embedding vector."""
        pass

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding generation."""
        return [self.embed_text(t) for t in texts]


class SimulatedEmbeddingProvider(BaseEmbeddingProvider):
    """
    Simulated embedding provider using deterministic feature hashing over
    token and character n-grams, projected into a 64-dimensional unit hypersphere.
    
    This provides reproducible vector representations where semantically and
    lexically related queries yield higher cosine similarity than unrelated queries,
    while explicitly labeling outputs as simulated.
    """

    def __init__(self, dimension: int = 64):
        self._dimension = dimension

    @property
    def provider_name(self) -> str:
        return "Simulated-Hashing-Embedder (64-dim)"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def is_simulated(self) -> bool:
        return True

    def embed_text(self, text: str) -> List[float]:
        cleaned = text.strip().lower()
        if not cleaned:
            return [0.0] * self._dimension

        vector = [0.0] * self._dimension
        tokens = cleaned.split()

        # Token hashing
        for token in tokens:
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dimension
            sign = 1.0 if (h // self._dimension) % 2 == 0 else -1.0
            vector[idx] += sign * 1.5

        # Character tri-grams hashing for morphological similarity
        if len(cleaned) >= 3:
            for i in range(len(cleaned) - 2):
                tri = cleaned[i : i + 3]
                h = int(hashlib.sha256(tri.encode("utf-8")).hexdigest(), 16)
                idx = h % self._dimension
                sign = 1.0 if (h // self._dimension) % 2 == 0 else -1.0
                vector[idx] += sign * 0.5

        # L2-normalization so dot product equals cosine similarity
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0.0:
            return [v / norm for v in vector]
        return [0.0] * self._dimension


# Phase 2 Extension Point Reference:
# class SentenceTransformersEmbeddingProvider(BaseEmbeddingProvider):
#     def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
#         from sentence_transformers import SentenceTransformer
#         self.model = SentenceTransformer(model_name)
#     ...
