from abc import ABC, abstractmethod
import math
import hashlib
import logging
import sys
from typing import List, Dict, Optional, Any

# Ensure torchvision image registration does not trigger broken C++ runtime on Windows
if "torchvision" not in sys.modules:
    sys.modules["torchvision"] = None
    sys.modules["torchvision.io"] = None

logger = logging.getLogger(__name__)

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


class SentenceTransformersEmbeddingProvider(BaseEmbeddingProvider):
    """
    Genuine dense semantic embedding provider using local SentenceTransformers models.
    Loads and caches the model once for high-throughput, low-latency reuse across requests.
    """

    _model_cache: Dict[str, Any] = {}

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._dimension: Optional[int] = None
        self._model = self._load_model(model_name)

    @classmethod
    def _load_model(cls, model_name: str) -> Any:
        if model_name not in cls._model_cache:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading SentenceTransformer model: {model_name}")
                model = SentenceTransformer(model_name)
                cls._model_cache[model_name] = model
            except Exception as e:
                logger.error(f"Failed to load embedding model '{model_name}': {e}")
                raise RuntimeError(
                    f"Could not initialize local embedding model '{model_name}': {e}"
                ) from e
        return cls._model_cache[model_name]

    @property
    def provider_name(self) -> str:
        return f"sentence-transformers/{self._model_name}"

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            # Query dimension from model or run sample text
            try:
                if hasattr(self._model, "get_embedding_dimension"):
                    self._dimension = self._model.get_embedding_dimension()
                else:
                    self._dimension = self._model.get_sentence_embedding_dimension()
            except Exception:
                sample = self._model.encode("dim check", convert_to_numpy=True)
                self._dimension = int(sample.shape[0])
        return self._dimension


    @property
    def is_simulated(self) -> bool:
        return False

    def embed_text(self, text: str) -> List[float]:
        if text is None:
            return [0.0] * self.dimension
        if not isinstance(text, str):
            raise TypeError(f"Expected string input for embedding, got {type(text).__name__}")

        cleaned = text.strip()
        if not cleaned:
            return [0.0] * self.dimension

        try:
            # normalize_embeddings=True yields unit-norm vectors for exact cosine similarity
            embedding = self._model.encode(
                cleaned,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return [float(x) for x in embedding.tolist()]
        except Exception as e:
            logger.error(f"Error generating embedding for text '{cleaned[:40]}...': {e}")
            raise RuntimeError(f"Embedding generation failed: {e}") from e

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if texts is None:
            return []
        if not isinstance(texts, list):
            raise TypeError(f"Expected list of strings for batch embedding, got {type(texts).__name__}")

        if not texts:
            return []

        cleaned_texts = []
        for t in texts:
            if t is None:
                cleaned_texts.append("")
            elif not isinstance(t, str):
                raise TypeError(f"Expected string element in texts list, got {type(t).__name__}")
            else:
                cleaned_texts.append(t.strip())

        try:
            embeddings = self._model.encode(
                cleaned_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return [[float(x) for x in emb.tolist()] for emb in embeddings]
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise RuntimeError(f"Batch embedding generation failed: {e}") from e



class SimulatedEmbeddingProvider(BaseEmbeddingProvider):
    """
    Simulated embedding provider using deterministic feature hashing over
    token and character n-grams, projected into a 64-dimensional unit hypersphere.
    Retained as an isolated mock provider for testing and fallback scenarios.
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
        if text is None:
            return [0.0] * self._dimension
        if not isinstance(text, str):
            raise TypeError(f"Expected string input, got {type(text).__name__}")

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

def get_embedding_provider(
    provider_type: str = "sentence-transformers",
    model_name: str = "all-MiniLM-L6-v2",
) -> BaseEmbeddingProvider:
    """Factory helper to obtain the configured embedding provider."""
    if provider_type == "sentence-transformers":
        return SentenceTransformersEmbeddingProvider(
            model_name=model_name
        )
    elif provider_type == "simulated":
        return SimulatedEmbeddingProvider()
    else:
        raise ValueError(
            f"Unknown embedding provider type: {provider_type}"
        )
