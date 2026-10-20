import pytest
from unittest.mock import patch, MagicMock
from app.cache.embeddings import (
    BaseEmbeddingProvider,
    SentenceTransformersEmbeddingProvider,
    SimulatedEmbeddingProvider,
    get_embedding_provider,
)

# ==============================================================================
# Genuine Dense Embedding Tests (sentence-transformers/all-MiniLM-L6-v2)
# ==============================================================================

def test_embedding_output_dimensions():
    """Verify that genuine dense embeddings produce exactly 384-dimensional vectors."""
    provider = SentenceTransformersEmbeddingProvider("all-MiniLM-L6-v2")
    assert provider.dimension == 384
    assert provider.is_simulated is False
    assert provider.provider_name == "sentence-transformers/all-MiniLM-L6-v2"

    vector = provider.embed_text("What is semantic caching?")
    assert isinstance(vector, list)
    assert len(vector) == 384
    assert all(isinstance(x, float) for x in vector)

def test_embedding_repeatability():
    """Verify that identical inputs produce identical numerical embedding vectors."""
    provider = SentenceTransformersEmbeddingProvider("all-MiniLM-L6-v2")
    query = "Repeatability test for deterministic dense vector generation."
    
    vec1 = provider.embed_text(query)
    vec2 = provider.embed_text(query)

    assert len(vec1) == len(vec2) == 384
    for a, b in zip(vec1, vec2):
        assert a == pytest.approx(b, abs=1e-6)

def test_embedding_unit_normalization():
    """Verify that embeddings are L2-normalized to unit hypersphere (sum of squares == 1.0)."""
    provider = SentenceTransformersEmbeddingProvider("all-MiniLM-L6-v2")
    vector = provider.embed_text("Normalize embedding vectors for cosine similarity.")
    
    norm_squared = sum(x * x for x in vector)
    assert pytest.approx(norm_squared, rel=1e-3) == 1.0

def test_embedding_batch_generation():
    """Verify batch embedding generation produces consistent vectors for all items."""
    provider = SentenceTransformersEmbeddingProvider("all-MiniLM-L6-v2")
    texts = [
        "First batch query",
        "Second query with different content",
        "Third query for latency testing",
    ]
    vectors = provider.embed_batch(texts)
    assert len(vectors) == 3
    for vec in vectors:
        assert len(vec) == 384
        norm_squared = sum(x * x for x in vec)
        assert pytest.approx(norm_squared, rel=1e-3) == 1.0

def test_embedding_empty_and_whitespace_input():
    """Verify that empty strings, whitespace, and None return zero-vectors safely."""
    provider = SentenceTransformersEmbeddingProvider("all-MiniLM-L6-v2")
    
    # None input
    zero_vec_none = provider.embed_text(None)
    assert len(zero_vec_none) == 384
    assert all(x == 0.0 for x in zero_vec_none)

    # Empty string
    zero_vec_empty = provider.embed_text("")
    assert len(zero_vec_empty) == 384
    assert all(x == 0.0 for x in zero_vec_empty)

    # Whitespace string
    zero_vec_ws = provider.embed_text("   \n\t  ")
    assert len(zero_vec_ws) == 384
    assert all(x == 0.0 for x in zero_vec_ws)

def test_embedding_invalid_type_input():
    """Verify that non-string inputs raise TypeError."""
    provider = SentenceTransformersEmbeddingProvider("all-MiniLM-L6-v2")
    
    with pytest.raises(TypeError, match="Expected string input"):
        provider.embed_text(12345)  # type: ignore

    with pytest.raises(TypeError, match="Expected list of strings"):
        provider.embed_batch("not a list")  # type: ignore

def test_model_caching_and_singleton_reuse():
    """Verify that loading the same model twice reuses the cached model instance."""
    p1 = SentenceTransformersEmbeddingProvider("all-MiniLM-L6-v2")
    p2 = SentenceTransformersEmbeddingProvider("all-MiniLM-L6-v2")

    assert p1._model is p2._model
    assert id(p1._model) == id(p2._model)

def test_embedding_factory():
    """Verify get_embedding_provider factory instantiates correct providers."""
    real_p = get_embedding_provider("sentence-transformers", "all-MiniLM-L6-v2")
    assert isinstance(real_p, SentenceTransformersEmbeddingProvider)
    assert real_p.is_simulated is False

    mock_p = get_embedding_provider("simulated")
    assert isinstance(mock_p, SimulatedEmbeddingProvider)
    assert mock_p.is_simulated is True

    with pytest.raises(ValueError, match="Unknown embedding provider type"):
        get_embedding_provider("unsupported-provider")


# ==============================================================================
# Isolated Unit Tests with Mocks (Failure Handling)
# ==============================================================================

def test_mock_model_loading_failure():
    """[MOCK TEST] Verify that an unresolvable or corrupted model name raises RuntimeError."""
    with patch("sentence_transformers.SentenceTransformer", side_effect=Exception("Model weights missing")):
        with pytest.raises(RuntimeError, match="Could not initialize local embedding model"):
            # Use unique model name to avoid hitting existing _model_cache
            SentenceTransformersEmbeddingProvider("nonexistent-corrupted-model-name-xyz")

def test_mock_inference_error_handling():
    """[MOCK TEST] Verify that an unexpected runtime failure during model.encode raises RuntimeError."""
    provider = SentenceTransformersEmbeddingProvider("all-MiniLM-L6-v2")
    
    with patch.object(provider._model, "encode", side_effect=RuntimeError("CUDA out of memory")):
        with pytest.raises(RuntimeError, match="Embedding generation failed"):
            provider.embed_text("Query that triggers inference failure")
