import os
import tempfile
import pytest
from app.cache.embeddings import (
    SentenceTransformersEmbeddingProvider,
    SimulatedEmbeddingProvider,
)
from app.cache.semantic import SemanticCache, cosine_similarity

# --- Genuine Dense Embedding Tests ---

def test_real_embedding_provider_properties():
    provider = SentenceTransformersEmbeddingProvider("all-MiniLM-L6-v2")
    assert provider.dimension == 384
    assert provider.is_simulated is False
    assert "sentence-transformers/all-MiniLM-L6-v2" in provider.provider_name

    vec = provider.embed_text("What is semantic caching?")
    assert len(vec) == 384
    
    # Check normalized unit vector
    norm = sum(x * x for x in vec)
    assert pytest.approx(norm, rel=1e-3) == 1.0

def test_real_embedding_batch():
    provider = SentenceTransformersEmbeddingProvider("all-MiniLM-L6-v2")
    texts = [
        "What is semantic caching?",
        "How to optimize LLM inference latency?",
    ]
    vecs = provider.embed_batch(texts)
    assert len(vecs) == 2
    assert len(vecs[0]) == 384
    assert len(vecs[1]) == 384

def test_real_semantic_paraphrase_hit():
    """Verify that actual sentence embeddings produce a similarity score and hit when threshold <= score."""
    provider = SentenceTransformersEmbeddingProvider("all-MiniLM-L6-v2")
    cache = SemanticCache(embedding_provider=provider)

    # Store base query
    cache.store(
        query="What is semantic caching in large language models?",
        response="Semantic caching stores embeddings and responses to reuse past LLM generations.",
    )

    # Paraphrased query
    paraphrase = "What is semantic caching in LLMs?"
    
    # Check baseline similarity with threshold=0.0
    baseline = cache.lookup(paraphrase, threshold=0.0)
    actual_score = baseline.similarity_score
    assert actual_score is not None
    assert actual_score > 0.60  # Meaningful semantic similarity

    # When threshold is set <= actual_score, it yields a cache hit
    hit_result = cache.lookup(paraphrase, threshold=round(actual_score - 0.05, 2))
    assert hit_result.is_hit is True
    assert hit_result.matched_entry is not None
    assert "Semantic caching stores" in hit_result.matched_entry.response
    assert hit_result.is_simulated is False


def test_real_semantic_unrelated_query_miss():
    """Verify that unrelated queries do not hit the semantic cache."""
    provider = SentenceTransformersEmbeddingProvider("all-MiniLM-L6-v2")
    cache = SemanticCache(embedding_provider=provider, default_threshold=0.70)

    cache.store(
        query="What is the capital of France?",
        response="The capital of France is Paris.",
    )

    unrelated = "How do quantum computers execute Shor's algorithm?"
    result = cache.lookup(unrelated)

    assert result.is_hit is False
    # Semantic similarity between geography and quantum computing should be very low
    if result.similarity_score is not None:
        assert result.similarity_score < 0.40

def test_threshold_sensitivity():
    """Verify that adjusting the threshold dynamically changes hit/miss decision based on embedding score."""
    provider = SentenceTransformersEmbeddingProvider("all-MiniLM-L6-v2")
    cache = SemanticCache(embedding_provider=provider)

    cache.store(
        query="How does vector indexing speed up search?",
        response="Vector indexes use approximate nearest neighbors algorithms like HNSW.",
    )

    query = "Why are vector indices faster for similarity lookup?"

    # Check exact score with no threshold constraint
    baseline = cache.lookup(query, threshold=0.0)
    score = baseline.similarity_score
    assert score is not None

    # Strict threshold higher than score -> Miss
    strict_res = cache.lookup(query, threshold=min(0.99, score + 0.05))
    assert strict_res.is_hit is False

    # Lenient threshold lower than score -> Hit
    lenient_res = cache.lookup(query, threshold=max(0.10, score - 0.05))
    assert lenient_res.is_hit is True

def test_cache_persistence_across_restarts():
    """Verify that cached entries survive application restarts using SQLite."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    try:
        provider = SentenceTransformersEmbeddingProvider("all-MiniLM-L6-v2")
        
        # Instance 1: Store entry into persistent cache
        cache1 = SemanticCache(embedding_provider=provider, db_path=db_path)
        cache1.store(
            query="Persistent query test",
            response="Persistent response test",
            metadata={"source": "disk"},
        )
        assert len(cache1._entries) == 1

        # Instance 2: Simulate restart by creating new instance pointing to same DB
        cache2 = SemanticCache(embedding_provider=provider, db_path=db_path)
        assert len(cache2._entries) == 1
        
        # Verify query can hit the reloaded cache
        hit = cache2.lookup("Persistent query test")
        assert hit.is_hit is True
        assert hit.matched_entry.response == "Persistent response test"
        assert hit.matched_entry.metadata.get("source") == "disk"
    finally:
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
        except PermissionError:
            pass

def test_empty_input_and_invalid_storage():
    """Verify safe handling of empty inputs and rejected empty responses."""
    provider = SentenceTransformersEmbeddingProvider("all-MiniLM-L6-v2")
    cache = SemanticCache(embedding_provider=provider)

    # Empty lookup returns miss
    res = cache.lookup("   ")
    assert res.is_hit is False

    # Storing empty response is rejected
    stored = cache.store("Query", "   ")
    assert stored is None
    assert len(cache._entries) == 0

# --- Simulated Provider Isolated Unit Tests ---

def test_simulated_embedding_provider_properties():
    provider = SimulatedEmbeddingProvider(dimension=64)
    assert provider.dimension == 64
    assert provider.is_simulated is True
    
    vec1 = provider.embed_text("What is semantic caching?")
    assert len(vec1) == 64
    norm = sum(x * x for x in vec1)
    assert pytest.approx(norm, rel=1e-3) == 1.0

def test_cosine_similarity():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    assert pytest.approx(cosine_similarity(v1, v2)) == 1.0

    v3 = [0.0, 1.0, 0.0]
    assert pytest.approx(cosine_similarity(v1, v3)) == 0.0

def test_semantic_similarity_diagnostic():
    provider = SentenceTransformersEmbeddingProvider("all-MiniLM-L6-v2")

    queries = [
        "What is semantic caching in large language models?",
        "Explain how semantic caching works for LLMs.",
        "How can a language model save computation by reusing answers from earlier similar questions?",
    ]

    vectors = provider.embed_batch(queries)

    print("\n--- Semantic Similarity Diagnostic ---")
    for i in range(len(queries)):
        for j in range(i + 1, len(queries)):
            score = cosine_similarity(vectors[i], vectors[j])
            print(f"Similarity Q{i + 1} vs Q{j + 1}: {score:.4f}")

    assert all(len(vector) == 384 for vector in vectors)


# --- Phase 2: Cache Edge-Case Tests ---

def test_cosine_similarity_edge_cases():
    assert cosine_similarity([], []) == 0.0
    assert cosine_similarity([1.0, 0.0], [1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert cosine_similarity([float("nan")], [1.0]) == 0.0
    assert cosine_similarity([float("inf")], [1.0]) == 0.0



def test_cache_lru_eviction():
    provider = SimulatedEmbeddingProvider(dimension=64)
    cache = SemanticCache(
        embedding_provider=provider,
        max_entries=2,
    )

    cache.store("Query A", "Response A")
    cache.store("Query B", "Response B")

    # Explicitly update A's access time to make B the least recently used.
    entry_a = next(
        entry for entry in cache._entries.values()
        if entry.query == "Query A"
    )
    entry_b = next(
        entry for entry in cache._entries.values()
        if entry.query == "Query B"
    )

    entry_a.last_accessed_at = 200.0
    entry_b.last_accessed_at = 100.0

    cache.store("Query C", "Response C")

    assert len(cache._entries) == 2

    cached_queries = {
        entry.query for entry in cache._entries.values()
    }

    assert "Query A" in cached_queries
    assert "Query C" in cached_queries
    assert "Query B" not in cached_queries


def test_clear_cache_and_stats():
    provider = SimulatedEmbeddingProvider(dimension=64)
    cache = SemanticCache(embedding_provider=provider)

    cache.store("Test query", "Test response")

    stats = cache.get_stats()
    assert stats["total_entries"] == 1
    assert stats["max_entries"] == cache.max_entries
    assert stats["is_simulated"] is True

    cache.clear()

    assert len(cache._entries) == 0
    assert cache.get_stats()["total_entries"] == 0


def test_embedding_failure_during_store_and_lookup():
    class FailingEmbeddingProvider(SimulatedEmbeddingProvider):
        def embed_text(self, text):
            raise RuntimeError("Simulated embedding failure")

    provider = FailingEmbeddingProvider(dimension=64)
    cache = SemanticCache(embedding_provider=provider)

    # Storage should fail safely.
    result = cache.store("Test query", "Test response")
    assert result is None
    assert len(cache._entries) == 0

    # Lookup should return a miss without raising an exception.
    # Add an entry with a working provider first.
    working_provider = SimulatedEmbeddingProvider(dimension=64)
    working_cache = SemanticCache(embedding_provider=working_provider)
    working_cache.store("Existing query", "Existing response")

    working_cache.embedding_provider = provider
    lookup_result = working_cache.lookup("A new query")

    assert lookup_result.is_hit is False
    assert lookup_result.matched_entry is None
