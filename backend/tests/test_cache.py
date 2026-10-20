import pytest
from app.cache.embeddings import SimulatedEmbeddingProvider
from app.cache.semantic import SemanticCache, cosine_similarity

def test_simulated_embedding_provider_properties():
    provider = SimulatedEmbeddingProvider(dimension=64)
    assert provider.dimension == 64
    assert provider.is_simulated is True
    
    vec1 = provider.embed_text("What is semantic caching?")
    assert len(vec1) == 64
    
    # Check unit norm
    norm = sum(x * x for x in vec1)
    assert pytest.approx(norm, rel=1e-3) == 1.0

def test_cosine_similarity():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    assert pytest.approx(cosine_similarity(v1, v2)) == 1.0

    v3 = [0.0, 1.0, 0.0]
    assert pytest.approx(cosine_similarity(v1, v3)) == 0.0

def test_semantic_cache_hit_and_miss():
    cache = SemanticCache(default_threshold=0.80)
    
    # Initially empty
    miss = cache.lookup("How to reduce LLM costs?")
    assert miss.is_hit is False
    assert miss.matched_entry is None
    assert miss.is_simulated is True

    # Store entry
    stored = cache.store(
        query="How to reduce LLM costs?",
        response="Use semantic caching and model routing.",
        metadata={"category": "optimization"}
    )
    assert stored.query == "How to reduce LLM costs?"

    # Exact query should hit with similarity ~ 1.0
    hit = cache.lookup("How to reduce LLM costs?")
    assert hit.is_hit is True
    assert hit.similarity_score is not None
    assert hit.similarity_score >= 0.99
    assert hit.matched_entry.response == "Use semantic caching and model routing."
    assert hit.matched_entry.hit_count == 1

def test_semantic_cache_lru_eviction():
    cache = SemanticCache(max_entries=2)
    cache.store("query 1", "res 1")
    cache.store("query 2", "res 2")
    assert len(cache._entries) == 2
    
    cache.store("query 3", "res 3")
    assert len(cache._entries) == 2
    # Oldest (query 1) was evicted
    assert all(e.query != "query 1" for e in cache._entries.values())
