
"""Measure local semantic-cache lookup performance without LLM calls."""

import statistics
import time

from app.cache import SemanticCache, get_embedding_provider

WARMUP_RUNS = 3
BENCHMARK_RUNS = 20

provider = get_embedding_provider(
    provider_type="sentence-transformers",
    model_name="all-MiniLM-L6-v2",
)

cache = SemanticCache(
    embedding_provider=provider,
    default_threshold=0.82,
    max_entries=100,
    db_path=None,
)

queries = [
    "What is semantic caching in RAG pipelines?",
    "How does cosine similarity compare text embeddings?",
    "How can I reduce inference costs for large language models?",
    "What is a database index and why is it useful?",
]

for i, query in enumerate(queries):
    cache.store(query, f"Benchmark response {i + 1}")

def measure_lookup(query, threshold=0.82):
    start = time.perf_counter()
    result = cache.lookup(query, threshold=threshold)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, result.is_hit

def report(label, query, expected_hit):
    for _ in range(WARMUP_RUNS):
        measure_lookup(query)

    times = []
    hits = 0

    for _ in range(BENCHMARK_RUNS):
        elapsed, hit = measure_lookup(query)
        times.append(elapsed)
        hits += int(hit)

    print(f"\n{label}")
    print(f"  Expected hit: {expected_hit}")
    print(f"  Observed hit rate: {hits}/{BENCHMARK_RUNS}")
    print(f"  Mean lookup latency: {statistics.mean(times):.3f} ms")
    print(f"  Median lookup latency: {statistics.median(times):.3f} ms")
    print(f"  Minimum lookup latency: {min(times):.3f} ms")
    print(f"  Maximum lookup latency: {max(times):.3f} ms")

print(f"Embedding provider: {provider.provider_name}")
print("Local cache benchmark; no LLM requests will be made.")

report(
    "Exact query",
    "What is semantic caching in RAG pipelines?",
    True,
)

report(
    "Paraphrased query",
    "Can you explain semantic caching in retrieval augmented generation?",
    False,  # Expected under the current 0.82 threshold; measure actual result.
)

report(
    "Unrelated query",
    "How do I bake a chocolate cake?",
    False,
)
