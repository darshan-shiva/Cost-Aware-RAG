
"""Evaluate semantic-cache quality across similarity thresholds.

Uses real embeddings and an in-memory cache. Does not call an LLM,
write to the application's SQLite database, or modify existing cache data.
Run from the backend directory:
    python evaluate_cache.py
"""

from collections import defaultdict

from app.cache import SemanticCache, get_embedding_provider


THRESHOLDS = [0.70, 0.75, 0.80, 0.82, 0.85, 0.90]

# Each group contains queries intended to mean the same thing.
QUERY_GROUPS = [
    [
        "What is semantic caching in RAG pipelines?",
        "Can you explain semantic caching in retrieval augmented generation?",
        "How does semantic caching work in a RAG system?",
    ],
    [
        "What is the purpose of cosine similarity in machine learning?",
        "Explain why cosine similarity is used to compare embeddings.",
        "How do embeddings use cosine similarity to measure relatedness?",
    ],
    [
        "How can I reduce inference costs for large language models?",
        "What techniques lower the cost of LLM inference?",
        "How do I make language model API calls less expensive?",
    ],
    [
        "What is a database index and why is it useful?",
        "Explain how indexes improve database query performance.",
        "Why do databases use indexes to speed up searches?",
    ],
        [
        "How does a Python dictionary work?",
        "Explain the purpose of dictionaries in Python.",
        "What are key-value mappings in Python used for?",
    ],
    [
        "What is overfitting in machine learning?",
        "Why does a machine learning model overfit its training data?",
        "Explain the problem of overfitting in predictive models.",
    ],
    [
        "How do REST APIs communicate with clients?",
        "Explain how a client sends requests to a REST API.",
        "What happens when an application calls a REST endpoint?",
    ],
    [
        "What is the difference between authentication and authorization?",
        "How do authentication and authorization differ?",
        "Explain authentication versus access control.",
    ],

]

# Each pair is deliberately intended to be unrelated.
NEGATIVE_PAIRS = [
    (
        "What is semantic caching in RAG pipelines?",
        "How do I bake a chocolate cake?",
    ),
    (
        "What is the purpose of cosine similarity in machine learning?",
        "What are the best exercises for stronger legs?",
    ),
    (
        "How can I reduce inference costs for large language models?",
        "When should I water tomato plants?",
    ),
    (
        "What is a database index and why is it useful?",
        "Explain the causes of the French Revolution.",
    ),
        (
        "How does a Python dictionary work?",
        "What causes earthquakes and volcanic eruptions?",
    ),
    (
        "What is overfitting in machine learning?",
        "How do I prepare a traditional vegetable curry?",
    ),
    (
        "How do REST APIs communicate with clients?",
        "What is the best way to grow roses indoors?",
    ),
    (
        "What is the difference between authentication and authorization?",
        "How far is the Moon from Earth?",
    ),
]


def main():
    provider = get_embedding_provider(
        provider_type="sentence-transformers",
        model_name="all-MiniLM-L6-v2",
    )

    # No db_path: all benchmark entries stay in memory.
    cache = SemanticCache(
        embedding_provider=provider,
        default_threshold=0.82,
        max_entries=100,
        db_path=None,
    )

    print(f"Embedding provider: {provider.provider_name}")
    print(f"Simulated embeddings: {provider.is_simulated}")
    print(f"Thresholds: {THRESHOLDS}")
    print()

    # Cache one representative query per meaning group.
    for group_index, group in enumerate(QUERY_GROUPS):
        cache.store(
            group[0],
            f"Benchmark response for semantic group {group_index + 1}",
        )

    # Gather similarity scores without changing the cache threshold.
    positive_scores = []
    for group in QUERY_GROUPS:
        for paraphrase in group[1:]:
            result = cache.lookup(paraphrase, threshold=1.0)
            if result.similarity_score is not None:
                positive_scores.append(result.similarity_score)

    negative_scores = []
    for query_a, query_b in NEGATIVE_PAIRS:
        temporary_cache = SemanticCache(
            embedding_provider=provider,
            max_entries=10,
            db_path=None,
        )
        temporary_cache.store(query_a, "Negative-pair benchmark response")
        result = temporary_cache.lookup(query_b, threshold=1.0)
        if result.similarity_score is not None:
            negative_scores.append(result.similarity_score)

    print("Similarity score distributions")
    print("-" * 64)
    print(
        f"Paraphrases: mean={sum(positive_scores) / max(1, len(positive_scores)):.3f}, "
        f"samples={len(positive_scores)}"
    )
    print(
        f"Unrelated pairs: mean={sum(negative_scores) / max(1, len(negative_scores)):.3f}, "
        f"samples={len(negative_scores)}"
    )
    print()

    # At each threshold, a paraphrase is a positive example.
    # An unrelated pair is a negative example.
    print("Threshold evaluation")
    print("-" * 78)
    print(
        f"{'Threshold':>9} {'TP':>4} {'FN':>4} {'FP':>4} {'TN':>4} "
        f"{'Precision':>10} {'Recall':>8} {'F1':>8}"
    )

    for threshold in THRESHOLDS:
        tp = sum(score >= threshold for score in positive_scores)
        fn = len(positive_scores) - tp
        fp = sum(score >= threshold for score in negative_scores)
        tn = len(negative_scores) - fp

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        print(
            f"{threshold:>9.2f} {tp:>4} {fn:>4} {fp:>4} {tn:>4} "
            f"{precision:>10.3f} {recall:>8.3f} {f1:>8.3f}"
        )

    print()
    print("Interpretation:")
    print("- TP: paraphrase correctly accepted as a cache hit.")
    print("- FN: paraphrase rejected even though it has the same intended meaning.")
    print("- FP: unrelated query incorrectly accepted as a cache hit.")
    print("- TN: unrelated query correctly rejected.")
    print("A higher threshold generally reduces false positives but may miss paraphrases.")
    print("These results reflect this small benchmark, not universal model performance.")


if __name__ == "__main__":
    main()
