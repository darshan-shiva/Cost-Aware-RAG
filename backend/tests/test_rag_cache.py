"""Semantic caching inside the RAG pipeline: exact and semantic hits,
threshold and evidence-based rejection, invalidation, and isolation."""

import pytest

from app.cache.guards import (
    contrastive_terms_conflict,
    uncovered_terms_conflict,
    normalize_query,
    salient_terms_conflict,
)
from tests.conftest import upload
from tests.pdf_utils import make_pdf

WARRANTY_Q = "How long is the Aurora X2 battery warranty?"
WARRANTY_PARAPHRASE = "How long does the Aurora X2 battery warranty last?"

MANUAL = make_pdf([
    "Aurora X2 battery warranty: the Aurora X2 battery warranty lasts 24 months from purchase.",
    "Aurora X2 charging: charge the Aurora X2 battery with the supplied 65 watt adapter only.",
])
GEOGRAPHY = make_pdf([
    "France facts. The capital of France is Paris. France uses the euro.",
    "Germany facts. The capital of Germany is Berlin. Germany borders Poland.",
])
RETURNS = make_pdf([
    "Laptop returns. The refund window for laptops is 30 days with the original box.",
    "Phone returns. The refund window for phones is 14 days and the phone must be unlocked.",
])


def ask(api, query, threshold=0.70, **kwargs):
    res = api.post(
        "/api/rag/query",
        json={"query": query, "min_relevance": 0.15, "similarity_threshold": threshold, **kwargs},
    )
    assert res.status_code == 200, res.text
    return res.json()


@pytest.fixture
def manual(api):
    return upload(api, "manual.pdf", MANUAL).json()["document"]["document_id"]


def test_exact_repeat_is_served_from_cache_without_model_call(api, manual):
    first = ask(api, WARRANTY_Q)
    second = ask(api, "  how long is the aurora x2 battery warranty  ")
    assert first["cache"]["hit"] is False
    assert second["cache"]["hit"] is True
    assert second["cache"]["match_type"] == "exact"
    assert second["cache"]["original_request_id"] == first["request_id"]
    assert len(api.rag.fake_llm.calls) == 1
    assert second["usage"]["estimated_cost"] == 0.0
    assert second["usage"]["estimated_cost_avoided"] == pytest.approx(first["usage"]["estimated_cost"])
    assert second["usage"]["token_source"] == "none"


def test_cached_answer_keeps_its_original_citations(api, manual):
    first = ask(api, WARRANTY_Q)
    second = ask(api, WARRANTY_Q)
    assert second["answer"] == first["answer"]
    assert second["citations"] == first["citations"]
    assert second["citations"][0]["page_number"] == 1
    assert second["citations"][0]["filename"] == "manual.pdf"


def test_semantic_paraphrase_hit_requires_same_evidence(api, manual):
    first = ask(api, WARRANTY_Q)
    second = ask(api, WARRANTY_PARAPHRASE)
    assert second["cache"]["hit"] is True
    assert second["cache"]["match_type"] == "semantic"
    assert 0.70 <= second["cache"]["similarity"] < 1.0
    assert second["citations"] == first["citations"]
    assert len(api.rag.fake_llm.calls) == 1


def test_similarity_below_threshold_is_a_miss(api, manual):
    ask(api, WARRANTY_Q)
    second = ask(api, WARRANTY_PARAPHRASE, threshold=0.99)
    assert second["cache"]["hit"] is False
    assert second["cache"]["similarity"] < 0.99
    assert len(api.rag.fake_llm.calls) == 2


def test_unrelated_question_is_a_miss(api, manual):
    ask(api, WARRANTY_Q)
    other = ask(api, "Which adapter should be used for charging?")
    assert other["cache"]["hit"] is False


def test_similar_questions_about_different_entities_do_not_share_answers(api):
    upload(api, "geography.pdf", GEOGRAPHY)
    france = ask(api, "What is the capital of France?")
    germany = ask(api, "What is the capital of Germany?")
    assert germany["cache"]["hit"] is False
    assert germany["cache"]["similarity"] >= 0.70  # embeddings alone would have matched
    assert germany["cache"]["rejection_reason"] == "different named entities"
    assert france["citations"][0]["page_number"] == 1
    assert germany["citations"][0]["page_number"] == 2


def test_lowercase_lookalike_questions_are_separated_by_retrieved_evidence(api):
    # No capitalised entities here, so only the evidence check can tell them apart.
    upload(api, "returns.pdf", RETURNS)
    laptops = ask(api, "what is the refund window for laptops")
    phones = ask(api, "what is the refund window for phones")
    assert phones["cache"]["hit"] is False
    assert phones["cache"]["similarity"] >= 0.70
    assert "different evidence" in phones["cache"]["rejection_reason"]
    assert laptops["citations"][0]["page_number"] == 1
    assert phones["citations"][0]["page_number"] == 2


def test_replacing_a_document_invalidates_cached_answers(api):
    upload(api, "policy.txt", b"Refund policy: refunds are processed within 14 days of the request.")
    first = ask(api, "How quickly are refunds processed?")
    assert "14 days" in first["answer"]
    assert api.rag.cache.get_stats()["entries_by_scope"] == {"rag:default:*": 1}

    upload(api, "policy.txt", b"Refund policy: refunds are processed within 30 days of the request.")
    assert api.rag.cache.get_stats()["entries_by_scope"] == {}
    second = ask(api, "How quickly are refunds processed?")
    assert second["cache"]["hit"] is False
    assert "30 days" in second["answer"]
    assert second["citations"][0]["document_version"] == 2


def test_deleting_a_document_invalidates_cached_answers(api, manual):
    ask(api, WARRANTY_Q)
    api.delete(f"/api/documents/{manual}")
    after = ask(api, WARRANTY_Q)
    assert after["cache"]["hit"] is False
    assert after["outcome"] == "no_documents"


def test_reindexing_invalidates_cached_answers(api, manual):
    ask(api, WARRANTY_Q)
    api.post(f"/api/documents/{manual}/reindex")
    after = ask(api, WARRANTY_Q)
    assert after["cache"]["hit"] is False
    assert after["citations"][0]["document_version"] == 2


def test_adding_a_document_changes_fingerprint(api, manual):
    ask(api, WARRANTY_Q)
    upload(api, "extra.txt", b"Aurora X2 battery warranty can be extended to 36 months for a fee.")
    after = ask(api, WARRANTY_Q)
    assert after["cache"]["hit"] is False


def test_collections_do_not_share_cache_entries(api):
    upload(api, "manual.pdf", MANUAL, collection="team-a")
    upload(api, "manual.pdf", MANUAL, collection="team-b")
    a = ask(api, WARRANTY_Q, collection="team-a")
    b = ask(api, WARRANTY_Q, collection="team-b")
    assert a["cache"]["hit"] is False and b["cache"]["hit"] is False
    assert a["citations"][0]["document_id"] != b["citations"][0]["document_id"]


def test_document_filter_has_its_own_cache_scope(api, manual):
    ask(api, WARRANTY_Q)
    filtered = ask(api, WARRANTY_Q, document_ids=[manual])
    assert filtered["cache"]["hit"] is False
    again = ask(api, WARRANTY_Q, document_ids=[manual])
    assert again["cache"]["hit"] is True


def test_general_mode_cache_never_serves_rag_queries(api, manual):
    api.qs._generate_response = lambda q, m: ("General knowledge answer", 5, 5)
    general = api.post("/api/query", json={"query": WARRANTY_Q})
    assert general.status_code == 200
    rag = ask(api, WARRANTY_Q)
    assert rag["cache"]["hit"] is False
    assert rag["answer"] != "General knowledge answer"


def test_force_refresh_bypasses_cache(api, manual):
    ask(api, WARRANTY_Q)
    fresh = ask(api, WARRANTY_Q, force_refresh=True)
    assert fresh["cache"]["hit"] is False and fresh["cache"]["bypassed"] is True
    assert len(api.rag.fake_llm.calls) == 2


def test_unusable_cache_entry_falls_back_to_fresh_generation(api, manual):
    ask(api, WARRANTY_Q)
    entry = next(iter(api.rag.cache._entries.values()))
    entry.metadata["citations"][0]["chunk_id"] = "doc_gone:v1:c0"
    after = ask(api, WARRANTY_Q)
    assert after["cache"]["hit"] is False
    assert after["cache"]["rejection_reason"] == "cached entry was unusable and was discarded"
    assert after["outcome"] == "answered"
    assert len(api.rag.fake_llm.calls) == 2


def test_cache_entries_survive_restart(api, manual, tmp_db_path):
    from app.cache.semantic import SemanticCache

    ask(api, WARRANTY_Q)
    reloaded = SemanticCache(
        embedding_provider=api.rag.embedding_provider, db_path=tmp_db_path
    )
    entries = list(reloaded._entries.values())
    assert len(entries) == 1
    assert entries[0].metadata["scope"] == "rag:default:*"
    assert entries[0].metadata["citations"][0]["filename"] == "manual.pdf"


# --------------------------------------------------------------- guard unit


@pytest.mark.parametrize(
    "a,b,conflict",
    [
        ("What is the capital of France?", "What is the capital of Germany?", "different named entities"),
        ("Revenue in 2023?", "Revenue in 2024?", "different numbers"),
        ("What is semantic caching in large language models?", "What is semantic caching in LLMs?", None),
        ("How does Python handle the GIL?", "How does the Python GIL work?", None),
        ("Explain how semantic caching works", "What is semantic caching?", None),
    ],
)
def test_salient_term_guard(a, b, conflict):
    assert salient_terms_conflict(a, b) == conflict


@pytest.mark.parametrize(
    "a,b,evidence,conflict",
    [
        (
            "How many vacation days do full-time employees get?",
            "How many vacation days do part-time employees get?",
            "Full-time employees receive 26 days. Part-time employees receive vacation pro rata.",
            True,
        ),
        (
            "How many vacation days do full-time employees get?",
            "How many paid vacation days do full-time employees receive?",
            "Full-time employees receive 26 days of paid vacation per year.",
            # "receive" and "paid" are in the passage, but "get" is not.
            False,
        ),
        (
            "How long is the Aurora X2 battery warranty?",
            "How long does the Aurora X2 battery warranty last?",
            "The Aurora X2 battery warranty lasts 24 months.",
            False,
        ),
    ],
)
def test_contrastive_term_guard(a, b, evidence, conflict):
    assert (contrastive_terms_conflict(a, b, evidence) is not None) is conflict


def test_contrasting_questions_over_one_passage_do_not_share_answers(api):
    upload(api, "leave.txt", (
        b"Time off. Full-time employees receive 26 days of paid vacation per year. "
        b"Part-time employees receive vacation pro rata to their contracted hours."
    ))
    ask(api, "How many vacation days do full-time employees get?")
    part = ask(api, "How many vacation days do part-time employees get?")
    assert part["cache"]["hit"] is False
    assert part["cache"]["similarity"] >= 0.70
    assert part["cache"]["rejection_reason"].startswith("contrasting terms")


def test_normalize_query():
    assert normalize_query("  What   IS this?? ") == "what is this"


def test_paraphrase_with_different_top_passage_is_not_reused(api, manual):
    # With the hashing embedder this paraphrase ranks the charging page first.
    # Reusing the warranty answer would cite evidence the new retrieval does
    # not support, so the cache refuses (conservative by design).
    ask(api, WARRANTY_Q)
    other = ask(api, "How long is the warranty on the Aurora X2 battery?")
    assert other["cache"]["hit"] is False
    assert other["cache"]["similarity"] >= 0.70
    assert other["cache"]["rejection_reason"].startswith("different evidence")


def test_uncovered_term_guard():
    evidence = (
        "The Aurora X3 is a 10,000 mAh battery. "
        "The Aurora X3 carries a 24 month limited warranty."
    )
    capacity_answer = "The Aurora X3 stores 10,000 mAh [S1]."
    # Same passages, but the cached answer never addresses the warranty.
    assert uncovered_terms_conflict(
        "What is the warranty of the Aurora X3?",
        "What is the capacity of the Aurora X3?",
        evidence,
        capacity_answer,
    )
    # A paraphrase whose extra words are not topics in the passage still passes.
    assert uncovered_terms_conflict(
        "What capacity does the Aurora X3 battery have?",
        "What is the capacity of the Aurora X3?",
        evidence,
        "The Aurora X3 battery stores 10,000 mAh [S1].",
    ) is None
