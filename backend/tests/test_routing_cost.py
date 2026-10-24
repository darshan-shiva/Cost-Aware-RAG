"""Routing rules, fallback behaviour and cost arithmetic."""

import pytest

from app.llm import GenerationFailed, LLMGateway, classify_error
from app.llm.gateway import EmptyCompletionError, LLMNotConfiguredError
from app.routing import CostAwareRouter, ModelProfile

LIGHT = ModelProfile(
    model_id="light-model", tier="lightweight", cost_input_per_1k=0.0001,
    cost_output_per_1k=0.0004, description="", max_context_tokens=4000,
)
HEAVY = ModelProfile(
    model_id="heavy-model", tier="complex", cost_input_per_1k=0.001,
    cost_output_per_1k=0.004, description="", max_context_tokens=16000,
)


def router(**kwargs):
    return CostAwareRouter(LIGHT, HEAVY, **{"context_token_threshold": 2500, "multi_doc_threshold": 2, **kwargs})


def test_short_factual_query_with_small_context_goes_lightweight():
    d = router().route("What is the warranty period?", context_tokens=600, source_document_count=1)
    assert d.selected_model.model_id == "light-model"
    assert d.fallback_model.model_id == "heavy-model"
    assert "94%" not in d.routing_reason  # no unverifiable savings claims


def test_large_context_goes_complex():
    d = router().route("What is the warranty period?", context_tokens=3000)
    assert d.selected_model.tier == "complex"
    assert any("large retrieved context" in f for f in d.factors)


def test_context_over_lightweight_limit_goes_complex():
    d = router(context_token_threshold=100000).route("Summary please", context_tokens=5000)
    assert d.selected_model.tier == "complex"
    assert "context exceeds lightweight model limit" in d.factors


def test_comparative_question_over_multiple_documents_goes_complex():
    q = "What are the differences in refund rules?"
    assert router().route(q, source_document_count=1).selected_model.tier == "lightweight"
    d = router().route(q, source_document_count=2)
    assert d.selected_model.tier == "complex"
    assert any("comparative question" in f for f in d.factors)


@pytest.mark.parametrize("policy,expected", [("fixed-lightweight", "light-model"), ("fixed-complex", "heavy-model")])
def test_fixed_policies_ignore_heuristics(policy, expected):
    d = router(policy=policy).route("Compare and analyze everything step by step", context_tokens=9999)
    assert d.selected_model.model_id == expected
    assert d.policy == policy


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError):
        router(policy="magic")


def test_unavailable_model_is_swapped_for_the_other_tier():
    d = router().route("What is the warranty period?", unavailable_models={"light-model"})
    assert d.selected_model.model_id == "heavy-model"
    assert d.fallback_model is None
    assert "not available" in d.routing_reason


@pytest.mark.parametrize(
    "model,tin,tout,expected",
    [
        (LIGHT, 1000, 1000, 0.0005),
        (LIGHT, 1500, 250, 0.00025),
        (HEAVY, 2000, 500, 0.004),
        (HEAVY, 0, 0, 0.0),
    ],
)
def test_cost_estimation(model, tin, tout, expected):
    assert router().estimate_cost(model, tin, tout) == pytest.approx(expected)


class _Status(Exception):
    def __init__(self, code):
        self.status_code = code


@pytest.mark.parametrize(
    "exc,category",
    [
        (_Status(429), "rate_limited"),
        (_Status(404), "model_unavailable"),
        (_Status(503), "model_unavailable"),
        (_Status(401), "auth_error"),
        (_Status(500), "provider_error"),
        (TimeoutError(), "timeout"),
        (EmptyCompletionError(), "empty_response"),
        (LLMNotConfiguredError(), "not_configured"),
    ],
)
def test_error_classification(exc, category):
    assert classify_error(exc) == category


def test_gateway_falls_back_once_then_stops():
    calls = []

    def fn(messages, model_id):
        calls.append(model_id)
        raise _Status(429)

    gw = LLMGateway(client=None, max_completion_tokens=10)
    decision = router().route("What is the warranty period?")
    with pytest.raises(GenerationFailed) as info:
        gw.generate([], decision, generate_fn=fn)
    assert calls == ["light-model", "heavy-model"]
    assert info.value.category == "rate_limited"


def test_gateway_fallback_can_be_disabled():
    calls = []

    def fn(messages, model_id):
        calls.append(model_id)
        raise _Status(429)

    gw = LLMGateway(client=None, max_completion_tokens=10, fallback_enabled=False)
    with pytest.raises(GenerationFailed):
        gw.generate([], router().route("Hi"), generate_fn=fn)
    assert calls == ["light-model"]


def test_gateway_uses_account_model_list_when_available():
    class Models:
        def list(self):
            class R:
                data = [type("M", (), {"id": "heavy-model"})()]
            return R()

    client = type("C", (), {"models": Models()})()
    gw = LLMGateway(client=client, max_completion_tokens=10)
    assert gw.unavailable_models(["light-model", "heavy-model"]) == {"light-model"}


def test_gateway_model_list_failure_assumes_available():
    class Models:
        def list(self):
            raise ConnectionError("offline")

    client = type("C", (), {"models": Models()})()
    gw = LLMGateway(client=client, max_completion_tokens=10)
    assert gw.unavailable_models(["light-model"]) == set()


def test_gateway_reads_provider_usage():
    class Completions:
        def create(self, **kwargs):
            msg = type("Msg", (), {"content": " Grounded answer [S1] "})()
            choice = type("Choice", (), {"message": msg, "finish_reason": "stop"})()
            usage = type("Usage", (), {"prompt_tokens": 321, "completion_tokens": 54})()
            return type("R", (), {"choices": [choice], "usage": usage})()

    chat = type("Chat", (), {"completions": Completions()})()
    gw = LLMGateway(client=type("C", (), {"chat": chat})(), max_completion_tokens=10)
    assert gw.complete([], "light-model") == ("Grounded answer [S1]", 321, 54, "stop")
