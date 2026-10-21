import pytest
from app.routing.cost_router import CostAwareRouter

def test_router_straightforward_query_routes_to_lightweight():
    router = CostAwareRouter()
    simple_query = "What is the capital of Japan?"
    decision = router.route(simple_query)

    assert decision.selected_model.tier == "lightweight"
    assert decision.estimated_complexity == "low"
    assert "LOW complexity" in decision.routing_reason
    assert decision.is_mock is True

def test_router_analytical_query_routes_to_complex():
    router = CostAwareRouter()
    complex_query = "Compare the tradeoffs between fine-tuning and retrieval-augmented generation for low latency enterprise systems."
    decision = router.route(complex_query)

    assert decision.selected_model.tier == "complex"
    assert decision.estimated_complexity == "high"
    assert "HIGH complexity" in decision.routing_reason
    assert "compare" in decision.routing_reason.lower() or "tradeoff" in decision.routing_reason.lower()

def test_router_cost_estimation():
    router = CostAwareRouter()
    # Lightweight: input $0.00015 / 1k, output $0.00060 / 1k
    cost = router.estimate_cost(
        router.lightweight,
        input_tokens=1000,
        output_tokens=1000,
    )
    # Verify cost estimation for 1,000 input and 1,000 output tokens.
    assert cost == pytest.approx(0.000375, rel=1e-4)
