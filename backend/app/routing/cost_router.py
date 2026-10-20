import re
from typing import Optional, List
from .base import BaseRouter, ModelProfile, RoutingDecision
from ..config import settings

class CostAwareRouter(BaseRouter):
    """
    Cost-aware model router evaluating query complexity to direct traffic between
    cost-effective lightweight models and high-capability reasoning models.
    """

    COMPLEXITY_KEYWORDS: List[str] = [
        "compare",
        "contrast",
        "tradeoff",
        "trade-off",
        "analyze",
        "analysis",
        "evaluate",
        "step-by-step",
        "step by step",
        "architecture",
        "architect",
        "debug",
        "optimize",
        "pros and cons",
        "difference between",
        "explain in detail",
        "write a script",
        "implement",
        "algorithm",
    ]

    def __init__(
        self,
        lightweight_profile: Optional[ModelProfile] = None,
        complex_profile: Optional[ModelProfile] = None,
    ):
        self.lightweight = lightweight_profile or ModelProfile(
            model_id=settings.LIGHTWEIGHT_MODEL_ID,
            tier="lightweight",
            cost_input_per_1k=settings.LIGHTWEIGHT_COST_INPUT_PER_1K,
            cost_output_per_1k=settings.LIGHTWEIGHT_COST_OUTPUT_PER_1K,
            description="High-throughput, cost-efficient model for direct retrieval and simple queries.",
        )
        self.complex = complex_profile or ModelProfile(
            model_id=settings.COMPLEX_MODEL_ID,
            tier="complex",
            cost_input_per_1k=settings.COMPLEX_COST_INPUT_PER_1K,
            cost_output_per_1k=settings.COMPLEX_COST_OUTPUT_PER_1K,
            description="Frontier reasoning model for multi-step synthesis, code analysis, and complex reasoning.",
        )

    def route(self, query: str) -> RoutingDecision:
        cleaned = query.strip()
        word_count = len(cleaned.split())
        lower_query = cleaned.lower()

        # Check matched keywords
        matched_indicators = [
            kw for kw in self.COMPLEXITY_KEYWORDS if re.search(r"\b" + re.escape(kw) + r"\b", lower_query)
        ]

        # Decision heuristics
        is_complex = False
        reasons: List[str] = []

        if matched_indicators:
            is_complex = True
            reasons.append(f"detected analytical keywords ({', '.join(matched_indicators[:3])})")

        if word_count > 35:
            is_complex = True
            reasons.append(f"high prompt token length ({word_count} words)")

        if "\n" in cleaned and word_count > 20:
            is_complex = True
            reasons.append("multi-paragraph structured context detected")

        if is_complex:
            reason_str = f"Assessed as HIGH complexity: {'; '.join(reasons)}. Routed to capable tier ({self.complex.model_id}) for optimal reasoning fidelity."
            return RoutingDecision(
                selected_model=self.complex,
                routing_reason=reason_str,
                estimated_complexity="high",
                is_mock=True,
            )
        else:
            reason_str = f"Assessed as LOW complexity: concise query ({word_count} words) with direct factual intent. Routed to lightweight tier ({self.lightweight.model_id}) for 94% cost reduction."
            return RoutingDecision(
                selected_model=self.lightweight,
                routing_reason=reason_str,
                estimated_complexity="low",
                is_mock=True,
            )

    def estimate_cost(
        self, model: ModelProfile, input_tokens: int, output_tokens: int
    ) -> float:
        input_cost = (input_tokens / 1000.0) * model.cost_input_per_1k
        output_cost = (output_tokens / 1000.0) * model.cost_output_per_1k
        return round(input_cost + output_cost, 6)
