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

    COMPARATIVE_KEYWORDS: List[str] = [
        "compare", "contrast", "difference", "differences", "versus", "vs",
        "both", "across", "all of", "each of", "summarize", "summarise",
    ]

    def __init__(
        self,
        lightweight_profile: Optional[ModelProfile] = None,
        complex_profile: Optional[ModelProfile] = None,
        policy: Optional[str] = None,
        context_token_threshold: Optional[int] = None,
        multi_doc_threshold: Optional[int] = None,
    ):
        self.lightweight = lightweight_profile or ModelProfile(
            model_id=settings.LIGHTWEIGHT_MODEL_ID,
            tier="lightweight",
            cost_input_per_1k=settings.LIGHTWEIGHT_COST_INPUT_PER_1K,
            cost_output_per_1k=settings.LIGHTWEIGHT_COST_OUTPUT_PER_1K,
            description="High-throughput, cost-efficient model for direct retrieval and simple queries.",
            max_context_tokens=settings.LIGHTWEIGHT_MAX_CONTEXT_TOKENS,
        )
        self.complex = complex_profile or ModelProfile(
            model_id=settings.COMPLEX_MODEL_ID,
            tier="complex",
            cost_input_per_1k=settings.COMPLEX_COST_INPUT_PER_1K,
            cost_output_per_1k=settings.COMPLEX_COST_OUTPUT_PER_1K,
            description="Frontier reasoning model for multi-step synthesis, code analysis, and complex reasoning.",
            max_context_tokens=settings.COMPLEX_MAX_CONTEXT_TOKENS,
        )
        self.policy = (policy or settings.ROUTING_POLICY).lower()
        if self.policy not in self.POLICIES:
            raise ValueError(
                f"Unknown routing policy '{self.policy}'. Expected one of {self.POLICIES}."
            )
        self.context_token_threshold = (
            context_token_threshold
            if context_token_threshold is not None
            else settings.ROUTING_CONTEXT_TOKEN_THRESHOLD
        )
        self.multi_doc_threshold = (
            multi_doc_threshold
            if multi_doc_threshold is not None
            else settings.ROUTING_MULTI_DOC_THRESHOLD
        )

    POLICIES = ("complexity", "fixed-lightweight", "fixed-complex")

    @property
    def profiles(self) -> List[ModelProfile]:
        return [self.lightweight, self.complex]

    def _other(self, profile: ModelProfile) -> ModelProfile:
        return self.complex if profile.tier == "lightweight" else self.lightweight

    def route(
        self,
        query: str,
        context_tokens: int = 0,
        source_document_count: int = 0,
        unavailable_models: Optional[set] = None,
    ) -> RoutingDecision:
        """Pick a model tier with a transparent rule set.

        Rules, in order: a fixed policy wins outright; otherwise the query is
        HIGH complexity if it contains analytical keywords, is long, is
        multi-paragraph, its retrieved context exceeds the configured token
        threshold or the lightweight model's context limit, or it is a
        comparative question over evidence from several documents. These are
        keyword and size heuristics, not a learned quality predictor.
        """
        cleaned = query.strip()
        word_count = len(cleaned.split())
        lower_query = cleaned.lower()
        unavailable = unavailable_models or set()

        if self.policy != "complexity":
            chosen = self.lightweight if self.policy == "fixed-lightweight" else self.complex
            factors = [f"fixed policy ({self.policy})"]
            reason = f"Fixed routing policy '{self.policy}': always {chosen.model_id}."
            complexity = "low" if chosen.tier == "lightweight" else "high"
            return self._finalize(chosen, reason, complexity, factors, unavailable)

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

        if context_tokens > self.context_token_threshold:
            is_complex = True
            reasons.append(
                f"large retrieved context (~{context_tokens} tokens > {self.context_token_threshold})"
            )

        if context_tokens > self.lightweight.max_context_tokens:
            is_complex = True
            reasons.append("context exceeds lightweight model limit")

        is_comparative = any(
            re.search(r"\b" + re.escape(kw) + r"\b", lower_query)
            for kw in self.COMPARATIVE_KEYWORDS
        )
        if is_comparative and source_document_count >= self.multi_doc_threshold:
            is_complex = True
            reasons.append(
                f"comparative question over {source_document_count} source documents"
            )

        if is_complex:
            reason_str = f"Assessed as HIGH complexity: {'; '.join(reasons)}. Routed to capable tier ({self.complex.model_id})."
            return self._finalize(self.complex, reason_str, "high", reasons, unavailable)

        reason_str = (
            f"Assessed as LOW complexity: concise query ({word_count} words) with direct factual intent"
            + (f" and ~{context_tokens} context tokens" if context_tokens else "")
            + f". Routed to lightweight tier ({self.lightweight.model_id}) because its configured per-token price is lower."
        )
        return self._finalize(
            self.lightweight, reason_str, "low", [f"concise query ({word_count} words)"], unavailable
        )

    def _finalize(
        self,
        chosen: ModelProfile,
        reason: str,
        complexity: str,
        factors: List[str],
        unavailable: set,
    ) -> RoutingDecision:
        fallback: Optional[ModelProfile] = self._other(chosen)
        if chosen.model_id in unavailable and fallback.model_id not in unavailable:
            reason += (
                f" {chosen.model_id} is not available on the configured account;"
                f" using {fallback.model_id} instead."
            )
            factors = factors + ["preferred model unavailable"]
            chosen, fallback = fallback, None
        elif fallback.model_id in unavailable or fallback.model_id == chosen.model_id:
            fallback = None
        return RoutingDecision(
            selected_model=chosen,
            routing_reason=reason,
            estimated_complexity=complexity,
            is_mock=True,
            policy=self.policy,
            factors=factors,
            fallback_model=fallback,
        )

    def estimate_cost(
        self, model: ModelProfile, input_tokens: int, output_tokens: int
    ) -> float:
        input_cost = (input_tokens / 1000.0) * model.cost_input_per_1k
        output_cost = (output_tokens / 1000.0) * model.cost_output_per_1k
        return round(input_cost + output_cost, 6)
