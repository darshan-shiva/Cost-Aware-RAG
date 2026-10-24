from abc import ABC, abstractmethod
from typing import Optional, List
from pydantic import BaseModel, Field

class ModelProfile(BaseModel):
    model_id: str
    tier: str  # "lightweight" or "complex"
    cost_input_per_1k: float
    cost_output_per_1k: float
    description: str
    max_context_tokens: int = 8000

class RoutingDecision(BaseModel):
    selected_model: ModelProfile
    routing_reason: str
    estimated_complexity: str  # "low", "medium", "high"
    is_mock: bool = True
    policy: str = "complexity"
    # Factors that drove the decision, for telemetry and the dashboard.
    factors: List[str] = Field(default_factory=list)
    # Next model to try if the selected one is unavailable or rate-limited.
    fallback_model: Optional[ModelProfile] = None

class BaseRouter(ABC):
    @abstractmethod
    def route(
        self,
        query: str,
        context_tokens: int = 0,
        source_document_count: int = 0,
        unavailable_models: Optional[set] = None,
    ) -> RoutingDecision:
        """Analyze query and determine appropriate model tier with routing reason."""
        pass

    @abstractmethod
    def estimate_cost(
        self, model: ModelProfile, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate estimated USD cost given input and output token counts."""
        pass
