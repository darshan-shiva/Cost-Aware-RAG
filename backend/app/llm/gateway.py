"""Groq chat-completion access with error classification and one fallback."""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

from ..routing.base import ModelProfile, RoutingDecision

logger = logging.getLogger("cost_aware_rag.llm")


class LLMNotConfiguredError(RuntimeError):
    """No provider API key is configured."""


class EmptyCompletionError(RuntimeError):
    """The provider answered but returned no usable text."""


# Error categories recorded in telemetry.
RATE_LIMITED = "rate_limited"
MODEL_UNAVAILABLE = "model_unavailable"
TIMEOUT = "timeout"
AUTH = "auth_error"
CONNECTION = "connection_error"
EMPTY = "empty_response"
NOT_CONFIGURED = "not_configured"
PROVIDER = "provider_error"

# Failures for which trying the other configured model can help.
FALLBACK_CATEGORIES = {RATE_LIMITED, MODEL_UNAVAILABLE}


def classify_error(exc: BaseException) -> str:
    if isinstance(exc, LLMNotConfiguredError):
        return NOT_CONFIGURED
    if isinstance(exc, EmptyCompletionError):
        return EMPTY
    if isinstance(exc, TimeoutError):
        return TIMEOUT
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if "Timeout" in name:
        return TIMEOUT
    if status == 429 or "RateLimit" in name:
        return RATE_LIMITED
    if status in (404,) or "NotFound" in name:
        return MODEL_UNAVAILABLE
    if status == 503 or "ServiceUnavailable" in name:
        return MODEL_UNAVAILABLE
    if status in (401, 403) or "Authentication" in name or "PermissionDenied" in name:
        return AUTH
    if "Connection" in name:
        return CONNECTION
    return PROVIDER


@dataclass
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    model: ModelProfile
    finish_reason: Optional[str] = None
    latency_ms: float = 0.0
    attempts: List[Dict[str, object]] = field(default_factory=list)

    @property
    def retry_count(self) -> int:
        return max(0, len(self.attempts) - 1)

    @property
    def fallback_used(self) -> bool:
        return len(self.attempts) > 1


class GenerationFailed(RuntimeError):
    def __init__(self, category: str, attempts: List[Dict[str, object]], cause: BaseException):
        super().__init__(f"Generation failed ({category})")
        self.category = category
        self.attempts = attempts
        self.cause = cause


# (messages, model_id) -> (text, input_tokens, output_tokens, finish_reason)
GenerateFn = Callable[[List[Dict[str, str]], str], tuple]


class LLMGateway:
    def __init__(
        self,
        client: Optional[object],
        max_completion_tokens: int,
        fallback_enabled: bool = True,
        availability_ttl_seconds: int = 600,
    ):
        self.client = client
        self.max_completion_tokens = max_completion_tokens
        self.fallback_enabled = fallback_enabled
        self.availability_ttl_seconds = availability_ttl_seconds
        self._available: Optional[Set[str]] = None
        self._available_at = 0.0
        self._lock = threading.Lock()

    @property
    def is_configured(self) -> bool:
        return self.client is not None

    def complete(self, messages: List[Dict[str, str]], model_id: str) -> tuple:
        """One provider call. Returns (text, input_tokens, output_tokens, finish_reason)."""
        if self.client is None:
            raise LLMNotConfiguredError("RAG_GROQ_API_KEY is not configured.")
        result = self.client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_completion_tokens=self.max_completion_tokens,
        )
        if not result.choices:
            raise EmptyCompletionError("The model provider returned no completion choices.")
        choice = result.choices[0]
        text = (choice.message.content or "").strip()
        if not text:
            raise EmptyCompletionError(
                "The model returned no visible text. "
                f"Finish reason: {choice.finish_reason!r}. "
                "Try increasing RAG_LLM_MAX_COMPLETION_TOKENS."
            )
        usage = result.usage
        return (
            text,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
            choice.finish_reason,
        )

    def unavailable_models(self, configured: List[str]) -> Set[str]:
        """Configured model IDs missing from the account's model list.

        The list is fetched at most once per TTL. If it cannot be fetched the
        result is empty: availability is then discovered from call errors.
        """
        if self.client is None:
            return set()
        now = time.time()
        with self._lock:
            fresh = self._available is not None and now - self._available_at < self.availability_ttl_seconds
            if not fresh:
                try:
                    listing = self.client.models.list()
                    self._available = {m.id for m in getattr(listing, "data", [])}
                except Exception as exc:
                    logger.warning("model_list_failed", extra={"error_type": type(exc).__name__})
                    self._available = None
                self._available_at = now
            available = self._available
        if not available:
            return set()
        return {m for m in configured if m not in available}

    def mark_unavailable(self, model_id: str) -> None:
        with self._lock:
            if self._available is not None:
                self._available.discard(model_id)

    def generate(
        self,
        messages: List[Dict[str, str]],
        decision: RoutingDecision,
        generate_fn: Optional[GenerateFn] = None,
    ) -> GenerationResult:
        """Call the routed model; on rate-limit or unavailability, try the
        decision's fallback model once. Never loops."""
        call = generate_fn or self.complete
        plan = [decision.selected_model]
        if self.fallback_enabled and decision.fallback_model is not None:
            plan.append(decision.fallback_model)

        attempts: List[Dict[str, object]] = []
        last_exc: Optional[BaseException] = None
        category = PROVIDER
        for index, model in enumerate(plan):
            started = time.perf_counter()
            try:
                text, in_tok, out_tok, finish = call(messages, model.model_id)
            except Exception as exc:
                category = classify_error(exc)
                attempts.append({
                    "model": model.model_id,
                    "error_category": category,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                })
                last_exc = exc
                if category == MODEL_UNAVAILABLE:
                    self.mark_unavailable(model.model_id)
                if category in FALLBACK_CATEGORIES and index + 1 < len(plan):
                    logger.warning(
                        "llm_fallback",
                        extra={"selected_model": model.model_id, "error_type": category},
                    )
                    continue
                raise GenerationFailed(category, attempts, exc) from exc
            latency = round((time.perf_counter() - started) * 1000, 2)
            attempts.append({"model": model.model_id, "error_category": None, "latency_ms": latency})
            return GenerationResult(
                text=text,
                input_tokens=int(in_tok or 0),
                output_tokens=int(out_tok or 0),
                model=model,
                finish_reason=finish,
                latency_ms=latency,
                attempts=attempts,
            )
        raise GenerationFailed(category, attempts, last_exc or RuntimeError("no models"))
