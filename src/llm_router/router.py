"""The router: classify -> cache -> model chain with circuit breakers
-> cost logging.

Flow for one prompt:
1. Semantic-cache lookup (hit = $0, no model call).
2. Complexity classification (learned TF-IDF model + heuristic blend).
3. Walk the ordered candidate models for that complexity class.
   A model with an open circuit breaker is skipped.
4. On success: reset the model's failure count, log the cost, store
   the answer in the cache.
5. On failure of every candidate: raise RouterError.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .backends import Backend, BackendError, Completion
from .cache import SemanticCache
from .classifier import ComplexityClassifier, ComplexityScore
from .cost_tracker import CostTracker
from .registry import ModelInfo, RouterConfig


class RouterError(Exception):
    """All candidate models failed for this request."""


@dataclass
class RouteResult:
    text: str
    model: str
    complexity: str
    complexity_score: float
    cached: bool
    cache_similarity: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    attempts: list[str] = field(default_factory=list)


@dataclass
class _Breaker:
    failures: int = 0
    opened_at: float | None = None


class Router:
    def __init__(self, config: RouterConfig, backend: Backend,
                 classifier: ComplexityClassifier | None = None,
                 cache: SemanticCache | None = None,
                 cost_tracker: CostTracker | None = None):
        self.config = config
        self.backend = backend
        self.classifier = classifier or ComplexityClassifier()
        self.cache = cache or SemanticCache(
            similarity_threshold=config.cache_threshold,
            ttl_seconds=config.cache_ttl_seconds,
            max_entries=config.cache_max_entries)
        self.cost_tracker = cost_tracker
        self._breakers: dict[str, _Breaker] = {}

    def _breaker_open(self, model: str) -> bool:
        breaker = self._breakers.get(model)
        if not breaker or breaker.opened_at is None:
            return False
        if time.time() - breaker.opened_at >= \
                self.config.circuit_breaker_cooldown_seconds:
            breaker.opened_at = None
            breaker.failures = 0
            return False
        return True

    def _record_failure(self, model: str) -> None:
        breaker = self._breakers.setdefault(model, _Breaker())
        breaker.failures += 1
        if breaker.failures >= self.config.circuit_breaker_failures:
            breaker.opened_at = time.time()

    def _record_success(self, model: str) -> None:
        self._breakers.pop(model, None)

    def breaker_state(self, model: str) -> str:
        return "open" if self._breaker_open(model) else "closed"

    def route(self, prompt: str, max_tokens: int = 512,
              use_cache: bool = True) -> RouteResult:
        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty")
        if use_cache:
            hit = self.cache.lookup(prompt)
            if hit.hit:
                if self.cost_tracker:
                    self.cost_tracker.log(
                        model=hit.model or "cache", input_tokens=0,
                        output_tokens=0, cost_usd=0.0, cached=True)
                return RouteResult(
                    text=hit.response or "", model=hit.model or "cache",
                    complexity="cached", complexity_score=0.0, cached=True,
                    cache_similarity=hit.similarity, input_tokens=0,
                    output_tokens=0, cost_usd=0.0, latency_ms=0,
                    attempts=[hit.model or "cache"])
        classification: ComplexityScore = self.classifier.predict(prompt)
        candidates = self.config.candidates_for(classification.label)
        if not candidates:
            raise RouterError(
                f"no models configured for complexity "
                f"'{classification.label}'")
        attempts: list[str] = []
        errors: list[str] = []
        for model in candidates:
            if self._breaker_open(model.name):
                attempts.append(f"{model.name} (circuit open, skipped)")
                continue
            attempts.append(model.name)
            try:
                completion: Completion = self.backend.complete(
                    model.name, prompt, max_tokens=max_tokens)
            except BackendError as exc:
                errors.append(f"{model.name}: {exc}")
                self._record_failure(model.name)
                continue
            self._record_success(model.name)
            cost = model.cost_usd(completion.input_tokens,
                                  completion.output_tokens)
            if self.cost_tracker:
                self.cost_tracker.log(
                    model=model.name,
                    input_tokens=completion.input_tokens,
                    output_tokens=completion.output_tokens,
                    cost_usd=cost, complexity=classification.label,
                    latency_ms=completion.latency_ms)
            if use_cache:
                self.cache.store(prompt, completion.text, model.name)
            return RouteResult(
                text=completion.text, model=model.name,
                complexity=classification.label,
                complexity_score=classification.score, cached=False,
                cache_similarity=0.0,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                cost_usd=cost, latency_ms=completion.latency_ms,
                attempts=attempts)
        raise RouterError(
            "all candidate models failed: " + "; ".join(errors))

    def stats(self) -> dict:
        budget = None
        if self.cost_tracker:
            status = self.cost_tracker.budget_status(
                self.config.monthly_budget_usd, self.config.warn_at_fraction)
            budget = {"month": status.month, "spent_usd": status.spent_usd,
                      "budget_usd": status.budget_usd,
                      "fraction_used": status.fraction_used,
                      "alert": status.alert}
        return {
            "cache": {"lookups": self.cache.lookups, "hits": self.cache.hits,
                      "hit_rate": self.cache.hit_rate,
                      "entries": self.cache.size},
            "cost": {
                "total_usd": self.cost_tracker.total_cost()
                if self.cost_tracker else 0.0,
                "requests": self.cost_tracker.request_count()
                if self.cost_tracker else 0,
                "by_model": self.cost_tracker.cost_by_model()
                if self.cost_tracker else [],
                "daily": self.cost_tracker.daily_rollup()
                if self.cost_tracker else [],
            },
            "budget": budget,
            "circuit_breakers": {
                name: self.breaker_state(name)
                for name in self.config.models
            },
            "models": {
                name: {"tier": m.tier, "provider": m.provider,
                       "price_input_per_1k": m.price_input_per_1k,
                       "price_output_per_1k": m.price_output_per_1k}
                for name, m in self.config.models.items()
            },
        }
