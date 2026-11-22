"""Model registry and routing configuration, loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "routing.yaml"


@dataclass
class ModelInfo:
    name: str
    provider: str
    tier: str
    price_input_per_1k: float
    price_output_per_1k: float
    latency_ms_p50: int
    context_window: int
    max_output_tokens: int

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        """Cost of one request in USD."""
        return round(
            input_tokens / 1000 * self.price_input_per_1k
            + output_tokens / 1000 * self.price_output_per_1k,
            6,
        )


@dataclass
class RouterConfig:
    models: dict[str, ModelInfo] = field(default_factory=dict)
    routing: dict[str, list[str]] = field(default_factory=dict)
    cache_threshold: float = 0.85
    cache_ttl_seconds: int = 3600
    cache_max_entries: int = 10000
    monthly_budget_usd: float = 100.0
    warn_at_fraction: float = 0.8
    circuit_breaker_failures: int = 3
    circuit_breaker_cooldown_seconds: int = 60
    request_timeout_seconds: int = 30

    @classmethod
    def load(cls, path: str | Path | None = None) -> "RouterConfig":
        path = Path(path) if path else DEFAULT_CONFIG
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        cfg = cls()
        for name, m in raw.get("models", {}).items():
            cfg.models[name] = ModelInfo(name=name, **m)
        cfg.routing = raw.get("routing", {})
        cache = raw.get("cache", {})
        cfg.cache_threshold = cache.get("similarity_threshold", 0.85)
        cfg.cache_ttl_seconds = cache.get("ttl_seconds", 3600)
        cfg.cache_max_entries = cache.get("max_entries", 10000)
        budget = raw.get("budget", {})
        cfg.monthly_budget_usd = budget.get("monthly_budget_usd", 100.0)
        cfg.warn_at_fraction = budget.get("warn_at_fraction", 0.8)
        rel = raw.get("reliability", {})
        cfg.circuit_breaker_failures = rel.get("circuit_breaker_failures", 3)
        cfg.circuit_breaker_cooldown_seconds = rel.get(
            "circuit_breaker_cooldown_seconds", 60)
        cfg.request_timeout_seconds = rel.get("request_timeout_seconds", 30)
        return cfg

    def candidates_for(self, complexity: str) -> list[ModelInfo]:
        """Ordered candidate models for a complexity class."""
        names = self.routing.get(complexity, [])
        return [self.models[n] for n in names if n in self.models]

    def cheapest_for(self, complexity: str) -> ModelInfo | None:
        cands = self.candidates_for(complexity)
        return cands[0] if cands else None
