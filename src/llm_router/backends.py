"""Pluggable model backends. MockBackend — deterministic, offline. OpenAICompatibleBackend — calls any OpenAI-compatible /v1/chat/completions endpoint via httpx."""
from __future__ import annotations
import hashlib
import os
import random
import time
from dataclasses import dataclass, field
from typing import Protocol
import httpx
class BackendError(Exception):
    """Raised when a backend fails (network, timeout, provider error)."""
@dataclass
class Completion:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
class Backend(Protocol):
    name: str
    def complete(self, model: str, prompt: str, max_tokens: int = 512) -> Completion:
        ...
def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)
@dataclass
class MockBackend:
    name: str = "mock"
    fail_models: set[str] = field(default_factory=set)
    fail_rate: float = 0.0
    base_latency_ms: int = 40
    _rng: random.Random = field(default_factory=lambda: random.Random(42))
    def complete(self, model: str, prompt: str, max_tokens: int = 512) -> Completion:
        start = time.monotonic()
        if model in self.fail_models:
            raise BackendError(f"mock backend: model {model} is down")
        if self.fail_rate > 0 and self._rng.random() < self.fail_rate:
            raise BackendError(f"mock backend: transient failure for {model}")
        digest = hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()[:8]
        text = (f"[mock:{model}] Response to your prompt ({estimate_tokens(prompt)} input tokens, ref {digest}): This is a deterministic mock completion. In production this would be the model's answer.")
        output_tokens = estimate_tokens(text)
        latency = int((time.monotonic() - start) * 1000) + self.base_latency_ms
        return Completion(text=text, model=model, input_tokens=estimate_tokens(prompt), output_tokens=output_tokens, latency_ms=latency)
@dataclass
class OpenAICompatibleBackend:
    name: str = "openai-compatible"
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: int = 30
    def __post_init__(self):
        self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = os.environ.get("OPENAI_BASE_URL", self.base_url)
    def complete(self, model: str, prompt: str, max_tokens: int = 512) -> Completion:
        if not self.api_key:
            raise BackendError("OPENAI_API_KEY is not set")
        start = time.monotonic()
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.post(f"{self.base_url.rstrip('/')}/chat/completions", headers={"Authorization": f"Bearer {self.api_key}"}, json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens},)
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise BackendError(f"provider request failed: {exc}") from exc
        data = resp.json()
        choice = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return Completion(text=choice, model=data.get("model", model), input_tokens=usage.get("prompt_tokens", estimate_tokens(prompt)), output_tokens=usage.get("completion_tokens", estimate_tokens(choice)), latency_ms=int((time.monotonic() - start) * 1000),)
