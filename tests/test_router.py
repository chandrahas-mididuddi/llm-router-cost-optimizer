"""Routing: complexity -> cheapest capable model, plus fallbacks."""
import pytest
from conftest import SIMPLE, MEDIUM, COMPLEX
from llm_router.backends import MockBackend
from llm_router.cache import SemanticCache
from llm_router.classifier import ComplexityClassifier
from llm_router.cost_tracker import CostTracker
from llm_router.registry import RouterConfig
from llm_router.router import Router, RouterError
def _router_with(fail_models=None, breaker_failures=3):
    config = RouterConfig.load()
    config.circuit_breaker_failures = breaker_failures
    return Router(config=config, backend=MockBackend(fail_models=fail_models or set()), classifier=ComplexityClassifier().train(), cache=SemanticCache(), cost_tracker=CostTracker(":memory:"))
def test_simple_prompt_goes_to_small_model(router):
    result = router.route(SIMPLE)
    assert result.model == "gpt-4o-mini"
    assert result.complexity == "simple"
    assert not result.cached
def test_complex_prompt_goes_to_large_model(router):
    result = router.route(COMPLEX)
    assert result.model in ("claude-3-opus", "gpt-4-turbo")
    assert result.complexity == "complex"
def test_empty_prompt_rejected(router):
    with pytest.raises(ValueError):
        router.route("   ")
def test_fallback_to_next_model_when_first_fails():
    router = _router_with(fail_models={"gpt-4o-mini"})
    result = router.route(SIMPLE)
    assert result.model == "claude-3-5-haiku"
    assert "gpt-4o-mini" in result.attempts
def test_all_models_failing_raises_router_error():
    router = _router_with(fail_models={"gpt-4o-mini", "claude-3-5-haiku"})
    with pytest.raises(RouterError, match="all candidate models failed"):
        router.route(SIMPLE)
def test_circuit_breaker_opens_after_consecutive_failures():
    router = _router_with(fail_models={"gpt-4o-mini", "claude-3-5-haiku"}, breaker_failures=2)
    for _ in range(2):
        with pytest.raises(RouterError):
            router.route(SIMPLE)
    assert router.breaker_state("gpt-4o-mini") == "open"
    assert router.breaker_state("claude-3-5-haiku") == "open"
def test_successful_call_resets_breaker():
    router = _router_with(fail_models={"gpt-4o-mini"})
    router.route(SIMPLE)
    assert router.breaker_state("gpt-4o-mini") == "closed"
    router.backend.fail_models = set()
    result = router.route("What is the capital of Spain?")
    assert result.model == "gpt-4o-mini"
