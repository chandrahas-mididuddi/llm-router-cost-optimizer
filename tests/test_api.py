"""FastAPI service: chat completions and stats endpoints."""
import pytest
from fastapi.testclient import TestClient
import llm_router.api as api
from llm_router.backends import MockBackend
from llm_router.cache import SemanticCache
from llm_router.classifier import ComplexityClassifier
from llm_router.cost_tracker import CostTracker
from llm_router.registry import RouterConfig
from llm_router.router import Router
@pytest.fixture()
def client(monkeypatch):
    router = Router(config=RouterConfig.load(), backend=MockBackend(), classifier=ComplexityClassifier().train(), cache=SemanticCache(), cost_tracker=CostTracker(":memory:"),)
    monkeypatch.setattr(api, "_ROUTER", router)
    return TestClient(api.app)
def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}
def test_chat_completion_routes_and_returns_openai_shape(client):
    resp = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "What is the capital of France?"}]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "gpt-4o-mini"
    assert body["complexity"] == "simple"
    assert body["cached"] is False
    assert body["cost_usd"] > 0
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert "usage" in body
def test_second_identical_request_is_cached(client):
    payload = {"messages": [{"role": "user", "content": "What is the capital of France?"}]}
    client.post("/v1/chat/completions", json=payload)
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.json()["cached"] is True
    assert resp.json()["cost_usd"] == 0.0
def test_stats_endpoint_reports_cost_and_cache(client):
    client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "What is the capital of France?"}]})
    stats = client.get("/stats").json()
    assert stats["cost"]["requests"] == 1
    assert stats["cost"]["total_usd"] > 0
    assert stats["cache"]["lookups"] == 1
    assert stats["budget"]["alert"] is None
    assert "gpt-4o-mini" in stats["models"]
def test_empty_messages_rejected(client):
    resp = client.post("/v1/chat/completions", json={"messages": []})
    assert resp.status_code == 422
