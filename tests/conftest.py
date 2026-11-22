import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llm_router.backends import MockBackend
from llm_router.cache import SemanticCache
from llm_router.classifier import ComplexityClassifier
from llm_router.cost_tracker import CostTracker
from llm_router.registry import RouterConfig
from llm_router.router import Router
@pytest.fixture(scope="session")
def classifier():
    return ComplexityClassifier().train()
@pytest.fixture()
def config():
    return RouterConfig.load()
@pytest.fixture()
def tracker():
    return CostTracker(":memory:")
@pytest.fixture()
def router(config, tracker):
    return Router(config=config, backend=MockBackend(), classifier=ComplexityClassifier().train(), cache=SemanticCache(), cost_tracker=tracker)
SIMPLE = "What is the capital of France?"
MEDIUM = "Explain how OAuth 2.0 authorization code flow works, step by step."
COMPLEX = ("Design a distributed rate limiter that works across 50 microservices with strict consistency requirements and analyze the failure modes under network partitions.")
