"""Semantic cache: identical / near-identical prompts hit, others miss."""
from llm_router.cache import SemanticCache
def test_identical_prompt_is_a_hit():
    cache = SemanticCache(similarity_threshold=0.85)
    cache.store("What is the capital of France?", "Paris", "gpt-4o-mini")
    result = cache.lookup("What is the capital of France?")
    assert result.hit
    assert result.response == "Paris"
    assert result.similarity == 1.0
def test_unrelated_prompt_is_a_miss():
    cache = SemanticCache(similarity_threshold=0.85)
    cache.store("What is the capital of France?", "Paris", "gpt-4o-mini")
    result = cache.lookup("Write a Python function to parse CSV files.")
    assert not result.hit
def test_hit_rate_reporting():
    cache = SemanticCache(similarity_threshold=0.85)
    cache.store("What is the capital of France?", "Paris", "gpt-4o-mini")
    cache.lookup("What is the capital of France?")
    cache.lookup("Explain quantum computing.")
    assert cache.lookups == 2
    assert cache.hits == 1
    assert cache.hit_rate == 0.5
def test_expired_entry_is_a_miss():
    cache = SemanticCache(similarity_threshold=0.85, ttl_seconds=0)
    cache.store("What is the capital of France?", "Paris", "gpt-4o-mini")
    assert not cache.lookup("What is the capital of France?").hit
def test_router_serves_second_identical_prompt_from_cache(router):
    first = router.route("What is the capital of France?")
    second = router.route("What is the capital of France?")
    assert not first.cached
    assert second.cached
    assert second.cost_usd == 0.0
    assert second.text == first.text
    assert router.cache.hit_rate == 0.5
