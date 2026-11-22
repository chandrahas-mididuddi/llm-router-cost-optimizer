"""Cost math, rollups and budget alerts."""
import pytest
from llm_router.cost_tracker import CostTracker
from llm_router.registry import RouterConfig
def test_model_cost_math():
    config = RouterConfig.load()
    mini = config.models["gpt-4o-mini"]
    assert mini.cost_usd(1000, 1000) == 0.00075
    assert mini.cost_usd(0, 0) == 0.0
def test_large_model_costs_more_than_small_for_same_tokens():
    config = RouterConfig.load()
    small = config.models["gpt-4o-mini"].cost_usd(1000, 500)
    large = config.models["claude-3-opus"].cost_usd(1000, 500)
    assert large > small * 10
def test_logging_and_daily_rollup(tracker):
    tracker.log("gpt-4o-mini", 100, 50, 0.000045, ts="2026-11-17")
    tracker.log("gpt-4o", 200, 100, 0.0015, ts="2026-11-17")
    tracker.log("gpt-4o-mini", 100, 50, 0.000045, ts="2026-11-18")
    assert tracker.total_cost() == pytest.approx(0.00159)
    rollup = tracker.daily_rollup()
    assert len(rollup) == 2
    assert rollup[0]["date"] == "2026-11-17"
    assert rollup[0]["requests"] == 2
    assert rollup[0]["cost_usd"] == pytest.approx(0.001545)
def test_cost_by_model(tracker):
    tracker.log("gpt-4o-mini", 100, 50, 0.000045)
    tracker.log("gpt-4o-mini", 100, 50, 0.000045)
    tracker.log("gpt-4o", 200, 100, 0.0015)
    by_model = tracker.cost_by_model()
    assert by_model[0]["model"] == "gpt-4o"
    assert by_model[0]["requests"] == 1
def test_budget_warn_at_80_percent(tracker):
    tracker.log("gpt-4o", 200, 100, 0.009, ts="2026-11-05")
    status = tracker.budget_status(monthly_budget_usd=0.01, warn_at_fraction=0.8, month="2026-11")
    assert status.fraction_used == 0.9
    assert status.alert == "warn"
def test_budget_exceeded(tracker):
    tracker.log("gpt-4o", 200, 100, 0.02, ts="2026-11-05")
    status = tracker.budget_status(monthly_budget_usd=0.01, warn_at_fraction=0.8, month="2026-11")
    assert status.alert == "exceeded"
def test_budget_no_alert_when_under_threshold(tracker):
    tracker.log("gpt-4o-mini", 100, 50, 0.001, ts="2026-11-05")
    status = tracker.budget_status(monthly_budget_usd=0.01, warn_at_fraction=0.8, month="2026-11")
    assert status.alert is None
def test_router_logs_cost_per_request(router, tracker):
    router.route("What is the capital of France?")
    assert tracker.request_count() == 1
    assert tracker.total_cost() > 0
    stats = router.stats()
    assert stats["cost"]["requests"] == 1
    assert stats["budget"]["alert"] is None
