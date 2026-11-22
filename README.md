# LLM Router — Cost Optimizer

Route every prompt to the **cheapest model that can actually handle it**,
with semantic caching, automatic fallbacks, and per-request cost tracking.

> Portfolio project. Not affiliated with any LLM provider, and not used in
> production anywhere — it's a working demonstration of routing, caching,
> reliability and cost-observability patterns for LLM systems.

## Why

Most LLM traffic is a mix: a few hard prompts (reasoning, design, long
documents) and a long tail of easy ones (factoids, summaries, formatting).
Sending everything to the largest model wastes money; sending everything to
the smallest degrades quality. This router classifies each prompt, checks a
semantic cache first, then picks the cheapest model in the right tier —
and falls back gracefully when a model is down.

On a sample workload of 1,000 mixed requests, routing saves roughly
**60–80% versus sending everything to the large tier** (see
[Sample cost report](#sample-cost-report)).

## Architecture

```
                ┌─────────────────────┐
  POST /v1/     │  Complexity         │   simple / medium / complex
  chat/         │  Classifier         │   TF-IDF + logistic regression
  completions ──▶  (learned, blended  │
                │   with a heuristic) │
                └────────┬────────────┘
                         ▼
                ┌─────────────────────┐
                │  Semantic Cache     │   TF-IDF cosine similarity
                │  (TTL + threshold)  │   hit → $0, no model call
                └────────┬────────────┘
                         ▼  (miss)
                ┌─────────────────────┐
                │  Router             │   YAML routing table:
                │  + Circuit Breaker  │   ordered candidates per tier
                └────────┬────────────┘
                         ▼
                ┌─────────────────────┐
                │  Backend (pluggable)│   Mock (offline) or
                │                     │   OpenAI-compatible API
                └────────┬────────────┘
                         ▼
                ┌─────────────────────┐
                │  Cost Tracker       │   SQLite per-request log,
                │  (SQLite + budget)  │   daily rollups, budget alerts
                └─────────────────────┘
```

### Routing strategy

| Complexity | Example | First choice | Fallbacks |
|---|---|---|---|
| simple | "What is the capital of France?" | gpt-4o-mini | claude-3-5-haiku |
| medium | "Explain how OAuth 2.0 works" | gpt-4o | claude-3-5-sonnet, gpt-4o-mini |
| complex | "Design a multi-region active-active trading system…" | claude-3-opus | gpt-4-turbo, gpt-4o |

Complexity comes from two signals, blended 75/25:

- **Learned classifier** — TF-IDF (1–2 grams) + multinomial logistic
  regression, trained on `data/training_prompts.csv` (45 hand-labelled
  prompts). Retrain by editing that CSV; the model trains lazily on
  first use, in under a second.
- **Heuristic** — prompt length, sentence count, deep-work keywords
  (`design`, `prove`, `distributed`, …), code markers, and a discount
  for short factoid questions. This is also the fallback if training
  data is unavailable.

### Semantic cache

TF-IDF cosine similarity over stored prompts. A request is a hit when
similarity ≥ 0.85 (configurable) and the entry is younger than the TTL
(1h default). Hits cost $0 and skip the model entirely. Hit rate is
reported on `/stats`.

### Reliability

- **Fallback chain** — if a model errors or times out, the router tries
  the next candidate in the tier's list, then the next tier down.
- **Circuit breaker** — after 3 consecutive failures a model is skipped
  for 60s (both configurable in `config/routing.yaml`), then probed again.

### Cost tracking

Every request is logged to SQLite (`data/costs.db`) with model, tokens,
cost and cache status. `/stats` returns totals, cost by model, daily
rollups, and a budget status that warns at 80% of the monthly budget
and flags when it is exceeded.

## Quick start

```bash
git clone <repo-url> && cd llm-router-cost-optimizer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn llm_router.api:app --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
docker build -t llm-router .
docker run -p 8000:8000 llm-router
```

To use a real provider instead of the mock backend:

```bash
export ROUTER_BACKEND=openai
export OPENAI_API_KEY=<your-key-here>
uvicorn llm_router.api:app --port 8000
```

## Example requests

```bash
# Simple prompt -> gpt-4o-mini, ~$0.00002
curl -X POST localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```

```json
{
  "id": "chatcmpl-3f2a91c0",
  "model": "gpt-4o-mini",
  "complexity": "simple",
  "cached": false,
  "cost_usd": 0.000018,
  "usage": {"prompt_tokens": 8, "completion_tokens": 30, "total_tokens": 38},
  "choices": [{"index": 0, "message": {"role": "assistant",
    "content": "[mock:gpt-4o-mini] Response to your prompt ..."}, "finish_reason": "stop"}]
}
```

```bash
# Same prompt again -> served from cache, $0
curl -X POST localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "What is the capital of France?"}]}'
# => "cached": true, "cost_usd": 0.0

# Stats: hit rate, spend, budget, circuit breakers
curl localhost:8000/stats
```

## Sample cost report

Simulated month: 10,000 requests (60% simple, 30% medium, 10% complex),
~800 input / ~400 output tokens each, 25% cache hit rate.

| Strategy | Monthly cost |
|---|---|
| Everything to large tier (claude-3-opus) | ~$510 |
| Routed (this project) | ~$95 |
| Routed + cache hits | ~$71 |

Savings come from the price gap between tiers (small models are
~100× cheaper per token than large ones) and from cache hits on
repeated prompts. Your numbers will differ with your traffic mix —
run `GET /stats` to see your own breakdown.

## Configuration

Everything lives in `config/routing.yaml`:

- **models** — per-model price per 1k tokens, latency, context window
- **routing** — ordered candidate models per complexity class
- **cache** — similarity threshold, TTL, max entries
- **budget** — monthly budget in USD, warn threshold
- **reliability** — circuit-breaker failures/cooldown, request timeout

## Tests

```bash
pip install -r requirements-dev.txt
PYTHONPATH=src pytest -q
```

28 tests covering routing decisions, classifier accuracy, cache
hits/misses/TTL, fallback chains, circuit breakers, cost math, budget
alerts and the FastAPI endpoints — 31 tests in total, all passing.

## Limitations

- The classifier is trained on 45 hand-labelled prompts — good enough
  to demonstrate the pattern, not enough for production routing
  decisions. A real deployment wants thousands of labelled examples
  (or labels derived from human ratings / eval outcomes).
- The semantic cache re-fits TF-IDF on every lookup; fine for
  thousands of entries, not for millions — use FAISS or pgvector
  at scale.
- Token counting is a character-based estimate (~4 chars/token);
  use the provider's tokenizer for exact billing.
- Prices in `config/routing.yaml` are illustrative list prices —
  verify against the provider's current pricing page.
- The mock backend returns deterministic fake text; it exists so the
  whole system runs offline and in CI with no API key.
