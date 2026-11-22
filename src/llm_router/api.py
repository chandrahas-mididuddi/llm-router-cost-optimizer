"""FastAPI service. POST /v1/chat/completions, GET /stats, GET /health."""
from __future__ import annotations
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .backends import MockBackend, OpenAICompatibleBackend
from .cache import SemanticCache
from .classifier import ComplexityClassifier
from .cost_tracker import CostTracker
from .registry import RouterConfig
from .router import RouteResult, Router, RouterError
_ROUTER: Router | None = None
def build_router() -> Router:
    config = RouterConfig.load(os.environ.get("ROUTER_CONFIG"))
    backend_name = os.environ.get("ROUTER_BACKEND", "mock").lower()
    if backend_name == "openai":
        backend = OpenAICompatibleBackend()
    else:
        backend = MockBackend()
    return Router(config=config, backend=backend, classifier=ComplexityClassifier(), cache=SemanticCache(similarity_threshold=config.cache_threshold, ttl_seconds=config.cache_ttl_seconds, max_entries=config.cache_max_entries), cost_tracker=CostTracker(os.environ.get("COST_DB", "data/costs.db")),)
def get_router() -> Router:
    global _ROUTER
    if _ROUTER is None:
        _ROUTER = build_router()
    return _ROUTER
@asynccontextmanager
async def lifespan(app: FastAPI):
    get_router()
    yield
app = FastAPI(title="LLM Router — Cost Optimizer", version="0.1.0", lifespan=lifespan)
class ChatMessage(BaseModel):
    role: str
    content: str
class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = 512
    use_cache: bool = True
class ChatCompletionResponse(BaseModel):
    id: str
    model: str
    complexity: str
    cached: bool
    cost_usd: float
    usage: dict
    choices: list[dict]
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(req: ChatCompletionRequest):
    prompt = "\n".join(m.content for m in req.messages if m.role == "user") or req.messages[-1].content
    try:
        result: RouteResult = get_router().route(prompt, max_tokens=req.max_tokens, use_cache=req.use_cache)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RouterError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ChatCompletionResponse(id=f"chatcmpl-{hash(prompt) & 0xFFFFFFFF:08x}", model=result.model, complexity=result.complexity, cached=result.cached, cost_usd=result.cost_usd, usage={"prompt_tokens": result.input_tokens, "completion_tokens": result.output_tokens, "total_tokens": result.input_tokens + result.output_tokens}, choices=[{"index": 0, "message": {"role": "assistant", "content": result.text}, "finish_reason": "stop"}],)
@app.get("/stats")
def stats() -> dict:
    return get_router().stats()
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("llm_router.api:app", host="0.0.0.0", port=8000, reload=False)
