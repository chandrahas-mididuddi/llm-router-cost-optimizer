"""Semantic cache: TF-IDF cosine similarity over past prompts. A request is a cache hit when a stored prompt is similar enough (cosine >= threshold) and not expired (TTL)."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
@dataclass
class CacheEntry:
    prompt: str
    response: str
    model: str
    created_at: float
    hits: int = 0
@dataclass
class CacheResult:
    hit: bool
    response: str | None = None
    matched_prompt: str | None = None
    similarity: float = 0.0
    model: str | None = None
class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.85, ttl_seconds: int = 3600, max_entries: int = 10000):
        self.threshold = similarity_threshold
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._entries: list[CacheEntry] = []
        self._lookups = 0
        self._hits = 0
        self._vectorizer: TfidfVectorizer | None = None
    @property
    def lookups(self) -> int:
        return self._lookups
    @property
    def hits(self) -> int:
        return self._hits
    @property
    def hit_rate(self) -> float:
        return round(self._hits / self._lookups, 4) if self._lookups else 0.0
    @property
    def size(self) -> int:
        return len(self._entries)
    def _live_entries(self, now: float) -> list[CacheEntry]:
        live = [e for e in self._entries if now - e.created_at <= self.ttl]
        if len(live) != len(self._entries):
            self._entries = live
        return live
    def lookup(self, prompt: str) -> CacheResult:
        now = time.time()
        self._lookups += 1
        live = self._live_entries(now)
        if not live:
            return CacheResult(hit=False)
        corpus = [e.prompt for e in live] + [prompt]
        vectorizer = TfidfVectorizer(stop_words="english").fit(corpus)
        matrix = vectorizer.transform(corpus)
        sims = cosine_similarity(matrix[-1], matrix[:-1])[0]
        best_idx = int(sims.argmax())
        best = float(sims[best_idx])
        if best >= self.threshold:
            entry = live[best_idx]
            entry.hits += 1
            self._hits += 1
            return CacheResult(hit=True, response=entry.response, matched_prompt=entry.prompt, similarity=round(best, 3), model=entry.model)
        return CacheResult(hit=False, similarity=round(best, 3))
    def store(self, prompt: str, response: str, model: str) -> None:
        now = time.time()
        self._live_entries(now)
        if len(self._entries) >= self.max_entries:
            self._entries.sort(key=lambda e: (e.hits, e.created_at))
            self._entries.pop(0)
        self._entries.append(CacheEntry(prompt=prompt, response=response, model=model, created_at=now))
    def clear(self) -> None:
        self._entries.clear()
        self._lookups = 0
        self._hits = 0
