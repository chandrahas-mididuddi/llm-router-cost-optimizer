"""Prompt complexity classification. Two signals are combined: 1. A fast heuristic (no model needed). 2. A learned TF-IDF + logistic-regression classifier trained on data/training_prompts.csv (45 labelled prompts, 3 classes). The classifier is trained lazily on first use and cached."""
from __future__ import annotations
import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
LABELS = ("simple", "medium", "complex")
_COMPLEX_KEYWORDS = ("design", "architect", "prove", "proof", "derive", "formal", "distributed", "consensus", "optimize", "optimization", "debug", "race condition", "security vulnerab", "threat model", "mathematical", "multi-region", "zero-trust", "compiler", "backpropagation", "queueing", "cap theorem", "microservices", "failure modes", "consistency", "rate limiter", "network partition", "trade-offs", "active-active",)
_CODE_MARKERS = ("```", "def ", "class ", "SELECT ", "function", "->", "{")
_QUESTION_WORDS = ("what", "who", "when", "where", "is", "are", "name", "list", "give")
@dataclass
class ComplexityScore:
    label: str
    score: float
    method: str
    probabilities: dict[str, float]
def heuristic_score(prompt: str) -> ComplexityScore:
    text = prompt.strip()
    lower = text.lower()
    words = re.findall(r"[a-zA-Z']+", lower)
    n_words = len(words)
    n_sentences = max(1, len(re.split(r"[.!?]\s+", text)))
    score = 0.0
    score += min(0.30, n_words / 300.0)
    score += min(0.10, (n_sentences - 1) * 0.05)
    hits = sum(1 for kw in _COMPLEX_KEYWORDS if kw in lower)
    score += min(0.55, hits * 0.15)
    if " and " in lower and any(w in lower for w in ("analyze", "discuss", "evaluate", "compare")):
        score += 0.10
    if any(m in prompt for m in _CODE_MARKERS):
        score += 0.15
    if n_words <= 12 and lower.split(" ", 1)[0] in _QUESTION_WORDS:
        score -= 0.15
    score = max(0.0, min(1.0, score))
    if score < 0.33:
        label = "simple"
    elif score < 0.66:
        label = "medium"
    else:
        label = "complex"
    probs = {lab: 0.15 for lab in LABELS}
    probs[label] = 0.70
    return ComplexityScore(label=label, score=score, method="heuristic", probabilities=probs)
class ComplexityClassifier:
    def __init__(self, training_csv: str | Path | None = None):
        self.training_csv = Path(training_csv) if training_csv else (Path(__file__).resolve().parents[2] / "data" / "training_prompts.csv")
        self._pipeline: Pipeline | None = None
    def train(self) -> "ComplexityClassifier":
        prompts, labels = [], []
        with open(self.training_csv, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                prompts.append(row["prompt"])
                labels.append(row["label"])
        self._pipeline = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")), ("clf", LogisticRegression(max_iter=1000, C=2.0)),])
        self._pipeline.fit(prompts, labels)
        return self
    @property
    def is_trained(self) -> bool:
        return self._pipeline is not None
    def _ensure_trained(self) -> None:
        if not self.is_trained:
            self.train()
    def predict(self, prompt: str) -> ComplexityScore:
        heuristic = heuristic_score(prompt)
        try:
            self._ensure_trained()
            assert self._pipeline is not None
            probs = self._pipeline.predict_proba([prompt])[0]
            learned = {lab: float(p) for lab, p in zip(self._pipeline.classes_, probs)}
        except Exception:
            return heuristic
        blended = {}
        for lab in LABELS:
            blended[lab] = (0.75 * learned.get(lab, 0.0) + 0.25 * heuristic.probabilities.get(lab, 0.0))
        label = max(blended, key=blended.get)
        score = blended["complex"] * 0.5 + blended["medium"] * 0.5 + blended["simple"] * 0.0
        total = sum(blended.values()) or 1.0
        probs = {k: round(v / total, 3) for k, v in blended.items()}
        return ComplexityScore(label=label, score=round(score, 3), method="learned", probabilities=probs)
    def train_accuracy(self) -> float:
        self._ensure_trained()
        assert self._pipeline is not None
        prompts, labels = [], []
        with open(self.training_csv, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                prompts.append(row["prompt"])
                labels.append(row["label"])
        return float(self._pipeline.score(prompts, labels))
def load_classifier() -> ComplexityClassifier:
    return ComplexityClassifier().train()
