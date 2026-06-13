"""
src/metrics.py
Evaluation metrics for LLM outputs.

Metrics computed per question:
  - keyword_hit_rate   : fraction of expected keywords found in the answer
  - hallucination_flag : True when keyword_hit_rate < threshold (proxy metric)
  - answer_relevancy   : cosine similarity of TF-IDF vectors (no external API needed)
  - latency_ms         : end-to-end call latency
  - cost_usd           : estimated API cost

Aggregate metrics:
  - hallucination_rate : % of responses flagged as hallucinations
  - p50_latency_ms     : median latency
  - p95_latency_ms     : 95th-percentile latency
  - total_cost_usd     : sum of per-query costs
  - avg_relevancy      : mean answer relevancy score
"""

from __future__ import annotations

import math
import re
from statistics import median
from typing import Any


# ── Per-question metrics ──────────────────────────────────────────────────────

def keyword_hit_rate(answer: str, expected_keywords: list[str]) -> float:
    """Fraction of expected keywords (case-insensitive substring) found."""
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords)


def is_hallucination(answer: str, expected_keywords: list[str], threshold: float = 0.5) -> bool:
    """Flag as hallucination if keyword hit rate is below threshold."""
    return keyword_hit_rate(answer, expected_keywords) < threshold


def answer_relevancy(answer: str, reference: str) -> float:
    """
    TF-IDF cosine similarity between answer and reference answer.
    Returns a float in [0, 1].
    """
    def tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z]+", text.lower())

    def tf(tokens: list[str]) -> dict[str, float]:
        counts: dict[str, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        total = len(tokens) or 1
        return {t: c / total for t, c in counts.items()}

    tokens_a, tokens_b = tokenize(answer), tokenize(reference)
    tf_a, tf_b = tf(tokens_a), tf(tokens_b)
    vocab = set(tf_a) | set(tf_b)

    # IDF over the two-doc "corpus"
    idf: dict[str, float] = {}
    for term in vocab:
        df = (term in tf_a) + (term in tf_b)
        idf[term] = math.log((2 + 1) / (df + 1)) + 1  # smoothed

    vec_a = [tf_a.get(t, 0) * idf[t] for t in vocab]
    vec_b = [tf_b.get(t, 0) * idf[t] for t in vocab]

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(x ** 2 for x in vec_a)) or 1e-9
    norm_b = math.sqrt(sum(x ** 2 for x in vec_b)) or 1e-9
    return round(dot / (norm_a * norm_b), 4)


# ── Aggregate metrics ─────────────────────────────────────────────────────────

def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate metrics from a list of per-question result dicts."""
    latencies = [r["latency_ms"] for r in results]
    sorted_lat = sorted(latencies)
    n = len(sorted_lat)

    def percentile(data: list[float], p: int) -> float:
        idx = max(0, math.ceil((p / 100) * len(data)) - 1)
        return data[idx]

    hallucinations = [r for r in results if r["hallucination_flag"]]
    hallucination_rate = len(hallucinations) / n if n else 0.0

    return {
        "total_questions": n,
        "hallucination_rate_pct": round(hallucination_rate * 100, 2),
        "hallucination_count": len(hallucinations),
        "avg_relevancy": round(sum(r["answer_relevancy"] for r in results) / n, 4) if n else 0,
        "p50_latency_ms": round(percentile(sorted_lat, 50), 2),
        "p95_latency_ms": round(percentile(sorted_lat, 95), 2),
        "total_cost_usd": round(sum(r["cost_usd"] for r in results), 6),
        "avg_cost_per_query_usd": round(sum(r["cost_usd"] for r in results) / n, 6) if n else 0,
    }
