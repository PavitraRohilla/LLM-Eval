"""
tests/test_metrics.py
Unit tests for src/metrics.py — no OpenAI calls needed.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from metrics import keyword_hit_rate, is_hallucination, answer_relevancy, aggregate


class TestKeywordHitRate:
    def test_all_keywords_found(self):
        assert keyword_hit_rate("Paris is the capital of France", ["Paris", "France"]) == 1.0

    def test_no_keywords_found(self):
        assert keyword_hit_rate("I don't know", ["Paris", "France"]) == 0.0

    def test_partial_match(self):
        rate = keyword_hit_rate("Paris is great", ["Paris", "France"])
        assert rate == 0.5

    def test_case_insensitive(self):
        assert keyword_hit_rate("paris is capital", ["Paris"]) == 1.0

    def test_empty_keywords_returns_one(self):
        assert keyword_hit_rate("anything", []) == 1.0

    def test_substring_match(self):
        # "evapor" should match "evaporates"
        assert keyword_hit_rate("Water evaporates from oceans", ["evapor"]) == 1.0


class TestHallucination:
    def test_not_hallucination_when_keywords_present(self):
        assert not is_hallucination("Paris is the capital", ["Paris"], threshold=0.5)

    def test_hallucination_when_no_keywords(self):
        assert is_hallucination("I have no idea", ["Paris", "France"], threshold=0.5)

    def test_hallucination_below_threshold(self):
        # 1/3 keywords found → 0.33 < 0.5 → hallucination
        assert is_hallucination("Only Paris mentioned", ["Paris", "France", "Europe"], threshold=0.5)

    def test_no_hallucination_at_exact_threshold(self):
        # 1/2 = 0.5, threshold 0.5 → NOT a hallucination (>= threshold)
        assert not is_hallucination("Paris is mentioned", ["Paris", "Lyon"], threshold=0.5)


class TestAnswerRelevancy:
    def test_identical_answers_score_one(self):
        text = "Paris is the capital of France"
        score = answer_relevancy(text, text)
        assert score > 0.95

    def test_completely_different_text_scores_low(self):
        score = answer_relevancy("cats and dogs", "quantum physics equations")
        assert score < 0.3

    def test_similar_text_scores_higher_than_different(self):
        reference = "A decorator wraps a function to modify its behavior"
        similar   = "Decorators are wrappers that modify function behavior"
        different = "The weather today is sunny and warm outside"
        assert answer_relevancy(similar, reference) > answer_relevancy(different, reference)

    def test_returns_float_between_zero_and_one(self):
        score = answer_relevancy("hello world", "goodbye world")
        assert 0.0 <= score <= 1.0


class TestAggregate:
    def _make_results(self, latencies, hall_flags, relevancies, costs):
        return [
            {
                "latency_ms": lat,
                "hallucination_flag": hall,
                "answer_relevancy": rel,
                "cost_usd": cost,
            }
            for lat, hall, rel, cost in zip(latencies, hall_flags, relevancies, costs)
        ]

    def test_hallucination_rate(self):
        results = self._make_results(
            [100, 200, 300, 400],
            [True, False, True, False],
            [0.8, 0.9, 0.7, 0.85],
            [0.001] * 4,
        )
        agg = aggregate(results)
        assert agg["hallucination_rate_pct"] == 50.0

    def test_zero_hallucinations(self):
        results = self._make_results([100, 200], [False, False], [0.9, 0.8], [0.001, 0.001])
        assert aggregate(results)["hallucination_rate_pct"] == 0.0

    def test_p95_latency(self):
        # With 4 items sorted [100,200,300,400], p95 index = ceil(0.95*4)-1 = 3 → 400
        results = self._make_results([300, 100, 400, 200], [False]*4, [0.8]*4, [0.001]*4)
        assert aggregate(results)["p95_latency_ms"] == 400

    def test_total_cost(self):
        results = self._make_results([100]*3, [False]*3, [0.8]*3, [0.01, 0.02, 0.03])
        assert abs(aggregate(results)["total_cost_usd"] - 0.06) < 1e-9
