"""
Unit tests for the metrics module.
"""

from __future__ import annotations

import time
import unittest

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_inference.metrics import (
    AggregateMetrics,
    MetricsCollector,
    RequestMetrics,
    DEFAULT_COST_PER_1K_TOKENS,
)


class TestRequestMetrics(unittest.TestCase):
    def test_total_tokens(self):
        m = RequestMetrics(
            request_id="r1", model="gpt-4o-mini",
            prompt_tokens=100, completion_tokens=50
        )
        self.assertEqual(m.total_tokens, 150)

    def test_estimated_cost_known_model(self):
        m = RequestMetrics(
            request_id="r1", model="gpt-4o-mini",
            prompt_tokens=1000, completion_tokens=1000
        )
        cost = m.estimated_cost_usd()
        # $0.00015/1k prompt + $0.0006/1k completion = $0.00075
        self.assertAlmostEqual(cost, 0.00075, places=6)

    def test_estimated_cost_unknown_model(self):
        m = RequestMetrics(
            request_id="r1", model="unknown-model",
            prompt_tokens=1000, completion_tokens=1000
        )
        self.assertAlmostEqual(m.estimated_cost_usd(), 0.0)


class TestAggregateMetrics(unittest.TestCase):
    def test_accuracy_none_when_no_graded(self):
        agg = AggregateMetrics()
        self.assertIsNone(agg.accuracy)

    def test_accuracy_calculated(self):
        agg = AggregateMetrics(correct_answers=3, graded_answers=5)
        self.assertAlmostEqual(agg.accuracy, 0.6)

    def test_mean_latency_zero_requests(self):
        agg = AggregateMetrics()
        self.assertAlmostEqual(agg.mean_latency_seconds, 0.0)

    def test_to_dict_keys(self):
        agg = AggregateMetrics(total_requests=1, successful_requests=1)
        d = agg.to_dict()
        expected_keys = {
            "total_requests", "successful_requests", "failed_requests",
            "total_prompt_tokens", "total_completion_tokens", "total_tokens",
            "total_latency_seconds", "mean_latency_seconds",
            "total_cost_usd", "accuracy",
        }
        self.assertEqual(set(d.keys()), expected_keys)


class TestMetricsCollector(unittest.TestCase):
    def test_record_context_manager(self):
        collector = MetricsCollector(model="gpt-4o-mini")
        with collector.record("req-1") as m:
            m.prompt_tokens = 100
            m.completion_tokens = 50

        agg = collector.aggregate()
        self.assertEqual(agg.total_requests, 1)
        self.assertEqual(agg.successful_requests, 1)
        self.assertEqual(agg.failed_requests, 0)
        self.assertEqual(agg.total_prompt_tokens, 100)
        self.assertEqual(agg.total_completion_tokens, 50)

    def test_record_captures_error(self):
        collector = MetricsCollector(model="gpt-4o-mini")
        try:
            with collector.record("req-err") as m:
                raise ValueError("simulated error")
        except ValueError:
            pass

        agg = collector.aggregate()
        self.assertEqual(agg.failed_requests, 1)

    def test_accuracy_tracking(self):
        collector = MetricsCollector(model="gpt-4o-mini")
        for correct in [True, True, False, True]:
            with collector.record("r") as m:
                m.is_correct = correct

        agg = collector.aggregate()
        self.assertEqual(agg.graded_answers, 4)
        self.assertAlmostEqual(agg.accuracy, 0.75)

    def test_cost_aggregation(self):
        collector = MetricsCollector(model="gpt-4o-mini")
        with collector.record("r1") as m:
            m.prompt_tokens = 1000
            m.completion_tokens = 1000

        agg = collector.aggregate()
        self.assertGreater(agg.total_cost_usd, 0.0)

    def test_add_method(self):
        collector = MetricsCollector(model="gpt-4o-mini")
        collector.add(RequestMetrics(request_id="r1", model="gpt-4o-mini"))
        self.assertEqual(collector.aggregate().total_requests, 1)

    def test_reset(self):
        collector = MetricsCollector(model="gpt-4o-mini")
        with collector.record("r1"):
            pass
        collector.reset()
        self.assertEqual(collector.aggregate().total_requests, 0)

    def test_compare_strategies(self):
        baseline = MetricsCollector(model="gpt-4o-mini")
        variant = MetricsCollector(model="gpt-4o-mini")

        with baseline.record("r1") as m:
            m.prompt_tokens = 500
            m.completion_tokens = 200
            m.is_correct = True

        with variant.record("r1") as m:
            m.prompt_tokens = 1000
            m.completion_tokens = 500
            m.is_correct = True

        comparison = baseline.compare_strategies(variant, "base", "new")
        self.assertIn("base", comparison)
        self.assertIn("new", comparison)
        self.assertIn("delta", comparison)
        # Variant uses more tokens → positive cost delta
        self.assertGreater(comparison["delta"]["cost_usd"], 0)


if __name__ == "__main__":
    unittest.main()
