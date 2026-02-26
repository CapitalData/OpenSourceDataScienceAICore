"""
Unit tests for the convergence_analysis module.
"""

from __future__ import annotations

import math
import unittest

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_inference.convergence_analysis import (
    ConvergenceAnalyzer,
    ConvergenceReport,
    _shannon_entropy,
)
from llm_inference.self_consistency import ConsistencyResult


def _make_result(counts: dict, final: str = "") -> ConsistencyResult:
    total = sum(counts.values())
    # Use the most common as final if not specified
    if not final and counts:
        final = max(counts, key=lambda k: counts[k])
    return ConsistencyResult(
        final_answer=final,
        confidence=counts.get(final, 0) / total if total else 0.0,
        answer_counts=counts,
        reasoning_paths=["path"] * total,
        num_samples=total,
    )


class TestShannonEntropy(unittest.TestCase):
    def test_uniform_two_choices(self):
        # 50/50 split → entropy = 1 bit
        entropy = _shannon_entropy({"A": 5, "B": 5})
        self.assertAlmostEqual(entropy, 1.0)

    def test_certain_one_choice(self):
        # All same → entropy = 0
        entropy = _shannon_entropy({"A": 10})
        self.assertAlmostEqual(entropy, 0.0)

    def test_empty(self):
        self.assertAlmostEqual(_shannon_entropy({}), 0.0)

    def test_four_uniform_choices(self):
        entropy = _shannon_entropy({"A": 1, "B": 1, "C": 1, "D": 1})
        self.assertAlmostEqual(entropy, 2.0)  # log2(4) = 2


class TestConvergenceAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = ConvergenceAnalyzer(convergence_threshold=0.6)

    def test_converged_result(self):
        result = _make_result({"Paris": 4, "London": 1})
        report = self.analyzer.analyse(result, question_id="q1")

        self.assertTrue(report.is_converged)
        self.assertEqual(report.final_answer, "Paris")
        self.assertAlmostEqual(report.agreement_rate, 0.8)
        self.assertGreater(report.entropy, 0.0)

    def test_diverged_result(self):
        result = _make_result({"A": 2, "B": 2, "C": 1})
        report = self.analyzer.analyse(result, question_id="q2")

        self.assertFalse(report.is_converged)
        self.assertNotEqual(report.divergence_note, "")

    def test_perfect_agreement(self):
        result = _make_result({"Yes": 5})
        report = self.analyzer.analyse(result)

        self.assertTrue(report.is_converged)
        self.assertAlmostEqual(report.entropy, 0.0)
        self.assertAlmostEqual(report.agreement_rate, 1.0)

    def test_empty_result(self):
        result = ConsistencyResult(
            final_answer="",
            confidence=0.0,
            answer_counts={},
            reasoning_paths=[],
            num_samples=0,
        )
        report = self.analyzer.analyse(result)
        self.assertFalse(report.is_converged)
        self.assertAlmostEqual(report.entropy, 0.0)

    def test_invalid_threshold_raises(self):
        with self.assertRaises(ValueError):
            ConvergenceAnalyzer(convergence_threshold=0.0)

    def test_analyse_batch(self):
        results = [
            _make_result({"A": 5}),
            _make_result({"B": 2, "C": 2, "D": 1}),  # majority = 2/5 = 0.4 < 0.6
        ]
        reports = self.analyzer.analyse_batch(results, question_ids=["qa", "qb"])
        self.assertEqual(len(reports), 2)
        self.assertEqual(reports[0].question_id, "qa")
        self.assertTrue(reports[0].is_converged)
        self.assertFalse(reports[1].is_converged)

    def test_batch_summary(self):
        results = [_make_result({"A": 5}), _make_result({"B": 2, "C": 2, "D": 1})]  # majority=0.4
        reports = self.analyzer.analyse_batch(results)
        summary = self.analyzer.batch_summary(reports)

        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["converged"], 1)
        self.assertEqual(summary["diverged"], 1)
        self.assertAlmostEqual(summary["convergence_rate"], 0.5)

    def test_batch_summary_empty(self):
        summary = self.analyzer.batch_summary([])
        self.assertEqual(summary["total"], 0)
        self.assertAlmostEqual(summary["convergence_rate"], 0.0)

    def test_report_summary_string(self):
        result = _make_result({"Yes": 5})
        report = self.analyzer.analyse(result, "q99")
        summary = report.summary()
        self.assertIn("q99", summary)
        self.assertIn("CONVERGED", summary)


if __name__ == "__main__":
    unittest.main()
