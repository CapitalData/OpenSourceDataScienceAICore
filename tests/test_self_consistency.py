"""
Unit tests for the self_consistency module.
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_inference.async_streaming import StreamResult
from llm_inference.self_consistency import (
    ConsistencyResult,
    SamplingConfig,
    SelfConsistencySampler,
    _extract_answer,
)


class TestExtractAnswer(unittest.TestCase):
    def test_plain_text(self):
        self.assertEqual(_extract_answer("The answer is 42"), "The answer is 42")

    def test_strips_answer_prefix(self):
        self.assertEqual(_extract_answer("Some reasoning\nAnswer: Paris"), "Paris")

    def test_strips_therefore_prefix(self):
        self.assertEqual(_extract_answer("...blah\nTherefore, 7"), "7")

    def test_regex_pattern(self):
        text = "After thinking... the answer is (B)"
        self.assertEqual(_extract_answer(text, r"the answer is \((\w)\)"), "B")

    def test_empty_string(self):
        self.assertEqual(_extract_answer(""), "")


class TestSelfConsistencySampler(unittest.IsolatedAsyncioTestCase):
    def _make_sampler(self, responses: list[str]) -> SelfConsistencySampler:
        """Create a sampler whose client returns *responses* in order."""
        idx = [0]

        async def _complete(messages, temperature=None, max_tokens=None):
            text = responses[idx[0] % len(responses)]
            idx[0] += 1
            return StreamResult(content=text, latency_seconds=0.0, model="gpt-4o-mini")

        mock_client = MagicMock()
        mock_client.complete = _complete

        sampler = SelfConsistencySampler(
            client=mock_client,
            sampling_config=SamplingConfig(num_samples=5, temperatures=[0.7]),
        )
        return sampler

    async def test_majority_vote(self):
        # 4 out of 5 say "Paris"
        responses = ["Paris", "Paris", "Paris", "Paris", "London"]
        sampler = self._make_sampler(responses)
        result = await sampler.sample([{"role": "user", "content": "Capital of France?"}])

        self.assertIsInstance(result, ConsistencyResult)
        self.assertEqual(result.final_answer, "Paris")
        self.assertAlmostEqual(result.confidence, 0.8)
        self.assertEqual(result.num_samples, 5)

    async def test_answer_counts(self):
        responses = ["A", "B", "A", "C", "A"]
        sampler = self._make_sampler(responses)
        result = await sampler.sample([{"role": "user", "content": "Pick a letter"}])

        self.assertEqual(result.answer_counts["A"], 3)
        self.assertEqual(result.answer_counts["B"], 1)
        self.assertEqual(result.answer_counts["C"], 1)

    async def test_majority_fraction_property(self):
        responses = ["X"] * 3 + ["Y"] * 2
        sampler = self._make_sampler(responses)
        result = await sampler.sample([{"role": "user", "content": "?"}])
        self.assertAlmostEqual(result.majority_fraction, 3 / 5)

    async def test_batch_sample(self):
        sampler = self._make_sampler(["42"])
        batch = [
            [{"role": "user", "content": "Q1"}],
            [{"role": "user", "content": "Q2"}],
        ]
        results = await sampler.batch_sample(batch)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIsInstance(r, ConsistencyResult)

    async def test_empty_responses_handled(self):
        """If all samples fail, should return an empty ConsistencyResult."""

        async def _failing_complete(messages, temperature=None, max_tokens=None):
            raise RuntimeError("API down")

        mock_client = MagicMock()
        mock_client.complete = _failing_complete

        sampler = SelfConsistencySampler(
            client=mock_client,
            sampling_config=SamplingConfig(num_samples=3, temperatures=[0.7]),
        )
        result = await sampler.sample([{"role": "user", "content": "?"}])
        self.assertEqual(result.final_answer, "")
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.num_samples, 0)


if __name__ == "__main__":
    unittest.main()
