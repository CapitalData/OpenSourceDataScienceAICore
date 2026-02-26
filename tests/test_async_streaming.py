"""
Unit tests for the async_streaming module.
Uses unittest.mock to avoid real API calls.
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_inference.async_streaming import (
    AsyncStreamingClient,
    StreamChunk,
    StreamResult,
    StreamingConfig,
)


def _make_delta(content: str, finish_reason: str | None = None):
    """Helper to build a mock stream event."""
    choice = MagicMock()
    choice.delta = MagicMock()
    choice.delta.content = content
    choice.finish_reason = finish_reason
    event = MagicMock()
    event.choices = [choice]
    return event


class _FakeStream:
    """Async iterator that yields pre-configured events."""

    def __init__(self, events):
        self._events = iter(events)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._events)
        except StopIteration:
            raise StopAsyncIteration


class TestStreamingConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = StreamingConfig()
        self.assertEqual(cfg.model, "gpt-4o-mini")
        self.assertEqual(cfg.max_retries, 3)

    def test_custom(self):
        cfg = StreamingConfig(model="gpt-4o", temperature=0.5, max_tokens=256)
        self.assertEqual(cfg.model, "gpt-4o")
        self.assertEqual(cfg.temperature, 0.5)


class TestAsyncStreamingClient(unittest.IsolatedAsyncioTestCase):
    def _make_client(self, events):
        """Build a client whose underlying API returns *events* as a stream."""
        fake_openai = MagicMock()
        fake_openai.chat = MagicMock()
        fake_openai.chat.completions = MagicMock()
        fake_openai.chat.completions.create = AsyncMock(
            return_value=_FakeStream(events)
        )
        return AsyncStreamingClient(
            config=StreamingConfig(max_retries=1),
            client=fake_openai,
        )

    async def test_stream_completion_yields_chunks(self):
        events = [
            _make_delta("Hello"),
            _make_delta(", "),
            _make_delta("world", finish_reason="stop"),
        ]
        client = self._make_client(events)
        messages = [{"role": "user", "content": "Say hello"}]

        chunks = []
        async for chunk in client.stream_completion(messages):
            chunks.append(chunk)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].content, "Hello")
        self.assertEqual(chunks[2].finish_reason, "stop")

    async def test_complete_returns_stream_result(self):
        events = [
            _make_delta("The answer is 42"),
            _make_delta("", finish_reason="stop"),
        ]
        client = self._make_client(events)
        messages = [{"role": "user", "content": "What is the answer?"}]

        result = await client.complete(messages)

        self.assertIsInstance(result, StreamResult)
        self.assertEqual(result.content, "The answer is 42")
        self.assertEqual(result.finish_reason, "stop")
        self.assertGreaterEqual(result.latency_seconds, 0.0)

    async def test_batch_complete_returns_all_results(self):
        def _make_events(text):
            return [_make_delta(text, finish_reason="stop")]

        fake_openai = MagicMock()
        call_count = [0]
        texts = ["Answer A", "Answer B", "Answer C"]

        async def _create(**kwargs):
            idx = call_count[0]
            call_count[0] += 1
            return _FakeStream(_make_events(texts[idx % len(texts)]))

        fake_openai.chat = MagicMock()
        fake_openai.chat.completions = MagicMock()
        fake_openai.chat.completions.create = _create

        client = AsyncStreamingClient(
            config=StreamingConfig(max_retries=1),
            client=fake_openai,
        )
        batch = [[{"role": "user", "content": f"Q{i}"}] for i in range(3)]
        results = await client.batch_complete(batch)

        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIsInstance(r, StreamResult)

    async def test_batch_complete_handles_errors_gracefully(self):
        import openai as _openai

        fake_openai = MagicMock()
        fake_openai.chat = MagicMock()
        fake_openai.chat.completions = MagicMock()
        fake_openai.chat.completions.create = AsyncMock(
            side_effect=Exception("network error")
        )

        client = AsyncStreamingClient(
            config=StreamingConfig(max_retries=1),
            client=fake_openai,
        )
        batch = [[{"role": "user", "content": "Q"}]]
        results = await client.batch_complete(batch)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].finish_reason, "error")


if __name__ == "__main__":
    unittest.main()
