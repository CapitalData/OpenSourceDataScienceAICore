"""
Async Streaming Client
======================
Core async streaming module using the OpenAI async API with retry logic,
concurrent request batching, and proper error handling.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

try:
    import openai
    from openai import AsyncOpenAI
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "openai is required. Install it with: pip install openai"
    ) from exc

logger = logging.getLogger(__name__)


@dataclass
class StreamingConfig:
    """Configuration for the async streaming client."""

    model: str = "gpt-4o-mini"
    max_tokens: int = 1024
    temperature: float = 0.7
    timeout: float = 60.0
    max_retries: int = 3
    retry_min_wait: float = 1.0
    retry_max_wait: float = 10.0
    max_concurrent_requests: int = 10


@dataclass
class StreamChunk:
    """A single streamed token chunk."""

    content: str
    finish_reason: Optional[str] = None
    usage: Optional[dict] = None


@dataclass
class StreamResult:
    """The complete result of a streaming generation request."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    model: str = ""
    finish_reason: str = ""


class AsyncStreamingClient:
    """
    Async streaming client for OpenAI completions.

    Supports:
    - Non-blocking API calls via ``AsyncOpenAI``
    - Streaming response handling
    - Concurrent request batching with a semaphore
    - Automatic retry with exponential back-off
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[StreamingConfig] = None,
        client: Optional[AsyncOpenAI] = None,
    ) -> None:
        self.config = config or StreamingConfig()
        # Allow injecting a pre-built client (useful for testing)
        self._client = client or AsyncOpenAI(api_key=api_key)
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

    async def stream_completion(
        self,
        messages: List[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a chat completion, yielding :class:`StreamChunk` objects.

        Args:
            messages: OpenAI-format message list.
            temperature: Override the config temperature for this call.
            max_tokens: Override the config max_tokens for this call.

        Yields:
            :class:`StreamChunk` for each delta received from the API.
        """
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens

        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(
                (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError)
            ),
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential(
                min=self.config.retry_min_wait,
                max=self.config.retry_max_wait,
            ),
            reraise=True,
        ):
            with attempt:
                async with self._semaphore:
                    stream = await self._client.chat.completions.create(
                        model=self.config.model,
                        messages=messages,
                        temperature=temp,
                        max_tokens=max_tok,
                        stream=True,
                        timeout=self.config.timeout,
                    )
                    async for event in stream:
                        choice = event.choices[0] if event.choices else None
                        if choice is None:
                            continue
                        delta_content = (
                            choice.delta.content if choice.delta.content else ""
                        )
                        yield StreamChunk(
                            content=delta_content,
                            finish_reason=choice.finish_reason,
                        )

    async def complete(
        self,
        messages: List[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> StreamResult:
        """
        Stream a completion and collect it into a single :class:`StreamResult`.

        Args:
            messages: OpenAI-format message list.
            temperature: Override the config temperature for this call.
            max_tokens: Override the config max_tokens for this call.

        Returns:
            :class:`StreamResult` with the full response text and usage stats.
        """
        start = time.monotonic()
        chunks: List[str] = []
        finish_reason = ""

        async for chunk in self.stream_completion(
            messages, temperature=temperature, max_tokens=max_tokens
        ):
            if chunk.content:
                chunks.append(chunk.content)
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason

        latency = time.monotonic() - start
        content = "".join(chunks)

        return StreamResult(
            content=content,
            latency_seconds=latency,
            model=self.config.model,
            finish_reason=finish_reason,
        )

    async def batch_complete(
        self,
        batch: List[List[dict]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> List[StreamResult]:
        """
        Run multiple completion requests concurrently.

        Args:
            batch: A list of message lists (one per request).
            temperature: Shared temperature override for all requests.
            max_tokens: Shared max_tokens override for all requests.

        Returns:
            List of :class:`StreamResult` in the same order as *batch*.
        """
        tasks = [
            self.complete(messages, temperature=temperature, max_tokens=max_tokens)
            for messages in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        processed: List[StreamResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Request %d failed: %s", i, result)
                processed.append(
                    StreamResult(content="", finish_reason="error")
                )
            else:
                processed.append(result)
        return processed
