"""
Self-Consistency Sampler
========================
Generates multiple independent reasoning paths for the same prompt and
aggregates the answers using majority voting to produce a more reliable
final answer with a confidence score.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .async_streaming import AsyncStreamingClient, StreamingConfig, StreamResult

logger = logging.getLogger(__name__)


@dataclass
class SamplingConfig:
    """Configuration for the self-consistency sampler."""

    num_samples: int = 5
    temperatures: List[float] = field(
        default_factory=lambda: [0.7, 0.8, 0.9, 1.0, 1.0]
    )
    answer_pattern: Optional[str] = None
    max_tokens: int = 512


@dataclass
class ConsistencyResult:
    """Aggregated result from self-consistency sampling."""

    final_answer: str
    confidence: float
    answer_counts: Dict[str, int]
    reasoning_paths: List[str]
    num_samples: int

    @property
    def majority_fraction(self) -> float:
        """Fraction of samples that agree with the final answer."""
        if self.num_samples == 0:
            return 0.0
        return self.answer_counts.get(self.final_answer, 0) / self.num_samples


def _extract_answer(text: str, pattern: Optional[str] = None) -> str:
    """
    Extract a short answer token from *text*.

    If *pattern* is provided it is used as a regex with a single capture
    group.  Otherwise the last non-empty line is returned after stripping
    common prefixes such as "Answer:" or "A:".
    """
    if pattern:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Fallback: last non-empty line, stripped of common prefixes
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return text.strip()
    last_line = lines[-1]
    for prefix in ("answer:", "a:", "therefore,", "so,", "thus,"):
        if last_line.lower().startswith(prefix):
            last_line = last_line[len(prefix):].strip()
            break
    return last_line


class SelfConsistencySampler:
    """
    Implements self-consistency sampling (Wang et al., 2022).

    Multiple reasoning paths are sampled at varying temperatures;
    the final answer is determined by majority vote, and a confidence
    score is derived from answer convergence.
    """

    def __init__(
        self,
        client: Optional[AsyncStreamingClient] = None,
        streaming_config: Optional[StreamingConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
    ) -> None:
        self.streaming_config = streaming_config or StreamingConfig()
        self.sampling_config = sampling_config or SamplingConfig()
        self.client = client or AsyncStreamingClient(config=self.streaming_config)

    async def _sample_one(
        self,
        messages: List[dict],
        temperature: float,
    ) -> StreamResult:
        """Draw a single reasoning sample at the given temperature."""
        return await self.client.complete(
            messages,
            temperature=temperature,
            max_tokens=self.sampling_config.max_tokens,
        )

    async def sample(
        self,
        messages: List[dict],
    ) -> ConsistencyResult:
        """
        Draw ``num_samples`` reasoning paths for *messages* and aggregate.

        Args:
            messages: OpenAI-format message list representing the prompt.

        Returns:
            :class:`ConsistencyResult` containing the majority answer,
            confidence score, and all reasoning paths.
        """
        cfg = self.sampling_config
        # Cycle through the configured temperatures
        temps = (
            cfg.temperatures * (cfg.num_samples // len(cfg.temperatures) + 1)
        )[: cfg.num_samples]

        tasks = [self._sample_one(messages, temp) for temp in temps]
        results: List[StreamResult] = await asyncio.gather(*tasks, return_exceptions=True)

        reasoning_paths: List[str] = []
        extracted_answers: List[str] = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning("Sample %d failed: %s", i, result)
                continue
            reasoning_paths.append(result.content)
            answer = _extract_answer(result.content, cfg.answer_pattern)
            extracted_answers.append(answer)

        if not extracted_answers:
            return ConsistencyResult(
                final_answer="",
                confidence=0.0,
                answer_counts={},
                reasoning_paths=reasoning_paths,
                num_samples=0,
            )

        answer_counts = Counter(extracted_answers)
        final_answer, top_count = answer_counts.most_common(1)[0]
        confidence = top_count / len(extracted_answers)

        return ConsistencyResult(
            final_answer=final_answer,
            confidence=confidence,
            answer_counts=dict(answer_counts),
            reasoning_paths=reasoning_paths,
            num_samples=len(extracted_answers),
        )

    async def batch_sample(
        self,
        batch: List[List[dict]],
    ) -> List[ConsistencyResult]:
        """
        Run self-consistency sampling for a list of prompts concurrently.

        Args:
            batch: List of message lists, one per question.

        Returns:
            List of :class:`ConsistencyResult` in the same order as *batch*.
        """
        tasks = [self.sample(messages) for messages in batch]
        return list(await asyncio.gather(*tasks))
