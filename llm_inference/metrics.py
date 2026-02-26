"""
Metrics Collection and Reporting
=================================
Track token usage, estimated cost, per-request latency, and accuracy
(when ground truth is available) for LLM inference workloads.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default cost per 1 000 tokens (USD) for common models.
# Update these figures as pricing changes.
DEFAULT_COST_PER_1K_TOKENS: Dict[str, Tuple[float, float]] = {
    # model: (prompt $/1k, completion $/1k)
    "gpt-4o": (0.005, 0.015),
    "gpt-4o-mini": (0.000150, 0.000600),
    "gpt-4-turbo": (0.010, 0.030),
    "gpt-3.5-turbo": (0.0005, 0.0015),
}


@dataclass
class RequestMetrics:
    """Metrics for a single LLM request."""

    request_id: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    is_correct: Optional[bool] = None
    error: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def estimated_cost_usd(
        self,
        cost_table: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> float:
        """Return estimated cost in USD based on token counts."""
        table = cost_table or DEFAULT_COST_PER_1K_TOKENS
        prompt_rate, completion_rate = table.get(self.model, (0.0, 0.0))
        return (
            self.prompt_tokens / 1000 * prompt_rate
            + self.completion_tokens / 1000 * completion_rate
        )


@dataclass
class AggregateMetrics:
    """Aggregate statistics across a collection of requests."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_latency_seconds: float = 0.0
    total_cost_usd: float = 0.0
    correct_answers: int = 0
    graded_answers: int = 0

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def mean_latency_seconds(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_seconds / self.total_requests

    @property
    def accuracy(self) -> Optional[float]:
        if self.graded_answers == 0:
            return None
        return self.correct_answers / self.graded_answers

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_latency_seconds": round(self.total_latency_seconds, 4),
            "mean_latency_seconds": round(self.mean_latency_seconds, 4),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "accuracy": self.accuracy,
        }


class MetricsCollector:
    """
    Collects per-request metrics and produces aggregate reports.

    Usage::

        collector = MetricsCollector(model="gpt-4o-mini")
        with collector.record("req-1") as m:
            result = await client.complete(messages)
            m.prompt_tokens = result.prompt_tokens
            m.completion_tokens = result.completion_tokens

        report = collector.aggregate()
        print(report.to_dict())
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        cost_table: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        self.model = model
        self.cost_table = cost_table or DEFAULT_COST_PER_1K_TOKENS
        self._records: List[RequestMetrics] = []

    # ------------------------------------------------------------------
    # Context-manager helper
    # ------------------------------------------------------------------

    class _RecordContext:
        """Context manager returned by :meth:`MetricsCollector.record`."""

        def __init__(
            self,
            collector: "MetricsCollector",
            request_id: str,
            model: str,
        ) -> None:
            self._collector = collector
            self.metrics = RequestMetrics(request_id=request_id, model=model)
            self._start: float = 0.0

        def __enter__(self) -> RequestMetrics:
            self._start = time.monotonic()
            return self.metrics

        def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
            self.metrics.latency_seconds = time.monotonic() - self._start
            if exc_type is not None:
                self.metrics.error = str(exc_val)
            self._collector._records.append(self.metrics)
            return False  # do not suppress exceptions

    def record(self, request_id: str) -> "_RecordContext":
        """
        Return a context manager that times the block and stores metrics.

        Args:
            request_id: A unique identifier for this request.

        Returns:
            A context manager whose ``__enter__`` value is a
            :class:`RequestMetrics` object you can populate.
        """
        return self._RecordContext(self, request_id, self.model)

    def add(self, metrics: RequestMetrics) -> None:
        """Manually add a pre-populated :class:`RequestMetrics` record."""
        self._records.append(metrics)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def aggregate(self) -> AggregateMetrics:
        """
        Compute aggregate statistics across all recorded requests.

        Returns:
            :class:`AggregateMetrics` summarising cost, latency, tokens,
            and accuracy (if ground-truth flags were set).
        """
        agg = AggregateMetrics()
        for m in self._records:
            agg.total_requests += 1
            if m.error:
                agg.failed_requests += 1
            else:
                agg.successful_requests += 1
            agg.total_prompt_tokens += m.prompt_tokens
            agg.total_completion_tokens += m.completion_tokens
            agg.total_latency_seconds += m.latency_seconds
            agg.total_cost_usd += m.estimated_cost_usd(self.cost_table)
            if m.is_correct is not None:
                agg.graded_answers += 1
                if m.is_correct:
                    agg.correct_answers += 1
        return agg

    def compare_strategies(
        self,
        other: "MetricsCollector",
        label_self: str = "baseline",
        label_other: str = "variant",
    ) -> Dict[str, object]:
        """
        Compare two :class:`MetricsCollector` instances.

        Returns a dict with per-strategy metrics plus delta values for
        cost, latency, and accuracy.
        """
        a = self.aggregate()
        b = other.aggregate()

        def _delta(x: Optional[float], y: Optional[float]) -> Optional[float]:
            if x is None or y is None:
                return None
            return round(y - x, 6)

        return {
            label_self: a.to_dict(),
            label_other: b.to_dict(),
            "delta": {
                "cost_usd": _delta(a.total_cost_usd, b.total_cost_usd),
                "mean_latency_seconds": _delta(
                    a.mean_latency_seconds, b.mean_latency_seconds
                ),
                "accuracy": _delta(a.accuracy, b.accuracy),
            },
        }

    def reset(self) -> None:
        """Clear all recorded metrics."""
        self._records.clear()
