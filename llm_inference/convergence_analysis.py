"""
Convergence Analysis
====================
Tools to analyse how consistently a set of reasoning paths converge on
the same answer.  Metrics include Shannon entropy, agreement rate, and a
structured convergence report.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .self_consistency import ConsistencyResult


@dataclass
class ConvergenceReport:
    """A detailed report on answer convergence for one question."""

    question_id: str
    final_answer: str
    num_samples: int
    agreement_rate: float
    entropy: float
    answer_distribution: Dict[str, float]
    is_converged: bool
    divergence_note: str = ""

    def summary(self) -> str:
        """Return a human-readable one-line summary."""
        status = "CONVERGED" if self.is_converged else "DIVERGED"
        return (
            f"[{self.question_id}] {status} | answer='{self.final_answer}' "
            f"| agreement={self.agreement_rate:.0%} | entropy={self.entropy:.3f}"
        )


def _shannon_entropy(counts: Dict[str, int]) -> float:
    """Compute Shannon entropy (in bits) from a frequency dict."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


class ConvergenceAnalyzer:
    """
    Analyses convergence of self-consistency sampling results.

    Args:
        convergence_threshold: Minimum agreement rate to consider a result
            *converged* (default: 0.6, i.e. 60 % of samples agree).
    """

    def __init__(self, convergence_threshold: float = 0.6) -> None:
        if not 0.0 < convergence_threshold <= 1.0:
            raise ValueError("convergence_threshold must be in (0, 1]")
        self.convergence_threshold = convergence_threshold

    def analyse(
        self,
        result: ConsistencyResult,
        question_id: str = "q0",
    ) -> ConvergenceReport:
        """
        Produce a :class:`ConvergenceReport` for a single sampling result.

        Args:
            result: Output from :class:`~llm_inference.SelfConsistencySampler`.
            question_id: A label used in the report.

        Returns:
            :class:`ConvergenceReport` with entropy, agreement rate and
            convergence verdict.
        """
        counts = result.answer_counts
        total = result.num_samples

        agreement_rate = result.majority_fraction
        entropy = _shannon_entropy(counts)

        # Normalised distribution (probability per answer)
        distribution: Dict[str, float] = (
            {ans: cnt / total for ans, cnt in counts.items()} if total > 0 else {}
        )

        is_converged = agreement_rate >= self.convergence_threshold
        divergence_note = ""
        if not is_converged:
            top_two = sorted(counts.items(), key=lambda x: -x[1])[:2]
            if len(top_two) >= 2:
                divergence_note = (
                    f"Top answers: '{top_two[0][0]}' ({top_two[0][1]}) "
                    f"vs '{top_two[1][0]}' ({top_two[1][1]})"
                )

        return ConvergenceReport(
            question_id=question_id,
            final_answer=result.final_answer,
            num_samples=total,
            agreement_rate=agreement_rate,
            entropy=entropy,
            answer_distribution=distribution,
            is_converged=is_converged,
            divergence_note=divergence_note,
        )

    def analyse_batch(
        self,
        results: List[ConsistencyResult],
        question_ids: Optional[List[str]] = None,
    ) -> List[ConvergenceReport]:
        """
        Analyse a list of sampling results.

        Args:
            results: List of :class:`~llm_inference.ConsistencyResult`.
            question_ids: Optional labels; defaults to ``q0``, ``q1``, …

        Returns:
            List of :class:`ConvergenceReport` in the same order.
        """
        ids = question_ids or [f"q{i}" for i in range(len(results))]
        return [
            self.analyse(result, qid) for result, qid in zip(results, ids)
        ]

    def batch_summary(
        self,
        reports: List[ConvergenceReport],
    ) -> Dict[str, object]:
        """
        Produce aggregate statistics across a batch of reports.

        Args:
            reports: List of :class:`ConvergenceReport`.

        Returns:
            Dict with keys: ``total``, ``converged``, ``diverged``,
            ``convergence_rate``, ``mean_entropy``, ``mean_agreement``.
        """
        if not reports:
            return {
                "total": 0,
                "converged": 0,
                "diverged": 0,
                "convergence_rate": 0.0,
                "mean_entropy": 0.0,
                "mean_agreement": 0.0,
            }
        total = len(reports)
        converged = sum(1 for r in reports if r.is_converged)
        return {
            "total": total,
            "converged": converged,
            "diverged": total - converged,
            "convergence_rate": converged / total,
            "mean_entropy": sum(r.entropy for r in reports) / total,
            "mean_agreement": sum(r.agreement_rate for r in reports) / total,
        }
