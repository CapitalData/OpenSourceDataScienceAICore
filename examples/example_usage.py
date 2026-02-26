"""
Example: LLM Inference System
==============================
This script demonstrates how to use the llm_inference package to:
  1. Stream completions asynchronously
  2. Apply self-consistency sampling for improved answer quality
  3. Analyse convergence of reasoning paths
  4. Collect and compare cost/latency/accuracy metrics

Set the OPENAI_API_KEY environment variable before running:
    export OPENAI_API_KEY="sk-..."

Optional Phoenix tracing (set PHOENIX_COLLECTOR_ENDPOINT to enable):
    pip install arize-phoenix-otel openinference-instrumentation-openai
    export PHOENIX_COLLECTOR_ENDPOINT="http://localhost:4317"
"""

from __future__ import annotations

import asyncio
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Phoenix / OpenTelemetry instrumentation
# ---------------------------------------------------------------------------
_PHOENIX_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
if _PHOENIX_ENDPOINT:
    try:
        from phoenix.otel import register as _phoenix_register
        from openinference.instrumentation.openai import OpenAIInstrumentor

        _tracer_provider = _phoenix_register(
            project_name="llm-inference-demo",
            endpoint=_PHOENIX_ENDPOINT,
        )
        OpenAIInstrumentor().instrument(tracer_provider=_tracer_provider)
        logger.info("Phoenix tracing enabled → %s", _PHOENIX_ENDPOINT)
    except ImportError:
        logger.warning(
            "Phoenix packages not installed. "
            "Run: pip install arize-phoenix-otel openinference-instrumentation-openai"
        )
else:
    logger.info(
        "Phoenix tracing disabled. Set PHOENIX_COLLECTOR_ENDPOINT to enable."
    )

# ---------------------------------------------------------------------------
# Import the llm_inference package
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_inference import (
    AsyncStreamingClient,
    StreamingConfig,
    SelfConsistencySampler,
    SamplingConfig,
    ConvergenceAnalyzer,
    MetricsCollector,
)

# ---------------------------------------------------------------------------
# Demo questions (few-shot math / reasoning)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Think step-by-step and end your answer with 'Answer: <value>'."
)

QUESTIONS = [
    "If a train travels 120 km in 2 hours, what is its average speed in km/h?",
    "What is 15% of 200?",
    "A rectangle has length 8 cm and width 5 cm. What is its area?",
]


def _build_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


# ---------------------------------------------------------------------------
# 1. Streaming demo
# ---------------------------------------------------------------------------
async def demo_streaming(client: AsyncStreamingClient) -> None:
    logger.info("=== Demo 1: Async Streaming ===")
    messages = _build_messages(QUESTIONS[0])
    print(f"\nQuestion: {QUESTIONS[0]}\nStreaming response: ", end="", flush=True)
    async for chunk in client.stream_completion(messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()  # newline after stream


# ---------------------------------------------------------------------------
# 2. Self-consistency sampling demo
# ---------------------------------------------------------------------------
async def demo_self_consistency(
    sampler: SelfConsistencySampler,
    collector: MetricsCollector,
) -> None:
    logger.info("=== Demo 2: Self-Consistency Sampling ===")
    for i, question in enumerate(QUESTIONS):
        messages = _build_messages(question)
        with collector.record(f"sc-q{i}") as m:
            result = await sampler.sample(messages)

        print(f"\nQ: {question}")
        print(f"  Final answer : {result.final_answer}")
        print(f"  Confidence   : {result.confidence:.0%}")
        print(f"  Votes        : {result.answer_counts}")


# ---------------------------------------------------------------------------
# 3. Convergence analysis demo
# ---------------------------------------------------------------------------
async def demo_convergence(sampler: SelfConsistencySampler) -> None:
    logger.info("=== Demo 3: Convergence Analysis ===")
    analyzer = ConvergenceAnalyzer(convergence_threshold=0.6)
    batch = [_build_messages(q) for q in QUESTIONS]
    results = await sampler.batch_sample(batch)
    reports = analyzer.analyse_batch(results, question_ids=[f"q{i}" for i in range(len(QUESTIONS))])

    for report in reports:
        print(report.summary())
        if report.divergence_note:
            print(f"  ↳ Divergence: {report.divergence_note}")

    summary = analyzer.batch_summary(reports)
    print(f"\nBatch convergence summary: {json.dumps(summary, indent=2)}")


# ---------------------------------------------------------------------------
# 4. Cost / latency / accuracy tradeoff demo
# ---------------------------------------------------------------------------
async def demo_metrics(client: AsyncStreamingClient) -> None:
    logger.info("=== Demo 4: Cost, Latency, and Accuracy Metrics ===")
    baseline = MetricsCollector(model="gpt-4o-mini")
    variant = MetricsCollector(model="gpt-4o-mini")

    # Simulate two strategies: single-sample vs. batch
    messages = _build_messages(QUESTIONS[0])

    # Baseline: single request
    with baseline.record("single") as m:
        result = await client.complete(messages, temperature=0.0)
        m.prompt_tokens = result.prompt_tokens
        m.completion_tokens = result.completion_tokens
        # (In a real scenario you'd compare against ground truth here)
        m.is_correct = True

    # Variant: batch of 3 requests (simulates higher quality but more cost)
    batch = [messages] * 3
    results = await client.batch_complete(batch, temperature=0.7)
    for j, r in enumerate(results):
        with variant.record(f"batch-{j}") as m:
            m.prompt_tokens = r.prompt_tokens
            m.completion_tokens = r.completion_tokens
            m.is_correct = True

    comparison = baseline.compare_strategies(variant, "single-sample", "batch-of-3")
    print(json.dumps(comparison, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning(
            "OPENAI_API_KEY not set. Skipping demos that require a live API."
        )
        print("Set OPENAI_API_KEY to run the live demos.")
        return

    streaming_cfg = StreamingConfig(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=512,
        max_retries=3,
    )
    sampling_cfg = SamplingConfig(
        num_samples=5,
        temperatures=[0.6, 0.7, 0.8, 0.9, 1.0],
        answer_pattern=r"Answer:\s*(.+)",
    )

    client = AsyncStreamingClient(api_key=api_key, config=streaming_cfg)
    sampler = SelfConsistencySampler(
        client=client,
        streaming_config=streaming_cfg,
        sampling_config=sampling_cfg,
    )
    collector = MetricsCollector(model=streaming_cfg.model)

    await demo_streaming(client)
    await demo_self_consistency(sampler, collector)
    await demo_convergence(sampler)
    await demo_metrics(client)

    agg = collector.aggregate()
    print("\n=== Overall Metrics ===")
    print(json.dumps(agg.to_dict(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
