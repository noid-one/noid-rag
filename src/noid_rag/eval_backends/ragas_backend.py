"""RAGAS evaluation backend — runs in-process via the ragas library."""

from __future__ import annotations

import asyncio
import math
import os
import threading
from typing import TYPE_CHECKING, Any

from noid_rag.models import EvalResult

if TYPE_CHECKING:
    from noid_rag.config import EvalConfig, LLMConfig

_ragas_lock = threading.Lock()


async def run_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str | None],
    eval_config: EvalConfig,
    llm_config: LLMConfig,
) -> list[EvalResult]:
    """Run RAGAS evaluation in-process.

    Requires the ``ragas`` optional dependency (``uv sync --extra eval``).
    """
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError:
        raise RuntimeError(
            "ragas is not installed. Install it with: uv sync --extra eval"
        )

    api_key = llm_config.api_key.get_secret_value()
    if not api_key:
        raise RuntimeError(
            "LLM API key is not configured. Set llm.api_key in your config or "
            "NOID_RAG_LLM__API_KEY environment variable."
        )

    metric_map = {
        "faithfulness": Faithfulness,
        "answer_relevancy": AnswerRelevancy,
        "context_precision": ContextPrecision,
        "context_recall": ContextRecall,
    }

    unknown = [m for m in eval_config.metrics if m not in metric_map]
    if unknown:
        import warnings

        warnings.warn(
            f"Unrecognized RAGAS metric(s) ignored: {unknown}. "
            f"Available: {sorted(metric_map)}",
            stacklevel=2,
        )

    selected = [metric_map[m]() for m in eval_config.metrics if m in metric_map]
    if not selected:
        raise ValueError(
            f"No valid RAGAS metrics found in {eval_config.metrics}. "
            f"Available: {list(metric_map)}"
        )

    samples = [
        SingleTurnSample(
            user_input=q,
            response=a,
            retrieved_contexts=c,
            reference=gt or "",
        )
        for q, a, c, gt in zip(questions, answers, contexts, ground_truths)
    ]

    dataset = EvaluationDataset(samples=samples)

    api_url = llm_config.api_url
    # RAGAS expects base URL without /chat/completions
    if api_url.endswith("/chat/completions"):
        api_url = api_url.rsplit("/chat/completions", 1)[0]

    def _run_with_env():
        """Run evaluate() in a thread with env vars protected by a lock."""
        with _ragas_lock:
            old_base = os.environ.get("OPENAI_BASE_URL")
            old_key = os.environ.get("OPENAI_API_KEY")
            try:
                os.environ["OPENAI_BASE_URL"] = api_url
                os.environ["OPENAI_API_KEY"] = api_key
                return evaluate(dataset=dataset, metrics=selected)
            finally:
                if old_base is not None:
                    os.environ["OPENAI_BASE_URL"] = old_base
                else:
                    os.environ.pop("OPENAI_BASE_URL", None)
                if old_key is not None:
                    os.environ["OPENAI_API_KEY"] = old_key
                else:
                    os.environ.pop("OPENAI_API_KEY", None)

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _run_with_env)

    return _parse_ragas_results(result, questions, answers, contexts, ground_truths)


def _parse_ragas_results(
    result: Any,
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str | None],
) -> list[EvalResult]:
    """Convert RAGAS result DataFrame to list of EvalResult."""
    df = result.to_pandas()
    results = []
    for pos, (_, row) in enumerate(df.iterrows()):
        if pos >= len(questions):
            break
        scores = {}
        for col in df.columns:
            if col not in ("user_input", "response", "retrieved_contexts", "reference"):
                val = row[col]
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    scores[col] = float(val)
        results.append(
            EvalResult(
                question=questions[pos],
                answer=answers[pos],
                contexts=contexts[pos],
                ground_truth=ground_truths[pos],
                scores=scores,
            )
        )
    return results
