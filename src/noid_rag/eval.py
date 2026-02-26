"""Evaluation orchestration — dataset loading and backend dispatch."""

from __future__ import annotations

import json
import logging
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from noid_rag.models import EvalQuestion, EvalResult, EvalSummary

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from noid_rag.api import NoidRag
    from noid_rag.config import EvalConfig, Settings


def load_dataset(path: Path) -> list[EvalQuestion]:
    """Load evaluation dataset from YAML or JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = path.suffix.lower()
    with open(path) as f:
        if suffix in (".yml", ".yaml"):
            data = yaml.safe_load(f)
        elif suffix == ".json":
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported dataset format: {suffix}. Use .yml, .yaml, or .json")

    if not isinstance(data, dict) or "questions" not in data:
        raise ValueError("Dataset must have a top-level 'questions' key.")

    questions = []
    for item in data["questions"]:
        if isinstance(item, str):
            questions.append(EvalQuestion(question=item))
        elif isinstance(item, dict):
            if "question" not in item:
                raise ValueError(f"Dataset entry is missing required 'question' key: {item!r}")
            questions.append(
                EvalQuestion(
                    question=item["question"],
                    ground_truth=item.get("ground_truth"),
                )
            )
        else:
            raise ValueError(f"Invalid question format: {item}")

    if not questions:
        raise ValueError("Dataset contains no questions.")

    return questions


async def run_evaluation(
    dataset_path: str | Path,
    eval_config: EvalConfig,
    settings: Settings,
    rag: NoidRag,
    top_k: int | None = None,
) -> EvalSummary:
    """Run full evaluation: RAG pipeline + scoring backend."""
    top_k = settings.search.top_k if top_k is None else top_k
    valid_backends = ("ragas", "promptfoo")
    if eval_config.backend not in valid_backends:
        raise ValueError(
            f"Unknown eval backend: {eval_config.backend!r}. "
            f"Choose from: {', '.join(valid_backends)}"
        )

    dataset_path = Path(dataset_path)
    questions = load_dataset(dataset_path)

    q_texts: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str | None] = []

    for q in questions:
        result = await rag.aanswer(q.question, top_k=top_k)
        q_texts.append(q.question)
        answers.append(result.answer)
        contexts.append([s.text for s in result.sources])
        ground_truths.append(q.ground_truth)

    if eval_config.backend == "ragas":
        from noid_rag.eval_backends.ragas_backend import run_ragas

        results = await run_ragas(
            q_texts,
            answers,
            contexts,
            ground_truths,
            eval_config,
            settings.llm,
        )
    else:  # "promptfoo" — the only other Literal value
        from noid_rag.eval_backends.promptfoo_backend import run_promptfoo

        results = await run_promptfoo(
            q_texts,
            answers,
            contexts,
            ground_truths,
            eval_config,
        )

    mean_scores = _compute_mean_scores(results)

    summary = EvalSummary(
        results=results,
        mean_scores=mean_scores,
        backend=eval_config.backend,
        model=settings.llm.model,
        total_questions=len(questions),
        dataset_path=str(dataset_path),
    )

    if eval_config.save_results:
        save_eval_results(summary, eval_config.results_dir)

    return summary


def _compute_mean_scores(results: list[EvalResult]) -> dict[str, float]:
    """Compute mean scores across all results."""
    if not results:
        return {}
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for r in results:
        for metric, score in r.scores.items():
            totals[metric] = totals.get(metric, 0.0) + score
            counts[metric] = counts.get(metric, 0) + 1
    return {m: totals[m] / counts[m] for m in totals}


def save_eval_results(summary: EvalSummary, results_dir: str) -> Path | None:
    """Save evaluation results as timestamped JSON. Returns None on filesystem errors."""
    from dataclasses import asdict

    try:
        dir_path = Path(results_dir).expanduser()
        dir_path.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        suffix = secrets.token_hex(4)
        out_path = dir_path / f"eval_{ts}_{suffix}.json"

        with open(out_path, "w") as f:
            json.dump(asdict(summary), f, indent=2)

        return out_path
    except OSError as exc:
        logger.warning("Could not save eval results to %s: %s", results_dir, exc)
        return None
