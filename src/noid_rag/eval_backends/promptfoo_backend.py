"""Promptfoo evaluation backend — runs via npx subprocess."""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from noid_rag.models import EvalResult

if TYPE_CHECKING:
    from noid_rag.config import EvalConfig

# Map noid-rag metric names → Promptfoo assertion types
_METRIC_MAP = {
    "faithfulness": "context-faithfulness",
    "answer_relevancy": "answer-relevance",
    "context_precision": "context-relevance",
    "context_recall": "context-recall",
}


def _build_assertions(metrics: list[str], threshold: float) -> list[dict]:
    """Build Promptfoo assertion list from metric names."""
    assertions = []
    for m in metrics:
        pf_type = _METRIC_MAP.get(m)
        if pf_type:
            assertions.append({"type": pf_type, "threshold": threshold})
    return assertions


async def run_promptfoo(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str | None],
    eval_config: EvalConfig,
) -> list[EvalResult]:
    """Run Promptfoo evaluation via CLI subprocess.

    Requires Node.js and npx available on PATH.
    """
    if not shutil.which("npx"):
        raise RuntimeError(
            "npx is not available. Install Node.js to use the promptfoo backend."
        )

    with tempfile.TemporaryDirectory(prefix="noid-rag-eval-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        config_path = tmp_path / "promptfoo_config.yaml"
        output_path = tmp_path / "results.json"

        tests = []
        for q, a, c, gt in zip(questions, answers, contexts, ground_truths):
            tests.append({
                "vars": {
                    "query": q,
                    "context": "\n---\n".join(c),
                    "answer": a,
                    "ground_truth": gt or "",
                },
                "assert": _build_assertions(
                    eval_config.metrics, eval_config.promptfoo_threshold
                ),
            })

        config = {
            "prompts": ["{{answer}}"],
            "providers": [{"id": "echo", "config": {"output": "{{answer}}"}}],
            "tests": tests,
        }

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        proc = await asyncio.create_subprocess_exec(
            "npx", "promptfoo", "eval",
            "-c", str(config_path),
            "-o", str(output_path),
            "--no-progress-bar",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise RuntimeError("promptfoo eval timed out after 300 seconds")

        if proc.returncode != 0:
            out = stdout.decode(errors="replace")[:2000]
            err = stderr.decode(errors="replace")[:2000]
            raise RuntimeError(
                f"promptfoo eval failed (exit {proc.returncode}):\n"
                f"stdout: {out}\nstderr: {err}"
            )

        return _parse_promptfoo_results(
            output_path, questions, answers, contexts, ground_truths,
            eval_config.metrics, eval_config.promptfoo_threshold,
        )


def _parse_promptfoo_results(
    output_path: Path,
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str | None],
    metrics: list[str],
    threshold: float,
) -> list[EvalResult]:
    """Parse Promptfoo JSON output into EvalResult list."""
    with open(output_path) as f:
        data = json.load(f)

    results_data = data.get("results", [])
    eval_results = []

    for i, row in enumerate(results_data):
        if i >= len(questions):
            break

        scores: dict[str, float] = {}
        passed: dict[str, bool] = {}

        for assertion in row.get("gradingResult", {}).get("componentResults", []):
            assertion_type = assertion.get("assertion", {}).get("type", "")
            score = assertion.get("score", 0.0)
            is_pass = assertion.get("pass", False)

            # Reverse-map promptfoo type to our metric name
            for metric_name, pf_type in _METRIC_MAP.items():
                if pf_type == assertion_type and metric_name in metrics:
                    scores[metric_name] = float(score)
                    passed[metric_name] = bool(is_pass)
                    break

        eval_results.append(
            EvalResult(
                question=questions[i],
                answer=answers[i],
                contexts=contexts[i],
                ground_truth=ground_truths[i],
                scores=scores,
                passed=passed,
            )
        )

    return eval_results
