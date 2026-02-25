"""RAGAS evaluation backend — runs via subprocess using ragas NumericMetric."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from noid_rag.models import EvalResult

if TYPE_CHECKING:
    from noid_rag.config import EvalConfig, LLMConfig

_METRIC_PROMPTS = {
    "faithfulness": (
        "Rate 0.0–1.0 how well the response is supported by the context. "
        "Response: {response} Context: {context}"
    ),
    "answer_relevancy": (
        "Rate 0.0–1.0 how relevant the response is to the question. "
        "Question: {question} Response: {response}"
    ),
    "context_precision": (
        "Rate 0.0–1.0 how relevant the retrieved context is to the question. "
        "Question: {question} Context: {context}"
    ),
    "context_recall": (
        "Rate 0.0–1.0 how well the context covers the reference answer. "
        "Reference: {reference} Context: {context}"
    ),
}

# Environment variable used to pass the API key to the eval subprocess.
# Never written to disk — passed only through the process environment.
_ENV_API_KEY = "NOID_RAG_EVAL_API_KEY"


def _build_eval_script(
    data_path: str,
    output_path: str,
    metrics: list[str],
    api_url: str,
    model: str,
    judge_max_tokens: int = 16,
    judge_temperature: float = 0.0,
) -> str:
    """Generate a Python script that runs ragas evaluation via NumericMetric.

    The API key is NOT embedded in the script; the subprocess inherits it from
    the ``NOID_RAG_EVAL_API_KEY`` environment variable set by the caller.
    """
    # Filter to known metrics
    known = [m for m in metrics if m in _METRIC_PROMPTS]

    metric_defs = []
    for m in known:
        prompt = _METRIC_PROMPTS[m]
        metric_defs.append(
            f'    NumericMetric(name="{m}", allowed_values=(0.0, 1.0),\n'
            f'                  prompt="{prompt}"),'
        )
    metrics_block = "\n".join(metric_defs)

    lines = [
        "import asyncio",
        "import json",
        "import os",
        "import sys",
        "",
        "try:",
        "    from ragas.metrics import NumericMetric",
        "except ImportError:",
        '    print("ragas is not installed", file=sys.stderr)',
        "    sys.exit(1)",
        "",
        "try:",
        "    from openai import AsyncOpenAI",
        "except ImportError:",
        '    print("openai is not installed", file=sys.stderr)',
        "    sys.exit(1)",
        "",
        f"DATA_PATH = {data_path!r}",
        f"OUTPUT_PATH = {output_path!r}",
        f"API_URL = {api_url!r}",
        f"MODEL = {model!r}",
        # API key is read from the environment at subprocess runtime — never stored on disk.
        f'API_KEY = os.environ.get("{_ENV_API_KEY}", "")',
        "",
        "metrics = [",
        metrics_block,
        "]",
        "",
        "SYSTEM_PROMPT = (",
        '    "You are an evaluation judge. '
        'Respond with ONLY a decimal number between 0.0 and 1.0."',
        ")",
        "",
        "",
        "async def main():",
        "    with open(DATA_PATH) as f:",
        "        data = json.load(f)",
        "",
        "    client = AsyncOpenAI(api_key=API_KEY, base_url=API_URL)",
        "    results = []",
        "",
        "    for row in data:",
        "        row_scores = {}",
        "        for metric in metrics:",
        "            prompt = metric.prompt.format(",
        '                question=row.get("question", ""),',
        '                response=row.get("answer", ""),',
        '                context=" ".join(row.get("contexts", [])),',
        '                reference=row.get("ground_truth") or "",',
        "            )",
        "            try:",
        "                resp = await client.chat.completions.create(",
        "                    model=MODEL,",
        "                    messages=[",
        '                        {"role": "system", "content": SYSTEM_PROMPT},',
        '                        {"role": "user", "content": prompt},',
        "                    ],",
        f"                    max_tokens={judge_max_tokens!r},",
        f"                    temperature={judge_temperature!r},",
        "                )",
        "                text = resp.choices[0].message.content.strip()",
        "                score = float(text)",
        "                score = max(0.0, min(1.0, score))",
        "            except Exception:",
        "                score = 0.0",
        "            row_scores[metric.name] = score",
        "        results.append(row_scores)",
        "",
        '    with open(OUTPUT_PATH, "w") as f:',
        "        json.dump(results, f, indent=2)",
        "",
        "",
        "asyncio.run(main())",
    ]
    return "\n".join(lines) + "\n"


async def run_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str | None],
    eval_config: EvalConfig,
    llm_config: LLMConfig,
) -> list[EvalResult]:
    """Run RAGAS evaluation via CLI subprocess.

    Requires the ``ragas`` optional dependency (``uv sync --extra eval``).
    """
    api_key = llm_config.api_key.get_secret_value()
    if not api_key:
        raise RuntimeError(
            "LLM API key is not configured. Set llm.api_key in your config or "
            "NOID_RAG_LLM__API_KEY environment variable."
        )

    unknown = [m for m in eval_config.metrics if m not in _METRIC_PROMPTS]
    if unknown:
        import warnings

        warnings.warn(
            f"Unrecognized RAGAS metric(s) ignored: {unknown}. "
            f"Available: {sorted(_METRIC_PROMPTS)}",
            stacklevel=2,
        )

    valid_metrics = [m for m in eval_config.metrics if m in _METRIC_PROMPTS]
    if not valid_metrics:
        raise ValueError(
            f"No valid RAGAS metrics found in {eval_config.metrics}. "
            f"Available: {list(_METRIC_PROMPTS)}"
        )

    api_url = llm_config.api_url
    # Strip /chat/completions — the OpenAI client adds it
    if api_url.endswith("/chat/completions"):
        api_url = api_url.rsplit("/chat/completions", 1)[0]

    with tempfile.TemporaryDirectory(prefix="noid-rag-ragas-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        data_file = tmp_path / "eval_data.json"
        output_file = tmp_path / "results.json"
        script_file = tmp_path / "run_eval.py"

        # Write input data
        data = []
        for q, a, c, gt in zip(questions, answers, contexts, ground_truths):
            data.append(
                {
                    "question": q,
                    "answer": a,
                    "contexts": c,
                    "ground_truth": gt or "",
                }
            )
        data_file.write_text(json.dumps(data, indent=2))

        # Write eval script — API key is NOT embedded; passed via env below.
        script = _build_eval_script(
            data_path=str(data_file),
            output_path=str(output_file),
            metrics=valid_metrics,
            api_url=api_url,
            model=llm_config.model,
            judge_max_tokens=eval_config.judge_max_tokens,
            judge_temperature=eval_config.judge_temperature,
        )
        script_file.write_text(script)

        # Pass the API key through the subprocess environment only — never on disk.
        subprocess_env = {**os.environ, _ENV_API_KEY: api_key}

        # Run via uv
        proc = await asyncio.create_subprocess_exec(
            "uv",
            "run",
            "python",
            str(script_file),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=subprocess_env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise RuntimeError("ragas eval timed out after 300 seconds")

        if proc.returncode != 0:
            err = stderr.decode(errors="replace")[:2000]
            out = stdout.decode(errors="replace")[:2000]
            raise RuntimeError(
                f"ragas eval failed (exit {proc.returncode}):\nstdout: {out}\nstderr: {err}"
            )

        return _parse_ragas_results(
            output_file,
            questions,
            answers,
            contexts,
            ground_truths,
        )


def _parse_ragas_results(
    output_path: Path,
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str | None],
) -> list[EvalResult]:
    """Parse JSON output from the ragas eval subprocess."""
    with open(output_path) as f:
        data = json.load(f)

    results = []
    for i, row_scores in enumerate(data):
        if i >= len(questions):
            break
        results.append(
            EvalResult(
                question=questions[i],
                answer=answers[i],
                contexts=contexts[i],
                ground_truth=ground_truths[i],
                scores={k: float(v) for k, v in row_scores.items()},
            )
        )
    return results
