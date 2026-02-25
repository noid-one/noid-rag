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
        "Rate 0.0–1.0 how much of the retrieved context contains information "
        "needed to produce the reference answer. "
        "Question: {question} Reference answer: {reference} Context: {context}"
    ),
    "context_recall": (
        "Rate 0.0–1.0 how well the context covers the reference answer. "
        "Reference: {reference} Context: {context}"
    ),
}

# Fallback prompt when ground_truth is empty — question-only relevance check.
_CONTEXT_PRECISION_FALLBACK = (
    "Rate 0.0–1.0 how relevant the retrieved context is to the question. "
    "Question: {question} Context: {context}"
)

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

    ``context_precision`` uses per-chunk evaluation: each chunk is scored
    individually against the reference answer and the final score is the mean.
    When ground_truth is empty, a question-only fallback prompt is used.
    Other metrics evaluate against the joined context as before.
    """
    # Filter to known metrics
    known = [m for m in metrics if m in _METRIC_PROMPTS]

    # Build metric definitions — context_precision uses a separate code path
    other_metrics = [m for m in known if m != "context_precision"]
    has_context_precision = "context_precision" in known

    metric_defs = []
    for m in other_metrics:
        prompt = _METRIC_PROMPTS[m]
        metric_defs.append(
            f'    NumericMetric(name="{m}", allowed_values=(0.0, 1.0),\n'
            f'                  prompt="{prompt}"),'
        )
    metrics_block = "\n".join(metric_defs)

    # Context precision prompts (with-reference and fallback)
    cp_prompt = _METRIC_PROMPTS["context_precision"]
    cp_fallback = _CONTEXT_PRECISION_FALLBACK

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
        f"HAS_CONTEXT_PRECISION = {has_context_precision!r}",
        f"CP_PROMPT = {cp_prompt!r}",
        f"CP_FALLBACK = {cp_fallback!r}",
        f"JUDGE_MAX_TOKENS = {judge_max_tokens!r}",
        f"JUDGE_TEMPERATURE = {judge_temperature!r}",
        "",
        "SYSTEM_PROMPT = (",
        '    "You are an evaluation judge. '
        'Respond with ONLY a decimal number between 0.0 and 1.0."',
        ")",
        "",
        "",
        "async def score_prompt(client, prompt_text, max_tokens, temperature):",
        '    """Call the LLM judge and return a 0.0–1.0 score."""',
        "    resp = await client.chat.completions.create(",
        "        model=MODEL,",
        "        messages=[",
        '            {"role": "system", "content": SYSTEM_PROMPT},',
        '            {"role": "user", "content": prompt_text},',
        "        ],",
        "        max_tokens=max_tokens,",
        "        temperature=temperature,",
        "    )",
        "    text = resp.choices[0].message.content.strip()",
        "    score = float(text)",
        "    return max(0.0, min(1.0, score))",
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
        "        question = row.get('question', '')",
        "        response = row.get('answer', '')",
        "        contexts = row.get('contexts', [])",
        "        reference = row.get('ground_truth') or ''",
        "        joined_context = ' '.join(contexts)",
        "",
        "        # Score non-context_precision metrics (joined context)",
        "        for metric in metrics:",
        "            prompt = metric.prompt.format(",
        "                question=question,",
        "                response=response,",
        "                context=joined_context,",
        "                reference=reference,",
        "            )",
        "            try:",
        "                score = await score_prompt(",
        "                    client, prompt, JUDGE_MAX_TOKENS, JUDGE_TEMPERATURE",
        "                )",
        "            except Exception as exc:",
        "                print(f'Eval error for {metric.name}: {exc}', file=sys.stderr)",
        "                score = 0.0",
        "            row_scores[metric.name] = score",
        "",
        "        # Per-chunk context_precision scoring",
        "        if HAS_CONTEXT_PRECISION:",
        "            if not contexts:",
        "                row_scores['context_precision'] = 0.0",
        "            else:",
        "                chunk_scores = []",
        "                for chunk in contexts:",
        "                    if reference:",
        "                        prompt = CP_PROMPT.format(",
        "                            question=question,",
        "                            reference=reference,",
        "                            context=chunk,",
        "                        )",
        "                    else:",
        "                        prompt = CP_FALLBACK.format(",
        "                            question=question,",
        "                            context=chunk,",
        "                        )",
        "                    try:",
        "                        s = await score_prompt(",
        "                            client, prompt, JUDGE_MAX_TOKENS, JUDGE_TEMPERATURE",
        "                        )",
        "                    except Exception as exc:",
        "                        print(",
        "                            f'Eval error for context_precision chunk: {exc}',",
        "                            file=sys.stderr,",
        "                        )",
        "                        s = 0.0",
        "                    chunk_scores.append(s)",
        "                row_scores['context_precision'] = sum(chunk_scores) / len(chunk_scores)",
        "",
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
