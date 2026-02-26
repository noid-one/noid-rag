"""Synthetic eval dataset generation from indexed documents."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

import httpx
import yaml
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from noid_rag.config import LLMConfig, Settings, VectorStoreConfig

logger = logging.getLogger(__name__)

_GENERATION_PROMPT = """\
You are an expert question-answer pair generator for evaluating \
retrieval-augmented generation (RAG) systems.

Given the following text chunk from a document, generate exactly \
{num_questions} question/answer pairs.

Requirements:
- Questions should be diverse: include factual, inferential, and comparison-style questions
- Each answer (ground_truth) MUST be fully derivable from the provided text alone
- Answers should be concise but complete
- Do NOT reference the text itself (e.g., avoid "according to the text...")

Text chunk:
---
{chunk_text}
---

Respond with ONLY a JSON array, no other text:
[{{"question": "...", "ground_truth": "..."}}]
"""


async def fetch_chunks(
    store_config: VectorStoreConfig,
    num_chunks: int,
    strategy: Literal["random", "diverse"] = "diverse",
) -> list[dict[str, Any]]:
    """Pull chunks from pgvector using the specified strategy."""
    from noid_rag.vectorstore import PgVectorStore

    async with PgVectorStore(config=store_config) as store:
        return await store.sample_chunks(limit=num_chunks, strategy=strategy)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=2, max=60),
    retry=retry_if_exception_type(
        (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)
    ),
)
async def _call_llm(
    llm_config: LLMConfig,
    model: str,
    chunk_text: str,
    num_questions: int,
    max_tokens: int = 2048,
    http_client: httpx.AsyncClient | None = None,
) -> list[dict[str, str]]:
    """Call the LLM to generate Q&A pairs from a single chunk."""
    prompt = _GENERATION_PROMPT.format(
        num_questions=num_questions,
        chunk_text=chunk_text,
    )

    client = http_client or httpx.AsyncClient(timeout=120.0)
    owns_client = http_client is None
    try:
        resp = await client.post(
            llm_config.api_url,
            json={
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
            headers={
                "Authorization": f"Bearer {llm_config.api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json()
    finally:
        if owns_client:
            await client.aclose()

    content = data["choices"][0]["message"]["content"]

    # Strip markdown code fences if present
    content = content.strip()
    if content.startswith("```"):
        # Remove opening fence (with optional language tag) and closing fence
        lines = content.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)

    pairs = json.loads(content)
    if not isinstance(pairs, list):
        raise ValueError(f"Expected JSON array, got {type(pairs).__name__}")
    return [
        {"question": p["question"], "ground_truth": p["ground_truth"]}
        for p in pairs
        if "question" in p and "ground_truth" in p
    ]


async def generate_qa_pairs(
    chunks: list[dict[str, Any]],
    llm_config: LLMConfig,
    model: str,
    questions_per_chunk: int = 3,
    num_questions: int | None = None,
    max_tokens: int = 2048,
    progress_callback: Any | None = None,
) -> list[dict[str, str]]:
    """Generate Q&A pairs from chunks using an LLM.

    Args:
        chunks: List of chunk dicts with 'text' field.
        llm_config: LLM configuration (api_url, api_key).
        model: Model name to use for generation.
        questions_per_chunk: How many Q&A pairs to request per chunk.
        num_questions: Total cap on questions. None = no cap.
        progress_callback: Called with (chunk_index,) after each chunk.

    Returns:
        List of {"question": ..., "ground_truth": ...} dicts.
    """
    all_pairs: list[dict[str, str]] = []
    failure_count = 0

    async with httpx.AsyncClient(timeout=120.0) as http_client:
        for i, chunk in enumerate(chunks):
            if num_questions and len(all_pairs) >= num_questions:
                break

            try:
                pairs = await _call_llm(
                    llm_config,
                    model,
                    chunk["text"],
                    questions_per_chunk,
                    max_tokens=max_tokens,
                    http_client=http_client,
                )
                all_pairs.extend(pairs)
            except Exception as exc:
                failure_count += 1
                logger.warning("Chunk %d/%d failed: %s", i + 1, len(chunks), exc)

            if progress_callback:
                progress_callback(i)

    if failure_count > 0:
        logger.warning("%d/%d chunks failed during generation", failure_count, len(chunks))

    # Trim to exact count if specified
    if num_questions and len(all_pairs) > num_questions:
        all_pairs = all_pairs[:num_questions]

    return all_pairs


def save_dataset(
    questions: list[dict[str, str]],
    output_path: Path,
) -> None:
    """Write Q&A pairs in eval dataset format (YAML or JSON)."""
    dataset = {"questions": questions}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    # Write to temp file then atomically replace
    fd, tmp_path = tempfile.mkstemp(dir=output_path.parent, suffix=suffix)
    try:
        with os.fdopen(fd, "w") as f:
            if suffix in (".yml", ".yaml"):
                yaml.dump(dataset, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            else:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, output_path)
    except BaseException:
        # Clean up temp file on any error
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


async def run_generate(
    settings: Settings,
    output: Path,
    num_questions: int | None = None,
    model: str | None = None,
    num_chunks: int | None = None,
    strategy: str | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """End-to-end generation: fetch chunks, call LLM, save dataset.

    Returns summary dict with counts and output path.
    """
    gen_config = settings.generate
    final_num_questions = num_questions or gen_config.num_questions
    final_model = model or gen_config.model
    final_strategy = strategy or gen_config.strategy
    qpc = gen_config.questions_per_chunk

    # Auto-calculate chunk count if not specified
    if num_chunks is None:
        num_chunks = max(1, -(-final_num_questions // qpc))  # ceil division

    chunks = await fetch_chunks(settings.vectorstore, num_chunks, final_strategy)  # type: ignore[arg-type]

    if not chunks:
        raise RuntimeError("No chunks found in the vector store. Ingest documents first.")

    pairs = await generate_qa_pairs(
        chunks,
        llm_config=settings.llm,
        model=final_model,
        questions_per_chunk=qpc,
        num_questions=final_num_questions,
        max_tokens=gen_config.max_tokens,
        progress_callback=progress_callback,
    )

    if not pairs:
        raise RuntimeError("No Q&A pairs were generated. Check your LLM configuration.")

    save_dataset(pairs, output)

    return {
        "questions_generated": len(pairs),
        "chunks_sampled": len(chunks),
        "model": final_model,
        "strategy": final_strategy,
        "output_path": str(output),
    }
