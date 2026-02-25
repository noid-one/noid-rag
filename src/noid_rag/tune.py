"""Bayesian hyperparameter optimization for RAG pipelines via Optuna."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Callable
from typing import Any

from noid_rag.config import Settings
from noid_rag.models import TuneResult

logger = logging.getLogger(__name__)

# Safe SQL identifier: letters/digits/underscores, must start with letter or underscore.
# Identical to VectorStoreConfig._validate_table_name's rule in config.py.
_SAFE_TABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")

# Known embedding model → dimension mappings
EMBEDDING_DIM_MAP: dict[str, int] = {
    "openai/text-embedding-3-small": 1536,
    "openai/text-embedding-3-large": 3072,
    "openai/text-embedding-ada-002": 1536,
}

# pgvector HNSW index limit
_HNSW_MAX_DIM = 2000


def _ingest_config_hash(chunker_params: dict[str, Any], embedding_model: str) -> str:
    """Deterministic hash of chunker + embedding combo for table naming."""
    payload = {"chunker": chunker_params, "embedding_model": embedding_model}
    key = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:8]


def _suggest_params(
    trial: Any, search_space: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Map search space config to Optuna suggest calls."""
    params: dict[str, dict[str, Any]] = {}
    for section, section_space in search_space.items():
        params[section] = {}
        for param_name, spec in section_space.items():
            full_name = f"{section}.{param_name}"
            if isinstance(spec, list):
                params[section][param_name] = trial.suggest_categorical(full_name, spec)
            elif isinstance(spec, dict) and "low" in spec and "high" in spec:
                low, high = spec["low"], spec["high"]
                step = spec.get("step")
                is_float = (
                    isinstance(low, float)
                    or isinstance(high, float)
                    or (step and isinstance(step, float))
                )
                if is_float:
                    kwargs = {"step": step} if step is not None else {}
                    val = trial.suggest_float(full_name, low, high, **kwargs)
                    params[section][param_name] = val
                else:
                    kwargs = {"step": step} if step is not None else {}
                    params[section][param_name] = trial.suggest_int(full_name, low, high, **kwargs)
            else:
                raise ValueError(
                    f"Invalid search space for {full_name!r}: expected a list of choices "
                    f"or a dict with 'low'/'high' keys, got {spec!r}"
                )
    return params


def _apply_trial_params(
    base_settings: Settings, trial_params: dict[str, dict[str, Any]]
) -> Settings:
    """Create a new Settings with trial params applied."""
    updates: dict[str, Any] = {}
    for section, section_params in trial_params.items():
        if hasattr(base_settings, section):
            current = getattr(base_settings, section)
            updates[section] = current.model_copy(update=section_params)
    return base_settings.model_copy(update=updates)


def _compute_composite_score(mean_scores: dict[str, float]) -> float:
    """Composite score = mean of all metric means."""
    if not mean_scores:
        return 0.0
    return sum(mean_scores.values()) / len(mean_scores)


async def _cleanup_tables(table_names: list[str], dsn: str) -> None:
    """Drop all temporary tuning tables (used on error/abort)."""
    import sqlalchemy
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(dsn)
    dropped = 0
    try:
        async with engine.begin() as conn:
            for table_name in table_names:
                # Validate each name before interpolating into raw SQL — same rule as
                # VectorStoreConfig._validate_table_name.  Names are constructed from
                # a validated base name + '_tune_' + 8-char hex digest, so this should
                # never fail in practice, but we guard explicitly to be safe.
                if not _SAFE_TABLE_RE.match(table_name):
                    logger.warning(
                        "Skipping unsafe table name during cleanup: %r", table_name
                    )
                    continue
                # Safe: table_name validated by _SAFE_TABLE_RE guard above.
                await conn.execute(
                    sqlalchemy.text(f"DROP TABLE IF EXISTS {table_name}")
                )
                dropped += 1
        logger.info("Cleaned up %d temporary tables", dropped)
    finally:
        await engine.dispose()


def run_tune(
    dataset_path: str,
    sources: list[str],
    settings: Settings,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> TuneResult:
    """Run Bayesian hyperparameter optimization.

    Args:
        dataset_path: Path to eval dataset YAML/JSON.
        sources: Document paths to ingest for each trial.
        settings: Base settings (tune.search_space defines the space).
        progress_callback: Called with (trial_number, total_trials, best_score).

    Returns:
        TuneResult with best params, score, and all trial data.
    """
    try:
        import optuna
    except ImportError:
        raise ImportError(
            "optuna is required for tuning. Install with: pip install 'noid-rag[tune]'"
        ) from None

    search_space = settings.tune.search_space
    if not search_space:
        raise ValueError(
            "No search space defined. Set tune.search_space in your config YAML."
        )

    max_trials = settings.tune.max_trials
    ingest_cache: dict[str, str] = {}  # hash -> table_name
    temp_tables: list[str] = []
    all_trials: list[dict[str, Any]] = []
    base_table = settings.vectorstore.table_name

    # PostgreSQL silently truncates identifiers longer than 63 bytes. The temp
    # table suffix is '_tune_' (6 chars) + 8-char hex digest = 14 chars, so the
    # base table name must be at most 49 characters.
    tune_suffix_len = 14  # len('_tune_') + len(8-char hex digest)
    max_base_len = 63 - tune_suffix_len
    if len(base_table) > max_base_len:
        raise ValueError(
            f"vectorstore.table_name {base_table!r} is too long for tuning. "
            f"Tuning appends a {tune_suffix_len}-char suffix; the base name "
            f"must be at most {max_base_len} characters to stay within "
            "PostgreSQL's 63-character identifier limit."
        )

    def objective(trial: optuna.trial.Trial) -> float:
        trial_params = _suggest_params(trial, search_space)
        trial_settings = _apply_trial_params(settings, trial_params)

        # Determine ingest cache key
        chunker_params = trial_params.get("chunker", {})
        embedding_model = trial_params.get("embedding", {}).get(
            "model", settings.embedding.model
        )
        cache_key = _ingest_config_hash(chunker_params, embedding_model)

        # Resolve embedding dimension and check HNSW compatibility
        embedding_dim = EMBEDDING_DIM_MAP.get(
            embedding_model, settings.vectorstore.embedding_dim
        )
        if embedding_dim > _HNSW_MAX_DIM:
            raise optuna.TrialPruned(
                f"Embedding model {embedding_model!r} produces {embedding_dim} dimensions, "
                f"which exceeds pgvector's HNSW index limit of {_HNSW_MAX_DIM}. "
                "Pruning this trial."
            )

        # Set up temp table
        table_name = f"{base_table}_tune_{cache_key}"
        trial_settings = trial_settings.model_copy(
            update={
                "vectorstore": trial_settings.vectorstore.model_copy(
                    update={
                        "table_name": table_name,
                        "embedding_dim": embedding_dim,
                    }
                ),
                "eval": trial_settings.eval.model_copy(update={"save_results": False}),
            }
        )

        from noid_rag.api import NoidRag

        rag = NoidRag(config=trial_settings)

        # Ingest if not cached
        if cache_key not in ingest_cache:
            for source in sources:
                asyncio.run(rag.aingest(source))
            ingest_cache[cache_key] = table_name
            temp_tables.append(table_name)

        # Evaluate
        summary = asyncio.run(rag.aeval(dataset_path))
        score = _compute_composite_score(summary.mean_scores)

        all_trials.append({
            "trial_number": trial.number,
            "params": trial_params,
            "score": score,
            "metric_scores": dict(summary.mean_scores),
        })

        if progress_callback:
            best_so_far = max(t["score"] for t in all_trials)
            progress_callback(trial.number + 1, max_trials, best_so_far)

        return score

    # Suppress Optuna's verbose logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())

    try:
        study.optimize(objective, n_trials=max_trials)
    finally:
        # Clean up temp tables
        if temp_tables and settings.vectorstore.dsn:
            try:
                asyncio.run(_cleanup_tables(temp_tables, settings.vectorstore.dsn))
            except Exception:
                logger.warning("Failed to clean up temporary tables: %s", temp_tables)

    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        raise ValueError(
            "All trials were pruned or failed — no completed trials to report. "
            "Check your search_space for incompatible models (e.g. embeddings "
            "exceeding pgvector's 2000-dimension HNSW limit)."
        )

    best_trial = study.best_trial
    best_params = _suggest_params(best_trial, search_space)

    return TuneResult(
        best_params=best_params,
        best_score=study.best_value,
        all_trials=all_trials,
        total_trials=len(all_trials),
        metrics_used=list(settings.eval.metrics),
    )


async def arun_tune(
    dataset_path: str,
    sources: list[str],
    settings: Settings,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> TuneResult:
    """Async wrapper for run_tune (runs in executor since Optuna is sync)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, run_tune, dataset_path, sources, settings, progress_callback
    )
