"""Bayesian hyperparameter optimization for RAG pipelines via Optuna."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
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


def _ingest_config_hash(
    chunker_params: dict[str, Any],
    embedding_model: str,
    parser_params: dict[str, Any] | None = None,
) -> str:
    """Deterministic hash of parser + chunker + embedding combo for table naming."""
    payload = {
        "chunker": chunker_params,
        "embedding_model": embedding_model,
        "parser": parser_params or {},
    }
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


def _compute_composite_score(
    mean_scores: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    """Composite score = weighted mean of all metric means.

    When *weights* is empty or ``None``, all metrics are weighted equally
    (backward-compatible default).  Otherwise each metric is scaled by its
    weight (defaulting to 1.0 for unlisted metrics) and the result is
    normalised by the total weight.
    """
    if not mean_scores:
        return 0.0
    if not weights:
        return sum(mean_scores.values()) / len(mean_scores)
    total = 0.0
    weight_sum = 0.0
    for metric, score in mean_scores.items():
        w = weights.get(metric, 1.0)
        total += score * w
        weight_sum += w
    return total / weight_sum if weight_sum else 0.0


async def _cleanup_stores(store_names: list[str], settings: Settings) -> None:
    """Drop all temporary tuning stores (used on error/abort).

    Dispatches by provider: pgvector uses raw SQL DROP TABLE,
    qdrant deletes collections via the client.
    """
    provider = settings.vectorstore.provider

    if provider == "zvec":
        import shutil

        from noid_rag.config import _SAFE_COLLECTION_RE

        data_dir = Path(settings.zvec.data_dir).expanduser()
        dropped = 0
        for name in store_names:
            if not _SAFE_COLLECTION_RE.match(name):
                logger.warning(
                    "Skipping unsafe collection name during cleanup: %r", name
                )
                continue
            collection_path = data_dir / name
            # Guard against path traversal even though names are generated internally
            if not collection_path.resolve().is_relative_to(data_dir.resolve()):
                logger.warning(
                    "Skipping collection path outside data_dir during cleanup: %r", name
                )
                continue
            if collection_path.exists():
                shutil.rmtree(collection_path)
                dropped += 1
        logger.info("Cleaned up %d temporary zvec collections", dropped)
    elif provider == "qdrant":
        from noid_rag.config import _SAFE_COLLECTION_RE
        from noid_rag.vectorstore_qdrant import make_raw_client

        client = None
        try:
            client = make_raw_client(settings.qdrant)
            dropped = 0
            for name in store_names:
                if not _SAFE_COLLECTION_RE.match(name):
                    logger.warning(
                        "Skipping unsafe collection name during cleanup: %r", name
                    )
                    continue
                if await client.collection_exists(name):
                    await client.delete_collection(name)
                    dropped += 1
            logger.info("Cleaned up %d temporary collections", dropped)
        finally:
            if client is not None:
                await client.close()
    else:
        from sqlalchemy import text as sa_text
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(settings.vectorstore.dsn)
        dropped = 0
        try:
            async with engine.begin() as conn:
                for table_name in store_names:
                    if not _SAFE_TABLE_RE.match(table_name):
                        logger.warning(
                            "Skipping unsafe table name during cleanup: %r", table_name
                        )
                        continue
                    await conn.execute(sa_text(f"DROP TABLE IF EXISTS {table_name}"))
                    dropped += 1
            logger.info("Cleaned up %d temporary tables", dropped)
        finally:
            await engine.dispose()


def _setup_trial(
    trial: Any,
    search_space: dict[str, dict[str, Any]],
    settings: Settings,
    base_table: str,
) -> tuple[dict[str, dict[str, Any]], Settings, str]:
    """Suggest params and configure trial settings.

    Returns (trial_params, trial_settings, cache_key).
    Raises optuna.TrialPruned if embedding dimensions exceed HNSW limit.
    """
    import optuna

    trial_params = _suggest_params(trial, search_space)
    trial_settings = _apply_trial_params(settings, trial_params)

    # Determine ingest cache key
    chunker_params = trial_params.get("chunker", {})
    parser_params = trial_params.get("parser", {})
    embedding_model = trial_params.get("embedding", {}).get("model", settings.embedding.model)
    cache_key = _ingest_config_hash(chunker_params, embedding_model, parser_params)

    # Resolve embedding dimension.  For local providers (zvec) the dimension is
    # fixed by the model and already set in vectorstore.embedding_dim; the
    # EMBEDDING_DIM_MAP only applies to API-based embedding models.
    if settings.embedding.provider == "zvec":
        embedding_dim = settings.vectorstore.embedding_dim
    else:
        embedding_dim = EMBEDDING_DIM_MAP.get(embedding_model, settings.vectorstore.embedding_dim)
    if settings.vectorstore.provider == "pgvector" and embedding_dim > _HNSW_MAX_DIM:
        raise optuna.TrialPruned(
            f"Embedding model {embedding_model!r} produces {embedding_dim} dimensions, "
            f"which exceeds pgvector's HNSW index limit of {_HNSW_MAX_DIM}. "
            "Pruning this trial."
        )

    # Set up temp store name
    store_name = f"{base_table}_tune_{cache_key}"
    if settings.vectorstore.provider == "zvec":
        trial_settings = trial_settings.model_copy(
            update={
                "vectorstore": trial_settings.vectorstore.model_copy(
                    update={"embedding_dim": embedding_dim}
                ),
                "zvec": trial_settings.zvec.model_copy(
                    update={"collection_name": store_name}
                ),
                "eval": trial_settings.eval.model_copy(update={"save_results": False}),
            }
        )
    elif settings.vectorstore.provider == "qdrant":
        trial_settings = trial_settings.model_copy(
            update={
                "vectorstore": trial_settings.vectorstore.model_copy(
                    update={"embedding_dim": embedding_dim}
                ),
                "qdrant": trial_settings.qdrant.model_copy(
                    update={"collection_name": store_name}
                ),
                "eval": trial_settings.eval.model_copy(update={"save_results": False}),
            }
        )
    else:
        trial_settings = trial_settings.model_copy(
            update={
                "vectorstore": trial_settings.vectorstore.model_copy(
                    update={
                        "table_name": store_name,
                        "embedding_dim": embedding_dim,
                    }
                ),
                "eval": trial_settings.eval.model_copy(update={"save_results": False}),
            }
        )

    return trial_params, trial_settings, cache_key


async def _ensure_ingested(
    cache_key: str,
    sources: list[str],
    rag: Any,
    ingest_cache: dict[str, str],
    table_name: str,
    temp_tables: list[str],
) -> None:
    """Ingest sources if this config hasn't been ingested yet."""
    if cache_key not in ingest_cache:
        # Track for cleanup BEFORE ingest so failed attempts are still removed.
        temp_tables.append(table_name)
        for source in sources:
            await rag.aingest(source)
        ingest_cache[cache_key] = table_name


async def _evaluate_trial(
    rag: Any,
    dataset_path: str,
    metric_weights: dict[str, float] | None,
) -> tuple[float, dict[str, float]]:
    """Run evaluation and compute composite score.

    Returns (composite_score, mean_scores).
    """
    summary = await rag.aeval(dataset_path)
    score = _compute_composite_score(summary.mean_scores, metric_weights)
    return score, dict(summary.mean_scores)


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
            "No search space defined. Set tune.search_space in your config YAML. "
            "See config.pgvector.yml or config.qdrant.yml for examples."
        )

    max_trials = settings.tune.max_trials
    ingest_cache: dict[str, str] = {}  # hash -> store_name
    temp_stores: list[str] = []
    all_trials: list[dict[str, Any]] = []

    provider = settings.vectorstore.provider
    if provider == "zvec":
        base_table = settings.zvec.collection_name
    elif provider == "qdrant":
        base_table = settings.qdrant.collection_name
    else:
        base_table = settings.vectorstore.table_name

    # The temp store suffix is '_tune_' (6 chars) + 8-char hex digest = 14 chars.
    # All backends have identifier length limits that must be respected.
    tune_suffix_len = 14  # len('_tune_') + len(8-char hex digest)
    if provider == "pgvector":
        max_base_len = 63 - tune_suffix_len  # PostgreSQL's 63-char identifier limit
        if len(base_table) > max_base_len:
            raise ValueError(
                f"vectorstore.table_name {base_table!r} is too long for tuning. "
                f"Tuning appends a {tune_suffix_len}-char suffix; the base name "
                f"must be at most {max_base_len} characters to stay within "
                "PostgreSQL's 63-character identifier limit."
            )
    elif provider == "qdrant":
        max_base_len = 255 - tune_suffix_len  # Qdrant's 255-char collection name limit
        if len(base_table) > max_base_len:
            raise ValueError(
                f"qdrant.collection_name {base_table!r} is too long for tuning. "
                f"Tuning appends a {tune_suffix_len}-char suffix; the base name "
                f"must be at most {max_base_len} characters to stay within "
                "Qdrant's 255-character collection name limit."
            )
    elif provider == "zvec":
        max_base_len = 255 - tune_suffix_len  # filesystem name limit
        if len(base_table) > max_base_len:
            raise ValueError(
                f"zvec.collection_name {base_table!r} is too long for tuning. "
                f"Tuning appends a {tune_suffix_len}-char suffix; the base name "
                f"must be at most {max_base_len} characters to stay within "
                "the 255-character filesystem name limit."
            )

    # Clean up stale tune collections from previous runs that may have
    # crashed or failed before their cleanup code could run.
    if provider == "zvec":
        import shutil

        data_dir = Path(settings.zvec.data_dir).expanduser()
        if data_dir.exists():
            prefix = f"{base_table}_tune_"
            for entry in data_dir.iterdir():
                if entry.is_dir() and entry.name.startswith(prefix):
                    shutil.rmtree(entry)
                    logger.debug("Removed stale tune collection: %s", entry.name)

    # Create a single event loop for all trials — avoids the overhead of
    # creating and destroying a loop per trial (which also prevents connection reuse).
    # set_event_loop is safe here because n_jobs=1 runs objective() in the
    # caller's thread; if n_jobs > 1 is ever supported, revisit this.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def objective(trial: optuna.trial.Trial) -> float:
        trial_params, trial_settings, cache_key = _setup_trial(
            trial, search_space, settings, base_table
        )

        from noid_rag.api import NoidRag

        rag = NoidRag(config=trial_settings)
        if provider == "zvec":
            store_name = trial_settings.zvec.collection_name
        elif provider == "qdrant":
            store_name = trial_settings.qdrant.collection_name
        else:
            store_name = trial_settings.vectorstore.table_name

        try:
            # Ingest if not cached
            loop.run_until_complete(
                _ensure_ingested(cache_key, sources, rag, ingest_cache, store_name, temp_stores)
            )

            # Evaluate
            metric_weights = settings.tune.metric_weights or None
            score, metric_scores = loop.run_until_complete(
                _evaluate_trial(rag, dataset_path, metric_weights)
            )
        finally:
            # Close shared HTTP clients to avoid ResourceWarning when the loop is closed.
            loop.run_until_complete(rag.close())

        all_trials.append(
            {
                "trial_number": trial.number,
                "params": trial_params,
                "score": score,
                "metric_scores": metric_scores,
            }
        )

        if progress_callback:
            best_so_far = max(t["score"] for t in all_trials)
            progress_callback(trial.number + 1, max_trials, best_so_far)

        return score

    # Suppress Optuna's verbose logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())

    try:
        study.optimize(objective, n_trials=max_trials, n_jobs=1)
    finally:
        # Clean up temp stores
        should_cleanup = bool(temp_stores) and (
            provider in ("qdrant", "zvec") or bool(settings.vectorstore.dsn)
        )
        if should_cleanup:
            try:
                loop.run_until_complete(_cleanup_stores(temp_stores, settings))
            except Exception:
                logger.warning("Failed to clean up temporary stores: %s", temp_stores)
        loop.close()
        asyncio.set_event_loop(None)

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
    """Async wrapper for run_tune (runs in executor since Optuna is sync).

    Uses a one-shot ThreadPoolExecutor so that run_tune's
    set_event_loop / set_event_loop(None) calls do not pollute the
    default executor's reusable worker threads.
    """
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        return await loop.run_in_executor(
            executor, run_tune, dataset_path, sources, settings, progress_callback
        )
