"""Tests for hyperparameter tuning module."""

from __future__ import annotations

import builtins
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from noid_rag.config import Settings
from noid_rag.models import EvalSummary, TuneResult

_original_import = builtins.__import__


# --- _ingest_config_hash ---


def test_ingest_config_hash_deterministic():
    from noid_rag.tune import _ingest_config_hash

    h1 = _ingest_config_hash({"max_tokens": 512}, "openai/text-embedding-3-small")
    h2 = _ingest_config_hash({"max_tokens": 512}, "openai/text-embedding-3-small")
    assert h1 == h2
    assert len(h1) == 8
    assert all(c in "0123456789abcdef" for c in h1)


def test_ingest_config_hash_differs_on_change():
    from noid_rag.tune import _ingest_config_hash

    h1 = _ingest_config_hash({"max_tokens": 512}, "openai/text-embedding-3-small")
    h2 = _ingest_config_hash({"max_tokens": 256}, "openai/text-embedding-3-small")
    h3 = _ingest_config_hash({"max_tokens": 512}, "openai/text-embedding-3-large")
    assert h1 != h2
    assert h1 != h3


# --- _suggest_params ---


def test_suggest_params_categorical():
    from noid_rag.tune import _suggest_params

    trial = MagicMock()
    trial.suggest_categorical.return_value = 512
    search_space = {"chunker": {"max_tokens": [256, 512, 1024]}}

    result = _suggest_params(trial, search_space)

    trial.suggest_categorical.assert_called_once_with("chunker.max_tokens", [256, 512, 1024])
    assert result == {"chunker": {"max_tokens": 512}}


def test_suggest_params_float_range():
    from noid_rag.tune import _suggest_params

    trial = MagicMock()
    trial.suggest_float.return_value = 0.1
    search_space = {"llm": {"temperature": {"low": 0.0, "high": 0.3, "step": 0.1}}}

    result = _suggest_params(trial, search_space)

    trial.suggest_float.assert_called_once_with("llm.temperature", 0.0, 0.3, step=0.1)
    assert result == {"llm": {"temperature": 0.1}}


def test_suggest_params_int_range():
    from noid_rag.tune import _suggest_params

    trial = MagicMock()
    trial.suggest_int.return_value = 5
    search_space = {"search": {"top_k": {"low": 3, "high": 15}}}

    result = _suggest_params(trial, search_space)

    trial.suggest_int.assert_called_once_with("search.top_k", 3, 15)
    assert result == {"search": {"top_k": 5}}


def test_suggest_params_invalid():
    from noid_rag.tune import _suggest_params

    trial = MagicMock()
    search_space = {"chunker": {"max_tokens": "invalid"}}

    with pytest.raises(ValueError, match="Invalid search space"):
        _suggest_params(trial, search_space)


# --- _apply_trial_params ---


def test_apply_trial_params():
    from noid_rag.tune import _apply_trial_params

    settings = Settings()
    trial_params = {
        "chunker": {"max_tokens": 256},
        "search": {"top_k": 10},
    }

    result = _apply_trial_params(settings, trial_params)

    assert result.chunker.max_tokens == 256
    assert result.search.top_k == 10
    # Original unchanged
    assert settings.chunker.max_tokens == 512
    assert settings.search.top_k == 5


def test_apply_trial_params_ignores_unknown_section():
    from noid_rag.tune import _apply_trial_params

    settings = Settings()
    trial_params = {"nonexistent": {"foo": "bar"}}

    result = _apply_trial_params(settings, trial_params)
    assert result.chunker.max_tokens == settings.chunker.max_tokens


# --- _compute_composite_score ---


def test_composite_score():
    from noid_rag.tune import _compute_composite_score

    scores = {"faithfulness": 0.8, "answer_relevancy": 0.6, "context_precision": 0.7}
    assert abs(_compute_composite_score(scores) - 0.7) < 1e-9


def test_composite_score_empty():
    from noid_rag.tune import _compute_composite_score

    assert _compute_composite_score({}) == 0.0


def test_composite_score_weighted():
    from noid_rag.tune import _compute_composite_score

    scores = {"faithfulness": 0.8, "context_precision": 0.4}
    weights = {"context_precision": 2.0, "faithfulness": 1.0}
    # weighted = (0.8*1.0 + 0.4*2.0) / (1.0 + 2.0) = 1.6 / 3.0
    expected = 1.6 / 3.0
    assert abs(_compute_composite_score(scores, weights) - expected) < 1e-9


def test_composite_score_weighted_missing_metric_defaults_to_1():
    """Metrics not listed in weights default to weight=1.0."""
    from noid_rag.tune import _compute_composite_score

    scores = {"faithfulness": 0.6, "answer_relevancy": 0.8}
    weights = {"faithfulness": 3.0}  # answer_relevancy defaults to 1.0
    # weighted = (0.6*3.0 + 0.8*1.0) / (3.0 + 1.0) = 2.6 / 4.0
    expected = 2.6 / 4.0
    assert abs(_compute_composite_score(scores, weights) - expected) < 1e-9


def test_composite_score_no_weights_is_equal():
    """Passing None or empty weights gives equal weighting (backward compat)."""
    from noid_rag.tune import _compute_composite_score

    scores = {"faithfulness": 0.8, "answer_relevancy": 0.6}
    assert _compute_composite_score(scores, None) == _compute_composite_score(scores)
    assert _compute_composite_score(scores, {}) == _compute_composite_score(scores)


# --- Ingest caching ---


def test_ingest_caching():
    """Same chunker+embedding combo should only ingest once."""
    from noid_rag.tune import run_tune

    settings = Settings()
    settings = settings.model_copy(
        update={
            "tune": settings.tune.model_copy(
                update={
                    "max_trials": 3,
                    "search_space": {
                        "search": {"top_k": [3, 5, 10]},
                    },
                }
            )
        }
    )

    mock_summary = EvalSummary(
        results=[],
        mean_scores={"faithfulness": 0.7, "answer_relevancy": 0.8},
        backend="ragas",
        model="test",
        total_questions=5,
        dataset_path="test.yml",
    )

    with patch("noid_rag.api.NoidRag") as mock_rag_cls:
        instance = MagicMock()
        instance.aingest = AsyncMock(return_value={"chunks_stored": 5, "document_id": "doc_test"})
        instance.aeval = AsyncMock(return_value=mock_summary)
        mock_rag_cls.return_value = instance

        result = run_tune("test.yml", ["doc1.pdf"], settings)

    # Only search params change, so ingest should happen exactly once
    assert instance.aingest.call_count == 1
    assert result.total_trials == 3


# --- Cleanup tables ---


@pytest.mark.asyncio
async def test_cleanup_tables():
    from contextlib import asynccontextmanager

    from noid_rag.tune import _cleanup_tables

    mock_conn = AsyncMock()

    @asynccontextmanager
    async def mock_begin():
        yield mock_conn

    mock_engine = MagicMock()
    mock_engine.begin = mock_begin
    mock_engine.dispose = AsyncMock()

    with patch("sqlalchemy.ext.asyncio.create_async_engine", return_value=mock_engine):
        await _cleanup_tables(
            ["docs_tune_abc12345", "docs_tune_def67890"],
            "postgresql+asyncpg://test",
        )

    assert mock_conn.execute.call_count == 2
    mock_engine.dispose.assert_called_once()


@pytest.mark.asyncio
async def test_cleanup_tables_skips_unsafe_names():
    """Table names that fail the safe-identifier check must be skipped, not executed."""
    from contextlib import asynccontextmanager

    from noid_rag.tune import _cleanup_tables

    mock_conn = AsyncMock()

    @asynccontextmanager
    async def mock_begin():
        yield mock_conn

    mock_engine = MagicMock()
    mock_engine.begin = mock_begin
    mock_engine.dispose = AsyncMock()

    with patch("sqlalchemy.ext.asyncio.create_async_engine", return_value=mock_engine):
        await _cleanup_tables(
            # safe name, unsafe name (contains semicolon), safe name
            ["docs_tune_abc12345", "bad; DROP TABLE users; --", "docs_tune_def67890"],
            "postgresql+asyncpg://test",
        )

    # Only the two safe names should have been executed
    assert mock_conn.execute.call_count == 2
    mock_engine.dispose.assert_called_once()


def test_cleanup_failure_is_swallowed(caplog):
    """A _cleanup_tables failure in the finally block must not propagate out of run_tune."""
    import logging

    from noid_rag.tune import run_tune

    settings = Settings()
    settings = settings.model_copy(
        update={
            "tune": settings.tune.model_copy(
                update={
                    "max_trials": 1,
                    "search_space": {"search": {"top_k": [5]}},
                }
            ),
            "vectorstore": settings.vectorstore.model_copy(
                update={"dsn": "postgresql+asyncpg://test"}
            ),
        }
    )

    mock_summary = EvalSummary(
        results=[],
        mean_scores={"faithfulness": 0.7},
        backend="ragas",
        model="test",
        total_questions=1,
        dataset_path="test.yml",
    )

    with patch("noid_rag.api.NoidRag") as mock_rag_cls:
        instance = MagicMock()
        instance.aingest = AsyncMock(return_value={"chunks_stored": 1, "document_id": "d"})
        instance.aeval = AsyncMock(return_value=mock_summary)
        mock_rag_cls.return_value = instance

        with patch(
            "noid_rag.tune._cleanup_tables",
            side_effect=RuntimeError("db gone"),
        ):
            with caplog.at_level(logging.WARNING, logger="noid_rag.tune"):
                # Must not raise even though cleanup failed
                result = run_tune("test.yml", ["doc.pdf"], settings)

    assert result.total_trials == 1
    assert any("Failed to clean up" in r.message for r in caplog.records)


# --- run_tune e2e (mocked) ---


def test_run_tune_e2e():
    from noid_rag.tune import run_tune

    settings = Settings()
    settings = settings.model_copy(
        update={
            "tune": settings.tune.model_copy(
                update={
                    "max_trials": 3,
                    "search_space": {
                        "search": {"top_k": [3, 5]},
                        "llm": {"temperature": {"low": 0.0, "high": 0.2, "step": 0.1}},
                    },
                }
            )
        }
    )

    mock_summary = EvalSummary(
        results=[],
        mean_scores={"faithfulness": 0.75, "answer_relevancy": 0.85},
        backend="ragas",
        model="test",
        total_questions=5,
        dataset_path="test.yml",
    )

    with patch("noid_rag.api.NoidRag") as mock_rag_cls:
        instance = MagicMock()
        instance.aingest = AsyncMock(return_value={"chunks_stored": 5, "document_id": "doc_test"})
        instance.aeval = AsyncMock(return_value=mock_summary)
        mock_rag_cls.return_value = instance

        progress_calls = []

        def on_progress(trial_num, total, best):
            progress_calls.append((trial_num, total, best))

        result = run_tune("test.yml", ["doc1.pdf"], settings, progress_callback=on_progress)

    assert isinstance(result, TuneResult)
    assert result.total_trials == 3
    assert result.best_score > 0
    assert len(result.all_trials) == 3
    assert len(progress_calls) == 3
    assert result.metrics_used == list(settings.eval.metrics)


# --- Error cases ---


def _mock_import_no_optuna(name, *args, **kwargs):
    if name == "optuna":
        raise ImportError("No module named 'optuna'")
    return _original_import(name, *args, **kwargs)


def test_missing_optuna_raises():
    import sys

    from noid_rag.tune import run_tune

    settings = Settings()
    settings = settings.model_copy(
        update={
            "tune": settings.tune.model_copy(
                update={
                    "max_trials": 1,
                    "search_space": {"search": {"top_k": [3, 5]}},
                }
            )
        }
    )

    with patch.dict(sys.modules, {"optuna": None}):
        with pytest.raises(ImportError, match="optuna is required"):
            with patch("builtins.__import__", side_effect=_mock_import_no_optuna):
                run_tune("test.yml", ["doc.pdf"], settings)


def test_empty_search_space_raises():
    """Empty search space should raise ValueError."""
    pytest.importorskip("optuna")

    from noid_rag.tune import run_tune

    settings = Settings()
    # search_space defaults to empty dict

    with pytest.raises(ValueError, match="No search space defined"):
        run_tune("test.yml", ["doc.pdf"], settings)


def test_all_trials_pruned_raises():
    """If every trial is pruned, raise a clear ValueError."""
    pytest.importorskip("optuna")

    from noid_rag.tune import run_tune

    settings = Settings()
    settings = settings.model_copy(
        update={
            "tune": settings.tune.model_copy(
                update={
                    "max_trials": 3,
                    "search_space": {
                        "embedding": {"model": ["openai/text-embedding-3-large"]},
                    },
                }
            )
        }
    )

    with pytest.raises(ValueError, match="All trials were pruned or failed"):
        run_tune("test.yml", ["doc.pdf"], settings)


# --- TuneConfig validation ---


def test_tune_config_max_trials_must_be_positive():
    """max_trials <= 0 should be rejected by TuneConfig."""
    from pydantic import ValidationError

    from noid_rag.config import TuneConfig

    with pytest.raises(ValidationError):
        TuneConfig(max_trials=0)

    with pytest.raises(ValidationError):
        TuneConfig(max_trials=-1)


def test_tune_config_max_trials_positive_accepted():
    from noid_rag.config import TuneConfig

    cfg = TuneConfig(max_trials=1)
    assert cfg.max_trials == 1


def test_tune_config_metric_weights():
    from noid_rag.config import TuneConfig

    cfg = TuneConfig(metric_weights={"context_precision": 2.0, "faithfulness": 1.0})
    assert cfg.metric_weights == {"context_precision": 2.0, "faithfulness": 1.0}


def test_tune_config_metric_weights_default_empty():
    from noid_rag.config import TuneConfig

    cfg = TuneConfig()
    assert cfg.metric_weights == {}


def test_tune_config_metric_weights_rejects_negative():
    from pydantic import ValidationError

    from noid_rag.config import TuneConfig

    with pytest.raises(ValidationError, match="must be > 0"):
        TuneConfig(metric_weights={"context_precision": -1.0})


def test_tune_config_metric_weights_rejects_zero():
    from pydantic import ValidationError

    from noid_rag.config import TuneConfig

    with pytest.raises(ValidationError, match="must be > 0"):
        TuneConfig(metric_weights={"faithfulness": 0.0})


# --- Table name length guard ---


def test_long_table_name_raises_before_first_trial():
    """base table name > 49 chars should raise ValueError before any trials run."""
    pytest.importorskip("optuna")

    from noid_rag.tune import run_tune

    settings = Settings()
    settings = settings.model_copy(
        update={
            "tune": settings.tune.model_copy(
                update={
                    "max_trials": 1,
                    "search_space": {"search": {"top_k": [3, 5]}},
                }
            ),
            "vectorstore": settings.vectorstore.model_copy(update={"table_name": "a" * 50}),
        }
    )

    with pytest.raises(ValueError, match="too long for tuning"):
        run_tune("test.yml", ["doc.pdf"], settings)


def test_hnsw_dim_limit_prunes_trial():
    """Embedding models exceeding pgvector's 2000-dim HNSW limit are pruned."""
    optuna = pytest.importorskip("optuna")

    from noid_rag.tune import run_tune

    settings = Settings()
    settings = settings.model_copy(
        update={
            "tune": settings.tune.model_copy(
                update={
                    "max_trials": 4,
                    "search_space": {
                        "embedding": {
                            "model": [
                                "openai/text-embedding-3-small",
                                "openai/text-embedding-3-large",
                            ]
                        },
                    },
                }
            )
        }
    )

    mock_summary = EvalSummary(
        results=[],
        mean_scores={"faithfulness": 0.8},
        backend="ragas",
        model="test",
        total_questions=1,
        dataset_path="test.yml",
    )

    # Use GridSampler to deterministically test both models
    with (
        patch("noid_rag.api.NoidRag") as mock_rag_cls,
        patch(
            "optuna.samplers.TPESampler",
            return_value=optuna.samplers.GridSampler(
                {
                    "embedding.model": [
                        "openai/text-embedding-3-small",
                        "openai/text-embedding-3-large",
                    ]
                }
            ),
        ),
    ):
        instance = MagicMock()
        instance.aingest = AsyncMock(return_value={"chunks_stored": 5, "document_id": "doc_test"})
        instance.aeval = AsyncMock(return_value=mock_summary)
        mock_rag_cls.return_value = instance

        result = run_tune("test.yml", ["doc1.pdf"], settings)

    # GridSampler guarantees both models are tried. Only the small model completes;
    # the large model (3072 dims) is pruned.
    assert result.total_trials > 0
    for t in result.all_trials:
        assert t["params"]["embedding"]["model"] == "openai/text-embedding-3-small"


def test_table_name_at_max_length_is_accepted():
    """base table name of exactly 49 chars should not raise."""
    pytest.importorskip("optuna")

    from noid_rag.tune import run_tune

    settings = Settings()
    settings = settings.model_copy(
        update={
            "tune": settings.tune.model_copy(
                update={
                    "max_trials": 1,
                    "search_space": {"search": {"top_k": [5]}},
                }
            ),
            "vectorstore": settings.vectorstore.model_copy(update={"table_name": "a" * 49}),
        }
    )

    mock_summary = EvalSummary(
        results=[],
        mean_scores={"faithfulness": 0.8},
        backend="ragas",
        model="test",
        total_questions=1,
        dataset_path="test.yml",
    )

    with patch("noid_rag.api.NoidRag") as mock_rag_cls:
        instance = MagicMock()
        instance.aingest = AsyncMock(return_value={"chunks_stored": 1, "document_id": "d"})
        instance.aeval = AsyncMock(return_value=mock_summary)
        mock_rag_cls.return_value = instance

        result = run_tune("test.yml", ["doc.pdf"], settings)

    assert result.total_trials == 1
