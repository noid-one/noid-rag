"""Tests for the generate module."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, patch

import pytest


class TestGenerateQaPairsLogging:
    @pytest.mark.asyncio
    async def test_logs_warning_on_chunk_failure(self, caplog):
        from noid_rag.generate import generate_qa_pairs

        mock_config = AsyncMock()
        mock_config.api_key.get_secret_value.return_value = "test-key"
        mock_config.api_url = "https://example.com/v1/chat/completions"

        chunks = [{"text": "chunk 1"}, {"text": "chunk 2"}]

        with (
            patch(
                "noid_rag.generate._call_llm",
                side_effect=RuntimeError("LLM error"),
            ),
            caplog.at_level(logging.WARNING, logger="noid_rag.generate"),
        ):
            result = await generate_qa_pairs(
                chunks,
                llm_config=mock_config,
                model="test-model",
                questions_per_chunk=1,
            )

        assert result == []
        warning_messages = [r.message for r in caplog.records]
        assert any("Chunk 1/2 failed" in m for m in warning_messages)
        assert any("Chunk 2/2 failed" in m for m in warning_messages)
        assert any("2/2 chunks failed" in m for m in warning_messages)

    @pytest.mark.asyncio
    async def test_no_warning_on_all_success(self, caplog):
        from noid_rag.generate import generate_qa_pairs

        mock_config = AsyncMock()
        mock_config.api_key.get_secret_value.return_value = "test-key"
        mock_config.api_url = "https://example.com/v1/chat/completions"

        chunks = [{"text": "chunk 1"}]

        with (
            patch(
                "noid_rag.generate._call_llm",
                return_value=[{"question": "Q?", "ground_truth": "A."}],
            ),
            caplog.at_level(logging.WARNING, logger="noid_rag.generate"),
        ):
            result = await generate_qa_pairs(
                chunks,
                llm_config=mock_config,
                model="test-model",
                questions_per_chunk=1,
            )

        assert len(result) == 1
        assert not any("failed" in r.message for r in caplog.records)
