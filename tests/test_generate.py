"""Tests for the generate module."""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml


class TestCallLlm:
    @pytest.mark.asyncio
    async def test_strips_markdown_fences(self):
        from noid_rag.generate import _call_llm

        mock_config = MagicMock()
        mock_config.api_key.get_secret_value.return_value = "test-key"
        mock_config.api_url = "https://example.com/v1/chat/completions"

        fenced = '```json\n[{"question": "Q?", "ground_truth": "A."}]\n```'
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": fenced}}]}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp

        result = await _call_llm(
            mock_config, "test-model", "chunk text", 1, http_client=mock_client
        )
        assert len(result) == 1
        assert result[0]["question"] == "Q?"
        assert result[0]["ground_truth"] == "A."

    @pytest.mark.asyncio
    async def test_malformed_json_raises(self):
        from noid_rag.generate import _call_llm

        mock_config = MagicMock()
        mock_config.api_key.get_secret_value.return_value = "test-key"
        mock_config.api_url = "https://example.com/v1/chat/completions"

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "not json"}}]}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp

        with pytest.raises(json.JSONDecodeError):
            await _call_llm(mock_config, "test-model", "chunk text", 1, http_client=mock_client)

    @pytest.mark.asyncio
    async def test_non_array_response_raises(self):
        from noid_rag.generate import _call_llm

        mock_config = MagicMock()
        mock_config.api_key.get_secret_value.return_value = "test-key"
        mock_config.api_url = "https://example.com/v1/chat/completions"

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": '{"question": "Q?"}'}}]}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp

        with pytest.raises(ValueError, match="Expected JSON array"):
            await _call_llm(mock_config, "test-model", "chunk text", 1, http_client=mock_client)

    @pytest.mark.asyncio
    async def test_missing_fields_filtered(self):
        from noid_rag.generate import _call_llm

        mock_config = MagicMock()
        mock_config.api_key.get_secret_value.return_value = "test-key"
        mock_config.api_url = "https://example.com/v1/chat/completions"

        # One valid, one missing ground_truth, one missing question
        content = json.dumps(
            [
                {"question": "Q1?", "ground_truth": "A1."},
                {"question": "Q2?"},
                {"ground_truth": "A3."},
            ]
        )
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": content}}]}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp

        result = await _call_llm(
            mock_config, "test-model", "chunk text", 3, http_client=mock_client
        )
        assert len(result) == 1
        assert result[0]["question"] == "Q1?"


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


class TestGenerateQaPairsNumQuestionsCap:
    @pytest.mark.asyncio
    async def test_respects_num_questions_cap(self):
        from noid_rag.generate import generate_qa_pairs

        mock_config = MagicMock()
        mock_config.api_key.get_secret_value.return_value = "test-key"
        mock_config.api_url = "https://example.com/v1/chat/completions"

        chunks = [{"text": f"chunk {i}"} for i in range(10)]

        with patch(
            "noid_rag.generate._call_llm",
            return_value=[
                {"question": "Q1?", "ground_truth": "A1."},
                {"question": "Q2?", "ground_truth": "A2."},
            ],
        ):
            result = await generate_qa_pairs(
                chunks,
                llm_config=mock_config,
                model="test-model",
                questions_per_chunk=2,
                num_questions=3,
            )

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_progress_callback_called(self):
        from noid_rag.generate import generate_qa_pairs

        mock_config = MagicMock()
        mock_config.api_key.get_secret_value.return_value = "test-key"
        mock_config.api_url = "https://example.com/v1/chat/completions"

        chunks = [{"text": "chunk 1"}, {"text": "chunk 2"}]
        calls = []

        with patch(
            "noid_rag.generate._call_llm",
            return_value=[{"question": "Q?", "ground_truth": "A."}],
        ):
            await generate_qa_pairs(
                chunks,
                llm_config=mock_config,
                model="test-model",
                questions_per_chunk=1,
                progress_callback=lambda i: calls.append(i),
            )

        assert calls == [0, 1]


class TestSaveDataset:
    def test_save_json(self, tmp_path):
        from noid_rag.generate import save_dataset

        output = tmp_path / "out.json"
        questions = [{"question": "Q?", "ground_truth": "A."}]
        save_dataset(questions, output)

        loaded = json.loads(output.read_text())
        assert loaded["questions"] == questions

    def test_save_yaml(self, tmp_path):
        from noid_rag.generate import save_dataset

        output = tmp_path / "out.yml"
        questions = [{"question": "Q?", "ground_truth": "A."}]
        save_dataset(questions, output)

        loaded = yaml.safe_load(output.read_text())
        assert loaded["questions"] == questions

    def test_save_creates_parent_dirs(self, tmp_path):
        from noid_rag.generate import save_dataset

        output = tmp_path / "deep" / "nested" / "out.json"
        save_dataset([{"question": "Q?", "ground_truth": "A."}], output)
        assert output.exists()

    def test_atomic_write_no_partial_on_error(self, tmp_path):
        """If serialization fails, the output file should not exist."""
        from noid_rag.generate import save_dataset

        output = tmp_path / "out.json"

        class Unserializable:
            pass

        with pytest.raises(TypeError):
            save_dataset([{"question": Unserializable()}], output)

        assert not output.exists()
