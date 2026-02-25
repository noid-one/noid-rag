"""Tests for LLM client and answer generation."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import SecretStr

from noid_rag.config import LLMConfig
from noid_rag.models import AnswerResult, SearchResult


def _config_with_key(key="test-key", **kwargs):
    return LLMConfig(api_key=SecretStr(key), **kwargs)


class TestLLMClient:
    def test_init_with_config_key(self):
        from noid_rag.llm import LLMClient

        client = LLMClient(config=_config_with_key("my-key"))
        # Secret must NOT be stored as a plain string attribute
        assert not hasattr(client, "_api_key")
        # Key is still accessible via the SecretStr
        assert client.config.api_key.get_secret_value() == "my-key"

    def test_init_raises_without_key(self):
        from noid_rag.llm import LLMClient

        with pytest.raises(ValueError, match="No LLM API key"):
            LLMClient(config=LLMConfig())

    @pytest.mark.asyncio
    @patch("noid_rag.llm.httpx.AsyncClient")
    async def test_generate_sends_correct_request(self, mock_client_cls):
        from noid_rag.llm import LLMClient

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "The answer is 42."}}]}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        client = LLMClient(config=_config_with_key())
        result = await client.generate("What is the answer?", "Context: the answer is 42")

        assert result == "The answer is 42."
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json") or call_kwargs[0][1]
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][1]["role"] == "user"
        assert "What is the answer?" in body["messages"][1]["content"]

    @pytest.mark.asyncio
    @patch("noid_rag.llm.httpx.AsyncClient")
    async def test_generate_uses_configured_model(self, mock_client_cls):
        from noid_rag.llm import LLMClient

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "answer"}}]}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        client = LLMClient(config=_config_with_key(model="anthropic/claude-3-haiku"))
        await client.generate("q", "ctx")

        call_kwargs = mock_client.post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json") or call_kwargs[0][1]
        assert body["model"] == "anthropic/claude-3-haiku"

    @pytest.mark.asyncio
    @patch("noid_rag.llm.httpx.AsyncClient")
    async def test_generate_raises_on_malformed_response(self, mock_client_cls):
        from noid_rag.llm import LLMClient

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": []}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        client = LLMClient(config=_config_with_key())
        with pytest.raises(ValueError, match="Unexpected LLM response shape"):
            await client.generate("q", "ctx")

    @pytest.mark.asyncio
    @patch("noid_rag.llm.httpx.AsyncClient")
    async def test_generate_sends_authorization_header(self, mock_client_cls):
        """Bearer token must be present and contain the configured key."""
        from noid_rag.llm import LLMClient

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        client = LLMClient(config=_config_with_key("secret-key-xyz"))
        await client.generate("q", "ctx")

        call_kwargs = mock_client.post.call_args
        headers = (
            call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers") or call_kwargs[0][2]
        )
        assert "Authorization" in headers
        assert "secret-key-xyz" in headers["Authorization"]

    @pytest.mark.asyncio
    @patch("noid_rag.llm.httpx.AsyncClient")
    async def test_generate_does_not_retry_on_malformed_response(self, mock_client_cls):
        """ValueError from a malformed response is not retriable — must not loop."""
        from noid_rag.llm import LLMClient

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": []}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        client = LLMClient(config=_config_with_key())
        with pytest.raises(ValueError, match="Unexpected LLM response shape"):
            await client.generate("q", "ctx")

        # POST was called exactly once — no retry loop
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    @patch("noid_rag.llm.httpx.AsyncClient")
    async def test_generate_retries_on_http_status_error(self, mock_client_cls):
        """HTTPStatusError triggers the retry policy (up to 3 attempts)."""
        from tenacity import RetryError

        from noid_rag.llm import LLMClient

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Internal Server Error",
            request=MagicMock(),
            response=MagicMock(),
        )

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        client = LLMClient(config=_config_with_key())
        # After all retry attempts are exhausted tenacity wraps the cause in RetryError
        with pytest.raises(RetryError) as exc_info:
            await client.generate("q", "ctx")

        # The original HTTPStatusError must be the underlying cause
        assert isinstance(exc_info.value.last_attempt.exception(), httpx.HTTPStatusError)
        # POST was called 3 times (stop_after_attempt=3)
        assert mock_client.post.call_count == 3


class TestAnswerGeneration:
    @pytest.mark.asyncio
    @patch("noid_rag.llm.LLMClient", autospec=False)
    @patch("noid_rag.api.NoidRag.asearch")
    async def test_aanswer_returns_answer_result(self, mock_asearch, mock_llm_cls):
        from pydantic import SecretStr

        from noid_rag.api import NoidRag
        from noid_rag.config import Settings

        mock_results = [
            SearchResult(
                chunk_id="chk_1",
                text="The capital of France is Paris.",
                score=0.95,
                metadata={},
                document_id="doc_1",
            ),
        ]
        mock_asearch.return_value = mock_results

        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "Paris is the capital of France."
        mock_llm_cls.return_value = mock_llm

        settings = Settings(
            embedding={"api_key": "test-embed-key"},
            vectorstore={"dsn": "postgresql+asyncpg://test@localhost/test"},
            llm={"api_key": SecretStr("test-llm-key"), "model": "openai/gpt-4o-mini"},
        )
        rag = NoidRag(config=settings)

        result = await rag.aanswer("What is the capital of France?")

        assert isinstance(result, AnswerResult)
        assert result.answer == "Paris is the capital of France."
        assert result.model == "openai/gpt-4o-mini"
        assert len(result.sources) == 1
        assert result.sources[0].chunk_id == "chk_1"

    @pytest.mark.asyncio
    @patch("noid_rag.api.NoidRag.asearch")
    async def test_aanswer_empty_results_returns_without_calling_llm(self, mock_asearch):
        """When search finds nothing, aanswer must short-circuit — no LLM call."""
        from pydantic import SecretStr

        from noid_rag.api import NoidRag
        from noid_rag.config import Settings

        mock_asearch.return_value = []

        settings = Settings(
            embedding={"api_key": "test-embed-key"},
            vectorstore={"dsn": "postgresql+asyncpg://test@localhost/test"},
            llm={"api_key": SecretStr("test-llm-key"), "model": "openai/gpt-4o-mini"},
        )
        rag = NoidRag(config=settings)

        with patch("noid_rag.llm.LLMClient") as mock_llm_cls:
            result = await rag.aanswer("anything")

        # LLMClient must never be instantiated when there are no results
        mock_llm_cls.assert_not_called()

        assert isinstance(result, AnswerResult)
        assert result.sources == []
        assert result.model == "openai/gpt-4o-mini"
        assert "no relevant" in result.answer.lower()

    @pytest.mark.asyncio
    @patch("noid_rag.llm.LLMClient", autospec=False)
    @patch("noid_rag.api.NoidRag.asearch")
    async def test_aanswer_context_includes_all_sources(self, mock_asearch, mock_llm_cls):
        """Context passed to LLM must include text from every search result."""
        from pydantic import SecretStr

        from noid_rag.api import NoidRag
        from noid_rag.config import Settings

        mock_results = [
            SearchResult(
                chunk_id="chk_1",
                text="Paris is the capital.",
                score=0.9,
                metadata={},
                document_id="doc_1",
            ),
            SearchResult(
                chunk_id="chk_2",
                text="France is in Europe.",
                score=0.8,
                metadata={},
                document_id="doc_2",
            ),
        ]
        mock_asearch.return_value = mock_results

        captured_context: list[str] = []

        async def capture_generate(prompt: str, context: str) -> str:
            captured_context.append(context)
            return "synthesized answer"

        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = capture_generate
        mock_llm_cls.return_value = mock_llm

        settings = Settings(
            embedding={"api_key": "embed-key"},
            vectorstore={"dsn": "postgresql+asyncpg://test@localhost/test"},
            llm={"api_key": SecretStr("llm-key"), "model": "openai/gpt-4o-mini"},
        )
        rag = NoidRag(config=settings)
        await rag.aanswer("capital question")

        assert len(captured_context) == 1
        ctx = captured_context[0]
        assert "Paris is the capital." in ctx
        assert "France is in Europe." in ctx
        assert "doc_1" in ctx
        assert "doc_2" in ctx
