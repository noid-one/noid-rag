"""Tests for embedding client."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from noid_rag.config import EmbeddingConfig
from noid_rag.embeddings import EmbeddingClient
from noid_rag.models import Chunk


def _make_mock_response(data):
    """Create a mock httpx response."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = data
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def _patch_client_post(client, responses):
    """Patch the internal httpx client's post method with given responses."""
    if isinstance(responses, dict):
        responses = [responses]

    mock_resps = [_make_mock_response(r) for r in responses]

    mock_http = AsyncMock()
    mock_http.is_closed = False
    if len(mock_resps) == 1:
        mock_http.post.return_value = mock_resps[0]
    else:
        mock_http.post.side_effect = mock_resps

    client._client = mock_http
    return mock_http


class TestEmbeddingClientInit:
    def test_initializes_with_config(self):
        config = EmbeddingConfig(api_key="my-key")
        client = EmbeddingClient(config=config)
        assert client.config.api_key.get_secret_value() == "my-key"
        assert client._api_key == "my-key"

    def test_initializes_with_default_config(self, monkeypatch):
        monkeypatch.setenv("NOID_RAG_EMBEDDING__API_KEY", "test-key")
        client = EmbeddingClient()
        assert client.config.provider == "openrouter"

    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("NOID_RAG_EMBEDDING__API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No embedding API key configured"):
            EmbeddingClient()

    def test_api_key_from_config(self):
        config = EmbeddingConfig(api_key="config-key")
        client = EmbeddingClient(config=config)
        assert client._api_key == "config-key"

    def test_api_key_from_noid_env(self, monkeypatch):
        monkeypatch.setenv("NOID_RAG_EMBEDDING__API_KEY", "noid-env-key")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        config = EmbeddingConfig()
        client = EmbeddingClient(config=config)
        assert client._api_key == "noid-env-key"

    def test_api_key_from_openrouter_env(self, monkeypatch):
        monkeypatch.delenv("NOID_RAG_EMBEDDING__API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        config = EmbeddingConfig()
        client = EmbeddingClient(config=config)
        assert client._api_key == "or-key"

    def test_api_key_priority_config_first(self, monkeypatch):
        monkeypatch.setenv("NOID_RAG_EMBEDDING__API_KEY", "env-key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        config = EmbeddingConfig(api_key="config-key")
        client = EmbeddingClient(config=config)
        assert client._api_key == "config-key"


class TestEmbedTexts:
    @pytest.fixture
    def client(self):
        config = EmbeddingConfig(api_key="test-key", batch_size=2)
        return EmbeddingClient(config=config)

    @pytest.mark.asyncio
    async def test_embed_texts_returns_embeddings(self, client):
        _patch_client_post(
            client,
            {
                "data": [
                    {"index": 0, "embedding": [0.1] * 10},
                    {"index": 1, "embedding": [0.2] * 10},
                ]
            },
        )

        result = await client.embed_texts(["hello", "world"])
        assert len(result) == 2
        assert len(result[0]) == 10
        assert result[0] == [0.1] * 10
        assert result[1] == [0.2] * 10

    @pytest.mark.asyncio
    async def test_embed_texts_sorts_by_index(self, client):
        """API may return embeddings out of order; they should be sorted."""
        _patch_client_post(
            client,
            {
                "data": [
                    {"index": 1, "embedding": [0.2] * 10},
                    {"index": 0, "embedding": [0.1] * 10},
                ]
            },
        )

        result = await client.embed_texts(["hello", "world"])
        assert result[0] == [0.1] * 10
        assert result[1] == [0.2] * 10

    @pytest.mark.asyncio
    async def test_batch_splitting(self, client):
        """With batch_size=2, 3 texts should produce 2 API calls."""
        resp1 = {
            "data": [
                {"index": 0, "embedding": [0.1] * 10},
                {"index": 1, "embedding": [0.2] * 10},
            ]
        }
        resp2 = {
            "data": [
                {"index": 0, "embedding": [0.3] * 10},
            ]
        }
        mock_http = _patch_client_post(client, [resp1, resp2])

        result = await client.embed_texts(["a", "b", "c"])
        assert len(result) == 3
        assert result[2] == [0.3] * 10
        assert mock_http.post.call_count == 2


class TestEmbedChunks:
    @pytest.fixture
    def client(self):
        config = EmbeddingConfig(api_key="test-key", batch_size=2)
        return EmbeddingClient(config=config)

    @pytest.mark.asyncio
    async def test_embed_chunks_sets_embedding_in_place(self, client):
        chunks = [
            Chunk(text="chunk 1", document_id="doc_1"),
            Chunk(text="chunk 2", document_id="doc_1"),
        ]

        _patch_client_post(
            client,
            {
                "data": [
                    {"index": 0, "embedding": [0.1] * 10},
                    {"index": 1, "embedding": [0.2] * 10},
                ]
            },
        )

        result = await client.embed_chunks(chunks)

        # Should return the same chunk objects
        assert result is chunks
        assert all(c.embedding is not None for c in result)
        assert chunks[0].embedding == [0.1] * 10
        assert chunks[1].embedding == [0.2] * 10

    @pytest.mark.asyncio
    async def test_embed_chunks_returns_chunks(self, client):
        chunks = [Chunk(text="test", document_id="doc_1")]

        _patch_client_post(client, {"data": [{"index": 0, "embedding": [0.5] * 10}]})

        result = await client.embed_chunks(chunks)
        assert len(result) == 1
        assert isinstance(result[0], Chunk)


class TestEmbedQuery:
    @pytest.fixture
    def client(self):
        config = EmbeddingConfig(api_key="test-key", batch_size=2)
        return EmbeddingClient(config=config)

    @pytest.mark.asyncio
    async def test_embed_query_returns_single_embedding(self, client):
        _patch_client_post(client, {"data": [{"index": 0, "embedding": [0.5] * 10}]})

        result = await client.embed_query("test query")
        assert isinstance(result, list)
        assert len(result) == 10
        assert result == [0.5] * 10


class TestCircuitBreakerIntegration:
    @pytest.mark.asyncio
    async def test_open_circuit_raises_immediately_without_retry(self):
        """CircuitOpenError must propagate without triggering tenacity retry."""
        from noid_rag.circuit_breaker import CircuitOpenError

        config = EmbeddingConfig(api_key="test-key")
        client = EmbeddingClient(config=config)

        # Open the circuit manually
        client.circuit_breaker._state = __import__(
            "noid_rag.circuit_breaker", fromlist=["CircuitState"]
        ).CircuitState.OPEN
        import time

        client.circuit_breaker._opened_at = time.monotonic()

        mock_http = AsyncMock()
        mock_http.is_closed = False
        client._client = mock_http

        with pytest.raises(CircuitOpenError):
            await client._embed_batch(["hello"])

        # No HTTP call should have been made
        mock_http.post.assert_not_called()


class TestClientLifecycle:
    @pytest.mark.asyncio
    async def test_close_disposes_client(self):
        config = EmbeddingConfig(api_key="test-key")
        client = EmbeddingClient(config=config)

        mock_http = AsyncMock()
        mock_http.is_closed = False
        mock_http.aclose = AsyncMock()
        client._client = mock_http

        await client.close()
        mock_http.aclose.assert_called_once()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_context_manager(self):
        config = EmbeddingConfig(api_key="test-key")
        async with EmbeddingClient(config=config) as client:
            assert client._client is None  # lazily created
        assert client._client is None  # closed

    @pytest.mark.asyncio
    async def test_lazy_client_creation(self):
        config = EmbeddingConfig(api_key="test-key")
        client = EmbeddingClient(config=config)
        assert client._client is None
        http = client._get_client()
        assert http is not None
        assert client._client is http
        await client.close()
