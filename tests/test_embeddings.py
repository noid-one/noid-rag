"""Tests for embedding client."""


from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from noid_rag.config import EmbeddingConfig
from noid_rag.embeddings import EmbeddingClient
from noid_rag.models import Chunk


def _make_mock_client(responses):
    """Create a mock httpx.AsyncClient that returns the given responses in order.

    Each response is a dict with a "data" key containing embedding results.
    If a single response is given, it is used for all calls.
    """
    if isinstance(responses, dict):
        responses = [responses]

    mock_response_objects = []
    for resp_data in responses:
        mock_resp = MagicMock()
        mock_resp.json.return_value = resp_data
        mock_resp.raise_for_status = MagicMock()
        mock_response_objects.append(mock_resp)

    mock_client = AsyncMock()
    if len(mock_response_objects) == 1:
        mock_client.post.return_value = mock_response_objects[0]
    else:
        mock_client.post.side_effect = mock_response_objects

    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


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
    @patch("noid_rag.embeddings.httpx.AsyncClient")
    async def test_embed_texts_returns_embeddings(self, mock_client_cls, client):
        mock_client = _make_mock_client({
            "data": [
                {"index": 0, "embedding": [0.1] * 10},
                {"index": 1, "embedding": [0.2] * 10},
            ]
        })
        mock_client_cls.return_value = mock_client

        result = await client.embed_texts(["hello", "world"])
        assert len(result) == 2
        assert len(result[0]) == 10
        assert result[0] == [0.1] * 10
        assert result[1] == [0.2] * 10

    @pytest.mark.asyncio
    @patch("noid_rag.embeddings.httpx.AsyncClient")
    async def test_embed_texts_sorts_by_index(self, mock_client_cls, client):
        """API may return embeddings out of order; they should be sorted."""
        mock_client = _make_mock_client({
            "data": [
                {"index": 1, "embedding": [0.2] * 10},
                {"index": 0, "embedding": [0.1] * 10},
            ]
        })
        mock_client_cls.return_value = mock_client

        result = await client.embed_texts(["hello", "world"])
        assert result[0] == [0.1] * 10
        assert result[1] == [0.2] * 10

    @pytest.mark.asyncio
    @patch("noid_rag.embeddings.httpx.AsyncClient")
    async def test_batch_splitting(self, mock_client_cls, client):
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
        mock_client = _make_mock_client([resp1, resp2])
        mock_client_cls.return_value = mock_client

        result = await client.embed_texts(["a", "b", "c"])
        assert len(result) == 3
        assert result[2] == [0.3] * 10
        assert mock_client.post.call_count == 2


class TestEmbedChunks:
    @pytest.fixture
    def client(self):
        config = EmbeddingConfig(api_key="test-key", batch_size=2)
        return EmbeddingClient(config=config)

    @pytest.mark.asyncio
    @patch("noid_rag.embeddings.httpx.AsyncClient")
    async def test_embed_chunks_sets_embedding_in_place(self, mock_client_cls, client):
        chunks = [
            Chunk(text="chunk 1", document_id="doc_1"),
            Chunk(text="chunk 2", document_id="doc_1"),
        ]

        mock_client = _make_mock_client({
            "data": [
                {"index": 0, "embedding": [0.1] * 10},
                {"index": 1, "embedding": [0.2] * 10},
            ]
        })
        mock_client_cls.return_value = mock_client

        result = await client.embed_chunks(chunks)

        # Should return the same chunk objects
        assert result is chunks
        assert all(c.embedding is not None for c in result)
        assert chunks[0].embedding == [0.1] * 10
        assert chunks[1].embedding == [0.2] * 10

    @pytest.mark.asyncio
    @patch("noid_rag.embeddings.httpx.AsyncClient")
    async def test_embed_chunks_returns_chunks(self, mock_client_cls, client):
        chunks = [Chunk(text="test", document_id="doc_1")]

        mock_client = _make_mock_client({
            "data": [{"index": 0, "embedding": [0.5] * 10}]
        })
        mock_client_cls.return_value = mock_client

        result = await client.embed_chunks(chunks)
        assert len(result) == 1
        assert isinstance(result[0], Chunk)


class TestEmbedQuery:
    @pytest.fixture
    def client(self):
        config = EmbeddingConfig(api_key="test-key", batch_size=2)
        return EmbeddingClient(config=config)

    @pytest.mark.asyncio
    @patch("noid_rag.embeddings.httpx.AsyncClient")
    async def test_embed_query_returns_single_embedding(self, mock_client_cls, client):
        mock_client = _make_mock_client({
            "data": [{"index": 0, "embedding": [0.5] * 10}]
        })
        mock_client_cls.return_value = mock_client

        result = await client.embed_query("test query")
        assert isinstance(result, list)
        assert len(result) == 10
        assert result == [0.5] * 10
