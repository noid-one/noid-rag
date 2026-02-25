"""Tests for vector store (mocked -- no real DB)."""


from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from noid_rag.config import VectorStoreConfig
from noid_rag.models import Chunk, SearchResult
from noid_rag.vectorstore import PgVectorStore


def _make_search_result(chunk_id, score, text="text", doc_id="doc_1"):
    return SearchResult(
        chunk_id=chunk_id, text=text, score=score, metadata={}, document_id=doc_id
    )


class TestPgVectorStoreInit:
    def test_init_with_config(self):
        config = VectorStoreConfig(
            dsn="postgresql+asyncpg://test@localhost/test", embedding_dim=10
        )
        store = PgVectorStore(config=config)
        assert store.config.dsn == "postgresql+asyncpg://test@localhost/test"
        assert store.config.embedding_dim == 10
        assert store._engine is None

    def test_init_with_defaults(self):
        store = PgVectorStore()
        assert store.config.dsn == ""
        assert store.config.table_name == "documents"


def _make_async_cm(return_value):
    """Create a proper async context manager mock."""
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=return_value)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


class TestPgVectorStoreConnect:
    @pytest.fixture
    def config(self):
        return VectorStoreConfig(
            dsn="postgresql+asyncpg://test@localhost/test", embedding_dim=10
        )

    @pytest.mark.asyncio
    @patch("noid_rag.vectorstore.create_async_engine")
    async def test_connect_creates_engine(self, mock_create_engine, config):
        mock_conn = AsyncMock()
        mock_engine = MagicMock()
        mock_engine.begin.return_value = _make_async_cm(mock_conn)
        mock_create_engine.return_value = mock_engine

        store = PgVectorStore(config=config)
        await store.connect()

        mock_create_engine.assert_called_once()
        assert store._engine is not None


class TestPgVectorStoreContextManager:
    @pytest.mark.asyncio
    @patch("noid_rag.vectorstore.create_async_engine")
    async def test_context_manager(self, mock_create_engine):
        config = VectorStoreConfig(
            dsn="postgresql+asyncpg://test@localhost/test", embedding_dim=10
        )

        mock_conn = AsyncMock()
        mock_engine = MagicMock()
        mock_engine.begin.return_value = _make_async_cm(mock_conn)
        mock_engine.dispose = AsyncMock()
        mock_create_engine.return_value = mock_engine

        async with PgVectorStore(config=config) as store:
            assert store._engine is not None

        # After exiting, engine should be disposed
        mock_engine.dispose.assert_called_once()


class TestPgVectorStoreUpsert:
    @pytest.fixture
    def config(self):
        return VectorStoreConfig(
            dsn="postgresql+asyncpg://test@localhost/test", embedding_dim=10
        )

    @pytest.mark.asyncio
    async def test_upsert_requires_embedding(self, config):
        store = PgVectorStore(config=config)
        mock_conn = AsyncMock()
        store._engine = MagicMock()
        store._engine.begin.return_value = _make_async_cm(mock_conn)

        chunk = Chunk(text="test", document_id="doc_1", embedding=None)

        with pytest.raises(ValueError, match="no embedding"):
            await store.upsert([chunk])

    @pytest.mark.asyncio
    async def test_upsert_with_valid_chunks(self, config):
        store = PgVectorStore(config=config)
        mock_conn = AsyncMock()
        store._engine = MagicMock()
        store._engine.begin.return_value = _make_async_cm(mock_conn)

        chunks = [
            Chunk(text="chunk 1", document_id="doc_1", embedding=[0.1] * 10),
            Chunk(text="chunk 2", document_id="doc_1", embedding=[0.2] * 10),
        ]

        count = await store.upsert(chunks)
        assert count == 2
        assert mock_conn.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_upsert_empty_list(self, config):
        store = PgVectorStore(config=config)
        store._engine = MagicMock()
        count = await store.upsert([])
        assert count == 0


class TestPgVectorStoreSearch:
    @pytest.fixture
    def config(self):
        return VectorStoreConfig(
            dsn="postgresql+asyncpg://test@localhost/test", embedding_dim=10
        )

    @pytest.mark.asyncio
    async def test_search_returns_search_results(self, config):
        store = PgVectorStore(config=config)

        mock_row = MagicMock()
        mock_row.id = "chk_abc"
        mock_row.document_id = "doc_1"
        mock_row.text = "result text"
        mock_row.metadata = {"source_type": "pdf"}
        mock_row.score = 0.95

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]

        mock_conn = AsyncMock()
        mock_conn.execute.return_value = mock_result

        store._engine = MagicMock()
        store._engine.connect.return_value = _make_async_cm(mock_conn)

        results = await store.search([0.1] * 10, top_k=5)

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].chunk_id == "chk_abc"
        assert results[0].score == 0.95
        assert results[0].text == "result text"
        assert results[0].document_id == "doc_1"

    @pytest.mark.asyncio
    async def test_search_with_string_metadata(self, config):
        """metadata stored as string in DB should be JSON-parsed."""
        store = PgVectorStore(config=config)

        mock_row = MagicMock()
        mock_row.id = "chk_1"
        mock_row.document_id = "doc_1"
        mock_row.text = "text"
        mock_row.metadata = '{"key": "value"}'
        mock_row.score = 0.9

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]

        mock_conn = AsyncMock()
        mock_conn.execute.return_value = mock_result

        store._engine = MagicMock()
        store._engine.connect.return_value = _make_async_cm(mock_conn)

        results = await store.search([0.1] * 10)
        assert results[0].metadata == {"key": "value"}


class TestPgVectorStoreDelete:
    @pytest.mark.asyncio
    async def test_delete_by_document_id(self):
        config = VectorStoreConfig(
            dsn="postgresql+asyncpg://test@localhost/test", embedding_dim=10
        )
        store = PgVectorStore(config=config)

        mock_result = MagicMock()
        mock_result.rowcount = 3

        mock_conn = AsyncMock()
        mock_conn.execute.return_value = mock_result

        store._engine = MagicMock()
        store._engine.begin.return_value = _make_async_cm(mock_conn)

        count = await store.delete("doc_1")
        assert count == 3


class TestPgVectorStoreStats:
    @pytest.mark.asyncio
    async def test_stats_returns_dict(self):
        config = VectorStoreConfig(
            dsn="postgresql+asyncpg://test@localhost/test",
            embedding_dim=10,
            table_name="my_docs",
        )
        store = PgVectorStore(config=config)

        mock_total = MagicMock()
        mock_total.scalar.return_value = 100

        mock_docs = MagicMock()
        mock_docs.scalar.return_value = 10

        mock_conn = AsyncMock()
        mock_conn.execute.side_effect = [mock_total, mock_docs]

        store._engine = MagicMock()
        store._engine.connect.return_value = _make_async_cm(mock_conn)

        stats = await store.stats()
        assert stats["total_chunks"] == 100
        assert stats["total_documents"] == 10
        assert stats["table_name"] == "my_docs"
        assert stats["embedding_dim"] == 10


class TestPgVectorStoreClose:
    @pytest.mark.asyncio
    async def test_close_disposes_engine(self):
        config = VectorStoreConfig(
            dsn="postgresql+asyncpg://test@localhost/test", embedding_dim=10
        )
        store = PgVectorStore(config=config)
        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()
        store._engine = mock_engine

        await store.close()
        assert store._engine is None
        mock_engine.dispose.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_when_no_engine(self):
        store = PgVectorStore()
        await store.close()  # Should not raise
        assert store._engine is None


class TestPgVectorStoreKeywordSearch:
    @pytest.fixture
    def config(self):
        return VectorStoreConfig(
            dsn="postgresql+asyncpg://test@localhost/test", embedding_dim=10
        )

    @pytest.mark.asyncio
    async def test_keyword_search_returns_results(self, config):
        store = PgVectorStore(config=config)

        mock_row = MagicMock()
        mock_row.id = "chk_kw1"
        mock_row.document_id = "doc_1"
        mock_row.text = "keyword match text"
        mock_row.metadata = {"source_type": "pdf"}
        mock_row.score = 0.75

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]

        mock_conn = AsyncMock()
        mock_conn.execute.return_value = mock_result

        store._engine = MagicMock()
        store._engine.connect.return_value = _make_async_cm(mock_conn)

        results = await store.keyword_search("keyword match", top_k=5)

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].chunk_id == "chk_kw1"
        assert results[0].score == 0.75

        # Verify ts_rank query was used
        call_args = mock_conn.execute.call_args
        query_text = call_args[0][0].text
        assert "ts_rank" in query_text
        assert "plainto_tsquery" in query_text

    @pytest.mark.asyncio
    async def test_keyword_search_empty_results(self, config):
        store = PgVectorStore(config=config)

        mock_result = MagicMock()
        mock_result.fetchall.return_value = []

        mock_conn = AsyncMock()
        mock_conn.execute.return_value = mock_result

        store._engine = MagicMock()
        store._engine.connect.return_value = _make_async_cm(mock_conn)

        results = await store.keyword_search("nonexistent", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_keyword_search_with_filter_metadata(self, config):
        store = PgVectorStore(config=config)

        mock_row = MagicMock()
        mock_row.id = "chk_filtered"
        mock_row.document_id = "doc_1"
        mock_row.text = "filtered result"
        mock_row.metadata = {"source_type": "pdf"}
        mock_row.score = 0.6

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]

        mock_conn = AsyncMock()
        mock_conn.execute.return_value = mock_result

        store._engine = MagicMock()
        store._engine.connect.return_value = _make_async_cm(mock_conn)

        results = await store.keyword_search(
            "test", top_k=5, filter_metadata={"source_type": "pdf"}
        )

        assert len(results) == 1
        assert results[0].chunk_id == "chk_filtered"

        # Verify filter was included in query
        call_args = mock_conn.execute.call_args
        query_text = call_args[0][0].text
        assert "metadata->>'source_type'" in query_text
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
        assert params["filter_0"] == "pdf"

    @pytest.mark.asyncio
    async def test_keyword_search_rejects_unsafe_filter_key(self, config):
        store = PgVectorStore(config=config)
        store._engine = MagicMock()
        store._engine.connect.return_value = _make_async_cm(AsyncMock())

        with pytest.raises(ValueError, match="not a safe identifier"):
            await store.keyword_search(
                "test", filter_metadata={"Robert'; DROP TABLE --": "val"}
            )


class TestPgVectorStoreHybridSearch:
    @pytest.fixture
    def config(self):
        return VectorStoreConfig(
            dsn="postgresql+asyncpg://test@localhost/test", embedding_dim=10
        )

    @pytest.mark.asyncio
    async def test_hybrid_deduplicates_results(self, config):
        store = PgVectorStore(config=config)

        shared = _make_search_result("chk_shared", 0.9)
        vector_only = _make_search_result("chk_vec", 0.8)
        keyword_only = _make_search_result("chk_kw", 0.7)

        store.search = AsyncMock(return_value=[shared, vector_only])
        store.keyword_search = AsyncMock(return_value=[shared, keyword_only])

        results = await store.hybrid_search([0.1] * 10, "test", top_k=5)

        chunk_ids = [r.chunk_id for r in results]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicates found"
        assert "chk_shared" in chunk_ids
        assert "chk_vec" in chunk_ids
        assert "chk_kw" in chunk_ids

    @pytest.mark.asyncio
    async def test_hybrid_rrf_scoring(self, config):
        """Shared result should rank highest due to RRF from both lists."""
        store = PgVectorStore(config=config)

        shared = _make_search_result("chk_shared", 0.9)
        vec_only = _make_search_result("chk_vec", 0.8)
        kw_only = _make_search_result("chk_kw", 0.7)

        store.search = AsyncMock(return_value=[shared, vec_only])
        store.keyword_search = AsyncMock(return_value=[shared, kw_only])

        results = await store.hybrid_search([0.1] * 10, "test", top_k=5, rrf_k=60)

        # Shared result appears in both lists, so it should have the highest RRF score
        assert results[0].chunk_id == "chk_shared"
        # Its RRF score = 1/(60+1) + 1/(60+1) = 2/61
        expected_shared = 2.0 / 61.0
        assert abs(results[0].score - expected_shared) < 1e-9

    @pytest.mark.asyncio
    async def test_hybrid_respects_top_k(self, config):
        store = PgVectorStore(config=config)

        vec_results = [_make_search_result(f"v{i}", 0.9 - i * 0.1) for i in range(5)]
        kw_results = [_make_search_result(f"k{i}", 0.8 - i * 0.1) for i in range(5)]

        store.search = AsyncMock(return_value=vec_results)
        store.keyword_search = AsyncMock(return_value=kw_results)

        results = await store.hybrid_search([0.1] * 10, "test", top_k=3)
        assert len(results) == 3
