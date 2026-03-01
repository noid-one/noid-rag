"""Tests for the zvec vector store (mocked -- no real zvec)."""

from unittest.mock import MagicMock, patch

import pytest

from noid_rag.config import ZvecConfig
from noid_rag.models import Chunk, SearchResult


def _make_mock_zvec():
    """Create a mock zvec module."""
    zvec = MagicMock()
    zvec.Schema.return_value = MagicMock()
    zvec.Field.side_effect = lambda name, ftype: MagicMock(name=name, field_type=ftype)
    zvec.VectorField.side_effect = lambda name, vtype, *args: MagicMock(name=name)
    zvec.FieldType.STRING = "STRING"
    zvec.VectorType.FP32 = "FP32"
    zvec.VectorType.SPARSE_FP32 = "SPARSE_FP32"
    zvec.VectorQuery.side_effect = lambda name, **kwargs: MagicMock(name=name, **kwargs)
    zvec.Doc.side_effect = lambda **kwargs: kwargs
    zvec.BM25EmbeddingFunction.return_value = MagicMock()
    zvec.RrfReRanker.side_effect = lambda **kwargs: MagicMock(**kwargs)
    zvec.Collection = MagicMock()
    zvec.DefaultLocalDenseEmbedding.return_value = MagicMock()
    return zvec


@pytest.fixture(autouse=True)
def mock_zvec_import():
    """Patch _import_zvec to avoid needing zvec installed."""
    mock_zvec = _make_mock_zvec()
    with patch(
        "noid_rag.vectorstore_zvec._import_zvec",
        return_value=mock_zvec,
    ):
        yield mock_zvec


@pytest.fixture
def mock_collection():
    """A mock zvec collection."""
    collection = MagicMock()
    collection.upsert = MagicMock()
    collection.search_by_filter = MagicMock(return_value=[])
    collection.delete_by_filter = MagicMock()
    collection.query = MagicMock(return_value=[])
    collection.destroy = MagicMock()
    collection.close = MagicMock()
    return collection


@pytest.fixture
def config():
    return ZvecConfig(data_dir="/tmp/test-zvec", collection_name="test_docs")


@pytest.fixture
def store(config, mock_collection, mock_zvec_import):
    from noid_rag.vectorstore_zvec import ZvecVectorStore

    s = ZvecVectorStore(config=config, embedding_dim=10)
    s._collection = mock_collection
    s._zvec = mock_zvec_import
    s._bm25_fn = mock_zvec_import.BM25EmbeddingFunction()
    return s


class TestZvecVectorStoreInit:
    def test_init_with_config(self):
        from noid_rag.vectorstore_zvec import ZvecVectorStore

        config = ZvecConfig(data_dir="/tmp/zvec-data", collection_name="my_docs")
        s = ZvecVectorStore(config=config, embedding_dim=384)
        assert s.config.data_dir == "/tmp/zvec-data"
        assert s.config.collection_name == "my_docs"
        assert s.embedding_dim == 384
        assert s._collection is None

    def test_init_with_defaults(self):
        from noid_rag.vectorstore_zvec import ZvecVectorStore

        s = ZvecVectorStore()
        assert s.config.data_dir == "~/.noid-rag/zvec"
        assert s.config.collection_name == "documents"
        assert s.embedding_dim == 384


class TestZvecVectorStoreUpsert:
    @pytest.mark.asyncio
    async def test_upsert_with_valid_chunks(self, store, mock_collection):
        chunks = [
            Chunk(text="chunk 1", document_id="doc_1", embedding=[0.1] * 10),
            Chunk(text="chunk 2", document_id="doc_1", embedding=[0.2] * 10),
        ]

        count = await store.upsert(chunks)
        assert count == 2
        mock_collection.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_empty_list(self, store):
        count = await store.upsert([])
        assert count == 0

    @pytest.mark.asyncio
    async def test_upsert_requires_embedding(self, store):
        chunk = Chunk(text="test", document_id="doc_1", embedding=None)

        with pytest.raises(ValueError, match="no embedding"):
            await store.upsert([chunk])


class TestZvecVectorStoreReplaceDocument:
    @pytest.mark.asyncio
    async def test_replace_document_with_existing(self, store, mock_collection):
        # Simulate 2 existing chunks
        existing = [
            {"chunk_id": "old_1", "document_id": "doc_1"},
            {"chunk_id": "old_2", "document_id": "doc_1"},
        ]
        mock_collection.search_by_filter.return_value = existing

        chunks = [
            Chunk(text="new chunk", document_id="doc_1", embedding=[0.1] * 10),
        ]

        deleted, inserted = await store.replace_document("doc_1", chunks)
        assert deleted == 2
        assert inserted == 1

    @pytest.mark.asyncio
    async def test_replace_document_no_existing(self, store, mock_collection):
        mock_collection.search_by_filter.return_value = []

        chunks = [
            Chunk(text="first chunk", document_id="doc_new", embedding=[0.1] * 10),
        ]

        deleted, inserted = await store.replace_document("doc_new", chunks)
        assert deleted == 0
        assert inserted == 1
        mock_collection.delete_by_filter.assert_not_called()


class TestZvecVectorStoreSearch:
    @pytest.mark.asyncio
    async def test_search_returns_results(self, store, mock_collection):
        mock_collection.query.return_value = [
            (
                {
                    "chunk_id": "chk_abc",
                    "document_id": "doc_1",
                    "text": "result text",
                    "metadata_json": '{"source_type": "pdf"}',
                },
                0.95,
            )
        ]

        results = await store.search([0.1] * 10, top_k=5)

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].chunk_id == "chk_abc"
        assert results[0].score == 0.95
        assert results[0].document_id == "doc_1"

    @pytest.mark.asyncio
    async def test_search_empty_results(self, store, mock_collection):
        mock_collection.query.return_value = []

        results = await store.search([0.1] * 10, top_k=5)
        assert results == []


class TestZvecVectorStoreKeywordSearch:
    @pytest.mark.asyncio
    async def test_keyword_search_returns_results(self, store, mock_collection):
        mock_collection.query.return_value = [
            (
                {
                    "chunk_id": "chk_kw",
                    "document_id": "doc_1",
                    "text": "keyword result",
                    "metadata_json": "{}",
                },
                0.8,
            )
        ]

        results = await store.keyword_search("my search terms", top_k=5)

        assert len(results) == 1
        assert results[0].chunk_id == "chk_kw"

    @pytest.mark.asyncio
    async def test_keyword_search_no_bm25(self, store):
        store._bm25_fn = None
        results = await store.keyword_search("test", top_k=5)
        assert results == []


class TestZvecVectorStoreHybridSearch:
    @pytest.mark.asyncio
    async def test_hybrid_search_native(self, store, mock_collection, mock_zvec_import):
        mock_collection.query.return_value = [
            (
                {
                    "chunk_id": "chk_1",
                    "document_id": "doc_1",
                    "text": "hybrid result",
                    "metadata_json": "{}",
                },
                0.9,
            )
        ]

        results = await store.hybrid_search([0.1] * 10, "test query", top_k=5)

        assert len(results) == 1
        assert results[0].chunk_id == "chk_1"

    @pytest.mark.asyncio
    async def test_hybrid_search_fallback(self, store, mock_collection, mock_zvec_import):
        """Falls back to Python RRF when native hybrid fails."""
        # Remove RrfReRanker to trigger fallback
        del mock_zvec_import.RrfReRanker

        mock_collection.query.return_value = [
            (
                {
                    "chunk_id": "chk_1",
                    "document_id": "doc_1",
                    "text": "text 1",
                    "metadata_json": "{}",
                },
                0.9,
            )
        ]

        results = await store.hybrid_search([0.1] * 10, "test query", top_k=5)
        assert len(results) >= 1


class TestZvecVectorStoreDelete:
    @pytest.mark.asyncio
    async def test_delete_by_document_id(self, store, mock_collection):
        mock_collection.search_by_filter.return_value = [
            {"chunk_id": f"chk_{i}"} for i in range(3)
        ]

        count = await store.delete("doc_1")
        assert count == 3
        mock_collection.delete_by_filter.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_no_existing(self, store, mock_collection):
        mock_collection.search_by_filter.return_value = []

        count = await store.delete("doc_missing")
        assert count == 0
        mock_collection.delete_by_filter.assert_not_called()


class TestZvecVectorStoreDrop:
    @pytest.mark.asyncio
    async def test_drop_destroys_collection(self, store, mock_collection):
        with patch("shutil.rmtree"):
            await store.drop()
        mock_collection.destroy.assert_called_once()


class TestZvecVectorStoreStats:
    @pytest.mark.asyncio
    async def test_stats_returns_dict(self, store, mock_collection):
        mock_collection.search_by_filter.return_value = [
            {"chunk_id": "chk_1", "document_id": "doc_a", "text": "t1", "metadata_json": "{}"},
            {"chunk_id": "chk_2", "document_id": "doc_a", "text": "t2", "metadata_json": "{}"},
            {"chunk_id": "chk_3", "document_id": "doc_b", "text": "t3", "metadata_json": "{}"},
        ]

        stats = await store.stats()
        assert stats["total_chunks"] == 3
        assert stats["total_documents"] == 2
        assert stats["store_name"] == "test_docs"
        assert stats["embedding_dim"] == 10

    @pytest.mark.asyncio
    async def test_stats_empty_collection(self, store, mock_collection):
        mock_collection.search_by_filter.return_value = []

        stats = await store.stats()
        assert stats["total_chunks"] == 0
        assert stats["total_documents"] == 0


class TestZvecVectorStoreNotConnected:
    @pytest.mark.asyncio
    async def test_search_raises_if_not_connected(self):
        from noid_rag.vectorstore_zvec import ZvecVectorStore

        store = ZvecVectorStore()
        with pytest.raises(RuntimeError, match="not connected"):
            await store.search([0.1] * 10)

    @pytest.mark.asyncio
    async def test_delete_raises_if_not_connected(self):
        from noid_rag.vectorstore_zvec import ZvecVectorStore

        store = ZvecVectorStore()
        with pytest.raises(RuntimeError, match="not connected"):
            await store.delete("doc_1")


class TestZvecVectorStoreClose:
    @pytest.mark.asyncio
    async def test_close_releases_collection(self, store, mock_collection):
        await store.close()
        mock_collection.close.assert_called_once()
        assert store._collection is None

    @pytest.mark.asyncio
    async def test_close_when_no_collection(self):
        from noid_rag.vectorstore_zvec import ZvecVectorStore

        s = ZvecVectorStore()
        await s.close()
        assert s._collection is None


class TestZvecVectorStoreSampleChunks:
    @pytest.mark.asyncio
    async def test_sample_random(self, store, mock_collection):
        mock_collection.search_by_filter.return_value = [
            {
                "chunk_id": "chk_1",
                "document_id": "doc_1",
                "text": "sample text",
                "metadata_json": "{}",
            },
        ]

        results = await store.sample_chunks(limit=5, strategy="random")

        assert len(results) == 1
        assert results[0]["id"] == "chk_1"

    @pytest.mark.asyncio
    async def test_sample_diverse(self, store, mock_collection):
        mock_collection.search_by_filter.return_value = [
            {
                "chunk_id": f"chk_{i}",
                "document_id": f"doc_{i % 2}",
                "text": f"text {i}",
                "metadata_json": "{}",
            }
            for i in range(4)
        ]

        results = await store.sample_chunks(limit=3, strategy="diverse")

        assert len(results) == 3
        doc_ids = [r["document_id"] for r in results]
        assert "doc_0" in doc_ids
        assert "doc_1" in doc_ids


class TestZvecRrfMerge:
    def test_rrf_merge_basic(self):
        from noid_rag.vectorstore_zvec import ZvecVectorStore

        dense = [
            SearchResult(chunk_id="a", text="a", score=1.0, metadata={}, document_id="d1"),
            SearchResult(chunk_id="b", text="b", score=0.9, metadata={}, document_id="d1"),
        ]
        sparse = [
            SearchResult(chunk_id="b", text="b", score=1.0, metadata={}, document_id="d1"),
            SearchResult(chunk_id="c", text="c", score=0.9, metadata={}, document_id="d2"),
        ]

        merged = ZvecVectorStore._rrf_merge(dense, sparse, top_k=3, rrf_k=60)
        assert len(merged) == 3
        # "b" appears in both lists so should have highest score
        assert merged[0].chunk_id == "b"

    def test_rrf_merge_respects_top_k(self):
        from noid_rag.vectorstore_zvec import ZvecVectorStore

        dense = [
            SearchResult(
                chunk_id=f"chk_{i}", text=f"t{i}", score=1.0, metadata={}, document_id="d1"
            )
            for i in range(5)
        ]
        merged = ZvecVectorStore._rrf_merge(dense, [], top_k=2)
        assert len(merged) == 2
