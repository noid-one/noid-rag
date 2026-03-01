"""Tests for the zvec vector store (mocked -- no real zvec)."""

from unittest.mock import MagicMock, patch

import pytest

from noid_rag.config import ZvecConfig
from noid_rag.models import Chunk, SearchResult


def _make_mock_zvec():
    """Create a mock zvec module matching the real zvec 0.2.0 API."""
    zvec = MagicMock()
    # Schema types
    zvec.CollectionSchema.return_value = MagicMock()
    zvec.FieldSchema.side_effect = lambda name, dtype, **kw: MagicMock(name=name)
    zvec.VectorSchema.side_effect = lambda name, dtype, **kw: MagicMock(name=name)
    zvec.DataType.STRING = "STRING"
    zvec.DataType.VECTOR_FP32 = "VECTOR_FP32"
    zvec.DataType.SPARSE_VECTOR_FP32 = "SPARSE_VECTOR_FP32"
    # Index
    zvec.HnswIndexParam.return_value = MagicMock(m=16, ef_construction=200)
    # Query
    zvec.VectorQuery.side_effect = lambda **kwargs: MagicMock(**kwargs)
    zvec.RrfReRanker.side_effect = lambda rank_constant=60, topn=10, **kwargs: MagicMock(
        rank_constant=rank_constant, topn=topn
    )
    # Doc: replicate the real Doc(id, fields, vectors, score) shape
    zvec.Doc.side_effect = lambda **kwargs: MagicMock(**kwargs)
    # Collection creation
    zvec.create_and_open.return_value = MagicMock()
    zvec.open.return_value = MagicMock()
    # BM25 (with encoding_type parameter)
    bm25_fn = MagicMock()
    bm25_fn.embed.return_value = [[0.1, 0.2]]
    zvec.BM25EmbeddingFunction.return_value = bm25_fn
    return zvec


def _make_mock_splade():
    """Create a mock _SpladeEncoder that returns sparse dicts."""
    splade_fn = MagicMock()
    splade_fn.embed.return_value = {10: 0.5, 42: 1.2}
    return splade_fn


def _make_doc(chunk_id: str, document_id: str, text: str, score: float = 0.0):
    """Create a mock zvec Doc with .id, .score, .fields matching the real API."""
    doc = MagicMock()
    doc.id = chunk_id
    doc.score = score
    doc.fields = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "text": text,
        "metadata_json": "{}",
    }
    doc.vectors = {}
    return doc


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
    collection.optimize = MagicMock()
    collection.query = MagicMock(return_value=[])
    collection.delete_by_filter = MagicMock()
    collection.destroy = MagicMock()
    collection.flush = MagicMock()
    # stats is a property returning CollectionStats with .doc_count
    stats_obj = MagicMock()
    stats_obj.doc_count = 0
    type(collection).stats = property(lambda self: stats_obj)
    collection._stats_obj = stats_obj
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
    # Separate doc/query instances for BM25
    bm25_fn = mock_zvec_import.BM25EmbeddingFunction()
    s._bm25_doc_fn = bm25_fn
    s._bm25_query_fn = bm25_fn
    # SPLADE mock
    splade_fn = _make_mock_splade()
    s._splade_doc_fn = splade_fn
    s._splade_query_fn = splade_fn
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


class TestZvecVectorStoreConnect:
    @pytest.mark.asyncio
    async def test_connect_loads_splade_encoder(self, config, mock_zvec_import):
        from noid_rag.vectorstore_zvec import ZvecVectorStore, _SpladeEncoder

        mock_encoder = MagicMock()
        with patch.object(_SpladeEncoder, "_get_encoder", return_value=mock_encoder):
            store = ZvecVectorStore(config=config, embedding_dim=10)
            await store.connect()
            assert store._splade_doc_fn is not None
            assert store._splade_query_fn is not None

    @pytest.mark.asyncio
    async def test_connect_splade_unavailable_falls_back(self, config, mock_zvec_import):
        from noid_rag.vectorstore_zvec import ZvecVectorStore, _SpladeEncoder

        with patch.object(
            _SpladeEncoder, "_get_encoder", side_effect=ImportError("no sentence-transformers")
        ):
            store = ZvecVectorStore(config=config, embedding_dim=10)
            await store.connect()
            assert store._splade_doc_fn is None
            assert store._splade_query_fn is None


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
    async def test_upsert_calls_optimize(self, store, mock_collection):
        """optimize() must be called after upsert to build HNSW index."""
        chunks = [
            Chunk(text="chunk 1", document_id="doc_1", embedding=[0.1] * 10),
        ]

        await store.upsert(chunks)
        mock_collection.optimize.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_skip_optimize(self, store, mock_collection):
        """optimize=False skips the HNSW index build for batch callers."""
        chunks = [
            Chunk(text="chunk 1", document_id="doc_1", embedding=[0.1] * 10),
        ]

        await store.upsert(chunks, optimize=False)
        mock_collection.optimize.assert_not_called()

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
        existing = [
            _make_doc("old_1", "doc_1", "old text 1"),
            _make_doc("old_2", "doc_1", "old text 2"),
        ]
        mock_collection.query.return_value = existing

        chunks = [
            Chunk(text="new chunk", document_id="doc_1", embedding=[0.1] * 10),
        ]

        deleted, inserted = await store.replace_document("doc_1", chunks)
        assert deleted == 2
        assert inserted == 1

    @pytest.mark.asyncio
    async def test_replace_document_no_existing(self, store, mock_collection):
        mock_collection.query.return_value = []

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
            _make_doc("chk_abc", "doc_1", "result text", score=0.95),
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
            _make_doc("chk_kw", "doc_1", "keyword result", score=0.8),
        ]

        results = await store.keyword_search("my search terms", top_k=5)

        assert len(results) == 1
        assert results[0].chunk_id == "chk_kw"

    @pytest.mark.asyncio
    async def test_keyword_search_no_bm25(self, store):
        store._bm25_query_fn = None
        results = await store.keyword_search("test", top_k=5)
        assert results == []


class TestZvecVectorStoreHybridSearch:
    @pytest.mark.asyncio
    async def test_hybrid_search_native_three_way(self, store, mock_collection):
        """Native multi-vector search with dense + SPLADE + BM25."""
        mock_collection.query.return_value = [
            _make_doc("chk_1", "doc_1", "hybrid result", score=0.9),
        ]

        results = await store.hybrid_search([0.1] * 10, "test query", top_k=5)

        assert len(results) == 1
        assert results[0].chunk_id == "chk_1"

    @pytest.mark.asyncio
    async def test_hybrid_search_without_splade(self, store, mock_collection):
        """Falls back to dense + BM25 when SPLADE is unavailable."""
        store._splade_query_fn = None

        mock_collection.query.return_value = [
            _make_doc("chk_1", "doc_1", "text 1", score=0.9),
        ]

        results = await store.hybrid_search([0.1] * 10, "test query", top_k=5)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_hybrid_search_fallback(self, store, mock_collection, mock_zvec_import):
        """Falls back to Python RRF when native hybrid fails."""
        del mock_zvec_import.RrfReRanker

        mock_collection.query.return_value = [
            _make_doc("chk_1", "doc_1", "text 1", score=0.9),
        ]

        results = await store.hybrid_search([0.1] * 10, "test query", top_k=5)
        assert len(results) >= 1


class TestZvecVectorStoreDelete:
    @pytest.mark.asyncio
    async def test_delete_by_document_id(self, store, mock_collection):
        mock_collection.query.return_value = [
            _make_doc(f"chk_{i}", "doc_1", f"text {i}") for i in range(3)
        ]

        count = await store.delete("doc_1")
        assert count == 3
        mock_collection.delete_by_filter.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_no_existing(self, store, mock_collection):
        mock_collection.query.return_value = []

        count = await store.delete("doc_missing")
        assert count == 0
        mock_collection.delete_by_filter.assert_not_called()


class TestZvecSafeFilter:
    def test_safe_filter_accepts_hex_ids(self):
        from noid_rag.vectorstore_zvec import _safe_filter

        result = _safe_filter("doc_577a0c0789ba")
        assert result == "document_id='doc_577a0c0789ba'"

    def test_safe_filter_rejects_single_quotes(self):
        from noid_rag.vectorstore_zvec import _safe_filter

        with pytest.raises(ValueError, match="Unsafe document_id"):
            _safe_filter("it's a doc")

    def test_safe_filter_rejects_sql_injection(self):
        from noid_rag.vectorstore_zvec import _safe_filter

        with pytest.raises(ValueError, match="Unsafe document_id"):
            _safe_filter("'; DROP TABLE --")


class TestZvecVectorStoreDrop:
    @pytest.mark.asyncio
    async def test_drop_destroys_collection(self, store, mock_collection):
        with patch("shutil.rmtree"):
            await store.drop()
        mock_collection.destroy.assert_called_once()


class TestZvecVectorStoreStats:
    @pytest.mark.asyncio
    async def test_stats_returns_dict(self, store, mock_collection):
        mock_collection._stats_obj.doc_count = 3
        mock_collection.query.return_value = [
            _make_doc("chk_1", "doc_a", "t1"),
            _make_doc("chk_2", "doc_a", "t2"),
            _make_doc("chk_3", "doc_b", "t3"),
        ]

        stats = await store.stats()
        assert stats["total_chunks"] == 3
        assert stats["total_documents"] == 2
        assert stats["store_name"] == "test_docs"
        assert stats["embedding_dim"] == 10

    @pytest.mark.asyncio
    async def test_stats_empty_collection(self, store, mock_collection):
        mock_collection._stats_obj.doc_count = 0

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
    async def test_close_flushes_and_releases(self, store, mock_collection):
        await store.close()
        mock_collection.flush.assert_called_once()
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
        mock_collection.query.return_value = [
            _make_doc("chk_1", "doc_1", "sample text"),
        ]

        results = await store.sample_chunks(limit=5, strategy="random")

        assert len(results) == 1
        assert results[0]["id"] == "chk_1"

    @pytest.mark.asyncio
    async def test_sample_diverse(self, store, mock_collection):
        mock_collection.query.return_value = [
            _make_doc(f"chk_{i}", f"doc_{i % 2}", f"text {i}")
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
            SearchResult(
                chunk_id="a", text="a", score=1.0, metadata={}, document_id="d1"
            ),
            SearchResult(
                chunk_id="b", text="b", score=0.9, metadata={}, document_id="d1"
            ),
        ]
        sparse = [
            SearchResult(
                chunk_id="b", text="b", score=1.0, metadata={}, document_id="d1"
            ),
            SearchResult(
                chunk_id="c", text="c", score=0.9, metadata={}, document_id="d2"
            ),
        ]

        merged = ZvecVectorStore._rrf_merge(dense, sparse, top_k=3, rrf_k=60)
        assert len(merged) == 3
        # "b" appears in both lists so should have highest score
        assert merged[0].chunk_id == "b"

    def test_rrf_merge_respects_top_k(self):
        from noid_rag.vectorstore_zvec import ZvecVectorStore

        dense = [
            SearchResult(
                chunk_id=f"chk_{i}",
                text=f"t{i}",
                score=1.0,
                metadata={},
                document_id="d1",
            )
            for i in range(5)
        ]
        merged = ZvecVectorStore._rrf_merge(dense, [], top_k=2)
        assert len(merged) == 2
