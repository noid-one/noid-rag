"""Tests for the Qdrant vector store (mocked -- no real Qdrant)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from noid_rag.config import QdrantConfig
from noid_rag.models import Chunk, SearchResult


def _make_mock_models():
    """Create a mock qdrant_client.models module."""
    m = MagicMock()
    # Make model constructors return MagicMock instances with the right attributes
    m.VectorParams.return_value = MagicMock()
    m.Distance.COSINE = "Cosine"
    m.HnswConfigDiff.return_value = MagicMock()
    m.SparseVectorParams.return_value = MagicMock()
    m.Modifier.IDF = "idf"
    m.PayloadSchemaType.KEYWORD = "keyword"
    m.PointStruct.side_effect = lambda **kwargs: MagicMock(**kwargs)
    m.Filter.side_effect = lambda **kwargs: MagicMock(**kwargs)
    m.FieldCondition.side_effect = lambda **kwargs: MagicMock(**kwargs)
    m.MatchValue.side_effect = lambda **kwargs: MagicMock(**kwargs)
    m.FilterSelector.side_effect = lambda **kwargs: MagicMock(**kwargs)
    m.SparseVector.side_effect = lambda **kwargs: MagicMock(**kwargs)
    m.Document.side_effect = lambda **kwargs: MagicMock(**kwargs)
    m.Prefetch.side_effect = lambda **kwargs: MagicMock(**kwargs)
    m.FusionQuery.side_effect = lambda **kwargs: MagicMock(**kwargs)
    m.Fusion.RRF = "rrf"
    return m


@pytest.fixture(autouse=True)
def mock_qdrant_import():
    """Patch _import_qdrant to avoid needing qdrant-client installed."""
    mock_client_cls = MagicMock()
    mock_models = _make_mock_models()
    with patch(
        "noid_rag.vectorstore_qdrant._import_qdrant",
        return_value=(mock_client_cls, mock_models),
    ):
        yield mock_client_cls, mock_models


@pytest.fixture
def mock_client():
    """A mock AsyncQdrantClient instance."""
    client = AsyncMock()
    client.collection_exists.return_value = False
    return client


@pytest.fixture
def config():
    return QdrantConfig(url="http://localhost:6333", collection_name="test_docs")


@pytest.fixture
def store(config, mock_client, mock_qdrant_import):
    from noid_rag.vectorstore_qdrant import QdrantVectorStore

    _, mock_models = mock_qdrant_import
    s = QdrantVectorStore(config=config, embedding_dim=10)
    s._client = mock_client
    s._models = mock_models
    return s


class TestQdrantVectorStoreInit:
    def test_init_with_config(self):
        from noid_rag.vectorstore_qdrant import QdrantVectorStore

        config = QdrantConfig(url="http://qdrant:6333", collection_name="my_docs")
        s = QdrantVectorStore(config=config, embedding_dim=768)
        assert s.config.url == "http://qdrant:6333"
        assert s.config.collection_name == "my_docs"
        assert s.embedding_dim == 768
        assert s._client is None

    def test_init_with_defaults(self):
        from noid_rag.vectorstore_qdrant import QdrantVectorStore

        s = QdrantVectorStore()
        assert s.config.url == "http://localhost:6333"
        assert s.config.collection_name == "documents"
        assert s.embedding_dim == 1536


class TestQdrantVectorStoreConnect:
    @pytest.mark.asyncio
    async def test_connect_creates_collection(
        self, config, mock_qdrant_import
    ):
        from noid_rag.vectorstore_qdrant import QdrantVectorStore

        mock_client_cls, _ = mock_qdrant_import
        mock_instance = AsyncMock()
        mock_instance.collection_exists.return_value = False
        mock_client_cls.return_value = mock_instance

        s = QdrantVectorStore(config=config, embedding_dim=10)
        await s.connect()

        assert s._client is not None
        mock_instance.create_collection.assert_called_once()
        mock_instance.create_payload_index.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_skips_if_collection_exists(
        self, config, mock_qdrant_import
    ):
        from noid_rag.vectorstore_qdrant import QdrantVectorStore

        mock_client_cls, _ = mock_qdrant_import
        mock_instance = AsyncMock()
        mock_instance.collection_exists.return_value = True
        mock_client_cls.return_value = mock_instance

        s = QdrantVectorStore(config=config, embedding_dim=10)
        await s.connect()

        mock_instance.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_tolerates_concurrent_creator(
        self, config, mock_qdrant_import
    ):
        """When another process creates the collection between exists-check and create,
        the 'already exists' error is caught but payload index is still attempted."""
        from noid_rag.vectorstore_qdrant import QdrantVectorStore

        mock_client_cls, _ = mock_qdrant_import
        mock_instance = AsyncMock()
        mock_instance.collection_exists.return_value = False
        mock_instance.create_collection.side_effect = Exception("collection already exists")
        mock_client_cls.return_value = mock_instance

        s = QdrantVectorStore(config=config, embedding_dim=10)
        await s.connect()  # must not raise

        # Payload index should still be attempted even when collection was created concurrently
        mock_instance.create_payload_index.assert_called_once()


class TestQdrantVectorStoreUpsert:
    @pytest.mark.asyncio
    async def test_upsert_with_valid_chunks(self, store, mock_client):
        chunks = [
            Chunk(text="chunk 1", document_id="doc_1", embedding=[0.1] * 10),
            Chunk(text="chunk 2", document_id="doc_1", embedding=[0.2] * 10),
        ]

        count = await store.upsert(chunks)
        assert count == 2
        mock_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_empty_list(self, store):
        count = await store.upsert([])
        assert count == 0

    @pytest.mark.asyncio
    async def test_upsert_requires_embedding(self, store):
        chunk = Chunk(text="test", document_id="doc_1", embedding=None)

        with pytest.raises(ValueError, match="no embedding"):
            await store.upsert([chunk])


class TestQdrantVectorStoreReplaceDocument:
    @pytest.mark.asyncio
    async def test_replace_document_inserts_before_deleting(self, store, mock_client):
        """Upsert must be called before delete to avoid data loss on failure."""
        call_order = []

        async def track_upsert(**kwargs):
            call_order.append("upsert")

        async def track_delete(**kwargs):
            call_order.append("delete")

        mock_client.upsert.side_effect = track_upsert
        mock_client.delete.side_effect = track_delete

        # scroll returns 3 existing point IDs then signals end-of-results
        old_point_1 = MagicMock()
        old_point_1.id = "uuid-old-1"
        old_point_2 = MagicMock()
        old_point_2.id = "uuid-old-2"
        old_point_3 = MagicMock()
        old_point_3.id = "uuid-old-3"
        mock_client.scroll.return_value = ([old_point_1, old_point_2, old_point_3], None)

        chunks = [
            Chunk(text="new chunk 1", document_id="doc_1", embedding=[0.1] * 10),
            Chunk(text="new chunk 2", document_id="doc_1", embedding=[0.2] * 10),
        ]

        deleted, inserted = await store.replace_document("doc_1", chunks)
        assert deleted == 3
        assert inserted == 2
        assert call_order == ["upsert", "delete"], (
            "upsert must precede delete to prevent data loss on failure"
        )

    @pytest.mark.asyncio
    async def test_replace_document_deletes_by_point_ids_not_filter(
        self, store, mock_client, mock_qdrant_import
    ):
        """Delete must target the pre-existing point IDs, not a document_id filter.

        Deleting by document_id filter would also sweep up the newly inserted
        chunks that share the same document_id, silently emptying the document.
        """
        _, mock_models = mock_qdrant_import

        old_point = MagicMock()
        old_point.id = "uuid-old-1"
        mock_client.scroll.return_value = ([old_point], None)

        chunks = [
            Chunk(text="new chunk", document_id="doc_1", embedding=[0.1] * 10),
        ]

        await store.replace_document("doc_1", chunks)

        # The delete call must use PointIdsList (specific IDs), not FilterSelector
        mock_models.PointIdsList.assert_called_once_with(points=["uuid-old-1"])
        mock_models.FilterSelector.assert_not_called()

    @pytest.mark.asyncio
    async def test_replace_document_empty_chunks(self, store, mock_client):
        # scroll returns 2 existing points
        old_point_1 = MagicMock()
        old_point_1.id = "uuid-old-1"
        old_point_2 = MagicMock()
        old_point_2.id = "uuid-old-2"
        mock_client.scroll.return_value = ([old_point_1, old_point_2], None)

        deleted, inserted = await store.replace_document("doc_1", [])
        assert deleted == 2
        assert inserted == 0
        # When there are no new chunks to insert, old chunks are still deleted
        mock_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_replace_document_no_existing_chunks_skips_delete(self, store, mock_client):
        """When no existing chunks are found, delete must not be called at all."""
        mock_client.scroll.return_value = ([], None)

        chunks = [
            Chunk(text="first chunk", document_id="doc_new", embedding=[0.1] * 10),
        ]

        deleted, inserted = await store.replace_document("doc_new", chunks)
        assert deleted == 0
        assert inserted == 1
        mock_client.delete.assert_not_called()


class TestQdrantVectorStoreSearch:
    @pytest.mark.asyncio
    async def test_search_returns_results(self, store, mock_client):
        mock_point = MagicMock()
        mock_point.id = "point-uuid"
        mock_point.score = 0.95
        mock_point.payload = {
            "chunk_id": "chk_abc",
            "document_id": "doc_1",
            "text": "result text",
            "metadata": {"source_type": "pdf"},
        }

        query_result = MagicMock()
        query_result.points = [mock_point]
        mock_client.query_points.return_value = query_result

        results = await store.search([0.1] * 10, top_k=5)

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].chunk_id == "chk_abc"
        assert results[0].score == 0.95
        assert results[0].document_id == "doc_1"


class TestQdrantVectorStoreHybridSearch:
    @pytest.mark.asyncio
    async def test_hybrid_search_uses_prefetch(self, store, mock_client):
        mock_point = MagicMock()
        mock_point.id = "point-uuid"
        mock_point.score = 0.9
        mock_point.payload = {
            "chunk_id": "chk_1",
            "document_id": "doc_1",
            "text": "hybrid result",
            "metadata": {},
        }

        query_result = MagicMock()
        query_result.points = [mock_point]
        mock_client.query_points.return_value = query_result

        results = await store.hybrid_search([0.1] * 10, "test query", top_k=5)

        assert len(results) == 1
        assert results[0].chunk_id == "chk_1"
        call_kwargs = mock_client.query_points.call_args.kwargs
        assert "prefetch" in call_kwargs
        assert len(call_kwargs["prefetch"]) == 2
        # Top-level query_filter should be set (belt-and-suspenders at fusion stage)
        assert "query_filter" in call_kwargs

    @pytest.mark.asyncio
    async def test_hybrid_search_rejects_unsafe_filter_key(self, store, mock_client):
        with pytest.raises(ValueError, match="not a safe identifier"):
            await store.hybrid_search(
                [0.1] * 10, "query", filter_metadata={"a; drop": "evil"}
            )


class TestQdrantVectorStoreKeywordSearch:
    @pytest.mark.asyncio
    async def test_keyword_search_uses_document_query(self, store, mock_client, mock_qdrant_import):
        """keyword_search must pass models.Document(text=query) not an empty SparseVector."""
        _, mock_models = mock_qdrant_import

        mock_point = MagicMock()
        mock_point.id = "point-uuid"
        mock_point.score = 0.8
        mock_point.payload = {
            "chunk_id": "chk_kw",
            "document_id": "doc_1",
            "text": "keyword result",
            "metadata": {},
        }
        query_result = MagicMock()
        query_result.points = [mock_point]
        mock_client.query_points.return_value = query_result

        results = await store.keyword_search("my search terms", top_k=5)

        assert len(results) == 1
        assert results[0].chunk_id == "chk_kw"

        # Verify the query argument was built via models.Document, not SparseVector
        mock_models.Document.assert_called_once_with(text="my search terms", model="Qdrant/bm25")
        mock_models.SparseVector.assert_not_called()

    @pytest.mark.asyncio
    async def test_keyword_search_rejects_unsafe_filter_key(self, store, mock_client):
        with pytest.raises(ValueError, match="not a safe identifier"):
            await store.keyword_search("query", filter_metadata={"x.y": "val"})


class TestQdrantBuildFilter:
    def test_none_returns_none(self, store):
        assert store._build_filter(None) is None

    def test_empty_dict_returns_none(self, store):
        assert store._build_filter({}) is None

    def test_single_key_prefixed(self, store, mock_qdrant_import):
        _, mock_models = mock_qdrant_import
        store._build_filter({"source_type": "pdf"})
        mock_models.FieldCondition.assert_called_once()
        call_kwargs = mock_models.FieldCondition.call_args.kwargs
        assert call_kwargs["key"] == "metadata.source_type"

    def test_multiple_keys_all_conditions(self, store, mock_qdrant_import):
        _, mock_models = mock_qdrant_import
        store._build_filter({"a": "1", "b": "2"})
        assert mock_models.FieldCondition.call_count == 2

    def test_rejects_unsafe_key(self, store):
        with pytest.raises(ValueError, match="not a safe identifier"):
            store._build_filter({"Robert'; DROP": "val"})

    @pytest.mark.asyncio
    async def test_search_rejects_unsafe_filter_key(self, store, mock_client):
        with pytest.raises(ValueError, match="not a safe identifier"):
            await store.search([0.1] * 10, filter_metadata={"x.y": "val"})

    def test_values_coerced_to_str(self, store, mock_qdrant_import):
        """Filter values must be coerced to str for parity with pgvector."""
        _, mock_models = mock_qdrant_import
        store._build_filter({"active": True})
        mock_models.MatchValue.assert_called_once()
        call_kwargs = mock_models.MatchValue.call_args.kwargs
        assert call_kwargs["value"] == "True"


class TestQdrantVectorStoreDelete:
    @pytest.mark.asyncio
    async def test_delete_by_document_id(self, store, mock_client):
        old_points = [MagicMock(id=f"uuid-{i}") for i in range(3)]
        mock_client.scroll.return_value = (old_points, None)

        count = await store.delete("doc_1")
        assert count == 3
        mock_client.scroll.assert_called_once()
        mock_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_no_existing_points_skips_delete(self, store, mock_client):
        mock_client.scroll.return_value = ([], None)

        count = await store.delete("doc_missing")
        assert count == 0
        mock_client.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_paginates_scroll(self, store, mock_client):
        """Delete scrolls all pages when results span multiple pages."""
        page1 = [MagicMock(id="uuid-0"), MagicMock(id="uuid-1")]
        page2 = [MagicMock(id="uuid-2")]
        mock_client.scroll.side_effect = [
            (page1, "cursor-after-page1"),
            (page2, None),
        ]

        count = await store.delete("doc_1")
        assert count == 3
        assert mock_client.scroll.call_count == 2
        mock_client.delete.assert_called_once()


class TestQdrantVectorStoreDrop:
    @pytest.mark.asyncio
    async def test_drop_deletes_collection(self, store, mock_client):
        mock_client.collection_exists.return_value = True

        await store.drop()
        mock_client.delete_collection.assert_called_once_with("test_docs")

    @pytest.mark.asyncio
    async def test_drop_noop_if_not_exists(self, store, mock_client):
        mock_client.collection_exists.return_value = False

        await store.drop()
        mock_client.delete_collection.assert_not_called()


class TestQdrantVectorStoreStats:
    @pytest.mark.asyncio
    async def test_stats_returns_dict(self, store, mock_client):
        mock_info = MagicMock()
        mock_info.vectors_count = 3
        mock_info.points_count = 3
        mock_client.get_collection.return_value = mock_info

        # Simulate two points from doc_a and one from doc_b, then end of scroll
        point_a1 = MagicMock()
        point_a1.payload = {"document_id": "doc_a"}
        point_a2 = MagicMock()
        point_a2.payload = {"document_id": "doc_a"}
        point_b1 = MagicMock()
        point_b1.payload = {"document_id": "doc_b"}
        # First scroll call returns all 3 points and signals no more pages
        mock_client.scroll.return_value = ([point_a1, point_a2, point_b1], None)

        stats = await store.stats()
        assert stats["total_chunks"] == 3
        assert stats["total_documents"] == 2  # two distinct document_ids
        assert stats["store_name"] == "test_docs"
        assert stats["embedding_dim"] == 10

    @pytest.mark.asyncio
    async def test_stats_with_none_points_count(self, store, mock_client):
        """Qdrant returns None for points_count on empty/new collections."""
        mock_info = MagicMock()
        mock_info.points_count = None
        mock_client.get_collection.return_value = mock_info
        mock_client.scroll.return_value = ([], None)

        stats = await store.stats()
        assert stats["total_chunks"] == 0
        assert stats["total_documents"] == 0

    @pytest.mark.asyncio
    async def test_stats_paginates_scroll(self, store, mock_client):
        """Stats scrolls all pages to count distinct document_ids."""
        mock_info = MagicMock()
        mock_info.points_count = 3
        mock_client.get_collection.return_value = mock_info

        page1 = [MagicMock(payload={"document_id": "doc_a"}),
                  MagicMock(payload={"document_id": "doc_a"})]
        page2 = [MagicMock(payload={"document_id": "doc_b"})]
        mock_client.scroll.side_effect = [
            (page1, "cursor"),
            (page2, None),
        ]

        stats = await store.stats()
        assert stats["total_documents"] == 2
        assert mock_client.scroll.call_count == 2


    @pytest.mark.asyncio
    async def test_stats_skips_scan_for_large_collections(self, store, mock_client):
        """Collections with >500k points skip the full document count scan."""
        mock_info = MagicMock()
        mock_info.points_count = 600_000
        mock_client.get_collection.return_value = mock_info

        stats = await store.stats()
        assert stats["total_chunks"] == 600_000
        assert stats["total_documents"] == -1
        # scroll should NOT be called — the scan was skipped
        mock_client.scroll.assert_not_called()


class TestQdrantVectorStoreNotConnected:
    @pytest.mark.asyncio
    async def test_search_raises_if_not_connected(self):
        from noid_rag.vectorstore_qdrant import QdrantVectorStore

        store = QdrantVectorStore()
        with pytest.raises(RuntimeError, match="not connected"):
            await store.search([0.1] * 10)

    @pytest.mark.asyncio
    async def test_delete_raises_if_not_connected(self):
        from noid_rag.vectorstore_qdrant import QdrantVectorStore

        store = QdrantVectorStore()
        with pytest.raises(RuntimeError, match="not connected"):
            await store.delete("doc_1")


class TestQdrantVectorStoreClose:
    @pytest.mark.asyncio
    async def test_close_closes_client(self, store, mock_client):
        await store.close()
        mock_client.close.assert_called_once()
        assert store._client is None

    @pytest.mark.asyncio
    async def test_close_when_no_client(self):
        from noid_rag.vectorstore_qdrant import QdrantVectorStore

        s = QdrantVectorStore()
        await s.close()
        assert s._client is None


class TestQdrantVectorStoreContextManager:
    @pytest.mark.asyncio
    async def test_context_manager(self, config, mock_qdrant_import):
        from noid_rag.vectorstore_qdrant import QdrantVectorStore

        mock_client_cls, _ = mock_qdrant_import
        mock_instance = AsyncMock()
        mock_instance.collection_exists.return_value = True
        mock_client_cls.return_value = mock_instance

        async with QdrantVectorStore(config=config, embedding_dim=10) as s:
            assert s._client is not None

        mock_instance.close.assert_called_once()


class TestQdrantVectorStoreSampleChunks:
    @pytest.mark.asyncio
    async def test_sample_random(self, store, mock_client):
        mock_point = MagicMock()
        mock_point.id = "point-uuid"
        mock_point.payload = {
            "chunk_id": "chk_1",
            "document_id": "doc_1",
            "text": "sample text",
            "metadata": {},
        }
        mock_client.scroll.return_value = ([mock_point], None)

        results = await store.sample_chunks(limit=5, strategy="random")

        assert len(results) == 1
        assert results[0]["id"] == "chk_1"

    @pytest.mark.asyncio
    async def test_sample_diverse(self, store, mock_client):
        points = []
        for i in range(4):
            p = MagicMock()
            p.id = f"point-{i}"
            p.payload = {
                "chunk_id": f"chk_{i}",
                "document_id": f"doc_{i % 2}",
                "text": f"text {i}",
                "metadata": {},
            }
            points.append(p)
        mock_client.scroll.return_value = (points, None)

        results = await store.sample_chunks(limit=3, strategy="diverse")

        assert len(results) == 3
        doc_ids = [r["document_id"] for r in results]
        assert "doc_0" in doc_ids
        assert "doc_1" in doc_ids
