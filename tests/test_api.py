"""Tests for NoidRag API."""

from unittest.mock import AsyncMock, patch

import pytest

from noid_rag.api import NoidRag
from noid_rag.config import Settings
from noid_rag.models import Chunk, Document, SearchResult


class TestNoidRagInit:
    def test_init_default(self):
        rag = NoidRag()
        assert isinstance(rag.settings, Settings)
        assert rag.settings.verbose is False

    def test_init_with_dict_merges_config(self):
        rag = NoidRag(config={"verbose": True})
        assert rag.settings.verbose is True

    def test_init_with_settings_object(self):
        s = Settings(verbose=True)
        rag = NoidRag(config=s)
        assert rag.settings is s
        assert rag.settings.verbose is True

    def test_init_with_none(self):
        rag = NoidRag(config=None)
        assert isinstance(rag.settings, Settings)


class TestNoidRagParse:
    @patch("noid_rag.parser.parse")
    def test_parse_delegates_to_parser(self, mock_parse):
        mock_doc = Document(source="/tmp/test.pdf", content="# Test")
        mock_parse.return_value = mock_doc

        rag = NoidRag()
        result = rag.parse("/tmp/test.pdf")

        assert result.content == "# Test"
        assert result.source == "/tmp/test.pdf"
        mock_parse.assert_called_once_with("/tmp/test.pdf", config=rag.settings.parser)


class TestNoidRagChunk:
    @patch("noid_rag.chunker.chunk")
    @patch("noid_rag.parser.parse")
    def test_chunk_delegates_to_parser_and_chunker(self, mock_parse, mock_chunk):
        mock_doc = Document(source="/tmp/test.pdf", content="# Test")
        mock_parse.return_value = mock_doc
        mock_chunk.return_value = [
            Chunk(text="chunk 1", document_id="doc_1"),
            Chunk(text="chunk 2", document_id="doc_1"),
        ]

        rag = NoidRag()
        result = rag.chunk("/tmp/test.pdf")

        assert len(result) == 2
        assert result[0].text == "chunk 1"
        assert result[1].text == "chunk 2"
        mock_parse.assert_called_once()
        mock_chunk.assert_called_once_with(mock_doc, config=rag.settings.chunker)


class TestNoidRagIngest:
    @pytest.mark.asyncio
    @patch("noid_rag.vectorstore.PgVectorStore")
    @patch("noid_rag.embeddings.EmbeddingClient")
    @patch("noid_rag.chunker.chunk")
    @patch("noid_rag.parser.parse")
    async def test_aingest_runs_full_pipeline(
        self, mock_parse, mock_chunk, mock_embed_cls, mock_store_cls
    ):
        # Set up parser
        mock_doc = Document(id="doc_test", source="/tmp/test.pdf", content="# Test")
        mock_parse.return_value = mock_doc

        # Set up chunker
        chunks = [
            Chunk(text="chunk 1", document_id="doc_test"),
            Chunk(text="chunk 2", document_id="doc_test"),
        ]
        mock_chunk.return_value = chunks

        # Set up embedding client
        mock_embed = AsyncMock()
        mock_embed.embed_chunks.return_value = chunks
        mock_embed_cls.return_value = mock_embed

        # Set up vector store
        mock_store = AsyncMock()
        mock_store.upsert.return_value = 2
        mock_store.__aenter__ = AsyncMock(return_value=mock_store)
        mock_store.__aexit__ = AsyncMock(return_value=False)
        mock_store_cls.return_value = mock_store

        rag = NoidRag()
        result = await rag.aingest("/tmp/test.pdf")

        assert result["chunks_stored"] == 2
        assert result["document_id"] == "doc_test"
        mock_embed.embed_chunks.assert_called_once_with(chunks)
        mock_store.upsert.assert_called_once_with(chunks)


class TestNoidRagSearch:
    @pytest.mark.asyncio
    @patch("noid_rag.vectorstore.PgVectorStore")
    @patch("noid_rag.embeddings.EmbeddingClient")
    async def test_asearch_embeds_query_and_searches(self, mock_embed_cls, mock_store_cls):
        # Set up embedding client
        mock_embed = AsyncMock()
        query_embedding = [0.1] * 10
        mock_embed.embed_query.return_value = query_embedding
        mock_embed_cls.return_value = mock_embed

        # Set up vector store
        expected_results = [
            SearchResult(
                chunk_id="chk_1",
                text="result",
                score=0.95,
                metadata={},
                document_id="doc_1",
            )
        ]
        mock_store = AsyncMock()
        mock_store.hybrid_search.return_value = expected_results
        mock_store.__aenter__ = AsyncMock(return_value=mock_store)
        mock_store.__aexit__ = AsyncMock(return_value=False)
        mock_store_cls.return_value = mock_store

        rag = NoidRag()
        results = await rag.asearch("test query", top_k=3)

        assert len(results) == 1
        assert results[0].score == 0.95
        mock_embed.embed_query.assert_called_once_with("test query")
        mock_store.hybrid_search.assert_called_once_with(
            query_embedding,
            "test query",
            top_k=3,
            rrf_k=60,
        )


class TestNoidRagBatch:
    @pytest.mark.asyncio
    @patch("noid_rag.batch.BatchProcessor")
    async def test_abatch_processes_directory(self, mock_processor_cls, tmp_path):
        # Create some test files
        (tmp_path / "a.pdf").touch()
        (tmp_path / "b.pdf").touch()

        # Set up mock batch processor
        from noid_rag.batch import BatchResult, FileResult

        mock_result = BatchResult(
            total=2,
            success=2,
            failed=0,
            files=[
                FileResult(path=str(tmp_path / "a.pdf"), status="success", chunks_count=5),
                FileResult(path=str(tmp_path / "b.pdf"), status="success", chunks_count=3),
            ],
        )

        mock_processor = AsyncMock()
        mock_processor.process.return_value = mock_result
        mock_processor_cls.return_value = mock_processor

        rag = NoidRag()
        result = await rag.abatch(tmp_path, pattern="*.pdf")

        assert result["total"] == 2
        assert result["success"] == 2
        assert len(result["files"]) == 2
        mock_processor.process.assert_called_once()
