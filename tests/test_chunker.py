"""Tests for document chunker."""

import sys
from unittest.mock import MagicMock, patch

from noid_rag.chunker import chunk
from noid_rag.config import ChunkerConfig
from noid_rag.models import Document


class TestFixedChunking:
    def test_fixed_chunk_returns_chunks(self, sample_document):
        config = ChunkerConfig(method="fixed", max_tokens=10, overlap=2)
        chunks = chunk(sample_document, config=config)
        assert len(chunks) > 0
        for c in chunks:
            assert c.document_id == sample_document.id
            assert c.text.strip()

    def test_fixed_chunk_metadata_includes_method(self, sample_document):
        config = ChunkerConfig(method="fixed", max_tokens=50, overlap=5)
        chunks = chunk(sample_document, config=config)
        for c in chunks:
            assert c.metadata.get("chunk_method") == "fixed"

    def test_fixed_chunk_splits_correctly(self):
        """Verify that fixed chunking produces predictable splits."""
        doc = Document(
            id="doc_split",
            source="test.txt",
            content="A" * 100,  # 100 chars
            metadata={},
        )
        # max_tokens=10, chars_per_token=4 => chunk_size=40
        # overlap=5 => overlap_chars=20
        config = ChunkerConfig(method="fixed", max_tokens=10, overlap=5)
        chunks = chunk(doc, config=config)
        assert len(chunks) >= 2
        # First chunk should be 40 chars
        assert len(chunks[0].text) == 40

    def test_fixed_chunk_preserves_document_metadata(self, sample_document):
        config = ChunkerConfig(method="fixed", max_tokens=50, overlap=5)
        chunks = chunk(sample_document, config=config)
        for c in chunks:
            # Document metadata should be propagated plus chunk_method
            assert "chunk_method" in c.metadata

    def test_empty_document_produces_no_chunks(self):
        doc = Document(
            id="doc_empty",
            source="empty.txt",
            content="",
            metadata={},
        )
        config = ChunkerConfig(method="fixed", max_tokens=50, overlap=5)
        chunks = chunk(doc, config=config)
        assert chunks == []

    def test_whitespace_only_document_produces_no_chunks(self):
        doc = Document(
            id="doc_ws",
            source="ws.txt",
            content="   \n  \n   ",
            metadata={},
        )
        config = ChunkerConfig(method="fixed", max_tokens=50, overlap=5)
        chunks = chunk(doc, config=config)
        assert chunks == []


class TestHybridChunking:
    def test_hybrid_falls_back_without_docling_doc(self, sample_document):
        """When _docling_doc is None, hybrid method falls back to fixed."""
        sample_document._docling_doc = None
        config = ChunkerConfig(method="hybrid", max_tokens=50, overlap=5)
        chunks = chunk(sample_document, config=config)
        assert len(chunks) > 0
        # Falls back to fixed, so chunk_method should be set
        for c in chunks:
            assert c.metadata.get("chunk_method") == "fixed"

    @patch.dict(sys.modules, {
        "docling_core": MagicMock(),
        "docling_core.transforms": MagicMock(),
        "docling_core.transforms.chunker": MagicMock(),
    })
    def test_hybrid_chunk_with_docling_doc(self, sample_document):
        mock_docling_doc = MagicMock()
        sample_document._docling_doc = mock_docling_doc

        mock_chunk = MagicMock()
        mock_chunk.text = "Chunk text"
        mock_chunk.meta = MagicMock()
        mock_chunk.meta.headings = ["Section 1"]
        mock_chunk.meta.page = 1

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [mock_chunk]

        mock_chunker_cls = sys.modules["docling_core.transforms.chunker"].HybridChunker
        mock_chunker_cls.return_value = mock_chunker

        config = ChunkerConfig(method="hybrid")
        chunks = chunk(sample_document, config=config)

        assert len(chunks) == 1
        assert chunks[0].text == "Chunk text"
        assert chunks[0].metadata.get("headings") == ["Section 1"]
        assert chunks[0].metadata.get("page") == 1

    @patch.dict(sys.modules, {
        "docling_core": MagicMock(),
        "docling_core.transforms": MagicMock(),
        "docling_core.transforms.chunker": MagicMock(),
    })
    def test_hybrid_chunk_multiple_chunks(self, sample_document):
        sample_document._docling_doc = MagicMock()

        mock_chunks = []
        for i in range(3):
            mc = MagicMock()
            mc.text = f"Chunk {i}"
            mc.meta = MagicMock()
            mc.meta.headings = [f"Heading {i}"]
            mc.meta.page = i
            mock_chunks.append(mc)

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = mock_chunks

        sys.modules["docling_core.transforms.chunker"].HybridChunker.return_value = mock_chunker

        config = ChunkerConfig(method="hybrid")
        chunks = chunk(sample_document, config=config)

        assert len(chunks) == 3
        for i, c in enumerate(chunks):
            assert c.text == f"Chunk {i}"
            assert c.document_id == sample_document.id

    @patch.dict(sys.modules, {
        "docling_core": MagicMock(),
        "docling_core.transforms": MagicMock(),
        "docling_core.transforms.chunker": MagicMock(),
    })
    def test_hybrid_chunk_no_meta(self, sample_document):
        """Chunk with meta=None should still work."""
        sample_document._docling_doc = MagicMock()

        mc = MagicMock()
        mc.text = "No meta chunk"
        mc.meta = None

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [mc]

        sys.modules["docling_core.transforms.chunker"].HybridChunker.return_value = mock_chunker

        config = ChunkerConfig(method="hybrid")
        chunks = chunk(sample_document, config=config)

        assert len(chunks) == 1
        assert "headings" not in chunks[0].metadata


class TestChunkIds:
    def test_chunk_ids_are_unique(self, sample_document):
        config = ChunkerConfig(method="fixed", max_tokens=10, overlap=2)
        chunks = chunk(sample_document, config=config)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_ids_start_with_chk(self, sample_document):
        config = ChunkerConfig(method="fixed", max_tokens=10, overlap=2)
        chunks = chunk(sample_document, config=config)
        for c in chunks:
            assert c.id.startswith("chk_")


class TestChunkDocumentId:
    def test_chunks_have_correct_document_id(self, sample_document):
        config = ChunkerConfig(method="fixed", max_tokens=50, overlap=5)
        chunks = chunk(sample_document, config=config)
        for c in chunks:
            assert c.document_id == sample_document.id
