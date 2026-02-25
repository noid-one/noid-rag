"""Tests for data models."""

from noid_rag.models import Chunk, Document, SearchResult, _content_id


class TestContentId:
    def test_generates_prefixed_id(self):
        result = _content_id("doc", "some-key")
        assert result.startswith("doc_")

    def test_deterministic(self):
        a = _content_id("doc", "same-key")
        b = _content_id("doc", "same-key")
        assert a == b

    def test_different_keys_produce_different_ids(self):
        a = _content_id("doc", "key-a")
        b = _content_id("doc", "key-b")
        assert a != b

    def test_hex_suffix_length(self):
        result = _content_id("chk", "key")
        suffix = result.split("_", 1)[1]
        assert len(suffix) == 12

    def test_different_prefixes(self):
        doc_id = _content_id("doc", "key")
        chk_id = _content_id("chk", "key")
        assert doc_id.startswith("doc_")
        assert chk_id.startswith("chk_")


class TestDocument:
    def test_creation_with_defaults(self):
        doc = Document(source="/tmp/test.pdf", content="Hello world")
        assert doc.id.startswith("doc_")
        assert doc.source == "/tmp/test.pdf"
        assert doc.content == "Hello world"
        assert doc.metadata == {}
        assert doc._docling_doc is None

    def test_creation_with_custom_id(self):
        doc = Document(id="doc_custom", source="file.txt", content="text")
        assert doc.id == "doc_custom"

    def test_custom_metadata(self):
        doc = Document(
            source="file.pdf",
            content="content",
            metadata={"source_type": "pdf", "page_count": 5, "custom_key": "value"},
        )
        assert doc.metadata["source_type"] == "pdf"
        assert doc.metadata["page_count"] == 5
        assert doc.metadata["custom_key"] == "value"

    def test_same_source_produces_same_id(self):
        a = Document(source="f.txt", content="c1")
        b = Document(source="f.txt", content="c2")
        assert a.id == b.id

    def test_different_sources_produce_different_ids(self):
        a = Document(source="a.txt", content="c")
        b = Document(source="b.txt", content="c")
        assert a.id != b.id

    def test_docling_doc_preserved(self):
        sentinel = object()
        doc = Document(source="f.txt", content="c", _docling_doc=sentinel)
        assert doc._docling_doc is sentinel


class TestChunk:
    def test_creation_with_defaults(self):
        chunk = Chunk(text="some text", document_id="doc_abc")
        assert chunk.id.startswith("chk_")
        assert chunk.text == "some text"
        assert chunk.document_id == "doc_abc"
        assert chunk.metadata == {}
        assert chunk.embedding is None

    def test_creation_with_custom_id(self):
        chunk = Chunk(id="chk_custom", text="text", document_id="doc_1")
        assert chunk.id == "chk_custom"

    def test_custom_metadata(self):
        chunk = Chunk(
            text="text",
            document_id="doc_1",
            metadata={"headings": ["Section 1"], "page": 3},
        )
        assert chunk.metadata["headings"] == ["Section 1"]
        assert chunk.metadata["page"] == 3

    def test_embedding_assignment(self):
        chunk = Chunk(text="text", document_id="doc_1")
        assert chunk.embedding is None
        chunk.embedding = [0.1, 0.2, 0.3]
        assert chunk.embedding == [0.1, 0.2, 0.3]

    def test_same_content_produces_same_id(self):
        a = Chunk(text="hello", document_id="doc_1")
        b = Chunk(text="hello", document_id="doc_1")
        assert a.id == b.id

    def test_different_content_produces_different_ids(self):
        a = Chunk(text="hello", document_id="doc_1")
        b = Chunk(text="world", document_id="doc_1")
        assert a.id != b.id


class TestSearchResult:
    def test_creation(self):
        sr = SearchResult(
            chunk_id="chk_abc",
            text="result text",
            score=0.92,
            metadata={"source_type": "pdf"},
            document_id="doc_xyz",
        )
        assert sr.chunk_id == "chk_abc"
        assert sr.text == "result text"
        assert sr.score == 0.92
        assert sr.metadata == {"source_type": "pdf"}
        assert sr.document_id == "doc_xyz"

    def test_score_range(self):
        sr = SearchResult(
            chunk_id="chk_1",
            text="text",
            score=0.0,
            metadata={},
            document_id="doc_1",
        )
        assert sr.score == 0.0

        sr2 = SearchResult(
            chunk_id="chk_2",
            text="text",
            score=1.0,
            metadata={},
            document_id="doc_2",
        )
        assert sr2.score == 1.0

    def test_metadata_dict(self):
        sr = SearchResult(
            chunk_id="chk_1",
            text="text",
            score=0.5,
            metadata={"key": "value", "nested": {"a": 1}},
            document_id="doc_1",
        )
        assert sr.metadata["key"] == "value"
        assert sr.metadata["nested"]["a"] == 1
