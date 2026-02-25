"""Shared test fixtures."""


import pytest

from noid_rag.config import Settings
from noid_rag.models import Chunk, Document, SearchResult


@pytest.fixture
def sample_document():
    """Return a Document with test content."""
    return Document(
        id="doc_test123456",
        source="/tmp/test.pdf",
        content=(
            "# Test Document\n\n"
            "This is a test document with some content.\n\n"
            "## Section 1\n\n"
            "First section text.\n\n"
            "## Section 2\n\n"
            "Second section text."
        ),
        metadata={"source_type": "pdf", "filename": "test.pdf", "page_count": 2},
    )


@pytest.fixture
def sample_chunks(sample_document):
    """Return a list of Chunks with test content."""
    return [
        Chunk(
            id="chk_aaa111222333",
            text="First section text.",
            document_id=sample_document.id,
            metadata={"source_type": "pdf"},
        ),
        Chunk(
            id="chk_bbb444555666",
            text="Second section text.",
            document_id=sample_document.id,
            metadata={"source_type": "pdf"},
        ),
    ]


@pytest.fixture
def sample_chunks_with_embeddings(sample_chunks):
    """Return chunks with fake embeddings set."""
    for i, chunk in enumerate(sample_chunks):
        chunk.embedding = [float(i)] * 10
    return sample_chunks


@pytest.fixture
def sample_search_results():
    """Return a list of SearchResults."""
    return [
        SearchResult(
            chunk_id="chk_aaa111",
            text="Result 1",
            score=0.95,
            metadata={"source_type": "pdf"},
            document_id="doc_test1",
        ),
        SearchResult(
            chunk_id="chk_bbb222",
            text="Result 2",
            score=0.85,
            metadata={"source_type": "html"},
            document_id="doc_test2",
        ),
    ]


@pytest.fixture
def settings():
    """Return default Settings."""
    return Settings()


@pytest.fixture
def tmp_output(tmp_path):
    """Return a temp directory for output files."""
    out = tmp_path / "output"
    out.mkdir()
    return out
