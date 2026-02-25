"""Data types flowing between noid-rag components."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


def _content_id(prefix: str, key: str) -> str:
    """Deterministic ID from a content key (e.g. file path, chunk text)."""
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return f"{prefix}_{digest}"


@dataclass
class Document:
    """Parsed document from Docling."""

    source: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default="")
    _docling_doc: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = _content_id("doc", self.source)


@dataclass
class Chunk:
    """A chunk of text from a document."""

    text: str
    document_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default="")
    embedding: list[float] | None = None

    def __post_init__(self) -> None:
        if not self.id:
            self.id = _content_id("chk", f"{self.document_id}:{self.text}")


@dataclass
class SearchResult:
    """A search result from the vector store."""

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]
    document_id: str


@dataclass
class AnswerResult:
    """LLM-synthesized answer with supporting sources."""

    answer: str
    sources: list[SearchResult]
    model: str
