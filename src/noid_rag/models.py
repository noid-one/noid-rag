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


@dataclass
class EvalQuestion:
    """A question from an evaluation dataset."""

    question: str
    ground_truth: str | None = None


@dataclass
class EvalResult:
    """Evaluation result for a single question."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str | None
    scores: dict[str, float] = field(default_factory=dict)
    passed: dict[str, bool] = field(default_factory=dict)


@dataclass
class EvalSummary:
    """Aggregated evaluation results."""

    results: list[EvalResult]
    mean_scores: dict[str, float]
    backend: str
    model: str
    total_questions: int
    dataset_path: str
