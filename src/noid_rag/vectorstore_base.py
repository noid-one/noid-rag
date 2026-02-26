"""Abstract vector store interface for noid-rag."""

from __future__ import annotations

from typing import Any, Protocol

from noid_rag.models import Chunk, SearchResult


class VectorStore(Protocol):
    """Protocol that all vector store backends must satisfy."""

    async def connect(self) -> None: ...
    async def close(self) -> None: ...

    async def __aenter__(self) -> VectorStore: ...
    async def __aexit__(self, *args: Any) -> None: ...

    async def upsert(self, chunks: list[Chunk]) -> int: ...

    async def replace_document(
        self, document_id: str, chunks: list[Chunk]
    ) -> tuple[int, int]:
        """Replace all chunks for a document. Returns (deleted, inserted).

        Atomicity depends on backend:
        - pgvector: fully atomic (single transaction).
        - Qdrant: best-effort (brief window where old and new chunks coexist).
        """
        ...

    async def search(
        self,
        embedding: list[float],
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]: ...

    async def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]: ...

    async def hybrid_search(
        self,
        embedding: list[float],
        query: str,
        top_k: int = 5,
        rrf_k: int = 60,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]: ...

    async def sample_chunks(
        self,
        limit: int = 10,
        strategy: str = "diverse",
    ) -> list[dict[str, Any]]: ...

    async def delete(self, document_id: str) -> int: ...
    async def drop(self) -> None: ...
    async def stats(self) -> dict[str, Any]: ...
