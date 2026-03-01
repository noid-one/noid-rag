"""Factory for creating vector store backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from noid_rag.config import Settings
    from noid_rag.vectorstore_base import VectorStore


def create_vectorstore(settings: Settings) -> VectorStore:
    """Create a vector store backend based on settings.vectorstore.provider."""
    provider = settings.vectorstore.provider
    if provider == "pgvector":
        from noid_rag.vectorstore import PgVectorStore

        return PgVectorStore(config=settings.vectorstore)
    elif provider == "qdrant":
        from noid_rag.vectorstore_qdrant import QdrantVectorStore

        return QdrantVectorStore(
            config=settings.qdrant,
            embedding_dim=settings.vectorstore.embedding_dim,
        )
    elif provider == "zvec":
        from noid_rag.vectorstore_zvec import ZvecVectorStore

        return ZvecVectorStore(
            config=settings.zvec,
            embedding_dim=settings.vectorstore.embedding_dim,
        )
    raise ValueError(f"Unknown vectorstore provider: {provider!r}")
