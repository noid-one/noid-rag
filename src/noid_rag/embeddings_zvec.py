"""Local embedding client using zvec's built-in embedding functions.

Uses the underlying ``sentence-transformers`` model (all-MiniLM-L6-v2, 384 dims)
via zvec's ``DefaultLocalDenseEmbedding`` for batch encoding.
No API key or network required — models auto-download on first use (~80MB).
Requires ``sentence-transformers`` (included in the ``zvec`` extra).
All synchronous model calls are wrapped with ``asyncio.to_thread()``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from noid_rag.config import EmbeddingConfig
from noid_rag.models import Chunk

logger = logging.getLogger(__name__)


def _import_zvec() -> Any:
    """Lazy import of zvec with clear error message."""
    try:
        import zvec

        return zvec
    except ImportError:
        raise ImportError(
            "zvec is required for local embeddings. "
            "Install with: pip install 'noid-rag[zvec]'"
        ) from None


class ZvecEmbeddingClient:
    """Local embedding client using zvec's built-in dense embedding model.

    Implements the same interface as EmbeddingClient for drop-in replacement.
    Uses the underlying sentence-transformers model directly for batch encoding
    (~2.5x faster than single-string calls through zvec's ``embed()`` wrapper).
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()
        self._embed_fn: Any | None = None
        self._model: Any | None = None
        self._zvec: Any | None = None

    def _get_model(self) -> Any:
        """Return the underlying SentenceTransformer model, creating lazily."""
        if self._model is None:
            zvec = _import_zvec()
            self._zvec = zvec
            embed_fn = zvec.DefaultLocalDenseEmbedding()
            self._embed_fn = embed_fn
            self._model = embed_fn._get_model()
        return self._model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts using batch encoding.

        Accesses the underlying ``SentenceTransformer.encode()`` for true batch
        processing instead of calling zvec's single-string ``embed()`` in a loop.
        """
        model = self._get_model()
        batch_size = self.config.batch_size

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = await asyncio.to_thread(
                model.encode, batch, show_progress_bar=False
            )
            for emb in embeddings:
                if hasattr(emb, "tolist"):
                    all_embeddings.append(emb.tolist())
                else:
                    all_embeddings.append(list(emb))

        return all_embeddings

    async def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Embed chunks, setting .embedding in-place. Returns the chunks."""
        texts = [c.text for c in chunks]
        embeddings = await self.embed_texts(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
        return chunks

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        result = await self.embed_texts([query])
        return result[0]

    async def close(self) -> None:
        """Release resources (no-op for local model, but matches interface)."""
        self._embed_fn = None
        self._model = None

    async def __aenter__(self) -> ZvecEmbeddingClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
