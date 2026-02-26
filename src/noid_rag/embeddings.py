"""Embedding generation via OpenRouter/OpenAI-compatible API."""

from __future__ import annotations

import os
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from noid_rag.circuit_breaker import CircuitBreaker
from noid_rag.config import EmbeddingConfig
from noid_rag.models import Chunk


class EmbeddingClient:
    """OpenAI-compatible embedding API client."""

    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()
        self._api_key = (
            self.config.api_key.get_secret_value()
            or os.environ.get("NOID_RAG_EMBEDDING__API_KEY", "")
            or os.environ.get("OPENROUTER_API_KEY", "")
        )
        if not self._api_key:
            raise ValueError(
                "No embedding API key configured. "
                "Set NOID_RAG_EMBEDDING__API_KEY in .env or environment."
            )
        self._client: httpx.AsyncClient | None = None
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5, cooldown_seconds=30.0, service_name="embedding-api"
        )

    def _get_client(self) -> httpx.AsyncClient:
        """Return the shared httpx client, creating it lazily."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1, max=60),
        retry=retry_if_exception_type(
            (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)
        ),
    )
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via API."""
        self.circuit_breaker.check()
        client = self._get_client()
        try:
            resp = await client.post(
                self.config.api_url,
                json={
                    "model": self.config.model,
                    "input": texts,
                },
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            # Sort by index to preserve order
            embeddings = sorted(data["data"], key=lambda x: x["index"])
            self.circuit_breaker.record_success()
            return [e["embedding"] for e in embeddings]
        except (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException):
            self.circuit_breaker.record_failure()
            raise

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, batching as needed."""
        all_embeddings: list[list[float]] = []
        batch_size = self.config.batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = await self._embed_batch(batch)
            all_embeddings.extend(embeddings)

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
        result = await self._embed_batch([query])
        return result[0]

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> EmbeddingClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
