"""Async Qdrant vector store with native BM25 hybrid search."""

from __future__ import annotations

import logging
import re
import uuid
from collections import defaultdict
from typing import Any

from noid_rag.config import QdrantConfig
from noid_rag.models import Chunk, SearchResult

logger = logging.getLogger(__name__)

_DENSE_VECTOR_NAME = "dense"
_SPARSE_VECTOR_NAME = "bm25"
_SAFE_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")


def _import_qdrant() -> tuple[Any, Any]:
    """Lazy import of qdrant_client with clear error message."""
    try:
        from qdrant_client import AsyncQdrantClient, models  # type: ignore[import-untyped]

        return AsyncQdrantClient, models
    except ImportError:
        raise ImportError(
            "qdrant-client is required for the Qdrant backend. "
            "Install with: uv pip install 'noid-rag[qdrant]'"
        ) from None


def make_raw_client(config: QdrantConfig) -> Any:
    """Create a raw AsyncQdrantClient without collection setup.

    Used by both QdrantVectorStore.connect() and the tune cleanup path.
    """
    client_cls, _ = _import_qdrant()
    api_key = config.api_key.get_secret_value() or None
    return client_cls(
        url=config.url,
        api_key=api_key,
        prefer_grpc=config.prefer_grpc,
        timeout=config.timeout,
    )


class QdrantVectorStore:
    """Async Qdrant vector store with native sparse BM25 vectors for hybrid search."""

    def __init__(self, config: QdrantConfig | None = None, embedding_dim: int = 1536):
        self.config = config or QdrantConfig()
        self.embedding_dim = embedding_dim
        self._client: Any | None = None
        self._models: Any | None = None

    async def connect(self) -> None:
        """Create client and ensure collection exists."""
        _, models = _import_qdrant()
        self._models = models
        self._client = make_raw_client(self.config)
        await self._ensure_collection()

    def _get_client(self) -> Any:
        if self._client is None:
            raise RuntimeError("QdrantVectorStore not connected. Call connect() first.")
        return self._client

    @property
    def _m(self) -> Any:
        """Shortcut for qdrant_client.models (available after connect)."""
        if self._models is None:
            _, models = _import_qdrant()
            self._models = models
        return self._models

    async def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist.

        Tolerates a concurrent creator by catching "already exists" errors.
        """
        client = self._get_client()
        m = self._m
        collection_name = self.config.collection_name

        if await client.collection_exists(collection_name):
            return

        try:
            await client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    _DENSE_VECTOR_NAME: m.VectorParams(
                        size=self.embedding_dim,
                        distance=m.Distance.COSINE,
                        hnsw_config=m.HnswConfigDiff(
                            m=self.config.hnsw_m,
                            # Qdrant SDK kwarg is 'ef_construct'; config uses 'hnsw_ef_construction'
                            ef_construct=self.config.hnsw_ef_construction,
                        ),
                    ),
                },
                sparse_vectors_config={
                    _SPARSE_VECTOR_NAME: m.SparseVectorParams(
                        modifier=m.Modifier.IDF,
                    ),
                },
            )
        except Exception as exc:
            if "already exists" not in str(exc).lower():
                raise
            # Collection created by concurrent process — fall through to
            # ensure the payload index exists regardless.

        try:
            await client.create_payload_index(
                collection_name=collection_name,
                field_name="document_id",
                field_schema=m.PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:
            if "already exists" in str(exc).lower():
                return
            raise

    async def upsert(self, chunks: list[Chunk]) -> int:
        """Upsert chunks into the collection. Returns count of upserted points."""
        if not chunks:
            return 0

        client = self._get_client()
        m = self._m
        collection_name = self.config.collection_name

        points = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.id} has no embedding")

            point = m.PointStruct(
                id=self._to_uuid(chunk.id),
                vector={
                    _DENSE_VECTOR_NAME: chunk.embedding,
                    _SPARSE_VECTOR_NAME: m.Document(
                        text=chunk.text, model="Qdrant/bm25"
                    ),
                },
                payload={
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                },
            )
            points.append(point)

        await client.upsert(
            collection_name=collection_name,
            points=points,
        )
        return len(points)

    async def replace_document(
        self, document_id: str, chunks: list[Chunk]
    ) -> tuple[int, int]:
        """Insert new chunks then delete old ones for a document.

        Scrolls the existing point IDs first, upserts the new chunks, then
        deletes only the previously collected IDs.  Deleting by specific IDs
        (rather than by document_id filter) ensures the newly inserted chunks
        are never swept up by the delete step.

        If the upsert raises, the original document is left intact because the
        delete step has not run yet.  During the brief overlap after upsert but
        before delete, both old and new chunks coexist; a concurrent search may
        return mixed results, but this is preferable to permanent data loss.

        Returns (deleted, inserted).
        """
        client = self._get_client()
        m = self._m
        collection_name = self.config.collection_name

        doc_filter = m.Filter(
            must=[
                m.FieldCondition(
                    key="document_id",
                    match=m.MatchValue(value=document_id),
                )
            ]
        )

        # Collect existing point IDs before inserting anything, so the delete
        # step targets only the pre-existing points, never the new ones.
        old_ids: list[Any] = []
        offset = None
        while True:
            batch, next_offset = await client.scroll(
                collection_name=collection_name,
                scroll_filter=doc_filter,
                limit=1000,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            for point in batch:
                old_ids.append(point.id)
            if next_offset is None:
                break
            offset = next_offset

        deleted = len(old_ids)

        # Insert first — if this raises, the original document is untouched.
        inserted = await self.upsert(chunks)

        # Delete only the pre-existing IDs, never the newly inserted ones.
        if old_ids:
            await client.delete(
                collection_name=collection_name,
                points_selector=m.PointIdsList(points=old_ids),
            )

        return deleted, inserted

    async def search(
        self,
        embedding: list[float],
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Dense vector search."""
        client = self._get_client()
        query_filter = self._build_filter(filter_metadata)

        results = await client.query_points(
            collection_name=self.config.collection_name,
            query=embedding,
            using=_DENSE_VECTOR_NAME,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        return [self._point_to_result(point) for point in results.points]

    async def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Sparse BM25 keyword search.

        Uses models.Document(text=query) so that Qdrant's server-side IDF
        modifier tokenises the text and computes the BM25 sparse vector.
        """
        client = self._get_client()
        m = self._m
        query_filter = self._build_filter(filter_metadata)

        results = await client.query_points(
            collection_name=self.config.collection_name,
            query=m.Document(text=query, model="Qdrant/bm25"),
            using=_SPARSE_VECTOR_NAME,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        return [self._point_to_result(point) for point in results.points]

    async def hybrid_search(
        self,
        embedding: list[float],
        query: str,
        top_k: int = 5,
        rrf_k: int = 60,  # noqa: ARG002 — Qdrant's native RRF does not expose k tuning
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Hybrid search using Qdrant's native prefetch + RRF fusion.

        Note: ``rrf_k`` is accepted for interface compatibility with the pgvector
        backend but is not forwarded to Qdrant — the server uses its own fixed RRF
        constant.  Use pgvector if you need to tune this parameter.
        """
        client = self._get_client()
        m = self._m
        query_filter = self._build_filter(filter_metadata)

        results = await client.query_points(
            collection_name=self.config.collection_name,
            prefetch=[
                m.Prefetch(
                    query=embedding,
                    using=_DENSE_VECTOR_NAME,
                    limit=top_k,
                    filter=query_filter,
                ),
                m.Prefetch(
                    query=m.Document(text=query, model="Qdrant/bm25"),
                    using=_SPARSE_VECTOR_NAME,
                    limit=top_k,
                    filter=query_filter,
                ),
            ],
            query=m.FusionQuery(fusion=m.Fusion.RRF),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        return [self._point_to_result(point) for point in results.points]

    async def sample_chunks(
        self,
        limit: int = 10,
        strategy: str = "diverse",
    ) -> list[dict[str, Any]]:
        """Sample chunks from the collection."""
        client = self._get_client()
        collection_name = self.config.collection_name

        if strategy == "random":
            results, _ = await client.scroll(
                collection_name=collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            return [
                {
                    "id": point.payload.get("chunk_id", str(point.id)),
                    "document_id": point.payload.get("document_id", ""),
                    "text": point.payload.get("text", ""),
                    "metadata": point.payload.get("metadata", {}),
                }
                for point in results
            ]
        else:
            # Diverse: over-fetch, then round-robin by document_id
            over_fetch = limit * 5
            results, _ = await client.scroll(
                collection_name=collection_name,
                limit=over_fetch,
                with_payload=True,
                with_vectors=False,
            )

            by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for point in results:
                payload = point.payload or {}
                by_doc[payload.get("document_id", "")].append(
                    {
                        "id": payload.get("chunk_id", str(point.id)),
                        "document_id": payload.get("document_id", ""),
                        "text": payload.get("text", ""),
                        "metadata": payload.get("metadata", {}),
                    }
                )

            sampled: list[dict[str, Any]] = []
            doc_lists = list(by_doc.values())
            idx = 0
            while len(sampled) < limit and doc_lists:
                for doc_chunks in list(doc_lists):
                    if idx < len(doc_chunks):
                        sampled.append(doc_chunks[idx])
                        if len(sampled) >= limit:
                            break
                    else:
                        doc_lists.remove(doc_chunks)
                idx += 1

            return sampled

    async def delete(self, document_id: str) -> int:
        """Delete all points for a document. Returns count of deleted points.

        Scrolls point IDs first, then deletes by explicit IDs to avoid TOCTOU
        races and to stay consistent with replace_document's approach.
        """
        client = self._get_client()
        m = self._m
        collection_name = self.config.collection_name

        doc_filter = m.Filter(
            must=[
                m.FieldCondition(
                    key="document_id",
                    match=m.MatchValue(value=document_id),
                )
            ]
        )

        ids: list[Any] = []
        offset = None
        while True:
            batch, next_offset = await client.scroll(
                collection_name=collection_name,
                scroll_filter=doc_filter,
                limit=1000,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            for point in batch:
                ids.append(point.id)
            if next_offset is None:
                break
            offset = next_offset

        if ids:
            await client.delete(
                collection_name=collection_name,
                points_selector=m.PointIdsList(points=ids),
            )

        return len(ids)

    async def drop(self) -> None:
        """Delete the collection."""
        client = self._get_client()
        collection_name = self.config.collection_name
        if await client.collection_exists(collection_name):
            await client.delete_collection(collection_name)

    async def stats(self) -> dict[str, Any]:
        """Get collection statistics.

        ``total_documents`` is approximated by scrolling payloads to count
        distinct ``document_id`` values.  For very large collections this may be
        slow; consider using the Qdrant dashboard for precise counts instead.
        """
        client = self._get_client()
        collection_name = self.config.collection_name
        info = await client.get_collection(collection_name)

        # Count distinct document_ids by scrolling payloads.
        # Skip the full scan for very large collections to avoid OOM/timeout.
        total_chunks = info.points_count or 0
        if total_chunks > 500_000:
            logger.warning(
                "stats(): collection has %d points; skipping full document count scan",
                total_chunks,
            )
            return {
                "total_chunks": total_chunks,
                "total_documents": -1,
                "store_name": collection_name,
                "embedding_dim": self.embedding_dim,
            }

        doc_ids: set[str] = set()
        offset = None
        while True:
            batch, next_offset = await client.scroll(
                collection_name=collection_name,
                limit=1000,
                offset=offset,
                with_payload=["document_id"],
                with_vectors=False,
            )
            for point in batch:
                if point.payload:
                    doc_id = point.payload.get("document_id")
                    if doc_id:
                        doc_ids.add(doc_id)
            if next_offset is None:
                break
            offset = next_offset

        return {
            # points_count == number of stored chunks.  vectors_count would
            # return 2× that value because each point holds two named vectors
            # (dense + sparse), so it must not be used here.
            "total_chunks": info.points_count or 0,
            "total_documents": len(doc_ids),
            "store_name": collection_name,
            "embedding_dim": self.embedding_dim,
        }

    async def close(self) -> None:
        """Close the client."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def __aenter__(self) -> QdrantVectorStore:
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    # --- Helpers ---

    @staticmethod
    def _to_uuid(chunk_id: str) -> str:
        """Deterministic UUID from chunk ID for Qdrant point IDs."""
        return str(uuid.uuid5(uuid.NAMESPACE_OID, chunk_id))

    def _build_filter(
        self,
        filter_metadata: dict[str, Any] | None,
    ) -> Any | None:
        """Convert metadata filter dict to Qdrant Filter."""
        if not filter_metadata:
            return None
        m = self._m
        conditions = []
        for key, value in filter_metadata.items():
            if not _SAFE_KEY_RE.match(key):
                raise ValueError(
                    f"filter_metadata key {key!r} is not a safe identifier. "
                    "Use only letters, digits, and underscores."
                )
            conditions.append(
                m.FieldCondition(
                    key=f"metadata.{key}",
                    match=m.MatchValue(value=str(value)),
                )
            )
        return m.Filter(must=conditions)

    @staticmethod
    def _point_to_result(point: Any) -> SearchResult:
        """Convert a Qdrant scored point to SearchResult."""
        payload = point.payload or {}
        return SearchResult(
            chunk_id=payload.get("chunk_id", str(point.id)),
            text=payload.get("text", ""),
            score=point.score if point.score is not None else 0.0,
            metadata=payload.get("metadata", {}),
            document_id=payload.get("document_id", ""),
        )
