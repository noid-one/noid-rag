"""Async pgvector store using SQLAlchemy + asyncpg."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from noid_rag.config import VectorStoreConfig
from noid_rag.models import Chunk, SearchResult

_SAFE_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")


class PgVectorStore:
    """Async PostgreSQL vector store with pgvector."""

    def __init__(self, config: VectorStoreConfig | None = None):
        self.config = config or VectorStoreConfig()
        self._engine: AsyncEngine | None = None

    async def connect(self) -> None:
        """Create engine and ensure table/indexes exist."""
        self._engine = create_async_engine(
            self.config.dsn,
            pool_size=self.config.pool_size,
            pool_pre_ping=True,
        )
        await self._ensure_table()

    async def _ensure_table(self) -> None:
        """Create table and indexes if they don't exist."""
        dim = self.config.embedding_dim
        table = self.config.table_name
        async with self._engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.execute(
                text(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id            TEXT PRIMARY KEY,
                    document_id   TEXT NOT NULL,
                    text          TEXT NOT NULL,
                    embedding     vector({dim}) NOT NULL,
                    metadata      JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at    TIMESTAMPTZ DEFAULT NOW(),
                    updated_at    TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            )
            # Create indexes if they don't exist
            await conn.execute(
                text(f"""
                CREATE INDEX IF NOT EXISTS idx_{table}_embedding_hnsw
                ON {table} USING hnsw (embedding vector_cosine_ops)
                WITH (m={self.config.hnsw_m}, ef_construction={self.config.hnsw_ef_construction})
            """)
            )
            await conn.execute(
                text(f"""
                CREATE INDEX IF NOT EXISTS idx_{table}_document_id ON {table} (document_id)
            """)
            )
            await conn.execute(
                text(f"""
                CREATE INDEX IF NOT EXISTS idx_{table}_metadata ON {table} USING gin (metadata)
            """)
            )
            # Full-text search: tsvector column + GIN index
            await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS tsv tsvector"))
            await conn.execute(
                text(f"""
                CREATE INDEX IF NOT EXISTS idx_{table}_tsv ON {table} USING gin(tsv)
            """)
            )

    async def upsert(self, chunks: list[Chunk]) -> int:
        """Upsert chunks into the store. Returns count of upserted rows."""
        if not chunks:
            return 0
        table = self.config.table_name
        now = datetime.now(timezone.utc)
        async with self._engine.begin() as conn:
            for chunk in chunks:
                if chunk.embedding is None:
                    raise ValueError(f"Chunk {chunk.id} has no embedding")
                embedding_str = "[" + ",".join(str(v) for v in chunk.embedding) + "]"
                await conn.execute(
                    text(f"""
                        INSERT INTO {table}
                            (id, document_id, text, embedding, metadata,
                             created_at, updated_at, tsv)
                        VALUES
                            (:id, :doc_id, :text, CAST(:embedding AS vector),
                             CAST(:metadata AS jsonb), :created_at, :updated_at,
                             to_tsvector(:fts_lang, :text))
                        ON CONFLICT (id) DO UPDATE SET
                            text = EXCLUDED.text,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata,
                            updated_at = EXCLUDED.updated_at,
                            tsv = to_tsvector(:fts_lang, EXCLUDED.text)
                    """),
                    {
                        "id": chunk.id,
                        "doc_id": chunk.document_id,
                        "text": chunk.text,
                        "embedding": embedding_str,
                        "metadata": json.dumps(chunk.metadata),
                        "created_at": now,
                        "updated_at": now,
                        "fts_lang": self.config.fts_language,
                    },
                )
        return len(chunks)

    async def search(
        self,
        embedding: list[float],
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar chunks by embedding vector."""
        table = self.config.table_name
        embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

        where_clause = ""
        params: dict[str, Any] = {"embedding": embedding_str, "top_k": top_k}

        if filter_metadata:
            conditions = []
            for i, (key, value) in enumerate(filter_metadata.items()):
                if not _SAFE_KEY_RE.match(key):
                    raise ValueError(
                        f"filter_metadata key {key!r} is not a safe identifier. "
                        "Use only letters, digits, and underscores."
                    )
                param_name = f"filter_{i}"
                conditions.append(f"metadata->>'{key}' = :{param_name}")
                params[param_name] = str(value)
            where_clause = "WHERE " + " AND ".join(conditions)

        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(f"""
                    SELECT id, document_id, text, metadata,
                           1 - (embedding <=> CAST(:embedding AS vector)) AS score
                    FROM {table}
                    {where_clause}
                    ORDER BY embedding <=> CAST(:embedding AS vector)
                    LIMIT :top_k
                """),
                params,
            )
            rows = result.fetchall()

        return [
            SearchResult(
                chunk_id=row.id,
                text=row.text,
                score=float(row.score),
                metadata=(
                    json.loads(row.metadata) if isinstance(row.metadata, str) else row.metadata
                ),
                document_id=row.document_id,
            )
            for row in rows
        ]

    async def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Full-text keyword search using ts_rank scoring."""
        table = self.config.table_name

        fts_lang = self.config.fts_language
        conditions = ["tsv @@ plainto_tsquery(:fts_lang, :query)"]
        params: dict[str, Any] = {"query": query, "top_k": top_k, "fts_lang": fts_lang}

        if filter_metadata:
            for i, (key, value) in enumerate(filter_metadata.items()):
                if not _SAFE_KEY_RE.match(key):
                    raise ValueError(
                        f"filter_metadata key {key!r} is not a safe identifier. "
                        "Use only letters, digits, and underscores."
                    )
                param_name = f"filter_{i}"
                conditions.append(f"metadata->>'{key}' = :{param_name}")
                params[param_name] = str(value)

        where_clause = "WHERE " + " AND ".join(conditions)

        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(f"""
                    SELECT id, document_id, text, metadata,
                           ts_rank(tsv, plainto_tsquery(:fts_lang, :query)) AS score
                    FROM {table}
                    {where_clause}
                    ORDER BY score DESC
                    LIMIT :top_k
                """),
                params,
            )
            rows = result.fetchall()

        return [
            SearchResult(
                chunk_id=row.id,
                text=row.text,
                score=float(row.score),
                metadata=(
                    json.loads(row.metadata) if isinstance(row.metadata, str) else row.metadata
                ),
                document_id=row.document_id,
            )
            for row in rows
        ]

    async def hybrid_search(
        self,
        embedding: list[float],
        query: str,
        top_k: int = 5,
        rrf_k: int = 60,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Hybrid search combining vector + keyword results with Reciprocal Rank Fusion."""
        vector_results, keyword_results = await asyncio.gather(
            self.search(embedding, top_k=top_k, filter_metadata=filter_metadata),
            self.keyword_search(query, top_k=top_k, filter_metadata=filter_metadata),
        )

        # Build RRF scores keyed by chunk_id
        rrf_scores: dict[str, float] = {}
        result_map: dict[str, SearchResult] = {}

        for rank, r in enumerate(vector_results):
            rrf_scores[r.chunk_id] = rrf_scores.get(r.chunk_id, 0) + 1.0 / (rrf_k + rank + 1)
            result_map[r.chunk_id] = r

        for rank, r in enumerate(keyword_results):
            rrf_scores[r.chunk_id] = rrf_scores.get(r.chunk_id, 0) + 1.0 / (rrf_k + rank + 1)
            if r.chunk_id not in result_map:
                result_map[r.chunk_id] = r

        # Sort by fused score descending, take top_k
        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        return [replace(result_map[cid], score=rrf_scores[cid]) for cid in sorted_ids]

    async def sample_chunks(
        self,
        limit: int = 10,
        strategy: str = "diverse",
    ) -> list[dict[str, Any]]:
        """Sample chunks from the store.

        Strategies:
            random: ORDER BY RANDOM()
            diverse: sample evenly across distinct document_ids
        """
        table = self.config.table_name
        async with self._engine.connect() as conn:
            if strategy == "random":
                result = await conn.execute(
                    text(f"""
                        SELECT id, document_id, text, metadata
                        FROM {table}
                        ORDER BY RANDOM()
                        LIMIT :limit
                    """),
                    {"limit": limit},
                )
            else:
                # Diverse: rank rows within each document, then round-robin
                result = await conn.execute(
                    text(f"""
                        WITH ranked AS (
                            SELECT id, document_id, text, metadata,
                                   ROW_NUMBER() OVER (
                                       PARTITION BY document_id ORDER BY RANDOM()
                                   ) AS rn
                            FROM {table}
                        )
                        SELECT id, document_id, text, metadata
                        FROM ranked
                        ORDER BY rn, RANDOM()
                        LIMIT :limit
                    """),
                    {"limit": limit},
                )

            rows = result.fetchall()

        return [
            {
                "id": row.id,
                "document_id": row.document_id,
                "text": row.text,
                "metadata": (
                    json.loads(row.metadata) if isinstance(row.metadata, str) else row.metadata
                ),
            }
            for row in rows
        ]

    async def delete(self, document_id: str) -> int:
        """Delete all chunks for a document. Returns count of deleted rows."""
        table = self.config.table_name
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(f"DELETE FROM {table} WHERE document_id = :doc_id"),
                {"doc_id": document_id},
            )
            return result.rowcount

    async def stats(self) -> dict[str, Any]:
        """Get store statistics."""
        table = self.config.table_name
        async with self._engine.connect() as conn:
            total = await conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            total_count = total.scalar()

            docs = await conn.execute(text(f"SELECT COUNT(DISTINCT document_id) FROM {table}"))
            doc_count = docs.scalar()

        return {
            "total_chunks": total_count,
            "total_documents": doc_count,
            "table_name": table,
            "embedding_dim": self.config.embedding_dim,
        }

    async def close(self) -> None:
        """Close the engine and connection pool."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None

    async def __aenter__(self) -> PgVectorStore:
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
