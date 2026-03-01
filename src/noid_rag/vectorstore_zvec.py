"""Async zvec vector store with BM25 hybrid search.

zvec is a file-based, in-process vector database (no server required).
All synchronous zvec calls are wrapped with ``asyncio.to_thread()``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from noid_rag.config import ZvecConfig
from noid_rag.models import Chunk, SearchResult

logger = logging.getLogger(__name__)


def _import_zvec() -> Any:
    """Lazy import of zvec with clear error message."""
    try:
        import zvec

        return zvec
    except ImportError:
        raise ImportError(
            "zvec is required for the zvec backend. "
            "Install with: pip install 'noid-rag[zvec]'"
        ) from None


class ZvecVectorStore:
    """Async zvec vector store with built-in BM25 for hybrid search.

    zvec stores data on the local filesystem and requires no external server.
    Dense and sparse (BM25) vectors are stored together for hybrid retrieval.
    """

    def __init__(self, config: ZvecConfig | None = None, embedding_dim: int = 384):
        self.config = config or ZvecConfig()
        self.embedding_dim = embedding_dim
        self._zvec: Any | None = None
        self._collection: Any | None = None
        self._bm25_fn: Any | None = None

    async def connect(self) -> None:
        """Open or create the zvec collection on disk."""
        zvec = _import_zvec()
        self._zvec = zvec

        data_dir = Path(self.config.data_dir).expanduser()
        data_dir.mkdir(parents=True, exist_ok=True)
        collection_path = data_dir / self.config.collection_name

        schema = zvec.Schema(
            fields=[
                zvec.Field("document_id", zvec.FieldType.STRING),
                zvec.Field("chunk_id", zvec.FieldType.STRING),
                zvec.Field("text", zvec.FieldType.STRING),
                zvec.Field("metadata_json", zvec.FieldType.STRING),
            ],
            vectors=[
                zvec.VectorField("embedding", zvec.VectorType.FP32, self.embedding_dim),
                zvec.VectorField("bm25_sparse", zvec.VectorType.SPARSE_FP32),
            ],
        )

        index_params = {}
        if self.config.index_type == "hnsw":
            index_params = {
                "index_type": "hnsw",
                "hnsw_m": self.config.hnsw_m,
                "hnsw_ef_construction": self.config.hnsw_ef_construction,
            }

        def _open() -> Any:
            if collection_path.exists():
                return zvec.Collection.open(str(collection_path))
            return zvec.Collection.create(str(collection_path), schema=schema, **index_params)

        self._collection = await asyncio.to_thread(_open)

        try:
            self._bm25_fn = zvec.BM25EmbeddingFunction()
        except AttributeError:
            logger.warning("zvec.BM25EmbeddingFunction not available; BM25 search will be limited")
            self._bm25_fn = None

    def _get_collection(self) -> Any:
        if self._collection is None:
            raise RuntimeError("ZvecVectorStore not connected. Call connect() first.")
        return self._collection

    async def upsert(self, chunks: list[Chunk]) -> int:
        """Upsert chunks into the collection. Returns count of upserted docs."""
        if not chunks:
            return 0

        zvec = self._zvec
        collection = self._get_collection()

        docs = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.id} has no embedding")

            doc = {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "metadata_json": json.dumps(chunk.metadata),
                "embedding": chunk.embedding,
            }

            # Add BM25 sparse vector if available
            if self._bm25_fn is not None:
                try:
                    doc["bm25_sparse"] = self._bm25_fn.encode(chunk.text)
                except Exception:
                    logger.debug("Failed to encode BM25 sparse vector for chunk %s", chunk.id)

            docs.append(zvec.Doc(**doc) if hasattr(zvec, "Doc") else doc)

        await asyncio.to_thread(collection.upsert, docs)
        return len(docs)

    async def replace_document(
        self, document_id: str, chunks: list[Chunk]
    ) -> tuple[int, int]:
        """Replace all chunks for a document: delete old, then insert new.

        Because zvec's delete_by_filter targets all records matching the filter
        (including any newly upserted chunks with the same document_id), the
        only safe ordering is: count + delete first, then insert.  This means
        there is a brief window where the document has no chunks in the store,
        but it avoids silently losing newly ingested data.

        Returns (deleted, inserted).
        """
        collection = self._get_collection()

        # Count existing chunks before any mutation
        old_results = await asyncio.to_thread(
            collection.search_by_filter,
            f"document_id='{document_id}'",
            limit=100_000,
        )
        deleted = len(old_results) if old_results else 0

        # Delete old chunks first to avoid clobbering the new ones
        if deleted > 0:
            await asyncio.to_thread(
                collection.delete_by_filter, f"document_id='{document_id}'"
            )

        # Insert new chunks after deletion
        inserted = await self.upsert(chunks)

        return deleted, inserted

    async def search(
        self,
        embedding: list[float],
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Dense vector search."""
        collection = self._get_collection()
        zvec = self._zvec

        fetch_k = top_k * 3 if filter_metadata else top_k

        results = await asyncio.to_thread(
            collection.query,
            zvec.VectorQuery("embedding", vector=embedding),
            topk=fetch_k,
        )

        return self._results_to_search_results(results, top_k, filter_metadata)

    async def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Sparse BM25 keyword search."""
        if self._bm25_fn is None:
            return []

        collection = self._get_collection()
        zvec = self._zvec

        fetch_k = top_k * 3 if filter_metadata else top_k

        try:
            sparse_query = self._bm25_fn.encode(query)
            results = await asyncio.to_thread(
                collection.query,
                zvec.VectorQuery("bm25_sparse", vector=sparse_query),
                topk=fetch_k,
            )
        except Exception:
            logger.debug("BM25 keyword search failed, returning empty results")
            return []

        return self._results_to_search_results(results, top_k, filter_metadata)

    async def hybrid_search(
        self,
        embedding: list[float],
        query: str,
        top_k: int = 5,
        rrf_k: int = 60,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Hybrid search using dense + sparse with RRF fusion.

        Tries zvec's native multi-vector query with RrfReRanker first;
        falls back to Python-side RRF fusion if unavailable.
        """
        zvec = self._zvec
        collection = self._get_collection()

        fetch_k = top_k * 3 if filter_metadata else top_k

        # Try native multi-vector RRF
        if self._bm25_fn is not None and hasattr(zvec, "RrfReRanker"):
            try:
                sparse_query = self._bm25_fn.encode(query)
                results = await asyncio.to_thread(
                    collection.query,
                    [
                        zvec.VectorQuery("embedding", vector=embedding),
                        zvec.VectorQuery("bm25_sparse", vector=sparse_query),
                    ],
                    topk=fetch_k,
                    reranker=zvec.RrfReRanker(k=rrf_k),
                )
                return self._results_to_search_results(results, top_k, filter_metadata)
            except Exception:
                logger.debug("Native zvec hybrid search failed, falling back to Python RRF")

        # Fallback: run dense + sparse in parallel, merge with Python RRF
        dense_task = self.search(embedding, top_k=fetch_k, filter_metadata=filter_metadata)
        sparse_task = self.keyword_search(query, top_k=fetch_k, filter_metadata=filter_metadata)
        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)

        return self._rrf_merge(dense_results, sparse_results, top_k, rrf_k)

    async def sample_chunks(
        self,
        limit: int = 10,
        strategy: str = "diverse",
    ) -> list[dict[str, Any]]:
        """Sample chunks from the collection."""
        collection = self._get_collection()

        if strategy == "random":
            results = await asyncio.to_thread(
                collection.search_by_filter, "", limit=limit
            )
            return [self._doc_to_dict(doc) for doc in (results or [])][:limit]
        else:
            # Diverse: over-fetch, round-robin by document_id
            over_fetch = limit * 5
            results = await asyncio.to_thread(
                collection.search_by_filter, "", limit=over_fetch
            )

            by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for doc in results or []:
                d = self._doc_to_dict(doc)
                by_doc[d["document_id"]].append(d)

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
        """Delete all chunks for a document. Returns count of deleted chunks."""
        collection = self._get_collection()

        results = await asyncio.to_thread(
            collection.search_by_filter,
            f"document_id='{document_id}'",
            limit=100_000,
        )
        count = len(results) if results else 0

        if count > 0:
            await asyncio.to_thread(
                collection.delete_by_filter, f"document_id='{document_id}'"
            )

        return count

    async def drop(self) -> None:
        """Destroy the collection and remove data directory."""
        if self._collection is not None:
            try:
                await asyncio.to_thread(self._collection.destroy)
            except Exception:
                logger.debug("collection.destroy() failed, falling back to rmtree")

        data_dir = Path(self.config.data_dir).expanduser()
        collection_path = data_dir / self.config.collection_name
        if collection_path.exists():
            shutil.rmtree(collection_path)

    async def stats(self) -> dict[str, Any]:
        """Get collection statistics."""
        collection = self._get_collection()

        results = await asyncio.to_thread(
            collection.search_by_filter, "", limit=100_000
        )
        all_docs = results or []
        total_chunks = len(all_docs)

        doc_ids: set[str] = set()
        for doc in all_docs:
            doc_id = self._get_field(doc, "document_id")
            if doc_id:
                doc_ids.add(doc_id)

        return {
            "total_chunks": total_chunks,
            "total_documents": len(doc_ids),
            "store_name": self.config.collection_name,
            "embedding_dim": self.embedding_dim,
        }

    async def close(self) -> None:
        """Release the collection handle."""
        if self._collection is not None:
            try:
                await asyncio.to_thread(self._collection.close)
            except Exception:
                pass
            self._collection = None

    async def __aenter__(self) -> ZvecVectorStore:
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    # --- Helpers ---

    @staticmethod
    def _get_field(doc: Any, field: str) -> str:
        """Extract a field from a zvec result doc (dict or object)."""
        if isinstance(doc, dict):
            return doc.get(field, "")
        return getattr(doc, field, "")

    def _doc_to_dict(self, doc: Any) -> dict[str, Any]:
        """Convert a zvec result doc to a standard dict."""
        metadata_str = self._get_field(doc, "metadata_json")
        try:
            metadata = json.loads(metadata_str) if metadata_str else {}
        except (json.JSONDecodeError, TypeError):
            metadata = {}

        return {
            "id": self._get_field(doc, "chunk_id"),
            "document_id": self._get_field(doc, "document_id"),
            "text": self._get_field(doc, "text"),
            "metadata": metadata,
        }

    def _doc_to_search_result(self, doc: Any, score: float = 0.0) -> SearchResult:
        """Convert a zvec result doc to a SearchResult."""
        d = self._doc_to_dict(doc)
        return SearchResult(
            chunk_id=d["id"],
            text=d["text"],
            score=score,
            metadata=d["metadata"],
            document_id=d["document_id"],
        )

    def _results_to_search_results(
        self,
        results: Any,
        top_k: int,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Convert zvec query results to SearchResult list with optional post-filtering."""
        if not results:
            return []

        search_results = []
        for item in results:
            # zvec results may be (doc, score) tuples or scored objects
            if isinstance(item, tuple) and len(item) == 2:
                doc, score = item
            else:
                doc = item
                score = getattr(item, "score", 0.0)

            sr = self._doc_to_search_result(doc, score)

            # Post-filter by metadata if needed
            if filter_metadata:
                if not all(
                    str(sr.metadata.get(k)) == str(v) for k, v in filter_metadata.items()
                ):
                    continue

            search_results.append(sr)
            if len(search_results) >= top_k:
                break

        return search_results

    @staticmethod
    def _rrf_merge(
        dense: list[SearchResult],
        sparse: list[SearchResult],
        top_k: int,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        """Reciprocal Rank Fusion merge of two result lists."""
        scores: dict[str, float] = {}
        result_map: dict[str, SearchResult] = {}

        for rank, r in enumerate(dense):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            result_map[r.chunk_id] = r

        for rank, r in enumerate(sparse):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            if r.chunk_id not in result_map:
                result_map[r.chunk_id] = r

        sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k]
        return [
            SearchResult(
                chunk_id=cid,
                text=result_map[cid].text,
                score=scores[cid],
                metadata=result_map[cid].metadata,
                document_id=result_map[cid].document_id,
            )
            for cid in sorted_ids
        ]
