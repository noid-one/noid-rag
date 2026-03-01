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

# zvec enforces a maximum topk of 1024 per query call.
_ZVEC_MAX_TOPK = 1024


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

        # Try to init BM25 before schema creation so we know whether to
        # include the sparse vector field (zvec fields are not nullable).
        try:
            self._bm25_fn = zvec.BM25EmbeddingFunction()
        except Exception as exc:
            logger.warning("BM25 not available (%s); keyword/hybrid search will be limited", exc)
            self._bm25_fn = None

        # Build HNSW index params if configured.
        # COSINE metric is correct for embedding similarity; IP (inner product) is only
        # equivalent to cosine when vectors are L2-normalised, which is not guaranteed here.
        index_param = None
        if self.config.index_type == "hnsw":
            index_param = zvec.HnswIndexParam(
                m=self.config.hnsw_m,
                ef_construction=self.config.hnsw_ef_construction,
                metric_type=zvec.MetricType.COSINE,
            )

        vector_fields = [
            zvec.VectorSchema(
                "embedding",
                zvec.DataType.VECTOR_FP32,
                dimension=self.embedding_dim,
                index_param=index_param,
            ),
        ]
        if self._bm25_fn is not None:
            vector_fields.append(
                zvec.VectorSchema("bm25_sparse", zvec.DataType.SPARSE_VECTOR_FP32)
            )

        schema = zvec.CollectionSchema(
            name=self.config.collection_name,
            fields=[
                zvec.FieldSchema("document_id", zvec.DataType.STRING),
                zvec.FieldSchema("chunk_id", zvec.DataType.STRING),
                zvec.FieldSchema("text", zvec.DataType.STRING),
                zvec.FieldSchema("metadata_json", zvec.DataType.STRING),
            ],
            vectors=vector_fields,
        )

        str_path = str(collection_path)

        def _open() -> Any:
            if collection_path.exists():
                return zvec.open(str_path)
            return zvec.create_and_open(str_path, schema)

        self._collection = await asyncio.to_thread(_open)

    def _get_collection(self) -> Any:
        if self._collection is None:
            raise RuntimeError("ZvecVectorStore not connected. Call connect() first.")
        return self._collection

    def _get_zvec(self) -> Any:
        """Return the imported zvec module, raising if not connected."""
        if self._zvec is None:
            raise RuntimeError("ZvecVectorStore not connected. Call connect() first.")
        return self._zvec

    async def upsert(self, chunks: list[Chunk]) -> int:
        """Upsert chunks into the collection. Returns count of upserted docs."""
        if not chunks:
            return 0

        zvec = self._get_zvec()
        collection = self._get_collection()

        docs = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.id} has no embedding")

            vectors: dict[str, Any] = {"embedding": chunk.embedding}
            if self._bm25_fn is not None:
                try:
                    vectors["bm25_sparse"] = self._bm25_fn.embed(chunk.text)
                except Exception:
                    logger.debug("Failed to encode BM25 sparse vector for chunk %s", chunk.id)

            doc = zvec.Doc(
                id=chunk.id,
                fields={
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "metadata_json": json.dumps(chunk.metadata),
                },
                vectors=vectors,
            )
            docs.append(doc)

        await asyncio.to_thread(collection.upsert, docs)
        return len(docs)

    async def replace_document(
        self, document_id: str, chunks: list[Chunk]
    ) -> tuple[int, int]:
        """Replace all chunks for a document: delete old, then insert new.

        Because zvec's delete_by_filter targets all records matching the filter
        (including any newly upserted chunks with the same document_id), the
        only safe ordering is: count + delete first, then insert.

        Returns (deleted, inserted).
        """
        collection = self._get_collection()

        # Count existing chunks before any mutation
        old_results = await asyncio.to_thread(
            collection.query,
            filter=f"document_id='{document_id}'",
            topk=_ZVEC_MAX_TOPK,
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
        zvec = self._get_zvec()

        fetch_k = top_k * 3 if filter_metadata else top_k

        results = await asyncio.to_thread(
            collection.query,
            zvec.VectorQuery(field_name="embedding", vector=embedding),
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
        zvec = self._get_zvec()

        fetch_k = top_k * 3 if filter_metadata else top_k

        try:
            sparse_vector = self._bm25_fn.embed(query)
            results = await asyncio.to_thread(
                collection.query,
                zvec.VectorQuery(field_name="bm25_sparse", vector=sparse_vector),
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
        zvec = self._get_zvec()
        collection = self._get_collection()

        fetch_k = top_k * 3 if filter_metadata else top_k

        # Try native multi-vector RRF
        if self._bm25_fn is not None and hasattr(zvec, "RrfReRanker"):
            try:
                sparse_vector = self._bm25_fn.embed(query)
                results = await asyncio.to_thread(
                    collection.query,
                    [
                        zvec.VectorQuery(field_name="embedding", vector=embedding),
                        zvec.VectorQuery(
                            field_name="bm25_sparse", vector=sparse_vector
                        ),
                    ],
                    topk=fetch_k,
                    reranker=zvec.RrfReRanker(rank_constant=rrf_k, topn=fetch_k),
                )
                return self._results_to_search_results(results, top_k, filter_metadata)
            except Exception:
                logger.debug(
                    "Native zvec hybrid search failed, falling back to Python RRF"
                )

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
                collection.query, topk=limit
            )
            return [self._doc_to_dict(doc) for doc in (results or [])][:limit]
        else:
            # Diverse: over-fetch, round-robin by document_id
            over_fetch = limit * 5
            results = await asyncio.to_thread(
                collection.query, topk=over_fetch
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
            collection.query,
            filter=f"document_id='{document_id}'",
            topk=_ZVEC_MAX_TOPK,
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

        # Use native stats for chunk count
        col_stats = await asyncio.to_thread(lambda: collection.stats)
        total_chunks = col_stats.doc_count

        # Count distinct document_ids by querying all docs.
        # NOTE: zvec enforces a hard cap of _ZVEC_MAX_TOPK (1024) results per query.
        # For collections with more than 1024 chunks the document count below will be
        # a lower-bound, not an exact figure.  The chunk count from native stats is
        # always accurate.
        doc_ids: set[str] = set()
        if total_chunks > 0:
            if total_chunks > _ZVEC_MAX_TOPK:
                logger.debug(
                    "stats(): collection has %d chunks but zvec query cap is %d; "
                    "total_documents may be undercounted",
                    total_chunks,
                    _ZVEC_MAX_TOPK,
                )
            results = await asyncio.to_thread(
                collection.query, topk=min(total_chunks, _ZVEC_MAX_TOPK)
            )
            for doc in results or []:
                doc_id = doc.fields.get("document_id", "") if doc.fields else ""
                if doc_id:
                    doc_ids.add(doc_id)

        return {
            "total_chunks": total_chunks,
            "total_documents": len(doc_ids),
            "store_name": self.config.collection_name,
            "embedding_dim": self.embedding_dim,
        }

    async def close(self) -> None:
        """Release the collection handle.

        zvec collections have no explicit close method; flush pending writes
        and release the reference so the GC can reclaim resources.
        """
        if self._collection is not None:
            try:
                await asyncio.to_thread(self._collection.flush)
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
    def _doc_to_dict(doc: Any) -> dict[str, Any]:
        """Convert a zvec Doc to a standard dict."""
        fields = doc.fields or {}
        metadata_str = fields.get("metadata_json", "")
        try:
            metadata = json.loads(metadata_str) if metadata_str else {}
        except (json.JSONDecodeError, TypeError):
            metadata = {}

        return {
            "id": fields.get("chunk_id", doc.id),
            "document_id": fields.get("document_id", ""),
            "text": fields.get("text", ""),
            "metadata": metadata,
        }

    def _doc_to_search_result(self, doc: Any) -> SearchResult:
        """Convert a zvec Doc to a SearchResult."""
        d = self._doc_to_dict(doc)
        return SearchResult(
            chunk_id=d["id"],
            text=d["text"],
            score=doc.score if doc.score is not None else 0.0,
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
        for doc in results:
            sr = self._doc_to_search_result(doc)

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
