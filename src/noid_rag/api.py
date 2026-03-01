"""NoidRag — programmatic API for agent skills."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from noid_rag.config import Settings
from noid_rag.models import AnswerResult, Chunk, Document, EvalSummary, SearchResult, TuneResult


class NoidRag:
    """High-level API for RAG operations.

    Sync methods use asyncio.run() internally. Async variants available.
    No Rich imports — CLI layer handles formatting.
    All returns are plain dicts/dataclasses.

    Supports async context manager for connection reuse:
        async with NoidRag() as rag:
            await rag.aingest("doc.pdf")
            result = await rag.aanswer("query")
    """

    def __init__(self, config: dict[str, Any] | Settings | None = None):
        if isinstance(config, Settings):
            self.settings = config
        elif isinstance(config, dict):
            self.settings = Settings.load(**config)
        else:
            self.settings = Settings.load()

        self._embed_client: Any | None = None
        self._llm_client: Any | None = None

    def _get_embed_client(self) -> Any:
        """Return a shared EmbeddingClient, creating lazily."""
        if self._embed_client is None:
            if self.settings.embedding.provider == "zvec":
                from noid_rag.embeddings_zvec import ZvecEmbeddingClient

                self._embed_client = ZvecEmbeddingClient(config=self.settings.embedding)
            else:
                from noid_rag.embeddings import EmbeddingClient

                self._embed_client = EmbeddingClient(config=self.settings.embedding)
        return self._embed_client

    def _get_llm_client(self) -> Any:
        """Return a shared LLMClient, creating lazily."""
        if self._llm_client is None:
            from noid_rag.llm import LLMClient

            self._llm_client = LLMClient(config=self.settings.llm)
        return self._llm_client

    async def close(self) -> None:
        """Close shared HTTP clients."""
        if self._embed_client is not None:
            await self._embed_client.close()
            self._embed_client = None
        if self._llm_client is not None:
            await self._llm_client.close()
            self._llm_client = None

    async def __aenter__(self) -> NoidRag:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    # --- Sync API ---

    def parse(self, source: str | Path) -> Document:
        """Parse a document, return Document dataclass."""
        from noid_rag.parser import parse as do_parse

        return do_parse(source, config=self.settings.parser)

    def chunk(self, source: str | Path) -> list[Chunk]:
        """Parse and chunk a document."""
        from noid_rag.chunker import chunk as do_chunk

        doc = self.parse(source)
        return do_chunk(doc, config=self.settings.chunker)

    def ingest(self, source: str | Path) -> dict[str, Any]:
        """Parse, chunk, embed, and store a document."""
        return asyncio.run(self.aingest(source))

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Hybrid search (vector + keyword with RRF)."""
        return asyncio.run(self.asearch(query, top_k=top_k))

    def answer(self, query: str, top_k: int | None = None) -> AnswerResult:
        """Search and synthesize an answer via LLM."""
        return asyncio.run(self.aanswer(query, top_k=top_k))

    def eval(self, dataset: str | Path, top_k: int | None = None) -> EvalSummary:
        """Evaluate RAG pipeline against a test dataset."""
        return asyncio.run(self.aeval(dataset, top_k=top_k))

    def generate(
        self,
        output: str | Path,
        num_questions: int | None = None,
        model: str | None = None,
        num_chunks: int | None = None,
        strategy: str | None = None,
    ) -> dict[str, Any]:
        """Generate a synthetic eval dataset from indexed documents."""
        return asyncio.run(
            self.agenerate(
                output,
                num_questions=num_questions,
                model=model,
                num_chunks=num_chunks,
                strategy=strategy,
            )
        )

    def tune(
        self,
        dataset: str | Path,
        sources: list[str | Path],
        progress_callback: Callable[[int, int, float], None] | None = None,
    ) -> TuneResult:
        """Run Bayesian hyperparameter optimization."""
        from noid_rag.tune import run_tune

        return run_tune(
            str(dataset),
            [str(s) for s in sources],
            self.settings,
            progress_callback=progress_callback,
        )

    async def atune(
        self,
        dataset: str | Path,
        sources: list[str | Path],
        progress_callback: Callable[[int, int, float], None] | None = None,
    ) -> TuneResult:
        """Async: run Bayesian hyperparameter optimization."""
        from noid_rag.tune import arun_tune

        return await arun_tune(
            str(dataset),
            [str(s) for s in sources],
            self.settings,
            progress_callback=progress_callback,
        )

    def reset(self) -> None:
        """Drop the vector store table so it can be recreated on next ingest."""
        asyncio.run(self.areset())

    def batch(self, directory: str | Path, pattern: str = "*") -> dict[str, Any]:
        """Batch process a directory."""
        return asyncio.run(self.abatch(directory, pattern=pattern))

    # --- Async API ---

    async def areset(self) -> None:
        """Async: drop the vector store."""
        from noid_rag.vectorstore_factory import create_vectorstore

        async with create_vectorstore(self.settings) as store:
            await store.drop()

    async def aingest(self, source: str | Path) -> dict[str, Any]:
        """Async: parse, chunk, embed, and store."""
        from noid_rag.chunker import chunk as do_chunk
        from noid_rag.vectorstore_factory import create_vectorstore

        doc = self.parse(source)
        chunks = do_chunk(doc, config=self.settings.chunker)

        embed_client = self._get_embed_client()
        await embed_client.embed_chunks(chunks)

        async with create_vectorstore(self.settings) as store:
            deleted, count = await store.replace_document(doc.id, chunks)

        return {"chunks_stored": count, "chunks_deleted": deleted, "document_id": doc.id}

    async def asearch(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Async: hybrid search (vector + keyword with RRF)."""
        from noid_rag.vectorstore_factory import create_vectorstore

        top_k = self.settings.search.top_k if top_k is None else top_k
        rrf_k = self.settings.search.rrf_k

        embed_client = self._get_embed_client()
        query_embedding = await embed_client.embed_query(query)

        async with create_vectorstore(self.settings) as store:
            return await store.hybrid_search(
                query_embedding,
                query,
                top_k=top_k,
                rrf_k=rrf_k,
            )

    async def aanswer(self, query: str, top_k: int | None = None) -> AnswerResult:
        """Async: search and synthesize an answer via LLM."""
        results = await self.asearch(query, top_k=top_k)  # top_k=None resolved in asearch

        if not results:
            return AnswerResult(
                answer="No relevant documents were found for your query.",
                sources=[],
                model=self.settings.llm.model,
            )

        context = "\n\n---\n\n".join(f"[Source: {r.document_id}]\n{r.text}" for r in results)

        llm = self._get_llm_client()
        answer_text = await llm.generate(query, context)

        return AnswerResult(
            answer=answer_text,
            sources=results,
            model=self.settings.llm.model,
        )

    async def aeval(self, dataset: str | Path, top_k: int | None = None) -> EvalSummary:
        """Async: evaluate RAG pipeline against a test dataset."""
        from noid_rag.eval import run_evaluation

        return await run_evaluation(  # top_k=None resolved in run_evaluation
            dataset,
            self.settings.eval,
            self.settings,
            self,
            top_k=top_k,
        )

    async def agenerate(
        self,
        output: str | Path,
        num_questions: int | None = None,
        model: str | None = None,
        num_chunks: int | None = None,
        strategy: str | None = None,
    ) -> dict[str, Any]:
        """Async: generate a synthetic eval dataset from indexed documents."""
        from noid_rag.generate import run_generate

        return await run_generate(
            settings=self.settings,
            output=Path(output),
            num_questions=num_questions,
            model=model,
            num_chunks=num_chunks,
            strategy=strategy,
        )

    async def abatch(self, directory: str | Path, pattern: str = "*") -> dict[str, Any]:
        """Async: batch process a directory.

        Reuses a single store connection for the entire batch to avoid
        per-file connection overhead (collection-existence checks, etc.).
        """
        from dataclasses import asdict

        from noid_rag.batch import BatchProcessor
        from noid_rag.chunker import chunk as do_chunk
        from noid_rag.vectorstore_factory import create_vectorstore

        directory = Path(directory)
        files = sorted(f for f in directory.glob(pattern) if f.is_file())

        processor = BatchProcessor(config=self.settings.batch)

        async with create_vectorstore(self.settings) as store:
            embed_client = self._get_embed_client()

            async def process_one(file_path: Path) -> dict[str, Any]:
                doc = self.parse(file_path)
                chunks = do_chunk(doc, config=self.settings.chunker)
                await embed_client.embed_chunks(chunks)
                deleted, count = await store.replace_document(doc.id, chunks)
                return {"chunks_stored": count, "chunks_deleted": deleted, "document_id": doc.id}

            result = await processor.process(files, process_one)
        return asdict(result)
