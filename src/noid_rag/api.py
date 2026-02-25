"""NoidRag — programmatic API for agent skills."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from noid_rag.config import Settings
from noid_rag.models import AnswerResult, Chunk, Document, EvalSummary, SearchResult


class NoidRag:
    """High-level API for RAG operations.

    Sync methods use asyncio.run() internally. Async variants available.
    No Rich imports — CLI layer handles formatting.
    All returns are plain dicts/dataclasses.
    """

    def __init__(self, config: dict[str, Any] | Settings | None = None):
        if isinstance(config, Settings):
            self.settings = config
        elif isinstance(config, dict):
            self.settings = Settings.load(**config)
        else:
            self.settings = Settings.load()

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

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Hybrid search (vector + keyword with RRF)."""
        return asyncio.run(self.asearch(query, top_k=top_k))

    def answer(self, query: str, top_k: int = 5) -> AnswerResult:
        """Search and synthesize an answer via LLM."""
        return asyncio.run(self.aanswer(query, top_k=top_k))

    def eval(self, dataset: str | Path, top_k: int = 5) -> EvalSummary:
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
                output, num_questions=num_questions, model=model,
                num_chunks=num_chunks, strategy=strategy,
            )
        )

    def batch(self, directory: str | Path, pattern: str = "*") -> dict[str, Any]:
        """Batch process a directory."""
        return asyncio.run(self.abatch(directory, pattern=pattern))

    # --- Async API ---

    async def aingest(self, source: str | Path) -> dict[str, Any]:
        """Async: parse, chunk, embed, and store."""
        from noid_rag.chunker import chunk as do_chunk
        from noid_rag.embeddings import EmbeddingClient
        from noid_rag.vectorstore import PgVectorStore

        doc = self.parse(source)
        chunks = do_chunk(doc, config=self.settings.chunker)

        embed_client = EmbeddingClient(config=self.settings.embedding)
        await embed_client.embed_chunks(chunks)

        async with PgVectorStore(config=self.settings.vectorstore) as store:
            count = await store.upsert(chunks)

        return {"chunks_stored": count, "document_id": doc.id}

    async def asearch(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Async: hybrid search (vector + keyword with RRF)."""
        from noid_rag.embeddings import EmbeddingClient
        from noid_rag.vectorstore import PgVectorStore

        embed_client = EmbeddingClient(config=self.settings.embedding)
        query_embedding = await embed_client.embed_query(query)

        async with PgVectorStore(config=self.settings.vectorstore) as store:
            return await store.hybrid_search(query_embedding, query, top_k=top_k)

    async def aanswer(self, query: str, top_k: int = 5) -> AnswerResult:
        """Async: search and synthesize an answer via LLM."""
        from noid_rag.llm import LLMClient

        results = await self.asearch(query, top_k=top_k)

        if not results:
            return AnswerResult(
                answer="No relevant documents were found for your query.",
                sources=[],
                model=self.settings.llm.model,
            )

        context = "\n\n---\n\n".join(
            f"[Source: {r.document_id}]\n{r.text}" for r in results
        )

        llm = LLMClient(config=self.settings.llm)
        answer_text = await llm.generate(query, context)

        return AnswerResult(
            answer=answer_text,
            sources=results,
            model=self.settings.llm.model,
        )

    async def aeval(self, dataset: str | Path, top_k: int = 5) -> EvalSummary:
        """Async: evaluate RAG pipeline against a test dataset."""
        from noid_rag.eval import run_evaluation

        return await run_evaluation(
            dataset, self.settings.eval, self.settings, self, top_k=top_k,
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
        """Async: batch process a directory."""
        from dataclasses import asdict

        from noid_rag.batch import BatchProcessor

        directory = Path(directory)
        files = sorted(f for f in directory.glob(pattern) if f.is_file())

        processor = BatchProcessor(config=self.settings.batch)

        async def process_one(file_path: Path) -> dict[str, Any]:
            return await self.aingest(file_path)

        result = await processor.process(files, process_one)
        return asdict(result)
