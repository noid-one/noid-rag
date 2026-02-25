"""Ingest command — parse, chunk, embed, and store."""

import asyncio
from pathlib import Path

import typer

from noid_rag.cli.app import app, state
from noid_rag.cli.display import console, print_error, print_success


@app.command()
def ingest(
    source: Path = typer.Argument(..., help="Path to document file"),
):
    """Parse, chunk, embed, and store a document."""
    from noid_rag.chunker import chunk as do_chunk
    from noid_rag.embeddings import EmbeddingClient
    from noid_rag.parser import parse as do_parse
    from noid_rag.vectorstore import PgVectorStore

    async def _ingest():
        with console.status(f"Parsing {source.name}..."):
            doc = do_parse(source, config=state.settings.parser)

        with console.status("Chunking..."):
            chunks = do_chunk(doc, config=state.settings.chunker)

        embed_client = EmbeddingClient(config=state.settings.embedding)
        with console.status(f"Embedding {len(chunks)} chunks..."):
            await embed_client.embed_chunks(chunks)

        async with PgVectorStore(config=state.settings.vectorstore) as store:
            with console.status("Storing..."):
                count = await store.upsert(chunks)

        return doc.id, count

    try:
        doc_id, count = asyncio.run(_ingest())
        print_success(f"Ingested {source.name}: {count} chunks stored (doc: {doc_id})")
    except Exception as e:
        print_error(f"Failed to ingest {source}: {e}")
        raise typer.Exit(1)
