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
    from noid_rag.vectorstore_factory import create_vectorstore

    async def _ingest():
        with console.status(f"Parsing {source.name}..."):
            doc = do_parse(source, config=state.settings.parser)

        with console.status("Chunking..."):
            chunks = do_chunk(doc, config=state.settings.chunker)

        async with EmbeddingClient(config=state.settings.embedding) as embed_client:
            with console.status(f"Embedding {len(chunks)} chunks..."):
                await embed_client.embed_chunks(chunks)

        async with create_vectorstore(state.settings) as store:
            with console.status("Storing..."):
                deleted, count = await store.replace_document(doc.id, chunks)

        return doc.id, count, deleted

    try:
        doc_id, count, deleted = asyncio.run(_ingest())
        msg = f"Ingested {source.name}: {count} chunks stored (doc: {doc_id})"
        if deleted:
            msg += f", {deleted} previous chunks replaced"
        print_success(msg)
    except Exception as e:
        print_error(f"Failed to ingest {source}: {e}")
        raise typer.Exit(1)
