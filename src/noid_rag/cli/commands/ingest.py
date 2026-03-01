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
    from noid_rag.api import NoidRag

    rag = NoidRag(config=state.settings)

    async def _ingest():
        from noid_rag.chunker import chunk as do_chunk

        with console.status(f"Parsing {source.name}..."):
            doc = rag.parse(source)

        with console.status("Chunking..."):
            chunks = do_chunk(doc, config=state.settings.chunker)

        embed_client = rag.get_embed_client()
        with console.status(f"Embedding {len(chunks)} chunks..."):
            await embed_client.embed_chunks(chunks)

        store = await rag._get_store()
        with console.status("Storing..."):
            deleted, count = await store.replace_document(doc.id, chunks)

        await rag.close()
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
