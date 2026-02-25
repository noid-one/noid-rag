"""Info command — show vector store info."""

import asyncio

import typer

from noid_rag.cli.app import app, state
from noid_rag.cli.display import print_error, print_stats


@app.command()
def info():
    """Show vector store statistics."""
    from noid_rag.vectorstore import PgVectorStore

    async def _info():
        async with PgVectorStore(config=state.settings.vectorstore) as store:
            return await store.stats()

    try:
        stats = asyncio.run(_info())
        print_stats(stats)
    except Exception as e:
        print_error(f"Failed to get info: {e}")
        raise typer.Exit(1)
