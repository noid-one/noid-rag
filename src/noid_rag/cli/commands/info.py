"""Info command — show vector store info."""

import asyncio

import typer

from noid_rag.cli.app import app, state
from noid_rag.cli.display import print_error, print_stats


@app.command()
def info():
    """Show vector store statistics."""
    from noid_rag.api import NoidRag

    async def _info():
        async with NoidRag(config=state.settings) as rag:
            return await rag.astats()

    try:
        stats = asyncio.run(_info())
        print_stats(stats)
    except Exception as e:
        print_error(f"Failed to get info: {e}")
        raise typer.Exit(1)
