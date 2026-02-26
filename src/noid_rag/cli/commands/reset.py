"""Reset command — drop and recreate the vector store table."""

import typer

from noid_rag.cli.app import app, state
from noid_rag.cli.display import print_error, print_success


@app.command()
def reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
):
    """Drop the vector store (removes all chunks and indexes)."""
    provider = state.settings.vectorstore.provider
    if provider == "qdrant":
        store_name = state.settings.qdrant.collection_name
    else:
        store_name = state.settings.vectorstore.table_name

    if not yes:
        typer.confirm(
            f"This will drop store '{store_name}' and all its data. Continue?", abort=True
        )

    try:
        from noid_rag.api import NoidRag

        rag = NoidRag(config=state.settings)
        rag.reset()
        print_success(f"Store '{store_name}' has been reset.")
    except Exception as e:
        print_error(f"Failed to reset: {e}")
        raise typer.Exit(1)
