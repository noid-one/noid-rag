"""Chunk command — chunk documents."""

from pathlib import Path
from typing import Optional

import typer

from noid_rag.cli.app import app, state
from noid_rag.cli.display import console, print_chunks, print_error, print_success


@app.command()
def chunk(
    source: Path = typer.Argument(..., help="Path to document file"),
    method: str = typer.Option("hybrid", "--method", "-m", help="Chunking method: hybrid or fixed"),
    max_tokens: int = typer.Option(512, "--max-tokens", "-t", help="Max tokens per chunk"),
    show: bool = typer.Option(False, "--show", "-s", help="Display chunks"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Export chunks to file"),
):
    """Parse and chunk a document."""
    from noid_rag.chunker import chunk as do_chunk
    from noid_rag.config import ChunkerConfig
    from noid_rag.export import export
    from noid_rag.parser import parse as do_parse

    try:
        with console.status(f"Parsing {source.name}..."):
            doc = do_parse(source, config=state.settings.parser)

        chunk_config = ChunkerConfig(method=method, max_tokens=max_tokens)
        with console.status("Chunking..."):
            chunks = do_chunk(doc, config=chunk_config)

        print_success(f"Created {len(chunks)} chunks from {source.name}")

        if show:
            print_chunks(chunks)

        if output:
            export(chunks, output)
            print_success(f"Exported to {output}")
    except Exception as e:
        print_error(f"Failed to chunk {source}: {e}")
        raise typer.Exit(1)
