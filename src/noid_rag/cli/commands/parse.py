"""Parse command — parse documents with Docling."""

from pathlib import Path
from typing import Optional

import typer

from noid_rag.cli.app import app, state
from noid_rag.cli.display import console, print_document, print_error, print_success


@app.command()
def parse(
    source: Path = typer.Argument(..., help="Path to document file"),
    show: bool = typer.Option(False, "--show", "-s", help="Display parsed content"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save markdown to file"),
):
    """Parse a document using Docling."""
    from noid_rag.parser import parse as do_parse

    try:
        with console.status(f"Parsing {source.name}..."):
            doc = do_parse(source, config=state.settings.parser)

        print_success(f"Parsed {source.name} ({len(doc.content)} chars, ID: {doc.id})")

        if show:
            print_document(doc)

        if output:
            output.write_text(doc.content)
            print_success(f"Saved to {output}")
    except Exception as e:
        print_error(f"Failed to parse {source}: {e}")
        raise typer.Exit(1)
