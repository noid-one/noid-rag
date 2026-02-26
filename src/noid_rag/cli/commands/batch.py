"""Batch command — batch process files."""

import asyncio
from pathlib import Path
from typing import Optional

import typer

from noid_rag.cli.app import app, state
from noid_rag.cli.display import console, create_progress, print_error, print_success, print_warning


@app.command()
def batch(
    directory: Path = typer.Argument(..., help="Directory containing documents"),
    pattern: str = typer.Option("*", "--pattern", "-p", help="File glob pattern"),
    dry_run: bool = typer.Option(False, "--dry-run", help="List files without processing"),
    retry_run: Optional[str] = typer.Option(
        None, "--retry", help="Retry failed files from a previous run ID"
    ),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Export results to file"),
):
    """Batch process documents in a directory."""
    from noid_rag.batch import BatchProcessor
    from noid_rag.chunker import chunk as do_chunk
    from noid_rag.embeddings import EmbeddingClient
    from noid_rag.export import export
    from noid_rag.parser import parse as do_parse
    from noid_rag.vectorstore_factory import create_vectorstore

    processor = BatchProcessor(config=state.settings.batch)

    # Collect files
    if retry_run:
        failed = processor.get_failed_files(retry_run)
        files = [Path(f) for f in failed]
        if not files:
            print_warning(f"No failed files found for run {retry_run}")
            raise typer.Exit(0)
    else:
        files = sorted(directory.glob(pattern))
        files = [f for f in files if f.is_file()]

    if not files:
        print_warning("No files found matching pattern.")
        raise typer.Exit(0)

    console.print(f"Found {len(files)} files to process")

    if dry_run:
        for f in files:
            console.print(f"  [dim]{f}[/dim]")
        return

    async def _batch():
        async with EmbeddingClient(config=state.settings.embedding) as embed_client:
            async with create_vectorstore(state.settings) as store:

                async def process_one(file_path: Path) -> dict:
                    doc = do_parse(file_path, config=state.settings.parser)
                    chunks = do_chunk(doc, config=state.settings.chunker)
                    await embed_client.embed_chunks(chunks)
                    deleted, count = await store.replace_document(doc.id, chunks)
                    return {
                        "chunks_stored": count,
                        "chunks_replaced": deleted,
                        "document_id": doc.id,
                    }

                with create_progress() as progress:
                    task = progress.add_task("Processing...", total=len(files))

                    def on_progress(filename: str, current: int, total: int):
                        progress.update(
                            task, completed=current, description=f"Processing {filename}"
                        )

                    result = await processor.process(files, process_one, progress=on_progress)

        return result

    try:
        result = asyncio.run(_batch())
        print_success(
            f"Batch complete: {result.success} succeeded, {result.failed} failed, "
            f"{result.skipped} skipped (run: {result.run_id})"
        )
        if result.failed > 0:
            print_warning(f"Retry failed files: noid-rag batch {directory} --retry {result.run_id}")

        if output:
            export(result.files, output)
    except Exception as e:
        print_error(f"Batch failed: {e}")
        raise typer.Exit(1)
