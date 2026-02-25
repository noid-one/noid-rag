"""Generate command — create synthetic eval datasets from indexed documents."""

import typer

from noid_rag.cli.app import app, state
from noid_rag.cli.display import print_error, print_generate_summary


@app.command(name="generate")
def generate_command(
    output: str = typer.Option(..., "--output", "-o", help="Output file path (.yml or .json)"),
    num_questions: int = typer.Option(
        None, "--num-questions", "-n", help="Total Q&A pairs to generate"
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Override generation model"),
    chunks: int | None = typer.Option(
        None, "--chunks", "-c", help="Number of source chunks to sample"
    ),
    strategy: str | None = typer.Option(
        None, "--strategy", "-s", help="Chunk selection: random or diverse"
    ),
):
    """Generate a synthetic eval dataset from indexed documents."""
    import asyncio
    from pathlib import Path

    from noid_rag.cli.display import create_progress

    try:
        if strategy and strategy not in ("random", "diverse"):
            print_error(f"Invalid strategy {strategy!r}. Choose from: random, diverse")
            raise typer.Exit(1)

        output_path = Path(output)
        if output_path.suffix.lower() not in (".yml", ".yaml", ".json"):
            print_error("Output file must be .yml, .yaml, or .json")
            raise typer.Exit(1)

        from noid_rag.generate import run_generate

        gen_config = state.settings.generate
        final_num_questions = num_questions or gen_config.num_questions
        qpc = gen_config.questions_per_chunk
        final_chunks = chunks or max(1, -(-final_num_questions // qpc))

        with create_progress() as progress:
            task = progress.add_task("Generating Q&A pairs...", total=final_chunks)

            def on_progress(chunk_index: int) -> None:
                progress.update(task, completed=chunk_index + 1)

            summary = asyncio.run(
                run_generate(
                    settings=state.settings,
                    output=output_path,
                    num_questions=num_questions,
                    model=model,
                    num_chunks=chunks,
                    strategy=strategy,
                    progress_callback=on_progress,
                )
            )

        print_generate_summary(summary)

    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Generation failed: {e}")
        if state.settings.verbose:
            import traceback

            from noid_rag.cli.display import error_console

            error_console.print(traceback.format_exc())
        raise typer.Exit(1)
