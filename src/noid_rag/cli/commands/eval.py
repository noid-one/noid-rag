"""Eval command — evaluate RAG pipeline quality."""

import typer

from noid_rag.cli.app import app, state
from noid_rag.cli.display import console, print_error, print_eval_summary, print_success


@app.command(name="eval")
def eval_command(
    dataset: str = typer.Argument(..., help="Path to YAML/JSON eval dataset"),
    backend: str | None = typer.Option(None, "--backend", "-b", help="ragas or promptfoo"),
    top_k: int | None = typer.Option(
        None, "--top-k", "-k", help="Contexts to retrieve per question",
    ),
    metrics: str | None = typer.Option(
        None, "--metrics", "-m", help="Comma-separated metric names"
    ),
    output: str | None = typer.Option(None, "--output", "-o", help="Export results to JSON"),
    no_save: bool = typer.Option(False, "--no-save", help="Skip saving to eval history"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Per-question breakdown"),
):
    """Evaluate RAG pipeline quality against a test dataset."""
    import asyncio

    from noid_rag.api import NoidRag

    try:
        valid_backends = ("ragas", "promptfoo")
        eval_config = state.settings.eval.model_copy()
        if backend:
            if backend not in valid_backends:
                print_error(
                    f"Invalid backend {backend!r}. "
                    f"Choose from: {', '.join(valid_backends)}"
                )
                raise typer.Exit(1)
            eval_config.backend = backend
        if metrics:
            eval_config.metrics = [m.strip() for m in metrics.split(",")]
        if no_save:
            eval_config.save_results = False

        rag = NoidRag(config=state.settings)

        with console.status("Running evaluation..."):
            from noid_rag.eval import run_evaluation

            summary = asyncio.run(
                run_evaluation(dataset, eval_config, state.settings, rag, top_k=top_k)
            )

        print_eval_summary(summary, verbose=verbose)

        if output:
            import json
            from dataclasses import asdict
            from pathlib import Path

            out_path = Path(output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(asdict(summary), f, indent=2)
            print_success(f"Results exported to {out_path}")

    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Evaluation failed: {e}")
        if state.settings.verbose:
            import traceback

            from noid_rag.cli.display import error_console

            error_console.print(traceback.format_exc())
        raise typer.Exit(1)
