"""Tune command — Bayesian hyperparameter optimization."""

from pathlib import Path
from typing import Optional

import typer

from noid_rag.cli.app import app, state
from noid_rag.cli.display import console, create_progress, print_error, print_success


@app.command(name="tune")
def tune_command(
    dataset: str = typer.Argument(..., help="Path to YAML/JSON eval dataset"),
    source: list[str] = typer.Option(
        ..., "--source", "-s", help="Document(s) to ingest for tuning"
    ),
    max_trials: Optional[int] = typer.Option(
        None, "--max-trials", "-n", help="Maximum number of trials"
    ),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Export results to JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show per-trial details"),
):
    """Optimize RAG parameters via Bayesian search (Optuna)."""
    from noid_rag.api import NoidRag

    try:
        settings = state.settings
        if max_trials is not None:
            settings = settings.model_copy(
                update={"tune": settings.tune.model_copy(update={"max_trials": max_trials})}
            )

        total = settings.tune.max_trials
        rag = NoidRag(config=settings)

        progress = create_progress()
        task_id = None

        def on_progress(trial_num: int, total_trials: int, best_score: float) -> None:
            nonlocal task_id
            if task_id is None:
                task_id = progress.add_task(f"Tuning ({total_trials} trials)", total=total_trials)
            progress.update(
                task_id,
                completed=trial_num,
                description=f"Trial {trial_num}/{total_trials} | Best: {best_score:.4f}",
            )

        with progress:
            result = rag.tune(dataset, source, progress_callback=on_progress)

        # Display results
        from rich.table import Table

        console.print()
        table = Table(title="Best Parameters", show_header=True)
        table.add_column("Section", style="cyan")
        table.add_column("Parameter", style="bold")
        table.add_column("Value", justify="right")

        for section, params in result.best_params.items():
            for param, value in params.items():
                table.add_row(section, param, str(value))

        console.print(table)
        console.print(
            f"\n[bold green]Best score:[/bold green] {result.best_score:.4f}"
            f"  [dim](mean of {', '.join(result.metrics_used)})[/dim]"
        )
        console.print(f"[dim]Trials: {result.total_trials}/{total}[/dim]")

        if verbose:
            console.print()
            trials_table = Table(title="All Trials", show_header=True)
            trials_table.add_column("#", style="dim", width=4)
            trials_table.add_column("Score", justify="right")
            trials_table.add_column("Parameters")

            for t in sorted(result.all_trials, key=lambda x: x["score"], reverse=True):
                param_parts = []
                for sec, params in t["params"].items():
                    for k, v in params.items():
                        param_parts.append(f"{sec}.{k}={v}")
                trials_table.add_row(
                    str(t["trial_number"]),
                    f"{t['score']:.4f}",
                    ", ".join(param_parts),
                )
            console.print(trials_table)

        if output:
            import json
            from dataclasses import asdict

            out_path = Path(output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(asdict(result), f, indent=2)
            print_success(f"Results exported to {out_path}")

    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Tuning failed: {e}")
        if state.settings.verbose:
            import traceback

            from noid_rag.cli.display import error_console

            error_console.print(traceback.format_exc())
        raise typer.Exit(1)
