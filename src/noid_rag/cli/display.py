"""Rich console display helpers for the CLI."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

console = Console()
error_console = Console(stderr=True)


def print_success(message: str) -> None:
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str) -> None:
    error_console.print(f"[red]✗[/red] {message}")


def print_warning(message: str) -> None:
    console.print(f"[yellow]![/yellow] {message}")


def print_document(doc: Any) -> None:
    """Display a parsed document."""
    from noid_rag.models import Document
    if not isinstance(doc, Document):
        return
    console.print(Panel(
        doc.content[:2000] + ("..." if len(doc.content) > 2000 else ""),
        title=f"[bold]{doc.source}[/bold]",
        subtitle=f"ID: {doc.id} | {len(doc.content)} chars",
    ))
    if doc.metadata:
        table = Table(title="Metadata", show_header=True)
        table.add_column("Key", style="cyan")
        table.add_column("Value")
        for k, v in doc.metadata.items():
            table.add_row(str(k), str(v))
        console.print(table)


def print_chunks(chunks: list[Any]) -> None:
    """Display chunks in a table."""
    table = Table(title=f"Chunks ({len(chunks)})", show_header=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("ID", style="cyan", width=16)
    table.add_column("Text", max_width=80)
    table.add_column("Tokens", justify="right", width=8)

    for i, chunk in enumerate(chunks):
        text_preview = chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
        # Rough token estimate
        tokens = len(chunk.text.split())
        table.add_row(str(i + 1), chunk.id, text_preview, str(tokens))

    console.print(table)


def print_search_results(results: list[Any]) -> None:
    """Display search results."""
    if not results:
        print_warning("No results found.")
        return

    for i, r in enumerate(results):
        console.print(Panel(
            r.text,
            title=f"[bold]#{i+1}[/bold] Score: {r.score:.4f}",
            subtitle=f"Chunk: {r.chunk_id} | Doc: {r.document_id}",
        ))


def print_answer_result(result: Any) -> None:
    """Display an LLM-synthesized answer with its sources."""
    from noid_rag.models import AnswerResult
    if not isinstance(result, AnswerResult):
        return

    if result.sources:
        console.print("[bold]Sources:[/bold]")
        for i, r in enumerate(result.sources):
            console.print(Panel(
                r.text,
                title=f"[bold]#{i+1}[/bold] Score: {r.score:.4f}",
                subtitle=f"Chunk: {r.chunk_id} | Doc: {r.document_id}",
            ))

    console.print(Panel(
        result.answer,
        title="[bold green]Answer[/bold green]",
        subtitle=f"Model: {result.model}",
        border_style="green",
    ))


def print_stats(stats: dict[str, Any]) -> None:
    """Display vector store statistics."""
    table = Table(title="Vector Store Stats", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    for k, v in stats.items():
        table.add_row(str(k), str(v))
    console.print(table)


def _score_color(score: float) -> str:
    """Return Rich color based on score threshold."""
    if score >= 0.7:
        return "green"
    elif score >= 0.4:
        return "yellow"
    return "red"


def _score_rating(score: float) -> str:
    """Return a text rating for a score."""
    if score >= 0.9:
        return "Excellent"
    elif score >= 0.7:
        return "Good"
    elif score >= 0.4:
        return "Fair"
    return "Poor"


def print_eval_summary(summary: Any, verbose: bool = False) -> None:
    """Display evaluation results."""
    from noid_rag.models import EvalSummary
    if not isinstance(summary, EvalSummary):
        return

    # Summary table
    table = Table(title="Evaluation Results", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Rating")

    for metric, score in summary.mean_scores.items():
        color = _score_color(score)
        rating = _score_rating(score)
        table.add_row(metric, f"[{color}]{score:.4f}[/{color}]", f"[{color}]{rating}[/{color}]")

    console.print(table)
    console.print(
        f"\n[dim]Questions: {summary.total_questions} | "
        f"Backend: {summary.backend} | "
        f"Model: {summary.model} | "
        f"Dataset: {summary.dataset_path}[/dim]"
    )

    if verbose:
        print_eval_details(summary)


def print_eval_details(summary: Any) -> None:
    """Display per-question evaluation breakdown."""
    for i, r in enumerate(summary.results):
        console.print()
        score_parts = [
            f"[{_score_color(s)}]{m}: {s:.3f}[/{_score_color(s)}]"
            for m, s in r.scores.items()
        ]
        scores_str = " | ".join(score_parts) if score_parts else "no scores"

        answer_preview = r.answer[:200] + "..." if len(r.answer) > 200 else r.answer
        console.print(Panel(
            f"[bold]Q:[/bold] {r.question}\n"
            f"[bold]A:[/bold] {answer_preview}\n"
            f"[bold]Scores:[/bold] {scores_str}",
            title=f"[bold]Question #{i + 1}[/bold]",
            subtitle=f"Contexts: {len(r.contexts)}",
        ))


def print_generate_summary(summary: dict[str, Any]) -> None:
    """Display generation results summary."""
    table = Table(title="Generation Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Questions generated", str(summary["questions_generated"]))
    table.add_row("Chunks sampled", str(summary["chunks_sampled"]))
    table.add_row("Model", summary["model"])
    table.add_row("Strategy", summary["strategy"])
    table.add_row("Output", summary["output_path"])

    console.print(table)
    print_success(f"Dataset saved to {summary['output_path']}")


def create_progress() -> Progress:
    """Create a Rich progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    )
