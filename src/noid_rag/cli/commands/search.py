"""Search command — hybrid search with optional LLM answer."""

import asyncio
from pathlib import Path

import typer

from noid_rag.cli.app import app, state
from noid_rag.cli.display import (
    print_answer_result,
    print_error,
    print_search_results,
    print_warning,
)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query text"),
    top_k: int | None = typer.Option(None, "--top-k", "-k", help="Number of results"),
    answer: bool = typer.Option(True, "--answer/--no-answer", help="Synthesize an answer via LLM"),
    output: str | None = typer.Option(None, "--output", "-o", help="Export results to file"),
):
    """Search for similar documents."""
    from noid_rag.api import NoidRag

    async def _search():
        async with NoidRag(config=state.settings) as rag:
            if answer and state.settings.llm.api_key.get_secret_value():
                result = await rag.aanswer(query, top_k=top_k)
                print_answer_result(result)

                if output:
                    from noid_rag.export import export

                    export(result.sources, Path(output))
            else:
                if answer:
                    print_warning(
                        "LLM API key not set — falling back to search results. "
                        "Set NOID_RAG_LLM__API_KEY to enable answer synthesis."
                    )
                results = await rag.asearch(query, top_k=top_k)
                print_search_results(results)

                if output:
                    from noid_rag.export import export

                    export(results, Path(output))

    try:
        asyncio.run(_search())
    except Exception as e:
        print_error(f"Search failed: {e}")
        raise typer.Exit(1)
