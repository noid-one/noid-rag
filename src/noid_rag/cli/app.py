"""Typer application with global options."""

from __future__ import annotations

from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Optional

import typer

from noid_rag.config import Settings

app = typer.Typer(
    name="noid-rag",
    help="Production RAG CLI & Agent Skill",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


# Global state
class State:
    settings: Settings = Settings()


state = State()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"noid-rag {pkg_version('noid-rag')}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: Optional[bool] = typer.Option(
        None, "--version", callback=_version_callback, is_eager=True, help="Show version and exit"
    ),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    """noid-rag: Production RAG CLI & Agent Skill."""
    state.settings = Settings.load(config_file=config, verbose=verbose)
