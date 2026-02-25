"""CLI entry point."""

# Import command modules so @app.command() decorators register
import noid_rag.cli.commands.batch  # noqa: F401, E402
import noid_rag.cli.commands.chunk  # noqa: F401, E402
import noid_rag.cli.commands.info  # noqa: F401, E402
import noid_rag.cli.commands.ingest  # noqa: F401, E402
import noid_rag.cli.commands.parse  # noqa: F401, E402
import noid_rag.cli.commands.search  # noqa: F401, E402
from noid_rag.cli.app import app


def main():
    app()


if __name__ == "__main__":
    main()
