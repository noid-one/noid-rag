"""Tests for the reset command and drop() method."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from noid_rag.config import VectorStoreConfig
from noid_rag.vectorstore import PgVectorStore


def _make_async_cm(return_value):
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=return_value)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


class TestDrop:
    @pytest.mark.asyncio
    @patch("noid_rag.vectorstore.create_async_engine")
    async def test_drop_executes_drop_table(self, mock_create_engine):
        mock_conn = AsyncMock()
        mock_engine = MagicMock()
        mock_engine.begin.return_value = _make_async_cm(mock_conn)
        mock_engine.dispose = AsyncMock()
        mock_create_engine.return_value = mock_engine

        config = VectorStoreConfig(
            dsn="postgresql+asyncpg://test@localhost/test",
            table_name="my_docs",
        )
        store = PgVectorStore(config=config)
        store._engine = mock_engine

        await store.drop()

        mock_conn.execute.assert_called_once()
        sql = str(mock_conn.execute.call_args[0][0].text)
        assert "DROP TABLE IF EXISTS my_docs" in sql


class TestNoidRagReset:
    @pytest.mark.asyncio
    @patch("noid_rag.vectorstore_factory.create_vectorstore")
    async def test_areset_calls_drop(self, mock_create_vs):
        from noid_rag.api import NoidRag

        mock_store = AsyncMock()
        mock_store.__aenter__ = AsyncMock(return_value=mock_store)
        mock_store.__aexit__ = AsyncMock(return_value=False)
        mock_create_vs.return_value = mock_store

        rag = NoidRag()
        await rag.areset()
        mock_store.drop.assert_called_once()

    @patch("noid_rag.api.NoidRag.areset", new_callable=AsyncMock)
    def test_reset_calls_areset(self, mock_areset):
        from noid_rag.api import NoidRag

        rag = NoidRag()
        rag.reset()
        mock_areset.assert_called_once()


class TestResetCLI:
    @patch("noid_rag.vectorstore_factory.create_vectorstore")
    def test_reset_with_yes_flag(self, mock_create_vs):
        from noid_rag.cli.main import app

        mock_store = AsyncMock()
        mock_store.__aenter__ = AsyncMock(return_value=mock_store)
        mock_store.__aexit__ = AsyncMock(return_value=False)
        mock_create_vs.return_value = mock_store

        runner = CliRunner()
        result = runner.invoke(app, ["reset", "--yes"])

        assert result.exit_code == 0
        mock_store.drop.assert_called_once()

    def test_reset_aborts_without_confirmation(self):
        from noid_rag.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["reset"], input="n\n")

        assert result.exit_code != 0

    @patch("noid_rag.vectorstore_factory.create_vectorstore")
    def test_reset_exits_1_on_error(self, mock_create_vs):
        from noid_rag.cli.main import app

        mock_store = AsyncMock()
        mock_store.__aenter__ = AsyncMock(return_value=mock_store)
        mock_store.__aexit__ = AsyncMock(return_value=False)
        mock_store.drop.side_effect = RuntimeError("connection refused")
        mock_create_vs.return_value = mock_store

        runner = CliRunner()
        result = runner.invoke(app, ["reset", "--yes"])

        assert result.exit_code == 1
