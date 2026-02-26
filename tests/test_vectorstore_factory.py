"""Tests for the vector store factory."""

from unittest.mock import MagicMock, patch

import pytest

from noid_rag.config import Settings


class TestCreateVectorstore:
    def test_pgvector_returns_pgvectorstore(self):
        from noid_rag.vectorstore_factory import create_vectorstore

        settings = Settings()
        store = create_vectorstore(settings)

        from noid_rag.vectorstore import PgVectorStore

        assert isinstance(store, PgVectorStore)

    def test_qdrant_returns_qdrantvectorstore(self):
        from noid_rag.vectorstore_factory import create_vectorstore

        settings = Settings()
        settings = settings.model_copy(
            update={
                "vectorstore": settings.vectorstore.model_copy(update={"provider": "qdrant"}),
            }
        )

        # Mock _import_qdrant so we don't need qdrant-client installed
        mock_models = MagicMock()
        with patch(
            "noid_rag.vectorstore_qdrant._import_qdrant",
            return_value=(MagicMock(), mock_models),
        ):
            store = create_vectorstore(settings)

        from noid_rag.vectorstore_qdrant import QdrantVectorStore

        assert isinstance(store, QdrantVectorStore)

    def test_unknown_provider_raises_valueerror(self):
        from noid_rag.vectorstore_factory import create_vectorstore

        settings = Settings()
        settings = settings.model_copy(
            update={
                "vectorstore": settings.vectorstore.model_copy(update={"provider": "pgvector"}),
            }
        )
        # Monkey-patch after construction
        object.__setattr__(settings.vectorstore, "provider", "unknown")

        with pytest.raises(ValueError, match="Unknown vectorstore provider"):
            create_vectorstore(settings)

    def test_pgvector_passes_vectorstore_config(self):
        from noid_rag.vectorstore_factory import create_vectorstore

        settings = Settings()
        settings = settings.model_copy(
            update={
                "vectorstore": settings.vectorstore.model_copy(
                    update={"embedding_dim": 768, "table_name": "test_docs"}
                ),
            }
        )

        store = create_vectorstore(settings)
        assert store.config.embedding_dim == 768
        assert store.config.table_name == "test_docs"

    def test_qdrant_passes_qdrant_config_and_embedding_dim(self):
        from noid_rag.vectorstore_factory import create_vectorstore

        settings = Settings()
        settings = settings.model_copy(
            update={
                "vectorstore": settings.vectorstore.model_copy(
                    update={"provider": "qdrant", "embedding_dim": 768}
                ),
                "qdrant": settings.qdrant.model_copy(
                    update={"collection_name": "my_docs", "url": "http://qdrant:6333"}
                ),
            }
        )

        with patch(
            "noid_rag.vectorstore_qdrant._import_qdrant",
            return_value=(MagicMock(), MagicMock()),
        ):
            store = create_vectorstore(settings)

        assert store.config.collection_name == "my_docs"
        assert store.config.url == "http://qdrant:6333"
        assert store.embedding_dim == 768
