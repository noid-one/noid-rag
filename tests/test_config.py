"""Tests for configuration loading."""

import pytest
import yaml

from noid_rag.config import (
    BatchConfig,
    ChunkerConfig,
    EmbeddingConfig,
    ParserConfig,
    Settings,
    VectorStoreConfig,
    _load_yaml_config,
)


class TestParserConfig:
    def test_defaults(self):
        c = ParserConfig()
        assert c.ocr_enabled is True
        assert c.ocr_engine == "easyocr"
        assert c.max_pages == 0

    def test_custom_values(self):
        c = ParserConfig(ocr_enabled=False, ocr_engine="tesseract", max_pages=10)
        assert c.ocr_enabled is False
        assert c.ocr_engine == "tesseract"
        assert c.max_pages == 10


class TestChunkerConfig:
    def test_defaults(self):
        c = ChunkerConfig()
        assert c.method == "hybrid"
        assert c.max_tokens == 512
        assert c.tokenizer == "BAAI/bge-small-en-v1.5"
        assert c.overlap == 50

    def test_fixed_method(self):
        c = ChunkerConfig(method="fixed", max_tokens=256, overlap=25)
        assert c.method == "fixed"
        assert c.max_tokens == 256
        assert c.overlap == 25

    def test_invalid_method_rejected(self):
        with pytest.raises(Exception):
            ChunkerConfig(method="typo")


class TestEmbeddingConfig:
    def test_defaults(self):
        c = EmbeddingConfig()
        assert c.provider == "openrouter"
        assert c.api_url == "https://openrouter.ai/api/v1/embeddings"
        assert c.api_key.get_secret_value() == ""
        assert c.model == "openai/text-embedding-3-small"
        assert c.batch_size == 64


class TestVectorStoreConfig:
    def test_defaults(self):
        c = VectorStoreConfig()
        assert c.dsn == ""
        assert c.table_name == "documents"
        assert c.embedding_dim == 1536
        assert c.pool_size == 20
        assert c.hnsw_m == 16
        assert c.hnsw_ef_construction == 64


class TestBatchConfig:
    def test_defaults(self):
        c = BatchConfig()
        assert c.max_retries == 3
        assert c.retry_min_wait == 1.0
        assert c.retry_max_wait == 60.0
        assert c.continue_on_error is True
        assert c.history_dir == "~/.noid-rag/history"


class TestLoadYamlConfig:
    def test_load_from_file(self, tmp_path):
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.dump({
            "verbose": True,
            "parser": {"ocr_enabled": False},
        }))
        data = _load_yaml_config(config_file)
        assert data["verbose"] is True
        assert data["parser"]["ocr_enabled"] is False

    def test_load_missing_file_returns_empty(self, tmp_path):
        data = _load_yaml_config(tmp_path / "nonexistent.yml")
        assert data == {}

    def test_load_empty_file(self, tmp_path):
        config_file = tmp_path / "empty.yml"
        config_file.write_text("")
        data = _load_yaml_config(config_file)
        assert data == {}


class TestSettings:
    def test_default_settings(self):
        s = Settings()
        assert s.verbose is False
        assert s.config_file is None
        assert isinstance(s.parser, ParserConfig)
        assert isinstance(s.chunker, ChunkerConfig)
        assert isinstance(s.embedding, EmbeddingConfig)
        assert isinstance(s.vectorstore, VectorStoreConfig)
        assert isinstance(s.batch, BatchConfig)

    def test_load_with_no_args(self):
        s = Settings.load()
        assert isinstance(s, Settings)
        assert s.verbose is False

    def test_load_merges_overrides(self):
        s = Settings.load(verbose=True)
        assert s.verbose is True

    def test_load_ignores_none_overrides(self):
        s = Settings.load(verbose=None)
        assert s.verbose is False

    def test_load_from_yaml(self, tmp_path):
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.dump({
            "verbose": True,
            "parser": {"ocr_engine": "tesseract"},
        }))
        s = Settings.load(config_file=config_file)
        assert s.verbose is True
        assert s.parser.ocr_engine == "tesseract"
        assert s.config_file == config_file

    def test_overrides_take_precedence_over_yaml(self, tmp_path):
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.dump({"verbose": True}))
        s = Settings.load(config_file=config_file, verbose=False)
        assert s.verbose is False

    def test_env_variable_override(self, monkeypatch):
        monkeypatch.setenv("NOID_RAG_VERBOSE", "true")
        s = Settings()
        assert s.verbose is True

    def test_env_nested_override(self, monkeypatch):
        monkeypatch.setenv("NOID_RAG_PARSER__OCR_ENABLED", "false")
        s = Settings()
        assert s.parser.ocr_enabled is False
